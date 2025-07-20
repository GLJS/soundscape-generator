#!/usr/bin/env python3
"""
Optimized FastAPI server for Audio Flamingo model inference.
Includes prefetching, pipeline parallelism, and model optimizations.
"""
from dotenv import load_dotenv

load_dotenv()
import os
import sys
import time
import asyncio
import logging
from pathlib import Path
from typing import Dict, Optional, Any, Tuple
from contextlib import asynccontextmanager
from collections import OrderedDict
from datetime import datetime
import traceback
from concurrent.futures import ThreadPoolExecutor
import threading
from queue import Queue
import gc

import torch
from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn
from huggingface_hub import snapshot_download

# Add flamingo directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'flamingo'))
import llava

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global variables
app_state = {
    "model": None,
    "generation_config": None,
    "prefetch_queue": None,
    "inference_queue": None,
    "result_queue": None,
    "audio_cache": OrderedDict(),  # Cache for loaded audio
    "stats": {
        "total_requests": 0,
        "successful_requests": 0,
        "failed_requests": 0,
        "total_processing_time": 0,
        "total_loading_time": 0,
        "total_inference_time": 0,
        "cache_hits": 0,
        "prefetch_hits": 0,
        "start_time": None
    },
    "response_cache": OrderedDict(),  # LRU cache for responses
    "cache_max_size": 100,
    "audio_cache_max_size": 50  # Separate limit for audio cache
}

# Configuration
class ServerConfig:
    MODEL_NAME = os.getenv("AUDIO_FLAMINGO_MODEL", "nvidia/audio-flamingo-3")
    DEVICE = os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", 8080))
    MAX_QUEUE_SIZE = int(os.getenv("MAX_QUEUE_SIZE", 100))
    REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", 300))
    CACHE_ENABLED = os.getenv("CACHE_ENABLED", "true").lower() == "true"
    WARMUP_ON_START = os.getenv("WARMUP_ON_START", "true").lower() == "true"
    
    # Optimization settings
    USE_TORCH_COMPILE = os.getenv("USE_TORCH_COMPILE", "true").lower() == "true"
    USE_MIXED_PRECISION = os.getenv("USE_MIXED_PRECISION", "true").lower() == "true"
    PREFETCH_SIZE = int(os.getenv("PREFETCH_SIZE", 3))
    NUM_LOADING_THREADS = int(os.getenv("NUM_LOADING_THREADS", 2))

config = ServerConfig()

# Request/Response models
class GenerateRequest(BaseModel):
    audio_path: str = Field(..., description="Path to audio file")
    prompt: str = Field(..., description="Prompt for generation")
    max_new_tokens: Optional[int] = Field(1024, description="Maximum tokens to generate")
    request_id: Optional[str] = Field(None, description="Optional request ID for tracking")

class GenerateResponse(BaseModel):
    caption: str
    processing_time_ms: float
    loading_time_ms: float
    inference_time_ms: float
    request_id: Optional[str] = None
    cached: bool = False
    prefetched: bool = False

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
    queue_lengths: Dict[str, int]
    stats: Dict[str, Any]

class StatsResponse(BaseModel):
    uptime_seconds: float
    total_requests: int
    successful_requests: int
    failed_requests: int
    average_processing_time_ms: float
    average_loading_time_ms: float
    average_inference_time_ms: float
    cache_hit_rate: float
    prefetch_hit_rate: float
    current_queue_lengths: Dict[str, int]

# Audio loading with prefetching
class AudioPrefetcher:
    """Handles audio loading and prefetching in separate threads"""
    
    def __init__(self, max_size: int = 3):
        self.max_size = max_size
        self.cache = OrderedDict()
        self.lock = threading.Lock()
        self.executor = ThreadPoolExecutor(max_workers=config.NUM_LOADING_THREADS)
        self.pending_loads = set()
        
    def _load_audio_sync(self, audio_path: str) -> Tuple[Any, float]:
        """Synchronously load audio file"""
        start_time = time.time()
        try:
            sound = llava.Sound(audio_path)
            load_time = (time.time() - start_time) * 1000
            return sound, load_time
        except Exception as e:
            logger.error(f"Failed to load audio {audio_path}: {e}")
            raise
    
    async def get_audio(self, audio_path: str) -> Tuple[Any, float, bool]:
        """Get audio, either from cache or load it"""
        # Check cache first
        with self.lock:
            if audio_path in self.cache:
                self.cache.move_to_end(audio_path)  # LRU
                app_state["stats"]["prefetch_hits"] += 1
                return self.cache[audio_path], 0, True  # No load time, was prefetched
        
        # Load audio
        loop = asyncio.get_event_loop()
        sound, load_time = await loop.run_in_executor(
            self.executor, 
            self._load_audio_sync, 
            audio_path
        )
        
        # Add to cache
        with self.lock:
            self.cache[audio_path] = sound
            self.cache.move_to_end(audio_path)
            
            # Evict if too large
            while len(self.cache) > self.max_size:
                self.cache.popitem(last=False)
        
        return sound, load_time, False
    
    def prefetch_async(self, audio_path: str):
        """Prefetch audio in background"""
        with self.lock:
            if audio_path in self.cache or audio_path in self.pending_loads:
                return
            self.pending_loads.add(audio_path)
        
        def _prefetch():
            try:
                sound, _ = self._load_audio_sync(audio_path)
                with self.lock:
                    if len(self.cache) < self.max_size:
                        self.cache[audio_path] = sound
                    self.pending_loads.discard(audio_path)
            except Exception as e:
                logger.warning(f"Prefetch failed for {audio_path}: {e}")
                with self.lock:
                    self.pending_loads.discard(audio_path)
        
        self.executor.submit(_prefetch)

# Pipeline stages
class InferencePipeline:
    """Three-stage pipeline for audio processing"""
    
    def __init__(self, model, generation_config, prefetcher):
        self.model = model
        self.generation_config = generation_config
        self.prefetcher = prefetcher
        
        # Queues for pipeline stages
        self.loading_queue = asyncio.Queue(maxsize=10)
        self.inference_queue = asyncio.Queue(maxsize=5)
        self.result_queue = asyncio.Queue(maxsize=10)
        
        # Mixed precision context
        if config.USE_MIXED_PRECISION and config.DEVICE == "cuda":
            self.autocast = torch.cuda.amp.autocast
        else:
            self.autocast = lambda: torch.cuda.amp.autocast(enabled=False)
    
    async def stage1_loading(self):
        """Stage 1: Load and preprocess audio"""
        while True:
            try:
                request_data, future = await self.loading_queue.get()
                
                try:
                    # Load audio (with prefetching)
                    sound, load_time, prefetched = await self.prefetcher.get_audio(
                        request_data["audio_path"]
                    )
                    
                    # Pass to inference stage
                    await self.inference_queue.put((
                        request_data, sound, load_time, prefetched, future
                    ))
                    
                except Exception as e:
                    future.set_exception(e)
                    
            except Exception as e:
                logger.error(f"Loading stage error: {e}")
                await asyncio.sleep(0.1)
    
    async def stage2_inference(self):
        """Stage 2: Model inference"""
        while True:
            try:
                data = await self.inference_queue.get()
                request_data, sound, load_time, prefetched, future = data
                
                try:
                    start_time = time.time()
                    
                    # Check response cache
                    cache_key = get_cache_key(
                        request_data["audio_path"], 
                        request_data["prompt"]
                    )
                    cached_result = cache_get(cache_key)
                    
                    if cached_result is not None:
                        inference_time = 0
                        result = cached_result
                        cached = True
                    else:
                        # Run inference with mixed precision
                        with self.autocast():
                            full_prompt = f"<sound>\n{request_data['prompt']}"
                            
                            # Update generation config
                            self.generation_config.max_new_tokens = request_data.get(
                                "max_new_tokens", 1024
                            )
                            
                            # Generate
                            result = self.model.generate_content(
                                [sound, full_prompt], 
                                generation_config=self.generation_config
                            )
                        
                        inference_time = (time.time() - start_time) * 1000
                        
                        # Cache result
                        cache_set(cache_key, result)
                        cached = False
                    
                    # Pass to result stage
                    await self.result_queue.put((
                        result, load_time, inference_time, 
                        prefetched, cached, future
                    ))
                    
                except Exception as e:
                    future.set_exception(e)
                    
            except Exception as e:
                logger.error(f"Inference stage error: {e}")
                await asyncio.sleep(0.1)
    
    async def stage3_results(self):
        """Stage 3: Process and return results"""
        while True:
            try:
                data = await self.result_queue.get()
                result, load_time, inference_time, prefetched, cached, future = data
                
                # Update stats
                app_state["stats"]["successful_requests"] += 1
                app_state["stats"]["total_loading_time"] += load_time
                app_state["stats"]["total_inference_time"] += inference_time
                app_state["stats"]["total_processing_time"] += load_time + inference_time
                
                # Set result
                future.set_result({
                    "caption": result,
                    "loading_time_ms": load_time,
                    "inference_time_ms": inference_time,
                    "processing_time_ms": load_time + inference_time,
                    "prefetched": prefetched,
                    "cached": cached
                })
                
            except Exception as e:
                logger.error(f"Result stage error: {e}")
                await asyncio.sleep(0.1)
    
    async def add_request(self, request_data: Dict[str, Any]) -> asyncio.Future:
        """Add request to pipeline"""
        future = asyncio.Future()
        await self.loading_queue.put((request_data, future))
        return future
    
    def get_queue_lengths(self) -> Dict[str, int]:
        """Get current queue lengths"""
        return {
            "loading": self.loading_queue.qsize(),
            "inference": self.inference_queue.qsize(),
            "results": self.result_queue.qsize()
        }

# Cache implementation
def get_cache_key(audio_path: str, prompt: str) -> str:
    """Generate cache key from request parameters"""
    return f"{audio_path}:{hash(prompt)}"

def cache_get(key: str) -> Optional[str]:
    """Get value from cache"""
    if not config.CACHE_ENABLED:
        return None
    
    if key in app_state["response_cache"]:
        app_state["response_cache"].move_to_end(key)
        app_state["stats"]["cache_hits"] += 1
        return app_state["response_cache"][key]
    return None

def cache_set(key: str, value: str):
    """Set value in cache"""
    if not config.CACHE_ENABLED:
        return
    
    app_state["response_cache"][key] = value
    app_state["response_cache"].move_to_end(key)
    
    if len(app_state["response_cache"]) > app_state["cache_max_size"]:
        app_state["response_cache"].popitem(last=False)

# Model loading and optimization
async def load_model():
    """Load and optimize Audio Flamingo model"""
    try:
        logger.info(f"Loading {config.MODEL_NAME} on {config.DEVICE}...")
        model_base = snapshot_download(repo_id=config.MODEL_NAME)
        model = llava.load(model_base, model_base=None)
        model = model.to(config.DEVICE)
        model.eval()
        
        # Apply torch.compile if enabled
        if config.USE_TORCH_COMPILE and hasattr(torch, 'compile'):
            logger.info("Compiling model with torch.compile()...")
            try:
                model = torch.compile(model, mode="reduce-overhead")
                logger.info("Model compilation successful")
            except Exception as e:
                logger.warning(f"torch.compile failed (using uncompiled model): {e}")
        
        generation_config = model.default_generation_config
        if not hasattr(generation_config, 'max_new_tokens'):
            generation_config.max_new_tokens = 1024
        
        app_state["model"] = model
        app_state["generation_config"] = generation_config
        
        # Warmup model
        if config.WARMUP_ON_START:
            await warmup_model()
        
        logger.info("Model loaded successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        logger.error(traceback.format_exc())
        return False

async def warmup_model():
    """Warm up model with dummy inference"""
    try:
        logger.info("Warming up model...")
        
        # Create dummy audio path
        dummy_audio = "/tmp/dummy_audio.wav"
        dummy_prompt = "Describe this audio"
        
        # Just warm up the model internals
        with torch.no_grad():
            if config.USE_MIXED_PRECISION and config.DEVICE == "cuda":
                with torch.cuda.amp.autocast():
                    _ = app_state["model"].tokenizer(dummy_prompt, return_tensors="pt")
            else:
                _ = app_state["model"].tokenizer(dummy_prompt, return_tensors="pt")
        
        logger.info("Model warmup completed")
    except Exception as e:
        logger.warning(f"Warmup failed (non-critical): {e}")

# Lifespan management
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle"""
    # Startup
    logger.info("Starting Optimized Audio Flamingo server...")
    
    # Initialize state
    app_state["stats"]["start_time"] = datetime.now()
    
    # Load model
    success = await load_model()
    if not success:
        raise RuntimeError("Failed to load model")
    
    # Initialize prefetcher and pipeline
    prefetcher = AudioPrefetcher(max_size=config.PREFETCH_SIZE)
    pipeline = InferencePipeline(
        app_state["model"], 
        app_state["generation_config"],
        prefetcher
    )
    
    app_state["prefetcher"] = prefetcher
    app_state["pipeline"] = pipeline
    
    # Start pipeline stages
    stage_tasks = [
        asyncio.create_task(pipeline.stage1_loading()),
        asyncio.create_task(pipeline.stage2_inference()),
        asyncio.create_task(pipeline.stage3_results())
    ]
    
    # Periodic tasks
    async def clear_cache_periodically():
        while True:
            await asyncio.sleep(300)  # Every 5 minutes
            if config.DEVICE == "cuda":
                torch.cuda.empty_cache()
                gc.collect()
                logger.info("Cleared GPU cache and ran garbage collection")
    
    cache_task = asyncio.create_task(clear_cache_periodically())
    
    yield
    
    # Shutdown
    logger.info("Shutting down Optimized Audio Flamingo server...")
    
    # Cancel tasks
    for task in stage_tasks:
        task.cancel()
    cache_task.cancel()
    
    # Cleanup
    if app_state["model"] is not None:
        del app_state["model"]
        if config.DEVICE == "cuda":
            torch.cuda.empty_cache()

# Create FastAPI app
app = FastAPI(
    title="Audio Flamingo Inference Server (Optimized)",
    version="2.0.0",
    description="Optimized HTTP server with prefetching and pipeline parallelism",
    lifespan=lifespan
)

# Endpoints
@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest, fastapi_request: Request):
    """Generate caption for audio file"""
    app_state["stats"]["total_requests"] += 1
    
    # Validate audio file exists
    if not Path(request.audio_path).exists():
        raise HTTPException(status_code=404, detail=f"Audio file not found: {request.audio_path}")
    
    # Add to pipeline
    try:
        future = await app_state["pipeline"].add_request({
            "audio_path": request.audio_path,
            "prompt": request.prompt,
            "max_new_tokens": request.max_new_tokens
        })
        
        # Prefetch next files if provided in header
        prefetch_header = fastapi_request.headers.get("X-Prefetch-Paths")
        if prefetch_header:
            paths = [p.strip() for p in prefetch_header.split(",")]
            for path in paths[:3]:  # Limit to 3 prefetches
                if Path(path).exists():
                    app_state["prefetcher"].prefetch_async(path)
        
        # Wait for result with timeout
        result = await asyncio.wait_for(future, timeout=config.REQUEST_TIMEOUT)
        
        return GenerateResponse(
            caption=result["caption"],
            processing_time_ms=result["processing_time_ms"],
            loading_time_ms=result["loading_time_ms"],
            inference_time_ms=result["inference_time_ms"],
            request_id=request.request_id,
            cached=result["cached"],
            prefetched=result["prefetched"]
        )
        
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Request timeout")
    except Exception as e:
        app_state["stats"]["failed_requests"] += 1
        logger.error(f"Generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint"""
    uptime = (datetime.now() - app_state["stats"]["start_time"]).total_seconds()
    
    return HealthResponse(
        status="healthy" if app_state["model"] is not None else "unhealthy",
        model_loaded=app_state["model"] is not None,
        device=config.DEVICE,
        queue_lengths=app_state["pipeline"].get_queue_lengths(),
        stats={
            "uptime_seconds": uptime,
            "total_requests": app_state["stats"]["total_requests"],
            "successful_requests": app_state["stats"]["successful_requests"],
            "failed_requests": app_state["stats"]["failed_requests"],
            "prefetch_size": config.PREFETCH_SIZE,
            "torch_compile": config.USE_TORCH_COMPILE,
            "mixed_precision": config.USE_MIXED_PRECISION
        }
    )

@app.get("/stats", response_model=StatsResponse)
async def stats():
    """Get detailed server statistics"""
    stats = app_state["stats"]
    uptime = (datetime.now() - stats["start_time"]).total_seconds()
    
    # Calculate averages
    avg_processing = 0
    avg_loading = 0
    avg_inference = 0
    
    if stats["successful_requests"] > 0:
        avg_processing = stats["total_processing_time"] / stats["successful_requests"]
        avg_loading = stats["total_loading_time"] / stats["successful_requests"]
        avg_inference = stats["total_inference_time"] / stats["successful_requests"]
    
    cache_hit_rate = 0
    if stats["total_requests"] > 0:
        cache_hit_rate = stats["cache_hits"] / stats["total_requests"]
    
    prefetch_hit_rate = 0
    if stats["successful_requests"] > 0:
        prefetch_hit_rate = stats["prefetch_hits"] / stats["successful_requests"]
    
    return StatsResponse(
        uptime_seconds=uptime,
        total_requests=stats["total_requests"],
        successful_requests=stats["successful_requests"],
        failed_requests=stats["failed_requests"],
        average_processing_time_ms=avg_processing,
        average_loading_time_ms=avg_loading,
        average_inference_time_ms=avg_inference,
        cache_hit_rate=cache_hit_rate,
        prefetch_hit_rate=prefetch_hit_rate,
        current_queue_lengths=app_state["pipeline"].get_queue_lengths()
    )

@app.post("/prefetch")
async def prefetch(paths: list[str]):
    """Prefetch audio files for upcoming requests"""
    prefetcher = app_state["prefetcher"]
    
    for path in paths:
        if Path(path).exists():
            prefetcher.prefetch_async(path)
    
    return {"status": "prefetching", "count": len(paths)}

# Main entry point
if __name__ == "__main__":
    uvicorn.run(
        "audio_flamingo_server_optimized:app",
        host=config.HOST,
        port=config.PORT,
        log_level="info",
        workers=1,
        timeout_keep_alive=300
    )