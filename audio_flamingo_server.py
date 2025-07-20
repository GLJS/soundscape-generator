#!/usr/bin/env python3
"""
Optimized FastAPI server for Audio Flamingo model inference.
Provides HTTP API for audio captioning with single-item processing.
"""
from dotenv import load_dotenv

load_dotenv()
import os
import sys
import time
import asyncio
import logging
from pathlib import Path
from typing import Dict, Optional, Any
from contextlib import asynccontextmanager
from collections import OrderedDict
from datetime import datetime
import traceback

import torch
from fastapi import FastAPI, HTTPException, BackgroundTasks
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
    "request_queue": None,
    "processing_lock": None,
    "stats": {
        "total_requests": 0,
        "successful_requests": 0,
        "failed_requests": 0,
        "total_processing_time": 0,
        "cache_hits": 0,
        "start_time": None
    },
    "cache": OrderedDict(),  # LRU cache for responses
    "cache_max_size": 100
}

# Configuration
class ServerConfig:
    MODEL_NAME = os.getenv("AUDIO_FLAMINGO_MODEL", "nvidia/audio-flamingo-3")
    DEVICE = os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", 8080))
    MAX_QUEUE_SIZE = int(os.getenv("MAX_QUEUE_SIZE", 100))
    REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", 300))  # 5 minutes
    CACHE_ENABLED = os.getenv("CACHE_ENABLED", "true").lower() == "true"
    WARMUP_ON_START = os.getenv("WARMUP_ON_START", "true").lower() == "true"

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
    request_id: Optional[str] = None
    cached: bool = False

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
    queue_length: int
    stats: Dict[str, Any]

class StatsResponse(BaseModel):
    uptime_seconds: float
    total_requests: int
    successful_requests: int
    failed_requests: int
    average_processing_time_ms: float
    cache_hit_rate: float
    current_queue_length: int

# Async queue for request processing
class RequestQueue:
    def __init__(self, max_size: int):
        self.queue = asyncio.Queue(maxsize=max_size)
        self.processing = False
        
    async def add_request(self, request_data: Dict[str, Any]) -> asyncio.Future:
        """Add request to queue and return future for result"""
        future = asyncio.Future()
        await self.queue.put((request_data, future))
        return future
    
    def get_length(self) -> int:
        """Get current queue length"""
        return self.queue.qsize()

# Cache implementation
def get_cache_key(audio_path: str, prompt: str) -> str:
    """Generate cache key from request parameters"""
    return f"{audio_path}:{hash(prompt)}"

def cache_get(key: str) -> Optional[str]:
    """Get value from cache"""
    if not config.CACHE_ENABLED:
        return None
    
    if key in app_state["cache"]:
        # Move to end (LRU)
        app_state["cache"].move_to_end(key)
        app_state["stats"]["cache_hits"] += 1
        return app_state["cache"][key]
    return None

def cache_set(key: str, value: str):
    """Set value in cache"""
    if not config.CACHE_ENABLED:
        return
    
    app_state["cache"][key] = value
    app_state["cache"].move_to_end(key)
    
    # Evict oldest if cache is too large
    if len(app_state["cache"]) > app_state["cache_max_size"]:
        app_state["cache"].popitem(last=False)

# Model loading and inference
async def load_model():
    """Load Audio Flamingo model"""
    try:
        logger.info(f"Loading {config.MODEL_NAME} on {config.DEVICE}...")
        model_base = snapshot_download(repo_id=config.MODEL_NAME)
        model = llava.load(model_base, model_base=None)
        model = model.to(config.DEVICE)
        
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
    """Warm up model with a dummy request"""
    try:
        logger.info("Warming up model...")
        # Create a small dummy audio file path (won't actually load it for warmup)
        dummy_prompt = "Describe this audio"
        
        # Just ensure model is ready without actual inference
        _ = app_state["model"].tokenizer
        logger.info("Model warmup completed")
    except Exception as e:
        logger.warning(f"Warmup failed (non-critical): {e}")

async def process_request(request_data: Dict[str, Any]) -> str:
    """Process a single request using the model"""
    audio_path = request_data["audio_path"]
    prompt = request_data["prompt"]
    max_new_tokens = request_data.get("max_new_tokens", 1024)
    
    # Check cache first
    cache_key = get_cache_key(audio_path, prompt)
    cached_result = cache_get(cache_key)
    if cached_result is not None:
        return cached_result
    
    # Load audio and generate
    try:
        sound = llava.Sound(audio_path)
        full_prompt = f"<sound>\n{prompt}"
        
        # Update generation config if needed
        generation_config = app_state["generation_config"]
        generation_config.max_new_tokens = max_new_tokens
        
        # Generate content
        response = app_state["model"].generate_content(
            [sound, full_prompt], 
            generation_config=generation_config
        )
        
        # Cache result
        cache_set(cache_key, response)
        
        return response
    except Exception as e:
        logger.error(f"Error processing request: {e}")
        raise

# Background task to process queue
async def process_queue():
    """Continuously process requests from the queue"""
    while True:
        try:
            # Get request from queue
            request_data, future = await app_state["request_queue"].queue.get()
            
            # Process with lock (ensure single GPU usage)
            async with app_state["processing_lock"]:
                start_time = time.time()
                try:
                    # Check if already cached
                    cache_key = get_cache_key(
                        request_data["audio_path"], 
                        request_data["prompt"]
                    )
                    cached_result = cache_get(cache_key)
                    
                    if cached_result is not None:
                        result = cached_result
                        cached = True
                    else:
                        result = await process_request(request_data)
                        cached = False
                    
                    processing_time = (time.time() - start_time) * 1000
                    
                    # Update stats
                    app_state["stats"]["successful_requests"] += 1
                    app_state["stats"]["total_processing_time"] += processing_time
                    
                    # Set result
                    future.set_result({
                        "caption": result,
                        "processing_time_ms": processing_time,
                        "cached": cached
                    })
                    
                except Exception as e:
                    app_state["stats"]["failed_requests"] += 1
                    future.set_exception(e)
                    
        except Exception as e:
            logger.error(f"Queue processing error: {e}")
            await asyncio.sleep(1)  # Prevent tight loop on error

# Lifespan management
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle"""
    # Startup
    logger.info("Starting Audio Flamingo server...")
    
    # Initialize state
    app_state["request_queue"] = RequestQueue(config.MAX_QUEUE_SIZE)
    app_state["processing_lock"] = asyncio.Lock()
    app_state["stats"]["start_time"] = datetime.now()
    
    # Load model
    success = await load_model()
    if not success:
        raise RuntimeError("Failed to load model")
    
    # Start queue processor
    queue_task = asyncio.create_task(process_queue())
    
    # Clear GPU cache periodically
    async def clear_cache_periodically():
        while True:
            await asyncio.sleep(30000)  # Every 5 minutes
            if config.DEVICE == "cuda":
                torch.cuda.empty_cache()
                logger.info("Cleared GPU cache")
    
    cache_task = asyncio.create_task(clear_cache_periodically())
    
    yield
    
    # Shutdown
    logger.info("Shutting down Audio Flamingo server...")
    queue_task.cancel()
    cache_task.cancel()
    
    # Clear model from memory
    if app_state["model"] is not None:
        del app_state["model"]
        if config.DEVICE == "cuda":
            torch.cuda.empty_cache()

# Create FastAPI app
app = FastAPI(
    title="Audio Flamingo Inference Server",
    version="1.0.0",
    description="Optimized HTTP server for Audio Flamingo model inference",
    lifespan=lifespan
)

# Endpoints
@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    """Generate caption for audio file"""
    app_state["stats"]["total_requests"] += 1
    
    # Validate audio file exists
    if not Path(request.audio_path).exists():
        raise HTTPException(status_code=404, detail=f"Audio file not found: {request.audio_path}")
    
    # Add to queue
    try:
        future = await app_state["request_queue"].add_request({
            "audio_path": request.audio_path,
            "prompt": request.prompt,
            "max_new_tokens": request.max_new_tokens
        })
        
        # Wait for result with timeout
        result = await asyncio.wait_for(future, timeout=config.REQUEST_TIMEOUT)
        
        return GenerateResponse(
            caption=result["caption"],
            processing_time_ms=result["processing_time_ms"],
            request_id=request.request_id,
            cached=result["cached"]
        )
        
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Request timeout")
    except Exception as e:
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
        queue_length=app_state["request_queue"].get_length(),
        stats={
            "uptime_seconds": uptime,
            "total_requests": app_state["stats"]["total_requests"],
            "successful_requests": app_state["stats"]["successful_requests"],
            "failed_requests": app_state["stats"]["failed_requests"]
        }
    )

@app.get("/stats", response_model=StatsResponse)
async def stats():
    """Get server statistics"""
    stats = app_state["stats"]
    uptime = (datetime.now() - stats["start_time"]).total_seconds()
    
    # Calculate averages
    avg_time = 0
    if stats["successful_requests"] > 0:
        avg_time = stats["total_processing_time"] / stats["successful_requests"]
    
    cache_hit_rate = 0
    if stats["total_requests"] > 0:
        cache_hit_rate = stats["cache_hits"] / stats["total_requests"]
    
    return StatsResponse(
        uptime_seconds=uptime,
        total_requests=stats["total_requests"],
        successful_requests=stats["successful_requests"],
        failed_requests=stats["failed_requests"],
        average_processing_time_ms=avg_time,
        cache_hit_rate=cache_hit_rate,
        current_queue_length=app_state["request_queue"].get_length()
    )

# Main entry point
if __name__ == "__main__":
    uvicorn.run(
        "audio_flamingo_server:app",
        host=config.HOST,
        port=config.PORT,
        log_level="info",
        # Single worker since we're doing GPU processing
        workers=1,
        # Increase timeout for long audio files
        timeout_keep_alive=300
    )