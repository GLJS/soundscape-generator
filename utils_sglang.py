#!/usr/bin/env python3
"""
SGLang-based generation utilities with audio support for Gemma multimodal models.
Uses SGLang's native API with audio_data parameter for multimodal generation.
"""
from dotenv import load_dotenv

load_dotenv()
import os
import requests
import json
import torch
import base64
import librosa
import soundfile as sf
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any, Union
import numpy as np
from tqdm import tqdm
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from torch.utils.data import Dataset, DataLoader

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AudioTextDataset(Dataset):
    """Dataset for audio-text pairs compatible with SGLang."""
    
    def __init__(
        self,
        texts: List[str],
        metadata: Optional[List[Dict]] = None,
        system_prompt: Optional[str] = None,
        user_prompt_template: Optional[str] = None,
        audio_paths: Optional[List[str]] = None,
        sample_rate: int = 16000,
        load_dual_sample_rates: bool = False,
        output_sample_rate: int = 48000
    ):
        """
        Initialize dataset compatible with the Gemma interface.
        
        Args:
            texts: List of input texts to process
            metadata: Optional list of metadata dicts for each text
            system_prompt: Optional system prompt to prepend
            user_prompt_template: Template for user prompts (use {text} placeholder)
            audio_paths: Optional list of audio file paths for multimodal input
            sample_rate: Sample rate for audio loading (16kHz for Gemma)
            load_dual_sample_rates: If True, load audio at both sample_rate and output_sample_rate
            output_sample_rate: Sample rate for output audio (typically 48kHz for FLAC)
        """
        self.texts = texts
        self.metadata = metadata or [{} for _ in texts]
        self.system_prompt = system_prompt or "You are a helpful assistant."
        self.user_prompt_template = user_prompt_template or "{text}"
        self.audio_paths = audio_paths
        self.sample_rate = sample_rate
        self.load_dual_sample_rates = load_dual_sample_rates
        self.output_sample_rate = output_sample_rate
        self.skipped_count = 0
        
        # Handle text-only mode
        if not self.audio_paths:
            self.audio_paths = [None] * len(self.texts)
            
        assert len(self.audio_paths) == len(self.texts) == len(self.metadata)
    
    def __len__(self):
        return len(self.audio_paths)
    
    def __getitem__(self, idx):
        item = {
            'audio_path': self.audio_paths[idx],
            'text': self.texts[idx],
            'metadata': self.metadata[idx],
            'messages': []  # For compatibility
        }
        
        # If we need dual sample rates, load audio here
        if self.load_dual_sample_rates and self.audio_paths[idx]:
            audio_path = self.audio_paths[idx]
            if audio_path and os.path.exists(audio_path):
                try:
                    # Load audio at original sample rate
                    audio, sr = librosa.load(audio_path, sr=None, mono=True)
                    
                    # Resample to both rates
                    if sr != self.output_sample_rate:
                        audio_output = librosa.resample(audio, orig_sr=sr, target_sr=self.output_sample_rate)
                    else:
                        audio_output = audio
                    
                    if sr != self.sample_rate:
                        audio_gemma = librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate)
                    else:
                        audio_gemma = audio
                    
                    item['audio'] = audio_gemma.astype(np.float32)
                    item['audio_output'] = audio_output.astype(np.float32)
                except Exception as e:
                    logger.error(f"Error loading audio {audio_path}: {e}")
                    item['audio'] = None
                    item['audio_output'] = None
        
        return item


def collate_batch(batch: List[Dict]) -> Dict[str, Any]:
    """Collate function for DataLoader."""
    result = {
        'audio_paths': [item['audio_path'] for item in batch],
        'texts': [item['text'] for item in batch],
        'metadata': [item['metadata'] for item in batch],
        'messages': [item.get('messages', []) for item in batch]
    }
    
    # Handle audio outputs if present
    if 'audio_output' in batch[0]:
        result['audio_outputs'] = [item.get('audio_output') for item in batch if item.get('audio_output') is not None]
        result['audios'] = [item.get('audio') for item in batch if item.get('audio') is not None]
    
    return result


class SGLangProcessor:
    """SGLang processor with audio support for Gemma multimodal models."""
    
    def __init__(
        self, 
        model_name: str = "google/gemma-3n-e4b-it",
        server_url: str = "http://127.0.0.1:30000",
        generation_config: Optional[Dict] = None,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        sample_rate: int = 16000
    ):
        """
        Initialize the SGLang processor with audio support.
        
        Args:
            model_name: Model name (for compatibility)
            server_url: URL of the SGLang server
            generation_config: Custom generation configuration
            max_retries: Maximum number of retries for failed requests
            retry_delay: Delay between retries in seconds
            sample_rate: Sample rate for audio processing
        """
        self.model_name = model_name
        self.server_url = server_url
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.sample_rate = sample_rate
        
        # Generation configuration
        self.generation_config = generation_config or {
            "max_new_tokens": 100,
            "temperature": 0.7,
            "top_p": 0.9,
        }
        
        # Check server availability
        self._check_server()
        
        logger.info(f"SGLang processor initialized with server at {server_url}")
        
    def _check_server(self):
        """Check if SGLang server is available."""
        try:
            response = requests.get(f"{self.server_url}/v1/models", timeout=5)
            if response.status_code == 200:
                models = response.json()
                logger.info(f"SGLang server is available. Models: {models}")
            else:
                raise ConnectionError(f"SGLang server returned status {response.status_code}")
        except Exception as e:
            raise ConnectionError(
                f"Cannot connect to SGLang server at {self.server_url}. "
                f"Please ensure the server is running with:\n"
                f"python -m sglang.launch_server --model-path {self.model_name} --attention-backend fa3\n"
                f"Error: {e}"
            )
    
    def _load_and_encode_audio(self, audio_path: str) -> str:
        """Load audio file and encode to base64."""
        try:
            # Load audio with librosa
            audio, sr = librosa.load(audio_path, sr=self.sample_rate)
            
            # Convert to wav format in memory
            import io
            buffer = io.BytesIO()
            sf.write(buffer, audio, self.sample_rate, format='WAV')
            buffer.seek(0)
            
            # Encode to base64
            audio_base64 = base64.b64encode(buffer.read()).decode('utf-8')
            return audio_base64
        except Exception as e:
            logger.error(f"Error loading audio {audio_path}: {e}")
            return None
    
    def _send_request(self, text: str, audio_path: Optional[str] = None, metadata: Optional[Dict] = None) -> Optional[str]:
        """Send a single request to SGLang server with optional audio."""
        # Prepare the base payload
        payload = {
            "text": text,
            "sampling_params": {
                "max_new_tokens": self.generation_config.get("max_new_tokens", 100),
                "temperature": self.generation_config.get("temperature", 0.7),
                "top_p": self.generation_config.get("top_p", 0.9),
            }
        }
        
        # Add audio data if available
        if audio_path and audio_path != 'None' and os.path.exists(audio_path):
            audio_base64 = self._load_and_encode_audio(audio_path)
            if audio_base64:
                payload["audio_data"] = audio_base64
            else:
                # Fallback to file path
                payload["audio_data"] = audio_path
        
        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    f"{self.server_url}/generate",
                    json=payload,
                    timeout=60  # Increased timeout for audio processing
                )
                
                if response.status_code == 200:
                    result = response.json()
                    return result.get("text", "").strip()
                else:
                    logger.warning(f"Request failed with status {response.status_code}: {response.text}")
                    
            except Exception as e:
                logger.warning(f"Request failed (attempt {attempt + 1}/{self.max_retries}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
        
        return None
    
    def generate_batch(self, batch: Dict[str, Any]) -> List[str]:
        """
        Generate text for a batch of inputs with audio support.
        
        Args:
            batch: Batch dict from collate_batch containing audio_paths, texts, and metadata
            
        Returns:
            List of generated texts
        """
        audio_paths = batch.get('audio_paths', [])
        texts = batch.get('texts', [])
        metadata = batch.get('metadata', [])
        
        # Prepare prompts with audio context
        requests_data = []
        for i, (audio_path, text, meta) in enumerate(zip(audio_paths, texts, metadata)):
            # Build the text prompt
            prompt_parts = []
            
            # Add system context if available
            if hasattr(self, 'system_prompt') and self.system_prompt:
                prompt_parts.append(self.system_prompt)
            
            # Add metadata context
            if meta.get('original_description'):
                prompt_parts.append(f"Audio description: {meta['original_description']}")
            if meta.get('original_filename'):
                prompt_parts.append(f"Filename: {meta['original_filename']}")
            
            # Add user prompt
            if hasattr(self, 'user_prompt_template') and self.user_prompt_template:
                user_prompt = self.user_prompt_template.format(text=text or "audio file")
            else:
                user_prompt = text or "Describe this audio in detail."
            
            prompt_parts.append(user_prompt)
            
            full_prompt = "\n".join(prompt_parts)
            
            requests_data.append({
                'text': full_prompt,
                'audio_path': audio_path,
                'metadata': meta
            })
        
        # Send requests in parallel for better throughput
        generated_texts = []
        with ThreadPoolExecutor(max_workers=min(len(requests_data), 10)) as executor:
            future_to_idx = {
                executor.submit(
                    self._send_request, 
                    req['text'], 
                    req['audio_path'], 
                    req['metadata']
                ): i 
                for i, req in enumerate(requests_data)
            }
            
            results = [None] * len(requests_data)
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    result = future.result()
                    results[idx] = result if result else "[Generation failed]"
                except Exception as e:
                    logger.error(f"Error generating text for index {idx}: {e}")
                    results[idx] = "[Generation error]"
            
            generated_texts = results
        
        return generated_texts


# Compatibility wrapper to match the original interface
class GemmaProcessor(SGLangProcessor):
    """Compatibility wrapper to use SGLang with the GemmaProcessor interface."""
    
    def __init__(self, *args, **kwargs):
        # Extract SGLang-specific parameters
        server_url = kwargs.pop('server_url', 'http://127.0.0.1:30000')
        
        # Extract prompts if provided
        self.system_prompt = kwargs.pop('system_prompt', None)
        self.user_prompt_template = kwargs.pop('user_prompt_template', None)
        
        # Remove parameters not used by SGLang
        kwargs.pop('device', None)
        kwargs.pop('torch_dtype', None)
        
        super().__init__(server_url=server_url, *args, **kwargs)
        
        # Add dummy attributes for compatibility
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.bfloat16
        self.processor = None  # Not used in SGLang version


def process_batch_with_gemma(
    batch: Dict[str, Any],
    processor: GemmaProcessor,
    system_prompt: Optional[str] = None,
    user_prompt_template: Optional[str] = None
) -> List[str]:
    """
    Process a single batch with the Gemma processor.
    
    Args:
        batch: Batch dict from collate_batch
        processor: GemmaProcessor instance
        system_prompt: System prompt for the model
        user_prompt_template: Template for user prompts
        
    Returns:
        List of generated texts
    """
    # Set prompts on processor if provided
    if system_prompt:
        processor.system_prompt = system_prompt
    if user_prompt_template:
        processor.user_prompt_template = user_prompt_template
    
    return processor.generate_batch(batch)


def process_dataset_with_gemma(
    texts: Optional[List[str]] = None,
    metadata: Optional[List[Dict]] = None,
    audio_paths: Optional[List[str]] = None,
    batch_size: int = 128,
    num_workers: int = 12,
    model_name: str = "google/gemma-3n-e4b-it",
    system_prompt: Optional[str] = None,
    user_prompt_template: Optional[str] = None,
    generation_config: Optional[Dict] = None,
    show_progress: bool = True,
    sample_rate: int = 16000,
    server_url: str = "http://127.0.0.1:30000"
) -> List[Tuple[str, str, Dict]]:
    """
    Process dataset with SGLang server supporting audio inputs.
    
    Args:
        texts: List of input texts
        metadata: Optional list of metadata dicts
        audio_paths: Optional list of audio file paths
        batch_size: Batch size for processing
        num_workers: Number of workers for DataLoader
        model_name: Model name (for compatibility)
        system_prompt: System prompt for the model
        user_prompt_template: Template for user prompts
        generation_config: Custom generation configuration
        show_progress: Whether to show progress bar
        sample_rate: Sample rate for audio processing
        server_url: URL of the SGLang server
        
    Returns:
        List of tuples (input, generated_text, metadata)
    """
    # Validate inputs
    if texts is None and audio_paths is None:
        raise ValueError("Either texts or audio_paths must be provided")
    
    # Default texts if using audio
    if audio_paths is not None and texts is None:
        texts = [""] * len(audio_paths)
    
    # Initialize processor
    processor = GemmaProcessor(
        model_name=model_name,
        generation_config=generation_config,
        server_url=server_url,
        sample_rate=sample_rate,
        system_prompt=system_prompt,
        user_prompt_template=user_prompt_template
    )
    
    # Create dataset and dataloader
    dataset = AudioTextDataset(
        audio_paths=audio_paths,
        texts=texts,
        metadata=metadata,
        sample_rate=sample_rate
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,  # Set to 0 for simplicity with audio loading
        collate_fn=collate_batch
    )
    
    # Process batches
    results = []
    iterator = dataloader
    if show_progress:
        iterator = tqdm(iterator, desc="Processing batches")
    
    for batch in iterator:
        # Generate outputs
        generated_texts = process_batch_with_gemma(
            batch, processor, system_prompt, user_prompt_template
        )
        
        # Collect results
        for audio_path, text, generated, meta in zip(
            batch['audio_paths'], batch['texts'], generated_texts, batch['metadata']
        ):
            input_ref = audio_path if audio_paths else text
            results.append((input_ref, generated, meta))
    
    return results


# Example usage
if __name__ == "__main__":
    # Example: Process audio files with text prompts
    sample_audio_paths = [
        "/path/to/dog_bark.wav",
        "/path/to/piano_melody.mp3",
        "/path/to/thunderstorm.flac"
    ]
    
    sample_texts = [
        "Dog barking sound",
        "Piano music",
        "Thunder and rain"
    ]
    
    sample_metadata = [
        {"original_filename": "dog_bark.wav", "duration": "3s"},
        {"original_filename": "piano_melody.mp3", "duration": "30s"},
        {"original_filename": "thunderstorm.flac", "duration": "60s"}
    ]
    
    try:
        results = process_dataset_with_gemma(
            audio_paths=sample_audio_paths,
            texts=sample_texts,
            metadata=sample_metadata,
            system_prompt="You are an expert audio analyst. Describe the audio content in detail.",
            user_prompt_template="Listen to this audio and provide a comprehensive description: {text}",
            batch_size=2,
            generation_config={
                "max_new_tokens": 150,
                "temperature": 0.7,
            }
        )
        
        for input_path, generated, meta in results:
            print(f"Audio: {input_path}")
            print(f"Metadata: {meta}")
            print(f"Generated: {generated}")
            print("-" * 80)
    except ConnectionError as e:
        print(f"Error: {e}")
        print("Please start the SGLang server first.")
        print("Example command:")
        print("python -m sglang.launch_server --model-path google/gemma-3n-e4b-it --attention-backend fa3")