#!/usr/bin/env python3
"""
Clean SGLang implementation for audio captioning with hybrid streaming.
"""
from dotenv import load_dotenv
load_dotenv()

import os
import requests
import json
import torch
import librosa
import soundfile as sf
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
import numpy as np
from tqdm import tqdm
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from torch.utils.data import Dataset, DataLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AudioTextDataset(Dataset):
    """Dataset for audio-text pairs with dual sample rate support."""
    
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
        self.texts = texts
        self.metadata = metadata or [{} for _ in texts]
        self.system_prompt = system_prompt
        self.user_prompt_template = user_prompt_template or "{text}"
        self.audio_paths = audio_paths or [None] * len(texts)
        self.sample_rate = sample_rate
        self.load_dual_sample_rates = load_dual_sample_rates
        self.output_sample_rate = output_sample_rate
        self.skipped_count = 0
            
        assert len(self.audio_paths) == len(self.texts) == len(self.metadata)
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        item = {
            'audio_path': self.audio_paths[idx],
            'text': self.texts[idx],
            'metadata': self.metadata[idx],
            'messages': []
        }
        
        # Build prompt in dataset
        text = self.texts[idx]
        meta = self.metadata[idx]
        
        # Build context
        context_parts = []
        if meta.get('original_filename'):
            context_parts.append(f"Filename: {meta['original_filename']}")
        if meta.get('original_description'):
            context_parts.append(f"Description: {meta['original_description']}")
        if text:
            context_parts.append(text)
        
        context = " | ".join(context_parts) if context_parts else "audio file"
        
        # Build full prompt
        prompt = ""
        if self.system_prompt:
            prompt += self.system_prompt + "\n\n"
        
        prompt += self.user_prompt_template.format(text=context)
        
        item['prompt'] = prompt
        
        # Load audio at dual sample rates if needed
        if self.load_dual_sample_rates and self.audio_paths[idx]:
            audio_path = self.audio_paths[idx]
            if audio_path and os.path.exists(audio_path):
                try:
                    audio, sr = librosa.load(audio_path, sr=None, mono=True)
                    
                    # Skip long files
                    if len(audio) / sr > 300:
                        logger.debug(f"Skipping long audio: {audio_path}")
                        return None
                    
                    # Resample for output
                    if sr != self.output_sample_rate:
                        audio_output = librosa.resample(audio, orig_sr=sr, target_sr=self.output_sample_rate)
                    else:
                        audio_output = audio
                    
                    item['audio_path'] = audio_path
                    item['audio_output'] = audio_output.astype(np.float32)
                except Exception as e:
                    logger.error(f"Error loading {audio_path}: {e}")
                    return None
        
        return item


def collate_batch(batch: List[Dict], processor=None) -> Dict[str, Any]:
    """Collate function that filters out None items."""
    # Filter out None items
    valid_items = [item for item in batch if item is not None]
    
    if not valid_items:
        return None
    
    result = {
        'audio_paths': [item['audio_path'] for item in valid_items],
        'texts': [item['text'] for item in valid_items],
        'metadata': [item['metadata'] for item in valid_items],
        'messages': [item.get('messages', []) for item in valid_items],
        'prompts': [item['prompt'] for item in valid_items],
        'audio_outputs': [item['audio_output'] for item in valid_items if 'audio_output' in item]
    }

    
    return result


class GemmaProcessor:
    """SGLang processor for Gemma model."""
    
    def __init__(
        self, 
        model_name: str = "google/gemma-3n-e4b-it",
        server_url: str = "http://127.0.0.1:30000",
        generation_config: Optional[Dict] = None,
        system_prompt: Optional[str] = None,
        user_prompt_template: Optional[str] = None,
        **kwargs
    ):
        self.model_name = model_name
        self.server_url = server_url
        self.system_prompt = system_prompt
        self.user_prompt_template = user_prompt_template
        self.generation_config = generation_config or {
            "max_new_tokens": 100,
            "temperature": 0.7,
            "top_p": 0.9,
        }
        
        # Compatibility attributes
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.bfloat16
        self.processor = None
        
        # Check server
        self._check_server()
        
    def _check_server(self):
        """Verify SGLang server is running."""
        try:
            response = requests.get(f"{self.server_url}/v1/models", timeout=120)
            if response.status_code == 200:
                logger.info(f"SGLang server ready: {response.json()}")
            else:
                raise ConnectionError(f"Server returned {response.status_code}")
        except Exception as e:
            raise ConnectionError(
                f"Cannot connect to SGLang server at {self.server_url}\n"
                f"Start server with: python -m sglang.launch_server --model-path {self.model_name} --attention-backend fa3"
            )
    
    def generate_batch(self, batch: Dict[str, Any]) -> List[str]:
        """Generate captions for a batch."""
        if batch is None:
            return []
            
        # Use pre-built prompts from dataset
        prompts = batch.get('prompts', [])
        audio_paths = batch.get('audio_paths', [])
        
        if not prompts:
            logger.warning("No prompts found in batch")
            return {k: [] for k in batch.keys()}, []
        
        # Generate responses
        try:
            response = requests.post(
                f"{self.server_url}/generate",
                json={"text": prompts, "audio_data": audio_paths, "sampling_params": self.generation_config},
                timeout=120
            )
            
            if response.status_code != 200:
                logger.error(f"Generation failed: {response.status_code}")
                return {k: [] for k in batch.keys()}, []
            
            # Filter successful results
            results = response.json()
            good_indices = [i for i, r in enumerate(results) if r.get("text", "").strip()]
            
            if not good_indices:
                return {k: [] for k in batch.keys()}, []
            
            # Filter batch to only successful items
            filtered_batch = {k: [v[i] for i in good_indices] for k, v in batch.items()}
            successful_results = [results[i].get("text", "").strip() for i in good_indices]
            
            return filtered_batch, successful_results
            
        except Exception as e:
            logger.error(f"Generation error: {e}")
            return {k: [] for k in batch.keys()}, []