#!/usr/bin/env python3
"""
LLM-based recaptioning utilities for audio datasets.
Provides efficient batch processing with multi-worker support.
"""
from dotenv import load_dotenv

load_dotenv()
import os
import sys
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
import pandas as pd
from tqdm import tqdm
import json
from huggingface_hub import snapshot_download
from functools import partial
import httpx
import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
import logging

# Add flamingo directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'flamingo'))
import llava
from llava.utils.media import extract_media
from llava.utils.tokenizer import tokenize_conversation
from llava.mm_utils import process_sounds, process_sound_masks
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AudioRecaptionDataset(Dataset):
    """Dataset for audio recaptioning tasks."""
    
    def __init__(self, audio_paths: List[Path], metadata: List[Dict], 
                 labels_df: Optional[pd.DataFrame] = None, recaptioner=None):
        """
        Args:
            audio_paths: List of paths to audio files
            metadata: List of metadata dicts corresponding to each audio
            labels_df: Optional DataFrame containing labels for prompt generation
            recaptioner: LLMRecaptioner instance for prompt generation
        """
        self.audio_paths = audio_paths
        self.metadata = metadata
        self.labels_df = labels_df
        self.recaptioner = recaptioner
        
    def __len__(self):
        return len(self.audio_paths)
    
    def __getitem__(self, idx):
        # Load audio as Sound object
        audio_path = self.audio_paths[idx]
        
        return {
            'metadata': self.metadata[idx],
            'audio_path': audio_path,
            'index': idx
        }


def collate_audio_batch_simple(batch: List[Dict]) -> Dict[str, Any]:
    """
    Simple collator for HTTP mode that just passes through data.
    
    Args:
        batch: List of items from dataset
        
    Returns:
        Dict with batch data
    """
    return {
        'metadata': [item['metadata'] for item in batch],
        'audio_paths': [item['audio_path'] for item in batch],
        'indices': [item['index'] for item in batch]
    }


def collate_audio_batch(batch: List[Dict], tokenizer, model_config) -> Dict[str, Any]:
    """
    Custom collator for audio batches that preprocesses data for batch generation.
    
    Args:
        batch: List of items from dataset
        tokenizer: Model tokenizer
        model_config: Model configuration
    
    Returns:
        Dict with preprocessed batch data ready for generation
    """
    return {
        'metadata': [item['metadata'] for item in batch],
        'audio_paths': [item['audio_path'] for item in batch],
        'indices': [item['index'] for item in batch]
    }


class LLMRecaptioner:
    """Handles LLM-based audio recaptioning with batching support."""
    
    def __init__(self, model_name: str = "nvidia/audio-flamingo-3", 
                 device: str = None, labels_df: Optional[pd.DataFrame] = None,
                 server_url: Optional[str] = None):
        """
        Initialize the recaptioner with specified model.
        
        Args:
            model_name: HuggingFace model identifier
            device: Device to run on (auto-detected if None)
            labels_df: DataFrame with labels for context-aware prompting
            server_url: Optional HTTP server URL for remote inference
        """
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.labels_df = labels_df
        self.model = None
        self.generation_config = None
        
        # HTTP client configuration
        self.server_url = server_url or os.getenv("AUDIO_FLAMINGO_SERVER_URL", "http://localhost:8080")
        self.use_http = bool(server_url) or os.getenv("USE_HTTP_SERVER", "false").lower() == "true"
        
        # Connection pooling for better performance
        if self.use_http:
            self.client = httpx.Client(
                timeout=httpx.Timeout(300.0),  # 5 minute timeout
                limits=httpx.Limits(max_keepalive_connections=10, max_connections=20)
            )
            logger.info(f"Using HTTP server at {self.server_url}")
    
    def __del__(self):
        """Cleanup HTTP client on deletion"""
        if hasattr(self, 'client') and self.client:
            self.client.close()
        
    def load_model(self):
        """Load the model (call this once per worker process)."""
        if self.use_http:
            # For HTTP mode, just check server health
            try:
                response = self.client.get(f"{self.server_url}/health")
                if response.status_code == 200:
                    health = response.json()
                    if health["model_loaded"]:
                        logger.info(f"HTTP server ready: {health}")
                    else:
                        raise RuntimeError("HTTP server model not loaded")
                else:
                    raise RuntimeError(f"HTTP server health check failed: {response.status_code}")
            except Exception as e:
                logger.error(f"Failed to connect to HTTP server: {e}")
                raise
        elif self.model is None:
            print(f"Loading {self.model_name} on {self.device}...")
            model_base = snapshot_download(repo_id=self.model_name)
            self.model = llava.load(model_base, model_base=None)
            self.model = self.model.to(self.device)
            self.generation_config = self.model.default_generation_config
            # Ensure max_new_tokens is set
            if not hasattr(self.generation_config, 'max_new_tokens'):
                self.generation_config.max_new_tokens = 1024
            self.tokenizer = self.model.tokenizer
            self.config = self.model.config
            
    def generate_prompt(self, metadata: Dict) -> str:
        """
        Generate context-aware prompt based on metadata and labels.
        
        Args:
            metadata: Metadata dict for the audio file
            
        Returns:
            Formatted prompt string
        """
        base_prompt = "Please describe this audio in detail. "
        
        # Add label-specific context if available
        if self.labels_df is not None and 'original_filename' in metadata:
            filename = metadata['original_filename']
            # Look up labels or categories from CSV
            label_info = self._get_label_info(filename)
            if label_info:
                base_prompt += f"Context: This audio is labeled as '{label_info}'. "
                
        base_prompt += "Include information about the sounds, their characteristics, duration, and any notable features."
        
        return base_prompt
    
    def _get_label_info(self, filename: str) -> Optional[str]:
        """Extract label information from labels DataFrame."""
        if self.labels_df is None:
            return None
            
        # Try to match filename in the dataframe
        mask = self.labels_df['filename'] == filename
        if mask.any():
            row = self.labels_df[mask].iloc[0]
            # Extract relevant label columns (customize based on your CSV structure)
            labels = []
            for col in ['category', 'label', 'description', 'tags']:
                if col in row and pd.notna(row[col]):
                    labels.append(str(row[col]))
            return ', '.join(labels) if labels else None
            
        return None
    
    def _http_generate(self, audio_path: str, prompt: str, max_retries: int = 3) -> str:
        """
        Generate caption using HTTP server with retry logic.
        
        Args:
            audio_path: Path to audio file
            prompt: Generation prompt
            max_retries: Number of retries on failure
            
        Returns:
            Generated caption
        """
        for attempt in range(max_retries):
            try:
                response = self.client.post(
                    f"{self.server_url}/generate",
                    json={
                        "audio_path": str(audio_path),
                        "prompt": prompt,
                        "max_new_tokens": getattr(self.generation_config, 'max_new_tokens', 1024) if self.generation_config else 1024
                    }
                )
                
                if response.status_code == 200:
                    result = response.json()
                    return result["caption"]
                elif response.status_code == 504:  # Timeout
                    if attempt < max_retries - 1:
                        wait_time = 2 ** attempt  # Exponential backoff
                        logger.warning(f"Request timeout, retrying in {wait_time}s...")
                        time.sleep(wait_time)
                        continue
                    else:
                        raise RuntimeError("Request timeout after retries")
                else:
                    raise RuntimeError(f"HTTP error {response.status_code}: {response.text}")
                    
            except httpx.ConnectError as e:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    logger.warning(f"Connection error, retrying in {wait_time}s: {e}")
                    time.sleep(wait_time)
                    continue
                else:
                    raise
        
        raise RuntimeError(f"Failed after {max_retries} attempts")
    
    def process_batch(self, batch: Dict[str, Any]) -> List[str]:
        """
        Process a batch of audio files and generate captions.
        Currently uses sequential processing due to batch processing issues.
        
        Args:
            batch: Preprocessed batch dict from DataLoader with collated data
            
        Returns:
            List of generated captions
        """
        # Ensure model is loaded
        self.load_model()
        
        captions = []
        
        # Process each audio file individually
        for i, (audio_path, metadata) in enumerate(zip(batch['audio_paths'], batch['metadata'])):
            try:
                # Generate prompt
                prompt = self.generate_prompt(metadata)
                
                if self.use_http:
                    # Use HTTP server
                    response = self._http_generate(audio_path, prompt)
                else:
                    # Use local model with generate_content
                    sound = llava.Sound(str(audio_path))
                    full_prompt = f"<sound>\n{prompt}"
                    response = self.model.generate_content([sound, full_prompt], generation_config=self.generation_config)
                
                captions.append(response)
                
            except Exception as e:
                print(f"[ERROR] Failed to process {audio_path}: {e}")
                # Fallback to filename
                fallback = audio_path.stem.replace('_', ' ').replace('-', ' ')
                captions.append(fallback)
                
        return captions


def process_dataset_with_llm(
    audio_paths: List[Path],
    metadata: List[Dict],
    batch_size: int = 64,
    num_workers: int = 8,
    labels_csv: Optional[str] = None,
    model_name: str = "nvidia/audio-flamingo-3",
    server_url: Optional[str] = None
) -> List[Tuple[Path, str, Dict]]:
    """
    Process entire dataset with LLM recaptioning using batch processing.
    
    Args:
        audio_paths: List of audio file paths
        metadata: List of metadata dicts
        batch_size: Batch size for processing
        num_workers: Number of DataLoader workers
        labels_csv: Optional path to CSV with labels
        model_name: Model to use for captioning
        server_url: Optional HTTP server URL for remote inference
        
    Returns:
        List of tuples (audio_path, caption, metadata)
    """
    # Load labels if provided
    labels_df = None
    if labels_csv and Path(labels_csv).exists():
        labels_df = pd.read_csv(labels_csv)
        print(f"Loaded {len(labels_df)} label entries from {labels_csv}")
    
    # Initialize recaptioner and load model once
    recaptioner = LLMRecaptioner(model_name=model_name, labels_df=labels_df, server_url=server_url)
    recaptioner.load_model()
    
    # Create dataset with recaptioner for prompt generation
    dataset = AudioRecaptionDataset(audio_paths, metadata, labels_df, recaptioner)
    
    # Create custom collate function with model info
    if recaptioner.use_http:
        # For HTTP mode, use simple collate that just passes through data
        collate_fn = collate_audio_batch_simple
    else:
        collate_fn = partial(
            collate_audio_batch, 
            tokenizer=recaptioner.tokenizer,
            model_config=recaptioner.config
        )
    
    # Create dataloader with multi-worker support for preprocessing
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0  # Keep workers alive between batches
    )
    
    # Process batches
    results = []
    with tqdm(total=len(dataset), desc="Recaptioning audio files") as pbar:
        for batch in dataloader:
            # Process batch on GPU
            captions = recaptioner.process_batch(batch)
            
            # Collect results
            for audio_path, caption, metadata in zip(
                batch['audio_paths'], captions, batch['metadata']
            ):
                results.append((audio_path, caption, metadata))
                    
            pbar.update(len(batch['audio_paths']))
    
    return results


def create_recaption_processor(
    dataset_name: str,
    audio_dir: str,
    metadata_path: str,
    output_dir: str,
    labels_csv: Optional[str] = None,
    batch_size: int = 64,
    num_workers: int = 8
):
    """
    Factory function to create a recaptioning processor for any dataset.
    
    Args:
        dataset_name: Name of the dataset
        audio_dir: Directory containing audio files
        metadata_path: Path to metadata
        output_dir: Output directory for processed data
        labels_csv: Optional CSV with labels for prompting
        batch_size: Batch size for LLM processing
        num_workers: Number of workers for data loading
        
    Returns:
        Function that processes the dataset with recaptioning
    """
    def processor():
        # Import the specific dataset processor
        module_name = f"laion_datasets.{dataset_name.lower()}"
        processor_class_name = f"{dataset_name}Processor"
        
        # Dynamically import and instantiate the processor
        module = __import__(module_name, fromlist=[processor_class_name])
        ProcessorClass = getattr(module, processor_class_name)
        
        # Create processor instance
        processor = ProcessorClass(
            audio_dir=audio_dir,
            metadata_path=metadata_path,
            output_dir=output_dir
        )
        
        # Override the match_audio_to_text method to use LLM recaptioning
        original_match = processor.match_audio_to_text
        
        def match_with_recaption(metadata_df):
            # Get original matches
            matched = original_match(metadata_df)
            
            # Extract audio paths and metadata
            audio_paths = [m[0] for m in matched]
            original_captions = [m[1] for m in matched]
            metadata_list = [m[2] for m in matched]
            
            # Run LLM recaptioning
            results = process_dataset_with_llm(
                audio_paths=audio_paths,
                metadata=metadata_list,
                batch_size=batch_size,
                num_workers=num_workers,
                labels_csv=labels_csv or metadata_path  # Use metadata as labels if not specified
            )
            
            return results
        
        # Replace the method
        processor.match_audio_to_text = match_with_recaption
        
        # Run the processor
        return processor.process_dataset()
    
    return processor