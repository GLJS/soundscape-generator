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

# Add flamingo directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'flamingo'))
import llava
from llava.utils.media import extract_media
from llava.utils.tokenizer import tokenize_conversation
from llava.mm_utils import process_sounds, process_sound_masks
import numpy as np


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
        sound = llava.Sound(str(audio_path))
        
        # Generate prompt using recaptioner if available
        if self.recaptioner:
            prompt = self.recaptioner.generate_prompt(self.metadata[idx])
        else:
            prompt = "Please describe this audio in detail. Include information about the sounds, their characteristics, duration, and any notable features."
        
        # Create conversation format expected by the model
        conversation = [{"from": "human", "value": [sound, prompt]}]
        
        return {
            'conversation': conversation,
            'metadata': self.metadata[idx],
            'audio_path': audio_path,
            'index': idx
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
    conversations = [item['conversation'] for item in batch]
    
    # Extract and process media for all items in batch
    all_media = {'sound': []}
    all_media_meta = {
        'sound_feature_masks': [], 
        'sound_embed_masks': []
    }
    
    # Process each conversation to extract media
    for conv in conversations:
        media, media_meta = extract_media(conv, model_config)
        
        # Process sounds
        if 'sound' in media:
            # Convert lists back to tensors and collect
            for sound_list in media["sound"]:
                # sound_list is a list representation of a tensor
                sound_tensor = torch.tensor(sound_list, dtype=torch.float32)
                all_media['sound'].append(sound_tensor)
            
            # Collect masks (they're already tensors)
            all_media_meta['sound_feature_masks'].extend(media_meta["sound_feature_masks"])
            all_media_meta['sound_embed_masks'].extend(media_meta["sound_embed_masks"])
    
    # Tokenize all conversations
    input_ids_list = []
    attention_mask_list = []
    
    for conv in conversations:
        # Replace media objects with tokens for tokenization
        conv_for_tokenize = []
        for msg in conv:
            new_msg = {"from": msg["from"], "value": msg["value"]}
            if isinstance(msg["value"], list):
                # Replace Sound objects with <sound> token
                text_parts = []
                for part in msg["value"]:
                    if isinstance(part, llava.Sound):
                        text_parts.append("<sound>")
                    else:
                        text_parts.append(part)
                new_msg["value"] = "\n".join(text_parts)
            conv_for_tokenize.append(new_msg)
        
        # Tokenize
        input_ids = tokenize_conversation(
            conv_for_tokenize, 
            tokenizer, 
            add_generation_prompt=True
        )
        input_ids_list.append(input_ids.squeeze(0))  # Remove batch dimension
    
    # Pad sequences to same length
    input_ids = torch.nn.utils.rnn.pad_sequence(
        input_ids_list, 
        batch_first=True, 
        padding_value=tokenizer.pad_token_id
    )
    
    # Create attention mask
    attention_mask = input_ids != tokenizer.pad_token_id
    
    # Media tensors are already processed above, no need to process again
    
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'media': all_media,
        'media_meta': all_media_meta,
        'metadata': [item['metadata'] for item in batch],
        'audio_paths': [item['audio_path'] for item in batch],
        'indices': [item['index'] for item in batch]
    }


class LLMRecaptioner:
    """Handles LLM-based audio recaptioning with batching support."""
    
    def __init__(self, model_name: str = "nvidia/audio-flamingo-3", 
                 device: str = None, labels_df: Optional[pd.DataFrame] = None):
        """
        Initialize the recaptioner with specified model.
        
        Args:
            model_name: HuggingFace model identifier
            device: Device to run on (auto-detected if None)
            labels_df: DataFrame with labels for context-aware prompting
        """
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.labels_df = labels_df
        self.model = None
        self.generation_config = None
        
    def load_model(self):
        """Load the model (call this once per worker process)."""
        if self.model is None:
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
        
        # Process each audio file individually using generate_content
        for i, (audio_path, metadata) in enumerate(zip(batch['audio_paths'], batch['metadata'])):
            try:
                # Create Sound object
                sound = llava.Sound(str(audio_path))
                
                # Generate prompt
                prompt = self.generate_prompt(metadata)
                full_prompt = f"<sound>\n{prompt}"
                
                # Generate caption using generate_content
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
    model_name: str = "nvidia/audio-flamingo-3"
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
        
    Returns:
        List of tuples (audio_path, caption, metadata)
    """
    # Load labels if provided
    labels_df = None
    if labels_csv and Path(labels_csv).exists():
        labels_df = pd.read_csv(labels_csv)
        print(f"Loaded {len(labels_df)} label entries from {labels_csv}")
    
    # Initialize recaptioner and load model once
    recaptioner = LLMRecaptioner(model_name=model_name, labels_df=labels_df)
    recaptioner.load_model()
    
    # Create dataset with recaptioner for prompt generation
    dataset = AudioRecaptionDataset(audio_paths, metadata, labels_df, recaptioner)
    
    # Create custom collate function with model info
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