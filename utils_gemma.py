#!/usr/bin/env python3
"""
Simplified Gemma-based text generation utilities with efficient batch processing.
Uses Gemma 3n model with processor.apply_chat_template for multimodal support.
"""
from dotenv import load_dotenv

load_dotenv()
import os
import torch
torch.set_float32_matmul_precision('high')
import torch._dynamo
torch._dynamo.config.cache_size_limit = 1000000000
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any, Union
import pandas as pd
from tqdm import tqdm
import json
from transformers import AutoProcessor, Gemma3nForConditionalGeneration
import logging
import librosa
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AudioTextDataset(Dataset):
    """Dataset for text generation tasks with optional audio support."""
    
    def __init__(
        self, 
        texts: List[str], 
        metadata: Optional[List[Dict]] = None,
        system_prompt: Optional[str] = None,
        user_prompt_template: Optional[str] = None,
        audio_paths: Optional[List[str]] = None,
        sample_rate: int = 16000
    ):
        """
        Args:
            texts: List of input texts to process
            metadata: Optional list of metadata dicts for each text
            system_prompt: Optional system prompt to prepend
            user_prompt_template: Template for user prompts (use {text} placeholder)
            audio_paths: Optional list of audio file paths for multimodal input
            sample_rate: Sample rate for audio loading
        """
        self.texts = texts
        self.metadata = metadata or [{} for _ in texts]
        self.system_prompt = system_prompt or "You are a helpful assistant."
        self.user_prompt_template = user_prompt_template or "{text}"
        self.audio_paths = audio_paths
        self.sample_rate = sample_rate
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        metadata = self.metadata[idx]
        
        # Format the user prompt
        user_prompt = self.user_prompt_template.format(text=text, **metadata)
        
        # Create messages in chat format
        if self.audio_paths is not None:
            # Load audio file for multimodal input
            audio_path = self.audio_paths[idx]
            try:
                audio, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True)
                audio = audio.astype(np.float32)
            except Exception as e:
                logger.warning(f"Failed to load audio {audio_path}: {e}")
                audio = np.zeros(self.sample_rate, dtype=np.float32)  # 1 second of silence
            
            # Multimodal message format
            messages = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": self.system_prompt}]
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "audio", "audio": audio},
                        {"type": "text", "text": user_prompt}
                    ]
                }
            ]
            
            return {
                'messages': messages,
                'audio': audio,
                'audio_path': audio_path,
                'text': text,
                'metadata': metadata,
                'index': idx
            }
        else:
            # Text-only format
            messages = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": self.system_prompt}]
                },
                {
                    "role": "user",
                    "content": [{"type": "text", "text": user_prompt}]
                }
            ]
            
            return {
                'messages': messages,
                'text': text,
                'metadata': metadata,
                'index': idx
            }


def collate_batch(batch: List[Dict], processor: Any) -> Dict[str, Any]:
    """
    Simple collator that processes messages with processor.apply_chat_template.
    
    Args:
        batch: List of items from dataset
        processor: Gemma processor
        
    Returns:
        Dict with processed batch data ready for generation
    """
    # Process each item individually then combine
    all_inputs = []
    
    for item in batch:
        messages = item['messages']
        
        # Use processor.apply_chat_template
        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        all_inputs.append(inputs)
    
    # Combine inputs by concatenating and padding
    # Find max length
    max_input_len = max(inp['input_ids'].shape[-1] for inp in all_inputs)
    max_input_features_len = max(inp['input_features'].shape[-2] for inp in all_inputs)  # sequence length dimension
    max_token_type_ids_len = max(inp['token_type_ids'].shape[-1] for inp in all_inputs)
    
    # Pad all inputs to max length
    padded_input_ids = []
    padded_attention_masks = []
    padded_token_type_ids = []
    padded_input_features = []
    padded_input_features_mask = []
    
    for inp in all_inputs:
        input_ids = inp['input_ids']
        attention_mask = inp['attention_mask']
        token_type_ids = inp['token_type_ids']
        input_features = inp['input_features']
        input_features_mask = inp['input_features_mask']
        
        # Calculate padding needed
        pad_len = max_input_len - input_ids.shape[-1]
        pad_features_len = max_input_features_len - input_features.shape[-2]  # sequence length dimension
        pad_token_type_ids_len = max_token_type_ids_len - token_type_ids.shape[-1]
        if pad_len > 0:
            # Left pad for decoder-only models
            input_ids = torch.nn.functional.pad(input_ids, (pad_len, 0), value=processor.tokenizer.pad_token_id)
            # Left pad attention_mask with 0s
            attention_mask = torch.nn.functional.pad(attention_mask, (pad_len, 0), value=0)
        
        if pad_features_len > 0:
            # Left pad on sequence dimension (dimension -2), no padding on feature dimension
            input_features = torch.nn.functional.pad(input_features, (0, 0, pad_features_len, 0), value=0)
            # Left pad mask with 0s to mask out the padded positions (2D tensor: [batch, seq_len])
            input_features_mask = torch.nn.functional.pad(input_features_mask, (pad_features_len, 0), value=0)
        
        if pad_token_type_ids_len > 0:
            token_type_ids = torch.nn.functional.pad(token_type_ids, (pad_token_type_ids_len, 0), value=0)
        
        padded_input_ids.append(input_ids)
        padded_attention_masks.append(attention_mask)
        padded_token_type_ids.append(token_type_ids)
        padded_input_features.append(input_features)
        padded_input_features_mask.append(input_features_mask)
        
    # Stack all tensors (don't move to device here - will be done in generate_batch)
    batch_dict = {
        'input_ids': torch.cat(padded_input_ids, dim=0),
        'attention_mask': torch.cat(padded_attention_masks, dim=0),
        'token_type_ids': torch.cat(padded_token_type_ids, dim=0),
        'input_features': torch.cat(padded_input_features, dim=0),
        'input_features_mask': torch.cat(padded_input_features_mask, dim=0),
        'texts': [item['text'] for item in batch],
        'metadata': [item['metadata'] for item in batch],
        'indices': [item['index'] for item in batch],
        'original_lengths': [inp['input_ids'].shape[-1] for inp in all_inputs]
    }
    
    # Add audio paths if present
    if 'audio_path' in batch[0]:
        batch_dict['audio_paths'] = [item['audio_path'] for item in batch]
    
    return batch_dict


class GemmaProcessor:
    """Simplified Gemma processor for text and audio generation."""
    
    def __init__(
        self, 
        model_name: str = "google/gemma-3n-e4b-it",
        device: str = None,
        torch_dtype: torch.dtype = torch.bfloat16,
        generation_config: Optional[Dict] = None
    ):
        """
        Initialize the Gemma model and processor.
        
        Args:
            model_name: HuggingFace model identifier
            device: Device to run on (auto-detected if None)
            torch_dtype: Data type for model weights
            generation_config: Custom generation configuration
        """
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.torch_dtype = torch_dtype
        
        # Load model and processor
        logger.info(f"Loading {model_name} on {self.device}...")
        
        self.model = Gemma3nForConditionalGeneration.from_pretrained(
            model_name,
            device_map="auto" if self.device == "cuda" else None,
            torch_dtype=torch_dtype,
        ).eval()
        
        self.processor = AutoProcessor.from_pretrained(model_name)
        
        # Set left padding for decoder-only models
        if hasattr(self.processor, 'tokenizer'):
            self.processor.tokenizer.padding_side = 'left'
            if self.processor.tokenizer.pad_token is None:
                self.processor.tokenizer.pad_token = self.processor.tokenizer.eos_token
        
        # Generation configuration
        self.generation_config = generation_config or {
            "max_new_tokens": 100,
            "do_sample": False,
        }
        self.split_on = "\nmodel\n"
        self.tokenized_split_on = self.processor.tokenizer(self.split_on, add_special_tokens=False).input_ids[0]
        
        logger.info(f"Model loaded successfully on {self.device}")
    
    def generate_batch(self, batch: Dict[str, Any]) -> List[str]:
        """
        Generate text for a batch of inputs.
        
        Args:
            batch: Batch dict from collate_batch
            
        Returns:
            List of generated texts
        """
        # Extract inputs and move to device with correct dtype
        input_ids = batch['input_ids'].to(self.model.device)
        attention_mask = batch['attention_mask'].to(self.model.device)
        token_type_ids = batch['token_type_ids'].to(self.model.device)
        input_features = batch['input_features'].to(self.model.device, dtype=self.torch_dtype)
        input_features_mask = batch['input_features_mask'].to(self.model.device)
        original_lengths = batch['original_lengths']
        
        # Generate
        with torch.inference_mode():
            generation = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                input_features=input_features,
                input_features_mask=input_features_mask,
                **self.generation_config
            )
        
        # Decode outputs, removing the input prompt
        # Pre-process all sequences to extract generated parts
        generated_sequences = []
        for i, (output, input_len) in enumerate(zip(generation, original_lengths)):
            # Extract only the generated part
            generated_ids = output[input_len:]
            split_index = torch.where(generated_ids == self.tokenized_split_on)[0]
            if split_index.numel() > 0:
                generated_ids = generated_ids[split_index[0]+3:]
            generated_sequences.append(generated_ids)
        
        # Use batch_decode for more efficient decoding
        generated_texts = self.processor.batch_decode(
            generated_sequences, 
            skip_special_tokens=True
        )
        generated_texts = [text.strip() for text in generated_texts]
        
        return generated_texts


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
    sample_rate: int = 16000
) -> List[Tuple[str, str, Dict]]:
    """
    Process dataset with Gemma model using batch processing.
    
    Args:
        texts: List of input texts (required if audio_paths not provided)
        metadata: Optional list of metadata dicts
        audio_paths: Optional list of audio file paths for multimodal input
        batch_size: Batch size for processing
        num_workers: Number of DataLoader workers
        model_name: Model to use for generation
        system_prompt: System prompt for the model
        user_prompt_template: Template for user prompts
        generation_config: Custom generation configuration
        show_progress: Whether to show progress bar
        sample_rate: Sample rate for audio loading
        
    Returns:
        List of tuples (input, generated_text, metadata)
        - For text mode: (input_text, generated_text, metadata)
        - For audio mode: (audio_path, generated_text, metadata)
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
        generation_config=generation_config
    )
    
    # Create dataset
    dataset = AudioTextDataset(
        texts=texts,
        metadata=metadata,
        system_prompt=system_prompt,
        user_prompt_template=user_prompt_template,
        audio_paths=audio_paths,
        sample_rate=sample_rate
    )
    
    # Create collate function
    def collate_fn(batch):
        return collate_batch(batch, processor.processor)
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers if audio_paths is None else min(num_workers, 2),
        pin_memory=torch.cuda.is_available()
    )
    
    # Process batches
    results = []
    desc = "Processing audio" if audio_paths else "Generating text"
    iterator = tqdm(dataloader, desc=desc) if show_progress else dataloader
    
    for batch in iterator:
        # Generate outputs
        generated_texts = processor.generate_batch(batch)
        
        # Collect results
        for audio_path, generated, metadata in zip(
            batch.get('audio_paths', batch['texts']), 
            generated_texts, 
            batch['metadata']
        ):
            results.append((audio_path, generated, metadata))
    
    return results


# Example usage
if __name__ == "__main__":
    # Example: Process a list of texts
    sample_texts = [
        "What is the capital of France?",
        "Explain quantum computing in simple terms.",
        "Write a haiku about spring."
    ]
    
    results = process_dataset_with_gemma(
        texts=sample_texts,
        user_prompt_template="Please answer the following question: {text}",
        batch_size=128,
        generation_config={
            "max_new_tokens": 200,
            "temperature": 0.7,
        }
    )
    
    for input_text, generated, _ in results:
        print(f"Input: {input_text}")
        print(f"Generated: {generated}")
        print("-" * 80)