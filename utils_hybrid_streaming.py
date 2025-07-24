#!/usr/bin/env python3
"""
Hybrid streaming processor that combines PyTorch DataLoader for efficient 
batch processing with queue-based streaming tar writing.
"""

import os
import queue
import threading
import io
import json
import tarfile
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Callable
import numpy as np
import soundfile as sf
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from dotenv import load_dotenv
import logging

load_dotenv()

# Dynamic import based on backend selection
USE_SGLANG = os.getenv("USE_SGLANG", "false").lower() == "true"
if USE_SGLANG:
    from utils_sglang import AudioTextDataset, collate_batch, GemmaProcessor
    logger = logging.getLogger(__name__)
    logger.info("Using SGLang backend for generation")
else:
    from utils_gemma import AudioTextDataset, collate_batch, GemmaProcessor
    logger = logging.getLogger(__name__)
    logger.info("Using local Gemma backend for generation")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HybridStreamingProcessor:
    """
    Combines DataLoader-based batch processing with streaming tar writing.
    Uses PyTorch DataLoader for efficient parallel audio loading and Gemma processing,
    while writing results to tar files asynchronously.
    """
    
    def __init__(self,
                 output_dir: str,
                 model_name: str = "google/gemma-3n-e4b-it",
                 system_prompt: Optional[str] = None,
                 user_prompt_template: Optional[str] = None,
                 generation_config: Optional[Dict] = None,
                 samples_per_tar: int = 2048,
                 batch_size: int = 128,
                 num_workers: int = 12,
                 queue_size: int = 3000,
                 target_sr_output: int = 48000,
                 target_sr_gemma: int = 16000,
                 sglang_server_url: str = "http://127.0.0.1:30000"):
        """
        Args:
            output_dir: Directory for output tar files
            model_name: Gemma model to use
            system_prompt: System prompt for generation
            user_prompt_template: Template for user prompts
            generation_config: Generation configuration for Gemma
            samples_per_tar: Number of samples per tar file
            batch_size: Batch size for GPU processing
            num_workers: Number of workers for DataLoader
            queue_size: Size of queue for tar writing
            target_sr_output: Sample rate for output audio (48kHz)
            target_sr_gemma: Sample rate for Gemma processing (16kHz)
            sglang_server_url: URL of SGLang server (only used if USE_SGLANG=true)
        """
        self.output_dir = Path(output_dir)
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.user_prompt_template = user_prompt_template
        self.generation_config = generation_config
        self.samples_per_tar = samples_per_tar
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.queue_size = queue_size
        self.target_sr_output = target_sr_output
        self.target_sr_gemma = target_sr_gemma
        
        # Initialize processor based on backend
        if USE_SGLANG:
            self.gemma_processor = GemmaProcessor(
                model_name=model_name,
                generation_config=generation_config,
                server_url=sglang_server_url,
                sample_rate=target_sr_gemma,
                system_prompt=system_prompt,
                user_prompt_template=user_prompt_template
            )
        else:
            self.gemma_processor = GemmaProcessor(
                model_name=model_name,
                generation_config=generation_config
            )
        
        # Queue for tar writing
        self.tar_queue = queue.Queue(maxsize=queue_size)
        
        # Control flags
        self.stop_writing = threading.Event()
        
        # Statistics
        self.stats = {
            'processed': 0,
            'written': 0,
            'failed': 0
        }
        print("Batch size: ", self.batch_size)
        print("Num workers: ", self.num_workers)
        print("Queue size: ", self.queue_size)
        print("Target SR output: ", self.target_sr_output)
        print("Target SR gemma: ", self.target_sr_gemma)
        print("Samples per tar: ", self.samples_per_tar)
        print("Generation config: ", self.generation_config)
        
    def tar_writer_thread(self, prefix: str = "data", split: Optional[str] = None):
        """Thread that writes processed samples to tar files."""
        logger.info("Starting tar writer thread")
        
        # Setup output directory
        output_dir = self.output_dir
        if split:
            output_dir = output_dir / split
        output_dir.mkdir(parents=True, exist_ok=True)
        
        tar_index = 0
        current_tar = None
        current_tar_path = None
        samples_in_tar = 0
        tar_summaries = []
        
        while not self.stop_writing.is_set() or not self.tar_queue.empty():
            try:
                # Get processed data with timeout
                result = self.tar_queue.get(timeout=1.0)
                
                if result is None:
                    # Close current tar if open
                    if current_tar:
                        current_tar.close()
                        tar_summaries.append({
                            'tar_file': current_tar_path.name,
                            'successful': samples_in_tar,
                            'failed': 0,
                            'total': samples_in_tar
                        })
                    break
                
                audio_48khz, caption, metadata = result
                
                # Create new tar if needed
                if current_tar is None or samples_in_tar >= self.samples_per_tar:
                    if current_tar:
                        current_tar.close()
                        tar_summaries.append({
                            'tar_file': current_tar_path.name,
                            'successful': samples_in_tar,
                            'failed': 0,
                            'total': samples_in_tar
                        })
                    
                    current_tar_path = output_dir / f"{tar_index:04d}.tar"
                    current_tar = tarfile.open(current_tar_path, "w")
                    samples_in_tar = 0
                    tar_index += 1
                
                # Write sample to tar
                self._write_sample_to_tar(current_tar, audio_48khz, caption, 
                                        metadata, tar_index - 1, samples_in_tar)
                samples_in_tar += 1
                self.stats['written'] += 1
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in tar writer: {e}")
                self.stats['failed'] += 1
        
        # Create sizes.json
        if tar_summaries:
            sizes = {s['tar_file']: s['successful'] for s in tar_summaries}
            with open(output_dir / "sizes.json", "w") as f:
                json.dump(sizes, f, indent=2)
        
        logger.info(f"Tar writer thread finished. Written: {self.stats['written']}")
    
    def _write_sample_to_tar(self, tar: tarfile.TarFile, audio_48khz: np.ndarray,
                           caption: str, metadata: Dict, tar_index: int, sample_index: int):
        """Write a single sample to tar file."""
        try:
            # Create key for this sample
            key = f"{tar_index:06d}_{sample_index:06d}"
            
            # Convert audio to FLAC bytes
            output_buffer = io.BytesIO()
            sf.write(output_buffer, audio_48khz, self.target_sr_output, 
                    format='FLAC', subtype='PCM_16')
            output_buffer.seek(0)
            audio_bytes = output_buffer.read()
            
            # Add audio file
            audio_info = tarfile.TarInfo(name=f"{key}.flac")
            audio_info.size = len(audio_bytes)
            tar.addfile(audio_info, io.BytesIO(audio_bytes))
            
            # Create JSON with caption and metadata
            json_data = {
                "text": caption,
                **metadata
            }
            json_bytes = json.dumps(json_data, ensure_ascii=False).encode('utf-8')
            
            # Add JSON file
            json_info = tarfile.TarInfo(name=f"{key}.json")
            json_info.size = len(json_bytes)
            tar.addfile(json_info, io.BytesIO(json_bytes))
            
        except Exception as e:
            logger.error(f"Failed to write sample to tar: {e}")
            raise
    
    def process_dataset(self, audio_files: List[Tuple[Path, str, Dict]], 
                       prefix: str = "data", split: Optional[str] = None,
                       show_progress: bool = True) -> Dict[str, int]:
        """
        Process dataset using DataLoader for batch processing and streaming tar writing.
        
        Args:
            audio_files: List of (audio_path, text, metadata) tuples
            prefix: Prefix for tar file naming
            split: Optional split name (train/valid/test)
            show_progress: Whether to show progress bars
            
        Returns:
            Dictionary with processing statistics
        """
        logger.info(f"Starting hybrid streaming pipeline for {len(audio_files)} files")
        
        # Extract components from audio_files
        audio_paths = [str(af[0]) for af in audio_files]
        texts = [af[1] for af in audio_files]
        metadata_list = [af[2] for af in audio_files]
        
        # Create dataset with dual sample rate loading
        dataset = AudioTextDataset(
            texts=texts,
            metadata=metadata_list,
            system_prompt=self.system_prompt,
            user_prompt_template=self.user_prompt_template,
            audio_paths=audio_paths,
            sample_rate=self.target_sr_gemma,
            load_dual_sample_rates=True,
            output_sample_rate=self.target_sr_output
        )
        
        # Create collate function
        if USE_SGLANG:
            # SGLang version doesn't need processor argument
            collate_fn = collate_batch
        else:
            # Local Gemma version needs processor
            def collate_fn(batch):
                return collate_batch(batch, self.gemma_processor.processor)
        
        # Create dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            prefetch_factor=2 if self.num_workers > 0 else None
        )
        
        # Start tar writer thread
        writer_thread = threading.Thread(
            target=self.tar_writer_thread,
            args=(prefix, split),
            name="TarWriter"
        )
        writer_thread.start()
        
        # Process batches
        try:
            desc = f"Processing {split or 'data'}"
            iterator = tqdm(dataloader, desc=desc) if show_progress else dataloader
            
            for batch in iterator:
                # Skip None batches (all items filtered out)
                if batch is None:
                    continue
                
                # Generate captions
                generated_texts = self.gemma_processor.generate_batch(batch)
                if generated_texts is None:
                    continue
                
                # Get audio outputs (48kHz versions)
                audio_outputs = batch['audio_outputs']
                
                # Submit results to tar queue
                for audio_48khz, caption, metadata in zip(
                    audio_outputs, generated_texts, batch['metadata']
                ):
                    self.tar_queue.put((audio_48khz, caption, metadata))
                    self.stats['processed'] += 1
                
                # Update progress description
                if show_progress:
                    iterator.set_postfix({
                        'processed': self.stats['processed'],
                        'written': self.stats['written'],
                        'queue': self.tar_queue.qsize()
                    })
            
            # Signal completion to tar writer
            self.tar_queue.put(None)
            
        except Exception as e:
            logger.error(f"Error in processing loop: {e}")
            self.stop_writing.set()
            raise
        
        # Wait for tar writer to finish
        writer_thread.join()
        
        # Log statistics
        logger.info(f"Processing complete. Processed: {self.stats['processed']}, "
                   f"Written: {self.stats['written']}, Failed: {self.stats['failed']}")
        
        # Log skipped files if any
        if hasattr(dataset, 'skipped_count') and dataset.skipped_count > 0:
            logger.warning(f"Skipped {dataset.skipped_count} files due to audio loading failures")
            self.stats['failed'] += dataset.skipped_count
        
        return self.stats