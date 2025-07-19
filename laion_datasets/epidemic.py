#!/usr/bin/env python3
"""
Convert Epidemic Sound Effects dataset from parquet files to WebDataset tar format.

Audio location: /scratch-shared/gwijngaard/laion/epidemicv2/
The parquet files contain both audio and metadata (text, tags, etc.)
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from pathlib import Path
from utils import DatasetProcessor, AudioProcessor, TarCreator
from typing import List, Tuple, Dict, Iterator
import argparse
from datasets import load_dataset, IterableDataset
import io
import json
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()


class EpidemicProcessor(DatasetProcessor):
    """Processor for Epidemic Sound Effects dataset - handles streaming from parquet files."""
    
    def __init__(self, audio_dir: str, output_dir: str):
        super().__init__(audio_dir, None, output_dir, task="AAC")
        
    def load_metadata(self) -> pd.DataFrame:
        """Not used for this dataset - metadata is in parquet files."""
        return pd.DataFrame()
        
    def create_parquet_iterator(self, split: str) -> Iterator[Dict]:
        """Create an iterator over parquet files for a given split."""
        parquet_dir = self.audio_dir / split
        
        if not parquet_dir.exists():
            print(f"Split directory not found: {parquet_dir}")
            return
            
        # Get all parquet files for this split
        parquet_files = sorted(parquet_dir.glob("*.parquet"))
        print(f"Found {len(parquet_files)} parquet files for {split} split")
        
        # Process each parquet file
        for parquet_file in tqdm(parquet_files, desc=f"Processing {split} parquet files"):
            try:
                # Load parquet file using datasets library in streaming mode
                dataset = load_dataset(
                    "parquet",
                    data_files=str(parquet_file),
                    split="train",
                    streaming=True
                )
                
                # Yield each sample from the dataset
                for sample in dataset:
                    yield sample
                    
            except Exception as e:
                print(f"Error processing {parquet_file}: {e}")
                continue
    
    def match_audio_to_text(self, metadata_df: pd.DataFrame = None) -> List[Tuple[bytes, str, Dict]]:
        """
        This method is not used in streaming mode.
        We'll override process_dataset instead.
        """
        return []
        
    def process_dataset(self, samples_per_tar: int = 2048):
        """Process the entire dataset into tar files using streaming."""
        # Process each split
        all_summaries = []
        for split in ['train', 'test']:
            print(f"\nProcessing {split} split...")
            
            # Create tar creator for this split
            tar_creator = TarCreator(
                self.output_dir, 
                prefix="epidemic",
                samples_per_tar=samples_per_tar,
                split=split
            )
            
            # Buffer for samples
            samples_buffer = []
            tar_index = 0
            
            # Process parquet files in streaming mode
            for sample in self.create_parquet_iterator(split):
                try:
                    # Extract filename and audio data from the sample
                    # Based on the structure: 'index' contains filename, 'audio' contains AudioDecoder object
                    
                    # Debug first sample to understand structure
                    if tar_index == 0 and len(samples_buffer) == 0:
                        print(f"First sample keys: {list(sample.keys())}")
                        print(f"Datasetname: {sample.get('datasetname')}")
                        print(f"Index: {sample.get('index')}")
                        print(f"Text: {sample.get('text', '')[:100]}...")
                    
                    # Extract filename from 'index' field
                    if 'index' in sample:
                        filename = sample['index']
                    else:
                        print(f"No 'index' field found in sample")
                        continue
                    
                    # Extract audio data
                    audio_data = None
                    if 'audio' in sample:
                        audio_obj = sample['audio']
                        
                        # Handle datasets.features._torchcodec.AudioDecoder object
                        if hasattr(audio_obj, 'get_all_samples'):
                            # This is the AudioDecoder from datasets library
                            try:
                                # Get all audio samples
                                audio_samples = audio_obj.get_all_samples()
                                
                                # Extract audio data and sample rate from AudioSamples object
                                if hasattr(audio_samples, 'data') and hasattr(audio_samples, 'sample_rate'):
                                    # Convert torch tensor to numpy
                                    audio_tensor = audio_samples.data
                                    sampling_rate = audio_samples.sample_rate
                                    
                                    # Convert to numpy array
                                    import numpy as np
                                    if hasattr(audio_tensor, 'numpy'):
                                        audio_array = audio_tensor.numpy()
                                    else:
                                        audio_array = np.array(audio_tensor)
                                    
                                    # Reshape if needed (remove channel dimension if mono)
                                    if audio_array.ndim > 1 and audio_array.shape[0] == 1:
                                        audio_array = audio_array.squeeze(0)
                                    
                                    # Convert numpy array to bytes
                                    import soundfile as sf
                                    buffer = io.BytesIO()
                                    sf.write(buffer, audio_array, sampling_rate, format='FLAC', subtype='PCM_16')
                                    buffer.seek(0)
                                    audio_data = buffer.read()
                                else:
                                    raise ValueError(f"AudioSamples object missing data or sample_rate attributes")
                                    
                            except Exception as e:
                                if tar_index == 0 and len(samples_buffer) < 5:
                                    print(f"Error decoding audio: {e}")
                                continue
                        
                        # Handle regular dictionary-style audio features
                        elif hasattr(audio_obj, 'array') and hasattr(audio_obj, 'sampling_rate'):
                            # Direct attribute access
                            audio_array = audio_obj.array
                            sampling_rate = audio_obj.sampling_rate
                            
                            # Convert numpy array to bytes
                            import numpy as np
                            import soundfile as sf
                            
                            # Ensure it's a numpy array
                            if not isinstance(audio_array, np.ndarray):
                                audio_array = np.array(audio_array)
                            
                            # Convert to FLAC bytes
                            buffer = io.BytesIO()
                            sf.write(buffer, audio_array, sampling_rate, format='FLAC', subtype='PCM_16')
                            buffer.seek(0)
                            audio_data = buffer.read()
                        
                        # Try dictionary-style access
                        elif isinstance(audio_obj, dict):
                            if 'bytes' in audio_obj:
                                audio_data = audio_obj['bytes']
                            elif 'array' in audio_obj:
                                # Handle dict with array
                                audio_array = np.array(audio_obj['array'])
                                sampling_rate = audio_obj.get('sampling_rate', 48000)
                                buffer = io.BytesIO()
                                sf.write(buffer, audio_array, sampling_rate, format='FLAC', subtype='PCM_16')
                                buffer.seek(0)
                                audio_data = buffer.read()
                        
                        # Alternative: if audio is already bytes
                        elif isinstance(audio_obj, bytes):
                            audio_data = audio_obj
                    
                    if audio_data is None:
                        # Only print for first few failures
                        if tar_index == 0 and len(samples_buffer) < 5:
                            print(f"Could not extract audio from sample. Audio type: {type(sample.get('audio'))}")
                        continue
                    
                    # Since the parquet already contains all needed data, use it directly
                    # Get the text caption from the sample
                    caption_text = sample.get('text', '')
                    if not caption_text and 'raw_text' in sample:
                        # Use raw_text if text is empty
                        raw_texts = sample['raw_text']
                        if isinstance(raw_texts, list) and len(raw_texts) > 0:
                            caption_text = raw_texts[0]
                    
                    # Extract tags
                    tags = sample.get('tag', [])
                    
                    # Skip if not epidemic data (check if index contains epidemic_sound_effects)
                    if 'epidemic_sound_effects' not in filename:
                        continue
                    
                    # Process audio
                    if isinstance(audio_data, bytes):
                        audio_bytes_io = io.BytesIO(audio_data)
                    else:
                        audio_bytes_io = audio_data
                        
                    processed_audio, audio_metadata = self.audio_processor.process_audio_file(audio_bytes_io)
                    
                    # Create sample metadata
                    sample_metadata = {
                        'split': split,
                        'original_filename': filename,
                        'task': self.task,
                        'tags': tags,
                        'datasetname': sample.get('datasetname', ''),
                        'audio_len': sample.get('audio_len', 0),
                        **audio_metadata
                    }
                    
                    # Add to buffer
                    samples_buffer.append({
                        'audio_bytes': processed_audio,
                        'text': caption_text,
                        'metadata': sample_metadata
                    })
                    
                    # Debug: print progress
                    if len(samples_buffer) == 1:
                        print(f"First sample successfully processed!")
                    if len(samples_buffer) % 100 == 0:
                        print(f"Successfully processed {len(samples_buffer)} samples...")
                    
                    # Write tar when buffer is full
                    if len(samples_buffer) >= samples_per_tar:
                        summary = tar_creator.create_tar_from_samples(samples_buffer, tar_index)
                        all_summaries.append(summary)
                        tar_index += 1
                        samples_buffer = []
                        
                except Exception as e:
                    print(f"Error processing sample: {e}")
                    continue
            
            # Write remaining samples
            if samples_buffer:
                summary = tar_creator.create_tar_from_samples(samples_buffer, tar_index)
                all_summaries.append(summary)
            
            # Create size file for this split
            split_summaries = [s for s in all_summaries if split in str(tar_creator.output_dir)]
            if split_summaries:
                tar_creator.create_size_file(split_summaries)
        
        # Summary
        total_successful = sum(s['successful'] for s in all_summaries)
        total_failed = sum(s['failed'] for s in all_summaries)
        
        print(f"\nProcessing complete!")
        print(f"Successfully processed: {total_successful}")
        print(f"Failed: {total_failed}")
        print(f"Created {len(all_summaries)} tar files in {self.output_dir}")
        
        return all_summaries


def main():
    parser = argparse.ArgumentParser(description="Convert Epidemic Sound Effects dataset to tar format")
    parser.add_argument("--audio-dir", type=str, 
                       default="/scratch-shared/gwijngaard/laion/epidemicv2",
                       help="Path to directory containing train/test parquet files")
    parser.add_argument("--output-dir", type=str,
                       default="/scratch-shared/gwijngaard/tar/epidemic",
                       help="Output directory for tar files")
    parser.add_argument("--samples-per-tar", type=int, default=2048,
                       help="Number of samples per tar file")
    
    args = parser.parse_args()
    
    # Create processor
    processor = EpidemicProcessor(
        audio_dir=args.audio_dir,
        output_dir=args.output_dir
    )
    
    # Process dataset
    processor.process_dataset(samples_per_tar=args.samples_per_tar)


if __name__ == "__main__":
    main()