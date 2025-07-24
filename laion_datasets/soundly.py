#!/usr/bin/env python3
"""
Convert Soundly dataset to WebDataset tar format.

Audio location: /scratch-shared/gwijngaard/laion/soundly/audio/
Note: This dataset only has audio files, filenames are used as prompts
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from pathlib import Path
from utils import DatasetProcessor
from typing import List, Tuple, Dict
import argparse
from dotenv import load_dotenv
from utils_gemma import process_dataset_with_gemma
from tqdm import tqdm
import json
load_dotenv()


class SoundlyProcessor(DatasetProcessor):
    """Processor for Soundly dataset."""
    
    def __init__(self, audio_dir: str, metadata_path: str, output_dir: str, 
                 use_recaption: bool = True, batch_size: int = 128, num_workers: int = 16):
        super().__init__(audio_dir, metadata_path, output_dir)
        self.use_recaption = use_recaption
        self.batch_size = batch_size
        self.num_workers = num_workers
        
    def load_metadata(self) -> pd.DataFrame:
        """Load metadata - returns empty DataFrame as Soundly has no CSV."""
        print(f"Loading audio files from {self.audio_dir}")
        # No metadata CSV for Soundly dataset
        return pd.DataFrame()
        
    def match_audio_to_text(self, metadata_df: pd.DataFrame) -> List[Tuple[Path, str, Dict]]:
        """Process audio files using filename as prompt text."""
        matched = []
        
        print(f"Processing audio files from {self.audio_dir}")
        
        # Find all .wav files recursively
        audio_files = list(self.audio_dir.rglob("*.wav"))
        print(f"  Found {len(audio_files)} audio files")
        
        if len(audio_files) > 0:
            # Collect audio paths and metadata
            audio_paths = []
            metadata_list = []
            
            for audio_path in tqdm(audio_files, desc="Collecting audio files"):
                try:
                    # Extract just the filename without parent directories
                    filename = audio_path.name
                    
                    metadata = {
                        'split': 'train',  # All files go to train split
                        'original_filename': filename,
                        'full_path': str(audio_path.relative_to(self.audio_dir))
                    }
                    
                    audio_paths.append(audio_path)
                    metadata_list.append(metadata)
                    
                except Exception as e:
                    print(f"    Error processing {audio_path}: {e}")
                    continue
            
            print(f"\n  Starting Gemma audio processing for {len(audio_paths)} files...")
            print(f"  Using {self.batch_size} batch size and {self.num_workers} workers")
            
            # Use filenames as text prompts
            texts = [metadata['original_filename'] for metadata in metadata_list]
            
            # Sort by text length for sequence bucketing (improves throughput ~2x)
            print("  Sorting by text length for sequence bucketing...")
            combined = list(zip(audio_paths, texts, metadata_list))
            combined_sorted = sorted(combined, key=lambda x: len(x[1]), reverse=True)
            audio_paths, texts, metadata_list = zip(*combined_sorted)
            audio_paths, texts, metadata_list = list(audio_paths), list(texts), list(metadata_list)
            
            # Process with Gemma using filenames as context
            gemma_results = process_dataset_with_gemma(
                audio_paths=audio_paths,
                texts=texts,  # Filenames as context
                metadata=metadata_list,
                system_prompt="You are a helpful assistant.",
                user_prompt_template="Describe in detail what you hear in the audio in max. 100 words. For context, use the following filename, but don't quote it directly: {text}",
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                generation_config={
                    "max_new_tokens": 200,
                    "temperature": 0.7,
                    "top_p": 0.9,
                }
            )
            
            # Results already in correct format (audio_path, generated_caption, metadata)
            matched.extend(gemma_results)
                            
        print(f"Total processed: {len(matched)} audio-text pairs")
        
        # Save to json as backup (convert Path objects to strings)
        matched_serializable = []
        for item in matched:
            # Convert Path objects to strings
            audio_path = str(item[0]) if hasattr(item[0], '__fspath__') else item[0]
            text = item[1]
            metadata = item[2]
            matched_serializable.append([audio_path, text, metadata])
        
        with open("matched_soundly.json", "w") as f:
            json.dump(matched_serializable, f)
        print(f"Saved {len(matched)} audio-text pairs to matched_soundly.json")
        
        return matched


def main():
    parser = argparse.ArgumentParser(description="Convert Soundly dataset to tar format")
    parser.add_argument("--audio-dir", type=str, 
                       default="/scratch-shared/gwijngaard/laion/soundly/audio",
                       help="Path to Soundly audio directory")
    parser.add_argument("--metadata", type=str,
                       default="/scratch-shared/gwijngaard/laion/soundly",
                       help="Path to metadata (not used for Soundly)")
    parser.add_argument("--output-dir", type=str,
                       default="/scratch-shared/gwijngaard/tar/soundly",
                       help="Output directory for tar files")
    parser.add_argument("--samples-per-tar", type=int, default=2048,
                       help="Number of samples per tar file")
    parser.add_argument("--batch-size", type=int, default=96,
                       help="Batch size for LLM processing")
    parser.add_argument("--num-workers", type=int, default=12,
                       help="Number of workers for data loading")
    parser.add_argument("--no-recaption", action="store_true",
                       help="Disable LLM recaptioning (use original captions)")
    
    args = parser.parse_args()
    
    # Create processor
    processor = SoundlyProcessor(
        audio_dir=args.audio_dir,
        metadata_path=args.metadata,
        output_dir=args.output_dir,
        use_recaption=not args.no_recaption,
        batch_size=args.batch_size,
        num_workers=args.num_workers)
    
    # Process dataset
    processor.process_dataset(samples_per_tar=args.samples_per_tar)


if __name__ == "__main__":
    main()