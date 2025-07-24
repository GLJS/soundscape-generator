#!/usr/bin/env python3
"""
Convert Downloaded SFX dataset to WebDataset tar format.

Audio location: /gpfs/scratch1/shared/gwijngaard/laion/downloaded_sfx/extracted
Note: This dataset only has audio files, filenames and directory names are used as prompts
Supports: wav, aif, aiff, flac, mp3 files
"""

import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from pathlib import Path
from utils import DatasetProcessor
from typing import List, Tuple, Dict
import argparse
from dotenv import load_dotenv
from utils_hybrid_streaming import HybridStreamingProcessor
from tqdm import tqdm
import json
import re
load_dotenv()


class DownloadedSFXProcessor(DatasetProcessor):
    """Processor for Downloaded SFX dataset."""
    
    # Generic directory names that don't provide useful context
    GENERIC_DIR_NAMES = {
        'sfx', 'fx', 'sound', 'sounds', 'audio', 'samples', 'loop', 'loops',
        'single hit', 'single_hit', 'one shot', 'oneshot', 'one_shot',
        'percussion', 'perc', 'kick', 'snare', 'hi hat', 'hihat', 'hi_hat',
        'bass', 'synth', 'vocal', 'vocals', 'clap', 'claps', 'drum', 'drums',
        'impact', 'impacts', 'movement', 'movements', 'friction', 'misc',
        'archived', 'impulse response', 'hardware', 'mechanics', 'tick',
        'chime', 'wav', 'aif', 'aiff', 'flac', 'mp3', 'stereo', 'mono',
        'part1', 'part2', 'part3', 'part4', 'part5', 'part6', 'part7', 'part8',
        'toolkit', 'bundle', 'pack', 'vol', 'volume', 'collection'
    }
    
    def __init__(self, audio_dir: str, metadata_path: str, output_dir: str, 
                 use_recaption: bool = True, batch_size: int = 128, num_workers: int = 16):
        super().__init__(audio_dir, metadata_path, output_dir)
        self.use_recaption = use_recaption
        self.batch_size = batch_size
        self.num_workers = num_workers
        
    def load_metadata(self) -> pd.DataFrame:
        """Load metadata - returns empty DataFrame as Downloaded SFX has no CSV."""
        print(f"Loading audio files from {self.audio_dir}")
        # No metadata CSV for Downloaded SFX dataset
        return pd.DataFrame()
    
    def is_useful_directory_name(self, dir_name: str) -> bool:
        """Check if a directory name contains useful information."""
        # Convert to lowercase for comparison
        dir_lower = dir_name.lower()
        
        # Remove common separators and clean up
        dir_cleaned = re.sub(r'[-_]', ' ', dir_lower)
        dir_cleaned = re.sub(r'\s+', ' ', dir_cleaned).strip()
        
        # Check if it's a generic name
        if dir_cleaned in self.GENERIC_DIR_NAMES:
            return False
        
        # Check if it's just numbers
        if dir_cleaned.isdigit():
            return False
        
        # Check if it's too short (less than 3 characters)
        if len(dir_cleaned) < 3:
            return False
        
        # Check if any word in the directory name is generic
        words = dir_cleaned.split()
        if all(word in self.GENERIC_DIR_NAMES for word in words):
            return False
        
        # If it contains meaningful words, it's useful
        return True
    
    def extract_useful_context(self, audio_path: Path) -> str:
        """Extract useful context from directory hierarchy."""
        # Get all parent directories up to the base directory
        parents = []
        current = audio_path.parent
        base = self.audio_dir
        
        while current != base and current.parent != current:
            parents.append(current.name)
            current = current.parent
            if current == base:
                break
        
        # Reverse to get top-down order
        parents.reverse()
        
        # Find useful directory names
        useful_dirs = []
        for dir_name in parents:
            if self.is_useful_directory_name(dir_name):
                # Clean up the directory name
                cleaned = re.sub(r'[-_]', ' ', dir_name)
                cleaned = re.sub(r'\s+', ' ', cleaned).strip()
                
                # Remove duplicate words (e.g., "ZTEKNO - GLOBAL FX/ZTEKNO - GLOBAL FX")
                if useful_dirs and cleaned in useful_dirs[-1]:
                    continue
                    
                useful_dirs.append(cleaned)
        
        # Combine useful directory names
        if useful_dirs:
            # Take the most specific (last) useful directory
            return useful_dirs[-1]
        else:
            return ""
        
    def match_audio_to_text(self, metadata_df: pd.DataFrame) -> List[Tuple[Path, str, Dict]]:
        """Process audio files using filename and directory name as prompt text."""
        matched = []
        
        print(f"Processing audio files from {self.audio_dir}")
        
        # Find all audio files with supported extensions
        extensions = ['.wav', '.aif', '.aiff', '.flac', '.mp3']
        audio_files = []
        
        for ext in extensions:
            # Case-insensitive search
            audio_files.extend(self.audio_dir.rglob(f"*{ext}"))
            audio_files.extend(self.audio_dir.rglob(f"*{ext.upper()}"))
        
        # Remove duplicates
        audio_files = list(set(audio_files))
        print(f"  Found {len(audio_files)} audio files")
        
        if len(audio_files) > 0:
            # Collect audio paths and metadata
            audio_paths = []
            metadata_list = []
            
            for audio_path in tqdm(audio_files, desc="Collecting audio files"):
                try:
                    # Extract filename
                    filename = audio_path.name
                    
                    # Extract useful context from directory hierarchy
                    dir_context = self.extract_useful_context(audio_path)
                    
                    # Combine directory context and filename if we have useful context
                    if dir_context:
                        combined_text = f"{dir_context} {filename}"
                    else:
                        combined_text = filename
                    
                    metadata = {
                        'split': 'train',  # All files go to train split
                        'original_filename': filename,
                        'directory_context': dir_context,
                        'combined_text': combined_text,
                        'full_path': str(audio_path.relative_to(self.audio_dir))
                    }
                    
                    audio_paths.append(audio_path)
                    metadata_list.append(metadata)
                    
                except Exception as e:
                    print(f"    Error processing {audio_path}: {e}")
                    continue
            
            print(f"\n  Starting Gemma audio processing for {len(audio_paths)} files...")
            print(f"  Using {self.batch_size} batch size and {self.num_workers} workers")
            
            # Use combined directory + filename as text prompts
            texts = [metadata['combined_text'] for metadata in metadata_list]
            
            # Sort by text length for sequence bucketing (improves throughput ~2x)
            print("  Sorting by text length for sequence bucketing...")
            combined = list(zip(audio_paths, texts, metadata_list))
            combined_sorted = sorted(combined, key=lambda x: len(x[1]), reverse=True)
            audio_paths, texts, metadata_list = zip(*combined_sorted)
            audio_paths, texts, metadata_list = list(audio_paths), list(texts), list(metadata_list)
            
            # Return matched samples without Gemma processing
            # The streaming processor will handle Gemma processing
            for audio_path, text, metadata in zip(audio_paths, texts, metadata_list):
                matched.append((audio_path, text, metadata))
            
            # Log matching summary
            print(f"\n  Matching summary:")
            print(f"    Total files found: {len(audio_paths)}")
            print(f"    Ready for processing: {len(matched)}")
                            
        print(f"\nTotal processed: {len(matched)} audio-text pairs")
        
        return matched
    
    def process_dataset(self, samples_per_tar: int = 2048):
        """Process the entire dataset into tar files using hybrid streaming approach."""
        # Load metadata
        metadata_df = self.load_metadata()
        
        # Match audio to text
        matched_samples = self.match_audio_to_text(metadata_df)
        
        # If not using recaption, fall back to parent class implementation
        if not self.use_recaption:
            return super().process_dataset(samples_per_tar)
        
        # Group samples by split
        samples_by_split = {}
        for audio_path, text, metadata in matched_samples:
            split = metadata.get('split', 'train')
            if split not in samples_by_split:
                samples_by_split[split] = []
            samples_by_split[split].append((audio_path, text, metadata))
        
        # Process each split with hybrid streaming processor
        all_stats = {}
        for split, split_samples in samples_by_split.items():
            print(f"\nProcessing {split} split with {len(split_samples)} samples...")
            
            # Create hybrid streaming processor
            processor = HybridStreamingProcessor(
                output_dir=self.output_dir,
                model_name="google/gemma-3n-e4b-it",
                system_prompt="You are a helpful assistant.",
                user_prompt_template="Describe in detail what you hear in the audio in max. 100 words. For context, use the following filename, but don't quote it directly: {text}",
                generation_config={
                    "max_new_tokens": 200,
                    "temperature": 0.7,
                    "top_p": 0.9,
                },
                samples_per_tar=samples_per_tar,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                sglang_server_url=os.getenv("SGLANG_SERVER_URL", "http://127.0.0.1:30000")
            )
            
            # Process with hybrid streaming
            stats = processor.process_dataset(
                audio_files=split_samples,
                prefix=self.__class__.__name__.lower().replace('processor', ''),
                split=split,
                show_progress=True
            )
            
            all_stats[split] = stats
        
        # Summary
        total_processed = sum(s['written'] for s in all_stats.values())
        total_failed = sum(s['failed'] for s in all_stats.values())
        
        print(f"\nProcessing complete!")
        print(f"Total samples attempted: {len(matched_samples)}")
        print(f"Successfully processed: {total_processed}")
        print(f"Failed: {total_failed}")
        
        return all_stats


def main():
    parser = argparse.ArgumentParser(description="Convert Downloaded SFX dataset to tar format")
    parser.add_argument("--audio-dir", type=str, 
                       default="/gpfs/scratch1/shared/gwijngaard/laion/downloaded_sfx/extracted",
                       help="Path to Downloaded SFX audio directory")
    parser.add_argument("--metadata", type=str,
                       default="/gpfs/scratch1/shared/gwijngaard/laion/downloaded_sfx",
                       help="Path to metadata (not used for Downloaded SFX)")
    parser.add_argument("--output-dir", type=str,
                       default="/scratch-shared/gwijngaard/tar/downloaded_sfx",
                       help="Output directory for tar files")
    parser.add_argument("--samples-per-tar", type=int, default=2048,
                       help="Number of samples per tar file")
    parser.add_argument("--batch-size", type=int, default=16,
                       help="Batch size for LLM processing")
    parser.add_argument("--num-workers", type=int, default=12,
                       help="Number of workers for data loading")
    parser.add_argument("--no-recaption", action="store_true",
                       help="Disable LLM recaptioning (use original captions)")
    
    args = parser.parse_args()
    
    # Create processor
    processor = DownloadedSFXProcessor(
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