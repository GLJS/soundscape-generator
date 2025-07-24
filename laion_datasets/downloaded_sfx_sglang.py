#!/usr/bin/env python3
"""
Downloaded SFX dataset processor using SGLang for caption generation.
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
from utils_hybrid_streaming_sglang import HybridStreamingProcessor
from tqdm import tqdm
import re

load_dotenv()


class DownloadedSFXProcessor(DatasetProcessor):
    """Processor for Downloaded SFX dataset with SGLang captioning."""
    
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
                 batch_size: int = 128, num_workers: int = 16):
        super().__init__(audio_dir, metadata_path, output_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        
    def load_metadata(self) -> pd.DataFrame:
        """No metadata CSV for this dataset."""
        print(f"Loading audio files from {self.audio_dir}")
        return pd.DataFrame()
    
    def is_useful_directory_name(self, dir_name: str) -> bool:
        """Check if a directory name contains useful information."""
        dir_lower = dir_name.lower()
        dir_cleaned = re.sub(r'[-_]', ' ', dir_lower)
        dir_cleaned = re.sub(r'\s+', ' ', dir_cleaned).strip()
        
        if dir_cleaned in self.GENERIC_DIR_NAMES:
            return False
        if dir_cleaned.isdigit():
            return False
        if len(dir_cleaned) < 3:
            return False
        
        words = dir_cleaned.split()
        if all(word in self.GENERIC_DIR_NAMES for word in words):
            return False
        
        return True
    
    def extract_useful_context(self, audio_path: Path) -> str:
        """Extract useful context from directory hierarchy."""
        parents = []
        current = audio_path.parent
        base = self.audio_dir
        
        while current != base and current.parent != current:
            parents.append(current.name)
            current = current.parent
            if current == base:
                break
        
        parents.reverse()
        
        useful_dirs = []
        for dir_name in parents:
            if self.is_useful_directory_name(dir_name):
                cleaned = re.sub(r'[-_]', ' ', dir_name)
                cleaned = re.sub(r'\s+', ' ', cleaned).strip()
                
                if useful_dirs and cleaned in useful_dirs[-1]:
                    continue
                    
                useful_dirs.append(cleaned)
        
        if useful_dirs:
            return useful_dirs[-1]
        else:
            return ""
        
    def match_audio_to_text(self, metadata_df: pd.DataFrame) -> List[Tuple[Path, str, Dict]]:
        """Process audio files using filename and directory as context."""
        matched = []
        
        print(f"Processing audio files from {self.audio_dir}")
        
        # Find all audio files
        extensions = ['.wav', '.aif', '.aiff', '.flac', '.mp3']
        audio_files = []
        
        for ext in extensions:
            audio_files.extend(self.audio_dir.rglob(f"*{ext}"))
            audio_files.extend(self.audio_dir.rglob(f"*{ext.upper()}"))
        
        audio_files = list(set(audio_files))
        print(f"  Found {len(audio_files)} audio files")
        
        if len(audio_files) > 0:
            for audio_path in tqdm(audio_files, desc="Collecting audio files"):
                try:
                    filename = audio_path.name
                    filename_clean = re.sub(r'[-_]', ' ', audio_path.stem)
                    filename_clean = re.sub(r'\d+$', '', filename_clean).strip()
                    
                    # Extract directory context
                    dir_context = self.extract_useful_context(audio_path)
                    
                    # Build text description
                    text_parts = []
                    if dir_context:
                        text_parts.append(f"Category: {dir_context}")
                    if filename_clean:
                        text_parts.append(f"Name: {filename_clean}")
                    
                    text = " | ".join(text_parts) if text_parts else filename
                    
                    metadata = {
                        'split': 'train',
                        'original_filename': filename,
                        'directory_context': dir_context,
                        'full_path': str(audio_path.relative_to(self.audio_dir))
                    }
                    
                    matched.append((audio_path, text, metadata))
                    
                except Exception as e:
                    print(f"Error processing {audio_path}: {e}")
                    continue
        
        print(f"Total matched: {len(matched)} audio files")
        return matched
    
    def process_dataset(self, samples_per_tar: int = 2048, server_url: str = "http://127.0.0.1:30000"):
        """Process dataset using SGLang hybrid streaming."""
        # Load metadata (empty for this dataset)
        metadata_df = self.load_metadata()
        
        # Match audio to text
        matched_samples = self.match_audio_to_text(metadata_df)
        
        # Group by split
        samples_by_split = {}
        for audio_path, text, metadata in matched_samples:
            split = metadata.get('split', 'train')
            if split not in samples_by_split:
                samples_by_split[split] = []
            samples_by_split[split].append((audio_path, text, metadata))
        
        # Process each split
        all_stats = {}
        for split, split_samples in samples_by_split.items():
            print(f"\nProcessing {split} split with {len(split_samples)} samples...")
            
            # Create hybrid streaming processor
            processor = HybridStreamingProcessor(
                output_dir=self.output_dir,
                model_name="google/gemma-3n-e4b-it",
                system_prompt="You are a helpful assistant.",
                user_prompt_template="Describe in detail what you hear in the audio in max. 100 words. Context: {text}",
                generation_config={
                    "max_new_tokens": 200,
                    "temperature": 0.7,
                    "top_p": 0.9,
                },
                samples_per_tar=samples_per_tar,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                sglang_server_url=server_url
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
    parser = argparse.ArgumentParser(description="Process Downloaded SFX with SGLang")
    parser.add_argument("--audio-dir", type=str, 
                       default="/gpfs/scratch1/shared/gwijngaard/laion/downloaded_sfx/extracted",
                       help="Path to Downloaded SFX audio directory")
    parser.add_argument("--metadata", type=str,
                       default="/gpfs/scratch1/shared/gwijngaard/laion/downloaded_sfx",
                       help="Path to metadata (not used)")
    parser.add_argument("--output-dir", type=str,
                       default="/scratch-shared/gwijngaard/tar/downloaded_sfx",
                       help="Output directory for tar files")
    parser.add_argument("--samples-per-tar", type=int, default=2048,
                       help="Number of samples per tar file")
    parser.add_argument("--batch-size", type=int, default=128,
                       help="Batch size for processing")
    parser.add_argument("--num-workers", type=int, default=16,
                       help="Number of workers for data loading")
    parser.add_argument("--server-url", type=str, default="http://127.0.0.1:30000",
                       help="SGLang server URL")
    
    args = parser.parse_args()
    
    # Create processor
    processor = DownloadedSFXProcessor(
        audio_dir=args.audio_dir,
        metadata_path=args.metadata,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    # Process dataset
    processor.process_dataset(
        samples_per_tar=args.samples_per_tar,
        server_url=args.server_url
    )


if __name__ == "__main__":
    main()