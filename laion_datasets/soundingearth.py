#!/usr/bin/env python3
"""
Convert SoundingEarth dataset to WebDataset tar format.

Audio location: /scratch-shared/gwijngaard/laion/SoundingEarth/data/
Metadata: /gpfs/work4/0/einf6190/data-preparation/data/SoundingEarth/metadata_enriched.csv
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
from utils_hybrid_streaming import HybridStreamingProcessor
from tqdm import tqdm
import json

load_dotenv()


class SoundingEarthProcessor(DatasetProcessor):
    """Processor for SoundingEarth dataset."""
    
    def __init__(self, audio_dir: str, metadata_path: str, output_dir: str,
                 use_recaption: bool = True, batch_size: int = 128, num_workers: int = 16):
        super().__init__(audio_dir, metadata_path, output_dir)
        self.use_recaption = use_recaption
        self.batch_size = batch_size
        self.num_workers = num_workers
        # Audio files are already extracted in data/ subdirectory
        self.audio_data_dir = self.audio_dir / "data"
        
    def load_metadata(self) -> pd.DataFrame:
        """Load enriched SoundingEarth metadata CSV."""
        print(f"Loading metadata from {self.metadata_path}")
        
        # Try enriched metadata first, fall back to regular metadata
        enriched_path = self.metadata_path / "metadata_enriched.csv"
        regular_path = self.metadata_path / "metadata.csv"
        
        if enriched_path.exists():
            df = pd.read_csv(enriched_path)
            print(f"  Loaded enriched metadata with {len(df)} entries")
        elif regular_path.exists():
            df = pd.read_csv(regular_path)
            print(f"  Loaded regular metadata with {len(df)} entries")
            print("  Warning: Using non-enriched metadata. Run enrich_soundingearth_metadata.py first for location data.")
        else:
            raise FileNotFoundError(f"No metadata found at {self.metadata_path}")
        
        # Keep all entries for Gemma processing
        print(f"Total entries: {len(df)}")
        return df
        
        
    def match_audio_to_text(self, metadata_df: pd.DataFrame) -> List[Tuple[Path, str, Dict]]:
        """Process audio files using metadata and generate captions with Gemma."""
        matched = []
        
        print(f"Processing audio files from {self.audio_data_dir}")
        
        # Check if metadata has split column
        has_split_column = 'split' in metadata_df.columns
        
        # If no split column, create splits - put train everywhere
        if not has_split_column:
            print("No split column found in metadata, assigning all samples to train split")
            metadata_df = metadata_df.copy()
            metadata_df['split'] = 'train'
        
        # Collect audio paths and metadata
        audio_paths = []
        metadata_list = []
        missing_count = 0
        
        for _, row in tqdm(metadata_df.iterrows(), total=len(metadata_df), desc="Matching audio files"):
            filename = row['file_name']
            if pd.isna(filename) or filename == '':
                missing_count += 1
                print(f"Missing filename for row {row['key']}")
                continue
            audio_path = self.audio_data_dir / filename
            
            if not audio_path.exists():
                missing_count += 1
                print(f"Missing audio file for row {row['key']}")
                continue
            
            metadata = {
                'split': row['split'],
                'original_filename': filename,
                'description': row.get('description', ''),
                'title': row.get('title', ''),
                'location': row.get('location', ''),
                'country': row.get('country', ''),
                'original_caption': row.get('caption', ''),
                'licenseurl': row.get('licenseurl', ''),
                'longitude': row.get('longitude', ''),
                'latitude': row.get('latitude', ''),
                'altitude': row.get('altitude', ''),
                'key': row.get('key', ''),
                'task': 'AAC'
            }
            
            audio_paths.append(audio_path)
            metadata_list.append(metadata)
        
        print(f"Found {len(audio_paths)} audio files, missing {missing_count}")
        
        if len(audio_paths) > 0 and self.use_recaption:
            print(f"\n  Starting Gemma audio processing for {len(audio_paths)} files...")
            print(f"  Using {self.batch_size} batch size and {self.num_workers} workers")
            
            # Prepare context texts from metadata
            texts = []
            for metadata in metadata_list:
                # Build comprehensive context
                context_parts = []
                
                if metadata.get('original_caption'):
                    context_parts.append(f"Caption: {metadata['original_caption']}")
                                
                if metadata.get('location'):
                    context_parts.append(f"Location: {metadata['location']}")
                
                if metadata.get('country'):
                    context_parts.append(f"Country: {metadata['country']}")
                
                texts.append(" | ".join(context_parts) if context_parts else "")
            
            # Sort by text length for sequence bucketing
            print("  Sorting by text length for sequence bucketing...")
            combined = list(zip(audio_paths, texts, metadata_list))
            combined_sorted = sorted(combined, key=lambda x: len(x[1]), reverse=True)
            audio_paths, texts, metadata_list = zip(*combined_sorted)
            audio_paths, texts, metadata_list = list(audio_paths), list(texts), list(metadata_list)
            
            # Return matched samples without Gemma processing
            # The streaming processor will handle Gemma processing
            for audio_path, text, metadata in zip(audio_paths, texts, metadata_list):
                matched.append((audio_path, text, metadata))
        else:
            # Use original captions
            for audio_path, metadata in zip(audio_paths, metadata_list):
                caption = metadata.get('original_caption', '')
                matched.append((audio_path, caption, metadata))
        
        print(f"Total processed: {len(matched)} audio-text pairs")
        
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
                user_prompt_template="Describe in detail what you hear in the audio in max. 100 words. Use the following context information if helpful, don't quote it directly: {text}",
                generation_config={
                    "max_new_tokens": 200,
                    "temperature": 0.7,
                    "top_p": 0.9,
                },
                samples_per_tar=samples_per_tar,
                batch_size=self.batch_size,
                num_workers=self.num_workers
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
    parser = argparse.ArgumentParser(description="Convert SoundingEarth dataset to tar format")
    parser.add_argument("--audio-dir", type=str, 
                       default="/scratch-shared/gwijngaard/laion/SoundingEarth",
                       help="Path to SoundingEarth directory containing data/ subdirectory")
    parser.add_argument("--metadata", type=str,
                       default="/gpfs/work4/0/einf6190/data-preparation/data/SoundingEarth",
                       help="Path to metadata directory")
    parser.add_argument("--output-dir", type=str,
                       default="/scratch-shared/gwijngaard/tar/soundingearth",
                       help="Output directory for tar files")
    parser.add_argument("--samples-per-tar", type=int, default=2048,
                       help="Number of samples per tar file")
    parser.add_argument("--batch-size", type=int, default=32,
                       help="Batch size for LLM processing")
    parser.add_argument("--num-workers", type=int, default=12,
                       help="Number of workers for data loading")
    parser.add_argument("--no-recaption", action="store_true",
                       help="Disable LLM recaptioning (use original captions)")
    
    args = parser.parse_args()
    
    # Create processor
    processor = SoundingEarthProcessor(
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