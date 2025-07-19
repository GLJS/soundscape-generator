#!/usr/bin/env python3
"""
Convert LAION-630k dataset to WebDataset tar format.

Audio location: /scratch-shared/gwijngaard/laion/laion-630k/ (currently empty)
Metadata: /gpfs/work4/0/einf6190/data-preparation/data/LAION630k/combined.csv
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

load_dotenv()


class LAION630kProcessor(DatasetProcessor):
    """Processor for LAION-630k dataset."""
    
    def load_metadata(self) -> pd.DataFrame:
        """Load LAION-630k metadata CSV."""
        print(f"Loading metadata from {self.metadata_path}")
        
        csv_path = self.metadata_path / "combined.csv"
        
        # Load in chunks due to large size
        chunk_size = 10000
        chunks = []
        
        for chunk in pd.read_csv(csv_path, chunksize=chunk_size):
            chunks.append(chunk)
            if len(chunks) * chunk_size >= 100000:  # Limit for testing
                break
                
        df = pd.concat(chunks, ignore_index=True)
        
        # Rename 'text' column to 'caption'
        df.rename(columns={'text': 'caption'}, inplace=True)
        
        # Build full file path
        df['full_path'] = df.apply(
            lambda x: f"{x['directory']}/{x['subdirectory']}/{x['file_name']}", 
            axis=1
        )
        
        print(f"Loaded {len(df)} entries")
        return df
        
    def match_audio_to_text(self, metadata_df: pd.DataFrame) -> List[Tuple[Path, str, Dict]]:
        """Match audio files to their captions."""
        matched = []
        missing_count = 0
        
        # Group by dataset (directory)
        for dataset, group_df in metadata_df.groupby('directory'):
            print(f"\nProcessing {dataset} dataset ({len(group_df)} files)...")
            
            # Check if dataset directory exists
            dataset_dir = self.audio_dir.parent / dataset
            if not dataset_dir.exists():
                print(f"  Dataset directory {dataset_dir} not found")
                missing_count += len(group_df)
                continue
                
            # Try to find audio files
            dataset_matched = 0
            for _, row in group_df.iterrows():
                # Try different possible locations
                possible_paths = [
                    self.audio_dir / row['full_path'],
                    dataset_dir / row['subdirectory'] / row['file_name'],
                    dataset_dir / row['file_name']
                ]
                
                audio_found = False
                for audio_path in possible_paths:
                    if audio_path.exists():
                        caption = row['caption']
                        metadata = {
                            'split': row.get('subdirectory', 'train'),
                            'original_filename': row['file_name'],
                            'dataset': row['directory'],
                            'id': row.get('id', ''),
                            'author': row.get('author', ''),
                            'collection_name': row.get('collection_name', ''),
                            'tag': row.get('tag', '')
                        }
                        matched.append((audio_path, caption, metadata))
                        dataset_matched += 1
                        audio_found = True
                        break
                        
                if not audio_found:
                    missing_count += 1
                    
            print(f"  Matched {dataset_matched} files from {dataset}")
                
        print(f"\nTotal matched: {len(matched)} audio-text pairs")
        print(f"Total missing: {missing_count}")
        
        return matched


def main():
    parser = argparse.ArgumentParser(description="Convert LAION-630k dataset to tar format")
    parser.add_argument("--audio-dir", type=str, 
                       default="/scratch-shared/gwijngaard/laion/laion-630k",
                       help="Path to audio files")
    parser.add_argument("--metadata", type=str,
                       default="/gpfs/work4/0/einf6190/data-preparation/data/LAION630k",
                       help="Path to metadata directory")
    parser.add_argument("--output-dir", type=str,
                       default="/scratch-shared/gwijngaard/tar/laion630k",
                       help="Output directory for tar files")
    parser.add_argument("--samples-per-tar", type=int, default=2048,
                       help="Number of samples per tar file")
    parser.add_argument("--dataset-filter", type=str, nargs="+",
                       help="Only process specific datasets")
    
    args = parser.parse_args()
    
    # Create processor
    processor = LAION630kProcessor(
        audio_dir=args.audio_dir,
        metadata_path=args.metadata,
        output_dir=args.output_dir,
        task="AAC")
    
    # Load metadata
    metadata_df = processor.load_metadata()
    
    # Filter by dataset if specified
    if args.dataset_filter:
        metadata_df = metadata_df[metadata_df['directory'].isin(args.dataset_filter)]
        print(f"Filtered to {len(metadata_df)} entries for datasets: {args.dataset_filter}")
    
    # Process dataset
    processor.process_dataset(samples_per_tar=args.samples_per_tar)


if __name__ == "__main__":
    main()