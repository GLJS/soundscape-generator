#!/usr/bin/env python3
"""
Convert Auto-ACD dataset to WebDataset tar format.

Audio location: /scratch-shared/gwijngaard/laion/Auto-ACD/flac/ (extracted FLAC files)
Metadata: /gpfs/work4/0/einf6190/data-preparation/data/AutoACD/train.csv and test.csv
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from pathlib import Path
from utils import DatasetProcessor, AudioProcessor, TarCreator
from typing import List, Tuple, Dict
import argparse
from dotenv import load_dotenv

load_dotenv()


class AutoACDProcessor(DatasetProcessor):
    """Processor for Auto-ACD dataset."""
    
    def __init__(self, audio_dir: str, metadata_path: str, output_dir: str):
        super().__init__(audio_dir, metadata_path, output_dir)
        self.flac_dir = self.audio_dir / "flac"
        
    def load_metadata(self) -> pd.DataFrame:
        """Load Auto-ACD metadata CSVs."""
        print(f"Loading metadata from {self.metadata_path}")
        
        # Load train and test CSVs
        train_df = pd.read_csv(self.metadata_path / "train.csv")
        train_df['split'] = 'train'
        
        test_df = pd.read_csv(self.metadata_path / "test.csv")
        test_df['split'] = 'test'
        
        # Combine
        df = pd.concat([train_df, test_df], ignore_index=True)
        
        # Add file extension
        df['file_name'] = df['youtube_id'] + '.flac'
        
        print(f"Loaded {len(df)} entries ({len(train_df)} train, {len(test_df)} test)")
        return df
        
    def match_audio_to_text(self, metadata_df: pd.DataFrame) -> List[Tuple[Path, str, Dict]]:
        """Match audio files to their captions."""
        matched = []
        missing_count = 0
        
        # Check if flac directory exists
        if not self.flac_dir.exists():
            print(f"ERROR: FLAC directory not found at {self.flac_dir}")
            return []
            
        print(f"Using FLAC files from {self.flac_dir}")
        
        # Index available files
        audio_files = {}
        for audio_file in self.flac_dir.glob("*.flac"):
            audio_files[audio_file.name] = audio_file
            
        print(f"Found {len(audio_files)} FLAC files")
        
        # Match with metadata
        for _, row in metadata_df.iterrows():
            filename = row['file_name']
            
            if filename in audio_files:
                audio_path = audio_files[filename]
                caption = row['caption']
                metadata = {
                    'split': row['split'],
                    'original_filename': filename,
                    'youtube_id': row['youtube_id'],
                    'task': 'AAC'
                }
                matched.append((audio_path, caption, metadata))
            else:
                missing_count += 1
                
        print(f"Matched {len(matched)} audio-text pairs")
        print(f"Missing audio files: {missing_count}")
        
        return matched


def main():
    parser = argparse.ArgumentParser(description="Convert Auto-ACD dataset to tar format")
    parser.add_argument("--audio-dir", type=str, 
                       default="/scratch-shared/gwijngaard/laion/Auto-ACD",
                       help="Path to directory containing flac/ subdirectory")
    parser.add_argument("--metadata", type=str,
                       default="/gpfs/work4/0/einf6190/data-preparation/data/AutoACD",
                       help="Path to directory containing train.csv and test.csv")
    parser.add_argument("--output-dir", type=str,
                       default="/scratch-shared/gwijngaard/tar/autoacd",
                       help="Output directory for tar files")
    parser.add_argument("--samples-per-tar", type=int, default=2048,
                       help="Number of samples per tar file")
    
    args = parser.parse_args()
    
    # Create processor
    processor = AutoACDProcessor(
        audio_dir=args.audio_dir,
        metadata_path=args.metadata,
        output_dir=args.output_dir)
    
    # Process dataset
    processor.process_dataset(samples_per_tar=args.samples_per_tar)


if __name__ == "__main__":
    main()