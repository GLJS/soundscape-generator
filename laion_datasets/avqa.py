#!/usr/bin/env python3
"""
Convert AVQA dataset to WebDataset tar format.

Audio location: /scratch-shared/gwijngaard/laion/AVQA/
Note: This dataset appears to be empty or requires special handling
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from pathlib import Path
from utils import DatasetProcessor, TarCreator
from typing import List, Tuple, Dict
import argparse
from dotenv import load_dotenv

load_dotenv()


class AVQAProcessor(DatasetProcessor):
    """Processor for AVQA dataset."""
    
    def load_metadata(self) -> pd.DataFrame:
        """Load AVQA metadata."""
        print(f"Checking for metadata in {self.metadata_path}")
        # AVQA might not have metadata in the standard location
        # This is a placeholder implementation
        return pd.DataFrame()
        
    def match_audio_to_text(self, metadata_df: pd.DataFrame) -> List[Tuple[Path, str, Dict]]:
        """Match audio files to their captions."""
        matched = []
        
        # Check what's in the AVQA directory
        print(f"Checking {self.audio_dir} for content...")
        
        if self.audio_dir.exists():
            files = list(self.audio_dir.rglob("*"))
            print(f"Found {len(files)} files/directories")
            
            # Log first few entries to understand structure
            for i, f in enumerate(files[:10]):
                print(f"  {f.relative_to(self.audio_dir)}")
        else:
            print(f"Directory {self.audio_dir} does not exist")
            
        return matched


def main():
    parser = argparse.ArgumentParser(description="Convert AVQA dataset to tar format")
    parser.add_argument("--audio-dir", type=str, 
                       default="/scratch-shared/gwijngaard/laion/AVQA",
                       help="Path to audio files")
    parser.add_argument("--metadata", type=str,
                       default="/gpfs/work4/0/einf6190/data-preparation/data/AVQA",
                       help="Path to metadata")
    parser.add_argument("--output-dir", type=str,
                       default="/scratch-shared/gwijngaard/tar/avqa",
                       help="Output directory for tar files")
    parser.add_argument("--samples-per-tar", type=int, default=2048,
                       help="Number of samples per tar file")
    
    args = parser.parse_args()
    
    # Create processor
    processor = AVQAProcessor(
        audio_dir=args.audio_dir,
        metadata_path=args.metadata,
        output_dir=args.output_dir,
        task="AQA")
    
    # Process dataset
    processor.process_dataset(samples_per_tar=args.samples_per_tar)


if __name__ == "__main__":
    main()