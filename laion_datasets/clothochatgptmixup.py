#!/usr/bin/env python3
"""
Convert ClothoChatGPTMixup dataset to WebDataset tar format.

Audio location: /scratch-shared/gwijngaard/laion/ClothoChatGPTMixup/audio/ (extracted folder)
Metadata: /gpfs/work4/0/einf6190/data-preparation/data/ClothoChatGPTMixup/mixed_audio_info.csv
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from pathlib import Path
from utils import DatasetProcessor, AudioProcessor, TarCreator, ArchiveExtractor
from typing import List, Tuple, Dict
import argparse
import tempfile
from dotenv import load_dotenv

load_dotenv()


class ClothoChatGPTMixupProcessor(DatasetProcessor):
    """Processor for ClothoChatGPTMixup dataset."""
    
    def __init__(self, audio_dir: str, metadata_path: str, output_dir: str):
        super().__init__(audio_dir, metadata_path, output_dir)
        
    def load_metadata(self) -> pd.DataFrame:
        """Load ClothoChatGPTMixup metadata CSV."""
        print(f"Loading metadata from {self.metadata_path}")
        df = pd.read_csv(self.metadata_path)
        print(f"Loaded {len(df)} entries")
        return df
        
        
    def match_audio_to_text(self, metadata_df: pd.DataFrame) -> List[Tuple[Path, str, Dict]]:
        """Match audio files to their captions."""
        matched = []
        missing_count = 0
        
        # Check if metadata has split column
        has_split_column = 'split' in metadata_df.columns
        
        # If no split column, create splits - put train everywhere
        if not has_split_column:
            print("No split column found in metadata, assigning all samples to train split")
            
            # Add split column to dataframe
            metadata_df = metadata_df.copy()
            metadata_df['split'] = 'train'  # Put train everywhere
        
        for _, row in metadata_df.iterrows():
            filename = row['filename']
            audio_path = self.audio_dir / filename
            
            if audio_path.exists():
                caption = row['combined_caption']
                metadata = {
                    'split': row['split'],  # Now guaranteed to exist
                    'original_filename': filename,
                    'task': 'AAC'
                }
                # Pass the Path object
                matched.append((audio_path, caption, metadata))
            else:
                missing_count += 1
                print(f"Missing audio file: {filename}")
                
        print(f"Matched {len(matched)} audio-text pairs")
        print(f"Missing audio files: {missing_count}")
        
        return matched
        


def main():
    parser = argparse.ArgumentParser(description="Convert ClothoChatGPTMixup dataset to tar format")
    parser.add_argument("--audio-dir", type=str, 
                       default="/scratch-shared/gwijngaard/laion/ClothoChatGPTMixup/audio",
                       help="Path to directory containing extracted audio files")
    parser.add_argument("--metadata", type=str,
                       default="/gpfs/work4/0/einf6190/data-preparation/data/ClothoChatGPTMixup/mixed_audio_info.csv",
                       help="Path to metadata CSV")
    parser.add_argument("--output-dir", type=str,
                       default="/scratch-shared/gwijngaard/tar/clothochatgptmixup",
                       help="Output directory for tar files")
    parser.add_argument("--samples-per-tar", type=int, default=2048,
                       help="Number of samples per tar file")
    
    args = parser.parse_args()
    
    # Create processor
    processor = ClothoChatGPTMixupProcessor(
        audio_dir=args.audio_dir,
        metadata_path=args.metadata,
        output_dir=args.output_dir
    )
    
    # Process dataset
    processor.process_dataset(samples_per_tar=args.samples_per_tar)


if __name__ == "__main__":
    main()