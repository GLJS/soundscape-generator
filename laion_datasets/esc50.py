#!/usr/bin/env python3
"""
Convert ESC50 dataset to WebDataset tar format.

Audio location: /scratch-shared/gwijngaard/laion/ESC50/audio/
Metadata: /gpfs/work4/0/einf6190/data-preparation/data/ESC50/esc50.csv
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
import random

load_dotenv()


class ESC50Processor(DatasetProcessor):
    """Processor for ESC50 dataset."""
    
    def __init__(self, audio_dir: str, metadata_path: str, output_dir: str):
        super().__init__(audio_dir, metadata_path, output_dir)
        self.audio_path = self.audio_dir / "audio"
        
    def load_metadata(self) -> pd.DataFrame:
        """Load ESC50 metadata CSV."""
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
        
        # If no split column, use fold information to create splits
        # ESC50 has 5 folds, typically fold 5 is used for test
        if not has_split_column:
            print("No split column found, using fold information to assign splits")
            metadata_df = metadata_df.copy()
            # Use fold 5 for test, fold 4 for valid, rest for train
            metadata_df['split'] = metadata_df['fold'].apply(
                lambda x: 'test' if x == 5 else ('valid' if x == 4 else 'train')
            )
        
        for _, row in metadata_df.iterrows():
            filename = row['filename']
            audio_file_path = self.audio_path / filename
            
            if audio_file_path.exists():
                # Pass the file path to the audio processor
                # ESC50 uses 'category' as the label/caption
                category = row['category'].replace('_', ' ')
                
                # Create varied captions
                caption_templates = [
                    f"This audio contains a {category}.",
                    f"The sound of a {category}.",
                    f"A recording of a {category}.",
                    f"This is the sound of a {category}.",
                    f"Audio recording of a {category}.",
                    f"You can hear a {category} in this recording.",
                    f"This recording features a {category}.",
                    f"The audio captures a {category}.",
                    f"Sound of a {category}.",
                    f"A {category} sound.",
                ]
                
                caption = random.choice(caption_templates)
                
                metadata = {
                    'split': row['split'],  # Now guaranteed to exist
                    'original_filename': filename,
                    'target': row.get('target', ''),
                    'fold': row.get('fold', ''),
                    'esc10': row.get('esc10', False),
                    'task': 'AAC'
                }
                matched.append((audio_file_path, caption, metadata))
            else:
                missing_count += 1
                print(f"Missing audio file: {audio_file_path}")
                
        print(f"Matched {len(matched)} audio-text pairs")
        print(f"Missing audio files: {missing_count}")
        
        return matched
        


def main():
    parser = argparse.ArgumentParser(description="Convert ESC50 dataset to tar format")
    parser.add_argument("--audio-dir", type=str, 
                       default="/scratch-shared/gwijngaard/laion/ESC50",
                       help="Path to directory containing audio folder")
    parser.add_argument("--metadata", type=str,
                       default="/gpfs/work4/0/einf6190/data-preparation/data/ESC50/esc50.csv",
                       help="Path to metadata CSV")
    parser.add_argument("--output-dir", type=str,
                       default="/scratch-shared/gwijngaard/tar/esc50",
                       help="Output directory for tar files")
    parser.add_argument("--samples-per-tar", type=int, default=2048,
                       help="Number of samples per tar file")
    
    args = parser.parse_args()
    
    # Create processor
    processor = ESC50Processor(
        audio_dir=args.audio_dir,
        metadata_path=args.metadata,
        output_dir=args.output_dir
    )
    
    # Process dataset
    processor.process_dataset(samples_per_tar=args.samples_per_tar)


if __name__ == "__main__":
    main()