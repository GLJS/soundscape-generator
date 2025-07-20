#!/usr/bin/env python3
"""
Convert UrbanSound8K dataset to WebDataset tar format.

Audio location: /gpfs/scratch1/shared/gwijngaard/laion/urbansound8k/
Metadata: /gpfs/scratch1/shared/gwijngaard/laion/urbansound8k/UrbanSound8K.csv
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


class UrbanSound8KProcessor(DatasetProcessor):
    """Processor for UrbanSound8K dataset."""
    
    def __init__(self, audio_dir: str, metadata_path: str, output_dir: str):
        super().__init__(audio_dir, metadata_path, output_dir)
        
    def load_metadata(self) -> pd.DataFrame:
        """Load UrbanSound8K metadata CSV."""
        print(f"Loading metadata from {self.metadata_path}")
        
        # UrbanSound8K CSV is in the same directory as audio
        csv_path = self.audio_dir / "UrbanSound8K.csv"
        df = pd.read_csv(csv_path)
        
        # Use fold 10 for test, fold 9 for validation, rest for train
        df['split'] = df['fold'].apply(
            lambda x: 'test' if x == 10 else ('valid' if x == 9 else 'train')
        )
        
        print(f"Loaded {len(df)} entries")
        print(f"Train: {len(df[df['split'] == 'train'])}, Valid: {len(df[df['split'] == 'valid'])}, Test: {len(df[df['split'] == 'test'])}")
        
        return df
        
    def create_caption(self, class_name: str) -> str:
        """Create varied captions for urban sound classes."""
        # Clean up class name (replace underscores with spaces)
        class_clean = class_name.replace('_', ' ')
        
        # Class-specific descriptions for more natural captions
        class_descriptions = {
            'air_conditioner': 'an air conditioner running',
            'car_horn': 'a car horn honking',
            'children_playing': 'children playing',
            'dog_bark': 'a dog barking',
            'drilling': 'drilling sounds',
            'engine_idling': 'an engine idling',
            'gun_shot': 'a gunshot',
            'jackhammer': 'a jackhammer operating',
            'siren': 'a siren wailing',
            'street_music': 'street music playing'
        }
        
        # Get the description or fall back to class name
        description = class_descriptions.get(class_name, class_clean)
        
        # Create varied caption templates
        caption_templates = [
            f"The sound of {description}.",
            f"Audio recording of {description}.",
            f"This recording captures {description}.",
            f"Urban sound: {description}.",
            f"You can hear {description} in this recording.",
            f"This audio contains {description}.",
            f"Sound of {description} in an urban environment.",
            f"Recording of {description}.",
            f"City sounds: {description}.",
            f"This is the sound of {description}.",
            f"Urban audio featuring {description}.",
            f"Environmental sound of {description}."
        ]
        
        return random.choice(caption_templates)
        
    def match_audio_to_text(self, metadata_df: pd.DataFrame) -> List[Tuple[Path, str, Dict]]:
        """Match audio files to their captions."""
        matched = []
        missing_count = 0
        
        for _, row in metadata_df.iterrows():
            filename = row['slice_file_name']
            fold = row['fold']
            
            # Audio files are organized in fold directories
            audio_file_path = self.audio_dir / f"fold{fold}" / filename
            
            if audio_file_path.exists():
                # Create caption from class name
                caption = self.create_caption(row['class'])
                
                metadata = {
                    'split': row['split'],
                    'original_filename': filename,
                    'fold': fold,
                    'class': row['class'],
                    'classID': row['classID'],
                    'fsID': row['fsID'],
                    'start': row['start'],
                    'end': row['end'],
                    'salience': row['salience'],
                    'task': 'AAC'
                }
                matched.append((audio_file_path, caption, metadata))
            else:
                missing_count += 1
                if missing_count <= 10:  # Only print first 10
                    print(f"Missing audio file: {audio_file_path}")
                
        print(f"Matched {len(matched)} audio-text pairs")
        if missing_count > 0:
            print(f"Missing audio files: {missing_count}")
        
        return matched


def main():
    parser = argparse.ArgumentParser(description="Convert UrbanSound8K dataset to tar format")
    parser.add_argument("--audio-dir", type=str, 
                       default="/gpfs/scratch1/shared/gwijngaard/laion/urbansound8k",
                       help="Path to UrbanSound8K directory")
    parser.add_argument("--metadata", type=str,
                       default="/gpfs/scratch1/shared/gwijngaard/laion/urbansound8k",
                       help="Path to metadata directory (same as audio-dir)")
    parser.add_argument("--output-dir", type=str,
                       default="/scratch-shared/gwijngaard/tar/urbansound8k",
                       help="Output directory for tar files")
    parser.add_argument("--samples-per-tar", type=int, default=2048,
                       help="Number of samples per tar file")
    
    args = parser.parse_args()
    
    # Create processor
    processor = UrbanSound8KProcessor(
        audio_dir=args.audio_dir,
        metadata_path=args.metadata,
        output_dir=args.output_dir
    )
    
    # Process dataset
    processor.process_dataset(samples_per_tar=args.samples_per_tar)


if __name__ == "__main__":
    main()