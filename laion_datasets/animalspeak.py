#!/usr/bin/env python3
"""
Convert AnimalSpeak dataset to WebDataset tar format.

Audio location: /scratch-shared/gwijngaard/laion/AnimalSpeak/audio/
Metadata: /gpfs/work4/0/einf6190/data-preparation/data/AnimalSpeak/AnimalSpeak_correct.csv
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


class AnimalSpeakProcessor(DatasetProcessor):
    """Processor for AnimalSpeak dataset."""
    
    def load_metadata(self) -> pd.DataFrame:
        """Load AnimalSpeak metadata CSV."""
        print(f"Loading metadata from {self.metadata_path}")
        df = pd.read_csv(self.metadata_path)
        print(f"Loaded {len(df)} entries")
        return df
        
    def match_audio_to_text(self, metadata_df: pd.DataFrame) -> List[Tuple[Path, str, Dict]]:
        """Match audio files to their captions."""
        matched = []
        audio_files = {}
        
        # Build index of available audio files
        print("Indexing audio files...")
        for audio_file in self.audio_dir.glob("*.mp3"):
            audio_files[audio_file.stem] = audio_file
        for audio_file in self.audio_dir.glob("*.wav"):
            audio_files[audio_file.stem] = audio_file
        for audio_file in self.audio_dir.glob("*.m4a"):
            audio_files[audio_file.stem] = audio_file
            
        print(f"Found {len(audio_files)} audio files")
        
        # Check if metadata has split column
        has_split_column = 'split' in metadata_df.columns
        
        # If no split column, create splits - put train everywhere
        if not has_split_column:
            print("No split column found in metadata, assigning all samples to train split")
            
            # Add split column to dataframe
            metadata_df = metadata_df.copy()
            metadata_df['split'] = 'train'  # Put train everywhere
        
        # Match with metadata
        missing_count = 0
        for _, row in metadata_df.iterrows():
            file_stem = Path(row['file_name']).stem
            
            if file_stem in audio_files:
                audio_path = audio_files[file_stem]
                caption = row['caption']
                metadata = {
                    'split': row['split'],  # Now guaranteed to exist
                    'original_filename': row['file_name'],
                    'task': 'STT'
                }
                matched.append((audio_path, caption, metadata))
            else:
                missing_count += 1
                
        print(f"Matched {len(matched)} audio-text pairs")
        print(f"Missing audio files: {missing_count}")
        
        return matched
        


def main():
    parser = argparse.ArgumentParser(description="Convert AnimalSpeak dataset to tar format")
    parser.add_argument("--audio-dir", type=str, 
                       default="/scratch-shared/gwijngaard/laion/AnimalSpeak/audio",
                       help="Path to audio files")
    parser.add_argument("--metadata", type=str,
                       default="/gpfs/work4/0/einf6190/data-preparation/data/AnimalSpeak/AnimalSpeak_correct.csv",
                       help="Path to metadata CSV")
    parser.add_argument("--output-dir", type=str,
                       default="/scratch-shared/gwijngaard/tar/animalspeak",
                       help="Output directory for tar files")
    parser.add_argument("--samples-per-tar", type=int, default=2048,
                       help="Number of samples per tar file")
    
    args = parser.parse_args()
    
    # Create processor
    processor = AnimalSpeakProcessor(
        audio_dir=args.audio_dir,
        metadata_path=args.metadata,
        output_dir=args.output_dir)
    
    # Process dataset
    processor.process_dataset(samples_per_tar=args.samples_per_tar)


if __name__ == "__main__":
    main()