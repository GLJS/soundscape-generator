#!/usr/bin/env python3
"""
Convert MedleySolos dataset to WebDataset tar format.

Audio location: /gpfs/scratch1/shared/gwijngaard/laion/medleysolos/data/
Metadata: /gpfs/work4/0/einf6190/data-preparation/data/medleysolos/Medley-solos-DB_metadata.csv
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


class MedleySolosProcessor(DatasetProcessor):
    """Processor for MedleySolos dataset."""
    
    def __init__(self, audio_dir: str, metadata_path: str, output_dir: str):
        super().__init__(audio_dir, metadata_path, output_dir)
        
    def load_metadata(self) -> pd.DataFrame:
        """Load MedleySolos metadata CSV."""
        print(f"Loading metadata from {self.metadata_path}")
        df = pd.read_csv(self.metadata_path)
        print(f"Loaded {len(df)} entries")
        print(f"Unique instruments: {df['instrument'].unique()}")
        print(f"Instrument counts: {df['instrument'].value_counts()}")
        return df
        
        
    def match_audio_to_text(self, metadata_df: pd.DataFrame) -> List[Tuple[Path, str, Dict]]:
        """Match audio files to their captions."""
        matched = []
        missing_count = 0
        
        for _, row in metadata_df.iterrows():
            # Construct filename from metadata
            subset = row['subset']
            instrument_id = row['instrument_id']
            uuid4 = row['uuid4']
            filename = f"Medley-solos-DB_{subset}-{instrument_id}_{uuid4}.wav"
            
            audio_file_path = self.audio_dir / filename
            
            if audio_file_path.exists():
                # Get instrument name
                instrument = row['instrument']
                
                # Create varied captions
                caption_templates = [
                    f"This audio contains a {instrument}.",
                    f"The sound of a {instrument}.",
                    f"A recording of a {instrument}.",
                    f"This is the sound of a {instrument}.",
                    f"Audio recording of a {instrument}.",
                    f"You can hear a {instrument} in this recording.",
                    f"This recording features a {instrument}.",
                    f"The audio captures a {instrument}.",
                    f"Sound of a {instrument}.",
                    f"A {instrument} sound.",
                ]
                
                caption = random.choice(caption_templates)
                
                metadata = {
                    'split': 'train' if subset == 'train' else 'test',
                    'subset': subset,
                    'original_filename': filename,
                    'instrument': instrument,
                    'instrument_id': instrument_id,
                    'song_id': row['song_id'],
                    'uuid4': uuid4,
                    'task': 'AAC'
                }
                matched.append((audio_file_path, caption, metadata))
            else:
                missing_count += 1
                if missing_count <= 10:  # Only print first 10 missing files
                    print(f"Missing audio file: {audio_file_path}")
                
        print(f"Matched {len(matched)} audio-text pairs")
        print(f"Missing audio files: {missing_count}")
        
        return matched
        

def main():
    parser = argparse.ArgumentParser(description="Convert MedleySolos dataset to tar format")
    parser.add_argument("--audio-dir", type=str, 
                       default="/gpfs/scratch1/shared/gwijngaard/laion/medleysolos/data",
                       help="Path to audio directory")
    parser.add_argument("--metadata", type=str,
                       default="/gpfs/work4/0/einf6190/data-preparation/data/medleysolos/Medley-solos-DB_metadata.csv",
                       help="Path to metadata CSV")
    parser.add_argument("--output-dir", type=str,
                       default="/scratch-shared/gwijngaard/tar/medleysolos",
                       help="Output directory for tar files")
    parser.add_argument("--samples-per-tar", type=int, default=2048,
                       help="Number of samples per tar file")
    
    args = parser.parse_args()
    
    # Create processor
    processor = MedleySolosProcessor(
        audio_dir=args.audio_dir,
        metadata_path=args.metadata,
        output_dir=args.output_dir
    )
    
    # Process dataset
    processor.process_dataset(samples_per_tar=args.samples_per_tar)


if __name__ == "__main__":
    main()