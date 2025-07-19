#!/usr/bin/env python3
"""
Convert SoundingEarth dataset to WebDataset tar format.

Audio location: /scratch-shared/gwijngaard/laion/SoundingEarth/data.tar.gz
Metadata: /gpfs/work4/0/einf6190/data-preparation/data/SoundingEarth/metadata.csv
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from pathlib import Path
from utils import DatasetProcessor, AudioProcessor, TarCreator
from typing import List, Tuple, Dict
import argparse
import tarfile
import os
from dotenv import load_dotenv

load_dotenv()


class SoundingEarthProcessor(DatasetProcessor):
    """Processor for SoundingEarth dataset."""
    
    def __init__(self, audio_dir: str, metadata_path: str, output_dir: str):
        super().__init__(audio_dir, metadata_path, output_dir)
        self.audio_archive = self.audio_dir / "data.tar.gz"
        self.extracted_audio = {}
        
    def load_metadata(self) -> pd.DataFrame:
        """Load SoundingEarth metadata CSV."""
        print(f"Loading metadata from {self.metadata_path}")
        
        csv_path = self.metadata_path / "metadata.csv"
        df = pd.read_csv(csv_path)
        
        # Drop rows with null captions
        df = df.dropna(subset=['caption'])
        
        print(f"Loaded {len(df)} entries with valid captions")
        return df
        
    def extract_audio_files(self):
        """Extract audio files from tar.gz archive into memory."""
        print(f"Extracting audio from {self.audio_archive}")
        
        if not self.audio_archive.exists():
            print(f"Audio archive {self.audio_archive} not found")
            return
            
        with tarfile.open(self.audio_archive, "r:gz") as tar:
            for member in tar.getmembers():
                if member.isfile() and any(member.name.endswith(ext) for ext in ['.wav', '.mp3', '.flac']):
                    f = tar.extractfile(member)
                    if f:
                        filename = member.name
                        self.extracted_audio[filename] = f.read()
                        
        print(f"Extracted {len(self.extracted_audio)} audio files")
        
    def match_audio_to_text(self, metadata_df: pd.DataFrame) -> List[Tuple[bytes, str, Dict]]:
        """Match audio files to their captions."""
        # First extract audio files if not already done
        if not self.extracted_audio:
            self.extract_audio_files()
            
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
            filename = row['file_name']
            
            # Try to find the audio file
            audio_found = False
            for audio_path, audio_bytes in self.extracted_audio.items():
                if filename in audio_path or os.path.basename(audio_path) == filename:
                    caption = row['caption']
                    metadata = {
                        'split': row['split'],  # Now guaranteed to exist
                        'original_filename': filename,
                        'task': 'AAC'
                    }
                    matched.append((audio_bytes, caption, metadata))
                    audio_found = True
                    break
                    
            if not audio_found:
                missing_count += 1
                
        print(f"Matched {len(matched)} audio-text pairs")
        print(f"Missing audio files: {missing_count}")
        
        return matched
        
        


def main():
    parser = argparse.ArgumentParser(description="Convert SoundingEarth dataset to tar format")
    parser.add_argument("--audio-dir", type=str, 
                       default="/scratch-shared/gwijngaard/laion/SoundingEarth",
                       help="Path to directory containing data.tar.gz")
    parser.add_argument("--metadata", type=str,
                       default="/gpfs/work4/0/einf6190/data-preparation/data/SoundingEarth",
                       help="Path to metadata directory")
    parser.add_argument("--output-dir", type=str,
                       default="/scratch-shared/gwijngaard/tar/soundingearth",
                       help="Output directory for tar files")
    parser.add_argument("--samples-per-tar", type=int, default=2048,
                       help="Number of samples per tar file")
    
    args = parser.parse_args()
    
    # Create processor
    processor = SoundingEarthProcessor(
        audio_dir=args.audio_dir,
        metadata_path=args.metadata,
        output_dir=args.output_dir)
    
    # Process dataset
    processor.process_dataset(samples_per_tar=args.samples_per_tar)


if __name__ == "__main__":
    main()