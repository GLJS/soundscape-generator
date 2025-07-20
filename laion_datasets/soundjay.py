#!/usr/bin/env python3
"""
Convert SoundJay dataset to WebDataset tar format.

Audio location: /scratch-shared/gwijngaard/laion/SoundJay/audio.tar.gz
Metadata: /gpfs/work4/0/einf6190/data-preparation/data/SoundJay/sound_descriptions.csv
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from pathlib import Path
from utils import DatasetProcessor
from typing import List, Tuple, Dict
import argparse
import tarfile
from dotenv import load_dotenv

load_dotenv()


class SoundJayProcessor(DatasetProcessor):
    """Processor for SoundJay dataset."""
    
    def __init__(self, audio_dir: str, metadata_path: str, output_dir: str, task: str = None):
        super().__init__(audio_dir, metadata_path, output_dir)
        self.audio_archive = self.audio_dir / "audio.tar.gz"
        self.extracted_audio = {}
        
    def load_metadata(self) -> pd.DataFrame:
        """Load SoundJay metadata CSV."""
        print(f"Loading metadata from {self.metadata_path}")
        
        csv_path = self.metadata_path / "sound_descriptions.csv"
        df = pd.read_csv(csv_path)
        
        # Rename columns to standard names
        df.rename(columns={'filename': 'file_name', 'description': 'caption'}, inplace=True)
        df['split'] = 'train'
        
        print(f"Loaded {len(df)} entries")
        return df
        
    def extract_audio_files(self):
        """Extract audio files from tar.gz archive into memory."""
        print(f"Extracting audio from {self.audio_archive}")
        
        if not self.audio_archive.exists():
            print(f"Audio archive {self.audio_archive} not found")
            # Try to find audio files directly
            return
            
        with tarfile.open(self.audio_archive, "r:gz") as tar:
            for member in tar.getmembers():
                if member.isfile() and any(member.name.endswith(ext) for ext in ['.wav', '.mp3', '.flac']):
                    f = tar.extractfile(member)
                    if f:
                        filename = os.path.basename(member.name)
                        self.extracted_audio[filename] = f.read()
                        
        print(f"Extracted {len(self.extracted_audio)} audio files")
        
    def match_audio_to_text(self, metadata_df: pd.DataFrame) -> List[Tuple[bytes, str, Dict]]:
        """Match audio files to their captions."""
        # First extract audio files if not already done
        if not self.extracted_audio:
            self.extract_audio_files()
            
        # If no extracted audio, try to find files directly
        if not self.extracted_audio:
            return self._match_direct_files(metadata_df)
            
        matched = []
        missing_count = 0
        
        for _, row in metadata_df.iterrows():
            filename = row['file_name']
            
            if filename in self.extracted_audio:
                audio_bytes = self.extracted_audio[filename]
                caption = row['caption']
                metadata = {
                    'split': row['split'],
                    'original_filename': filename
                }
                matched.append((audio_bytes, caption, metadata))
            else:
                missing_count += 1
                
        print(f"Matched {len(matched)} audio-text pairs")
        print(f"Missing audio files: {missing_count}")
        
        return matched
        
    def _match_direct_files(self, metadata_df: pd.DataFrame) -> List[Tuple[Path, str, Dict]]:
        """Match audio files directly from directory."""
        matched = []
        missing_count = 0
        
        # Index available audio files
        print("Indexing audio files in directory...")
        audio_files = {}
        
        for ext in ['.wav', '.mp3', '.flac']:
            for audio_file in self.audio_dir.rglob(f"*{ext}"):
                audio_files[audio_file.name] = audio_file
                
        print(f"Found {len(audio_files)} audio files")
        
        # Match with metadata
        for _, row in metadata_df.iterrows():
            filename = row['file_name']
            
            if filename in audio_files:
                audio_path = audio_files[filename]
                caption = row['caption']
                metadata = {
                    'split': row['split'],
                    'original_filename': filename
                }
                matched.append((audio_path, caption, metadata))
            else:
                missing_count += 1
                
        print(f"Matched {len(matched)} audio-text pairs")
        print(f"Missing audio files: {missing_count}")
        
        return matched


def main():
    parser = argparse.ArgumentParser(description="Convert SoundJay dataset to tar format")
    parser.add_argument("--audio-dir", type=str, 
                       default="/scratch-shared/gwijngaard/laion/SoundJay",
                       help="Path to directory containing audio.tar.gz")
    parser.add_argument("--metadata", type=str,
                       default="/gpfs/work4/0/einf6190/data-preparation/data/SoundJay",
                       help="Path to metadata directory")
    parser.add_argument("--output-dir", type=str,
                       default="/scratch-shared/gwijngaard/tar/soundjay",
                       help="Output directory for tar files")
    parser.add_argument("--samples-per-tar", type=int, default=2048,
                       help="Number of samples per tar file")
    
    args = parser.parse_args()
    
    # Create processor
    processor = SoundJayProcessor(
        audio_dir=args.audio_dir,
        metadata_path=args.metadata,
        output_dir=args.output_dir,
        task="AAC")
    
    # Process dataset
    processor.process_dataset(samples_per_tar=args.samples_per_tar)


if __name__ == "__main__":
    main()