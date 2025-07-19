#!/usr/bin/env python3
"""
Convert Auto-ACD dataset to WebDataset tar format.

Audio location: /scratch-shared/gwijngaard/laion/Auto-ACD/flac/ (extracted)
                /scratch-shared/gwijngaard/laion/Auto-ACD/flac.tar.bz2 (archive)
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
import tarfile
from dotenv import load_dotenv

load_dotenv()


class AutoACDProcessor(DatasetProcessor):
    """Processor for Auto-ACD dataset."""
    
    def __init__(self, audio_dir: str, metadata_path: str, output_dir: str):
        super().__init__(audio_dir, metadata_path, output_dir)
        self.flac_dir = self.audio_dir / "flac"
        self.flac_archive = self.audio_dir / "flac.tar.bz2"
        
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
        
        # Check if flac directory exists with extracted files
        if self.flac_dir.exists():
            print(f"Using extracted FLAC files from {self.flac_dir}")
            
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
                        'youtube_id': row['youtube_id']
                    }
                    matched.append((audio_path, caption, metadata))
                else:
                    missing_count += 1
                    
        else:
            # Need to use the tar.bz2 archive
            print(f"FLAC directory not found, will extract from {self.flac_archive}")
            
            if self.flac_archive.exists():
                # Extract on the fly
                matched = self._match_from_archive(metadata_df)
            else:
                print("ERROR: Neither flac directory nor flac.tar.bz2 archive found!")
                return []
                
        print(f"Matched {len(matched)} audio-text pairs")
        print(f"Missing audio files: {missing_count}")
        
        return matched
        
    def _match_from_archive(self, metadata_df: pd.DataFrame) -> List[Tuple[bytes, str, Dict]]:
        """Extract and match audio files from tar.bz2 archive."""
        matched = []
        
        print("Indexing flac.tar.bz2 archive...")
        
        # First, index the archive
        file_index = {}
        with tarfile.open(self.flac_archive, "r:bz2") as tar:
            for member in tar.getmembers():
                if member.isfile() and member.name.endswith('.flac'):
                    filename = os.path.basename(member.name)
                    file_index[filename] = member.name
                    
        print(f"Found {len(file_index)} FLAC files in archive")
        
        # Now match and extract as needed
        missing_count = 0
        
        with tarfile.open(self.flac_archive, "r:bz2") as tar:
            for _, row in metadata_df.iterrows():
                filename = row['file_name']
                
                if filename in file_index:
                    member_name = file_index[filename]
                    member = tar.getmember(member_name)
                    f = tar.extractfile(member)
                    
                    if f:
                        audio_bytes = f.read()
                        caption = row['caption']
                        metadata = {
                            'split': row['split'],
                            'original_filename': filename,
                            'youtube_id': row['youtube_id']
                        }
                        matched.append((audio_bytes, caption, metadata))
                else:
                    missing_count += 1
                    
        print(f"Missing files: {missing_count}")
        
        return matched


def main():
    parser = argparse.ArgumentParser(description="Convert Auto-ACD dataset to tar format")
    parser.add_argument("--audio-dir", type=str, 
                       default="/scratch-shared/gwijngaard/laion/Auto-ACD",
                       help="Path to directory containing flac/ or flac.tar.bz2")
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
        output_dir=args.output_dir,
        task="AAC")
    
    # Process dataset
    processor.process_dataset(samples_per_tar=args.samples_per_tar)


if __name__ == "__main__":
    main()