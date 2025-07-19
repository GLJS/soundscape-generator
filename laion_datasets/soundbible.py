#!/usr/bin/env python3
"""
Convert SoundBible dataset to WebDataset tar format.

Audio location: /scratch-shared/gwijngaard/laion/SoundBible/
Metadata: /gpfs/work4/0/einf6190/data-preparation/data/SoundBible/sb_final.json
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import json
from pathlib import Path
from utils import DatasetProcessor
from typing import List, Tuple, Dict
import argparse
from dotenv import load_dotenv

load_dotenv()


class SoundBibleProcessor(DatasetProcessor):
    """Processor for SoundBible dataset."""
    
    def load_metadata(self) -> pd.DataFrame:
        """Load SoundBible metadata JSON."""
        print(f"Loading metadata from {self.metadata_path}")
        
        json_path = self.metadata_path / "sb_final.json"
        
        with open(json_path, 'r') as f:
            data = json.load(f)
            
        # Extract data array
        df = pd.DataFrame(data['data'])
        
        # Rename columns
        df.rename(columns={'id': 'file_name'}, inplace=True)
        
        # Add .flac extension
        df['file_name'] = df['file_name'] + '.flac'
        df['split'] = 'train'
        
        # Drop unnecessary columns
        columns_to_drop = ['audio', 'download_link', 'href', 'title', 'author', 'description']
        df.drop(columns=[col for col in columns_to_drop if col in df.columns], inplace=True)
        
        print(f"Loaded {len(df)} entries")
        return df
        
    def match_audio_to_text(self, metadata_df: pd.DataFrame) -> List[Tuple[Path, str, Dict]]:
        """Match audio files to their captions."""
        matched = []
        missing_count = 0
        
        # Index available audio files
        print("Indexing audio files...")
        audio_files = {}
        
        # Look for audio files in the SoundBible directory
        for ext in ['.flac', '.wav', '.mp3']:
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
                # Try without extension
                basename = filename.replace('.flac', '')
                found = False
                for ext in ['.wav', '.mp3']:
                    alt_filename = basename + ext
                    if alt_filename in audio_files:
                        audio_path = audio_files[alt_filename]
                        caption = row['caption']
                        metadata = {
                            'split': row['split'],
                            'original_filename': filename
                        }
                        matched.append((audio_path, caption, metadata))
                        found = True
                        break
                        
                if not found:
                    missing_count += 1
                    
        print(f"Matched {len(matched)} audio-text pairs")
        print(f"Missing audio files: {missing_count}")
        
        return matched


def main():
    parser = argparse.ArgumentParser(description="Convert SoundBible dataset to tar format")
    parser.add_argument("--audio-dir", type=str, 
                       default="/scratch-shared/gwijngaard/laion/SoundBible",
                       help="Path to audio files")
    parser.add_argument("--metadata", type=str,
                       default="/gpfs/work4/0/einf6190/data-preparation/data/SoundBible",
                       help="Path to metadata directory")
    parser.add_argument("--output-dir", type=str,
                       default="/scratch-shared/gwijngaard/tar/soundbible",
                       help="Output directory for tar files")
    parser.add_argument("--samples-per-tar", type=int, default=2048,
                       help="Number of samples per tar file")
    
    args = parser.parse_args()
    
    # Create processor
    processor = SoundBibleProcessor(
        audio_dir=args.audio_dir,
        metadata_path=args.metadata,
        output_dir=args.output_dir)
    
    # Process dataset
    processor.process_dataset(samples_per_tar=args.samples_per_tar)


if __name__ == "__main__":
    main()