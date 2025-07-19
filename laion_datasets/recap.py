#!/usr/bin/env python3
"""
Convert RECAP dataset to WebDataset tar format.

Audio location: /scratch-shared/gwijngaard/laion/RECAP/captions_rewritten_from_existing_recap_paper.tar.gz
Metadata: /scratch-shared/gwijngaard/laion/RECAP/captions.json
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
import tarfile
from dotenv import load_dotenv

load_dotenv()


class RECAPProcessor(DatasetProcessor):
    """Processor for RECAP dataset."""
    
    def __init__(self, audio_dir: str, metadata_path: str, output_dir: str):
        super().__init__(audio_dir, metadata_path, output_dir)
        self.audio_archive = self.audio_dir / "captions_rewritten_from_existing_recap_paper.tar.gz"
        self.extracted_audio = {}
        
    def load_metadata(self) -> pd.DataFrame:
        """Load RECAP metadata JSON."""
        print(f"Loading metadata from {self.metadata_path}")
        
        json_path = self.audio_dir / "captions.json"  # In same directory as audio
        
        if json_path.exists():
            with open(json_path, 'r') as f:
                data = json.load(f)
                
            # Convert to dataframe
            if isinstance(data, list):
                df = pd.DataFrame(data)
            elif isinstance(data, dict):
                df = pd.DataFrame.from_dict(data, orient='index')
            else:
                print(f"Unexpected data format in {json_path}")
                return pd.DataFrame()
                
            df['split'] = 'train'
            print(f"Loaded {len(df)} entries")
            return df
        else:
            print(f"Metadata file {json_path} not found")
            return pd.DataFrame()
        
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
                        filename = os.path.basename(member.name)
                        self.extracted_audio[filename] = f.read()
                        
        print(f"Extracted {len(self.extracted_audio)} audio files")
        
    def match_audio_to_text(self, metadata_df: pd.DataFrame) -> List[Tuple[bytes, str, Dict]]:
        """Match audio files to their captions."""
        # First extract audio files if not already done
        if not self.extracted_audio:
            self.extract_audio_files()
            
        matched = []
        
        # If no metadata, use audio files directly
        if len(metadata_df) == 0:
            for filename, audio_bytes in self.extracted_audio.items():
                caption = filename.replace('.wav', '').replace('.mp3', '').replace('.flac', '')
                metadata = {
                    'split': 'train',
                    'original_filename': filename
                }
                matched.append((audio_bytes, caption, metadata))
        else:
            # Match with metadata
            missing_count = 0
            
            # Determine filename column
            filename_col = None
            for col in ['file_name', 'filename', 'audio_file', 'audio']:
                if col in metadata_df.columns:
                    filename_col = col
                    break
                    
            # Determine caption column
            caption_col = None
            for col in ['caption', 'text', 'description']:
                if col in metadata_df.columns:
                    caption_col = col
                    break
                    
            if filename_col and caption_col:
                for _, row in metadata_df.iterrows():
                    filename = row[filename_col]
                    
                    if filename in self.extracted_audio:
                        audio_bytes = self.extracted_audio[filename]
                        caption = row[caption_col]
                        metadata = {
                            'split': row.get('split', 'train'),
                            'original_filename': filename
                        }
                        matched.append((audio_bytes, caption, metadata))
                    else:
                        missing_count += 1
                        
                print(f"Missing audio files: {missing_count}")
                
        print(f"Matched {len(matched)} audio-text pairs")
        return matched


def main():
    parser = argparse.ArgumentParser(description="Convert RECAP dataset to tar format")
    parser.add_argument("--audio-dir", type=str, 
                       default="/scratch-shared/gwijngaard/laion/RECAP",
                       help="Path to RECAP directory")
    parser.add_argument("--metadata", type=str,
                       default="/scratch-shared/gwijngaard/laion/RECAP",
                       help="Path to metadata (same as audio-dir)")
    parser.add_argument("--output-dir", type=str,
                       default="/scratch-shared/gwijngaard/tar/recap",
                       help="Output directory for tar files")
    parser.add_argument("--samples-per-tar", type=int, default=2048,
                       help="Number of samples per tar file")
    
    args = parser.parse_args()
    
    # Create processor
    processor = RECAPProcessor(
        audio_dir=args.audio_dir,
        metadata_path=args.metadata,
        output_dir=args.output_dir,
        task="AAC")
    
    # Process dataset
    processor.process_dataset(samples_per_tar=args.samples_per_tar)


if __name__ == "__main__":
    main()