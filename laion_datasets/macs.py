#!/usr/bin/env python3
"""
Convert MACS dataset to WebDataset tar format.

Audio location: /scratch-shared/gwijngaard/laion/macs/
Metadata: /gpfs/work4/0/einf6190/data-preparation/data/macs/MACS.yaml
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import yaml
from pathlib import Path
from utils import DatasetProcessor
from typing import List, Tuple, Dict
import argparse
from dotenv import load_dotenv

load_dotenv()


class MACSProcessor(DatasetProcessor):
    """Processor for MACS dataset."""
    
    def load_metadata(self) -> pd.DataFrame:
        """Load MACS metadata YAML."""
        print(f"Loading metadata from {self.metadata_path}")
        
        yaml_path = self.metadata_path / "MACS.yaml"
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
            
        # Convert to dataframe
        records = []
        for item in data.get('files', []):
            filename = item['filename']
            for annotation in item.get('annotations', []):
                records.append({
                    'file_name': filename,
                    'caption': annotation.get('sentence', '')
                })
                
        df = pd.DataFrame(records)
        df['split'] = 'train'  # All MACS data is training
        
        print(f"Loaded {len(df)} entries")
        return df
        
    def match_audio_to_text(self, metadata_df: pd.DataFrame) -> List[Tuple[Path, str, Dict]]:
        """Match audio files to their captions."""
        matched = []
        missing_count = 0
        
        # Index available audio files
        print("Indexing audio files...")
        audio_files = {}
        
        # Look for audio files in the macs directory
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
                if missing_count <= 10:
                    print(f"Missing: {filename}")
                
        print(f"Matched {len(matched)} audio-text pairs")
        print(f"Missing audio files: {missing_count}")
        
        return matched


def main():
    parser = argparse.ArgumentParser(description="Convert MACS dataset to tar format")
    parser.add_argument("--audio-dir", type=str, 
                       default="/scratch-shared/gwijngaard/laion/macs",
                       help="Path to audio files")
    parser.add_argument("--metadata", type=str,
                       default="/gpfs/work4/0/einf6190/data-preparation/data/macs",
                       help="Path to metadata directory")
    parser.add_argument("--output-dir", type=str,
                       default="/scratch-shared/gwijngaard/tar/macs",
                       help="Output directory for tar files")
    parser.add_argument("--samples-per-tar", type=int, default=2048,
                       help="Number of samples per tar file")
    
    args = parser.parse_args()
    
    # Create processor
    processor = MACSProcessor(
        audio_dir=args.audio_dir,
        metadata_path=args.metadata,
        output_dir=args.output_dir,
        task="AAC")
    
    # Process dataset
    processor.process_dataset(samples_per_tar=args.samples_per_tar)


if __name__ == "__main__":
    main()