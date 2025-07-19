#!/usr/bin/env python3
"""
Convert GISE-51 dataset to WebDataset tar format.

Audio location: /scratch-shared/gwijngaard/laion/GISE-51/
Multiple tar.gz files for different splits
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


class GISE51Processor(DatasetProcessor):
    """Processor for GISE-51 dataset."""
    
    def __init__(self, audio_dir: str, metadata_path: str, output_dir: str):
        super().__init__(audio_dir, metadata_path, output_dir)
        # GISE-51 has multiple tar files
        self.tar_files = {
            'train': list(self.audio_dir.glob("train*.tar.gz")),
            'val': [self.audio_dir / "val.tar.gz"],
            'eval': [self.audio_dir / "eval.tar.gz"]
        }
        
    def load_metadata(self) -> pd.DataFrame:
        """Load GISE-51 metadata from meta.tar.gz."""
        print(f"Loading metadata from {self.audio_dir}")
        
        meta_archive = self.audio_dir / "meta.tar.gz"
        if not meta_archive.exists():
            print("meta.tar.gz not found, creating dummy metadata")
            return pd.DataFrame()
            
        # Extract metadata files
        records = []
        with tarfile.open(meta_archive, "r:gz") as tar:
            for member in tar.getmembers():
                if member.isfile() and member.name.endswith('.txt'):
                    f = tar.extractfile(member)
                    if f:
                        content = f.read().decode('utf-8')
                        # Parse metadata format (adjust based on actual format)
                        filename = os.path.basename(member.name).replace('.txt', '.flac')
                        records.append({
                            'file_name': filename,
                            'caption': content.strip(),
                            'split': 'train'  # Will be updated based on which archive contains it
                        })
                        
        df = pd.DataFrame(records)
        print(f"Loaded {len(df)} metadata entries")
        return df
        
    def match_audio_to_text(self, metadata_df: pd.DataFrame) -> List[Tuple[bytes, str, Dict]]:
        """Match audio files to their captions."""
        matched = []
        
        # Process each split
        for split, tar_list in self.tar_files.items():
            print(f"\nProcessing {split} split...")
            
            for tar_path in tar_list:
                if not tar_path.exists():
                    continue
                    
                print(f"  Processing {tar_path.name}...")
                
                with tarfile.open(tar_path, "r:gz") as tar:
                    for member in tar.getmembers():
                        if member.isfile() and member.name.endswith('.flac'):
                            f = tar.extractfile(member)
                            if f:
                                audio_bytes = f.read()
                                filename = os.path.basename(member.name)
                                
                                # Find matching metadata or use filename as caption
                                if len(metadata_df) > 0:
                                    meta_row = metadata_df[metadata_df['file_name'] == filename]
                                    if not meta_row.empty:
                                        caption = meta_row.iloc[0]['caption']
                                    else:
                                        caption = filename.replace('.flac', '')
                                else:
                                    caption = filename.replace('.flac', '')
                                    
                                metadata = {
                                    'split': split,
                                    'original_filename': filename,
                                    'source_archive': tar_path.name
                                }
                                
                                matched.append((audio_bytes, caption, metadata))
                                
        print(f"\nTotal matched: {len(matched)} audio-text pairs")
        return matched


def main():
    parser = argparse.ArgumentParser(description="Convert GISE-51 dataset to tar format")
    parser.add_argument("--audio-dir", type=str, 
                       default="/scratch-shared/gwijngaard/laion/GISE-51",
                       help="Path to GISE-51 directory")
    parser.add_argument("--metadata", type=str,
                       default="/scratch-shared/gwijngaard/laion/GISE-51",
                       help="Path to metadata (same as audio-dir for GISE-51)")
    parser.add_argument("--output-dir", type=str,
                       default="/scratch-shared/gwijngaard/tar/gise51",
                       help="Output directory for tar files")
    parser.add_argument("--samples-per-tar", type=int, default=2048,
                       help="Number of samples per tar file")
    
    args = parser.parse_args()
    
    # Create processor
    processor = GISE51Processor(
        audio_dir=args.audio_dir,
        metadata_path=args.metadata,
        output_dir=args.output_dir,
        task="SED")
    
    # Process dataset
    processor.process_dataset(samples_per_tar=args.samples_per_tar)


if __name__ == "__main__":
    main()