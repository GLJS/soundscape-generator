#!/usr/bin/env python3
"""
Convert SoundDescs dataset to WebDataset tar format.

Audio location: /scratch-shared/gwijngaard/laion/Sounddescs/data.tar.gz
Metadata: /gpfs/work4/0/einf6190/data-preparation/data/SoundDescs/
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import pickle
from pathlib import Path
from utils import DatasetProcessor
from typing import List, Tuple, Dict
import argparse
import tarfile
from dotenv import load_dotenv

load_dotenv()


class SoundDescsProcessor(DatasetProcessor):
    """Processor for SoundDescs dataset."""
    
    def __init__(self, audio_dir: str, metadata_path: str, output_dir: str):
        super().__init__(audio_dir, metadata_path, output_dir)
        self.audio_archive = self.audio_dir / "data.tar.gz"
        self.extracted_audio = {}
        
    def load_metadata(self) -> pd.DataFrame:
        """Load SoundDescs metadata from pickle files."""
        print(f"Loading metadata from {self.metadata_path}")
        
        # Load pickle files
        descriptions = pd.read_pickle(self.metadata_path / "descriptions.pkl")
        descriptions = pd.DataFrame.from_dict(descriptions, orient="index", columns=["description"])
        
        categories = pd.read_pickle(self.metadata_path / "categories.pkl")
        categories = pd.DataFrame.from_dict(categories, orient="index", 
                                          columns=["category1", "category2", "category3"])
        
        extra_info = pd.read_pickle(self.metadata_path / "extra_info.pkl")
        extra_info = pd.DataFrame.from_dict(extra_info, orient="index", columns=["extra_info"])
        
        # Load split files
        train = pd.read_csv(self.metadata_path / "train_list.txt", header=None, names=["id"])
        train["split"] = "train"
        
        val = pd.read_csv(self.metadata_path / "val_list.txt", header=None, names=["id"])
        val["split"] = "valid"
        
        test = pd.read_csv(self.metadata_path / "test_list.txt", header=None, names=["id"])
        test["split"] = "test"
        
        # Combine splits
        splits = pd.concat([train, val, test], ignore_index=True)
        
        # Merge all metadata
        df = pd.merge(splits, descriptions, left_on="id", right_index=True)
        df = pd.merge(df, categories, left_on="id", right_index=True, how="left")
        df = pd.merge(df, extra_info, left_on="id", right_index=True, how="left")
        
        # Rename columns
        df.rename(columns={"id": "file_name", "description": "caption"}, inplace=True)
        
        # Format filenames
        df["file_name"] = df["file_name"].str.upper() + ".wav"
        
        print(f"Loaded {len(df)} entries")
        return df
        
    def extract_audio_files(self):
        """Extract audio files from tar.gz archive into memory."""
        print(f"Extracting audio from {self.audio_archive}")
        
        if not self.audio_archive.exists():
            print(f"Audio archive {self.audio_archive} not found")
            return
            
        with tarfile.open(self.audio_archive, "r:gz") as tar:
            for member in tar.getmembers():
                if member.isfile() and member.name.endswith('.wav'):
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
        missing_count = 0
        
        for _, row in metadata_df.iterrows():
            filename = row['file_name']
            
            if filename in self.extracted_audio:
                audio_bytes = self.extracted_audio[filename]
                caption = row['caption']
                metadata = {
                    'split': row['split'],
                    'original_filename': filename,
                    'category1': row.get('category1', ''),
                    'category2': row.get('category2', ''),
                    'category3': row.get('category3', ''),
                    'extra_info': row.get('extra_info', '')
                }
                matched.append((audio_bytes, caption, metadata))
            else:
                missing_count += 1
                
        print(f"Matched {len(matched)} audio-text pairs")
        print(f"Missing audio files: {missing_count}")
        
        return matched


def main():
    parser = argparse.ArgumentParser(description="Convert SoundDescs dataset to tar format")
    parser.add_argument("--audio-dir", type=str, 
                       default="/scratch-shared/gwijngaard/laion/Sounddescs",
                       help="Path to directory containing data.tar.gz")
    parser.add_argument("--metadata", type=str,
                       default="/gpfs/work4/0/einf6190/data-preparation/data/SoundDescs",
                       help="Path to metadata directory")
    parser.add_argument("--output-dir", type=str,
                       default="/scratch-shared/gwijngaard/tar/sounddescs",
                       help="Output directory for tar files")
    parser.add_argument("--samples-per-tar", type=int, default=2048,
                       help="Number of samples per tar file")
    
    args = parser.parse_args()
    
    # Create processor
    processor = SoundDescsProcessor(
        audio_dir=args.audio_dir,
        metadata_path=args.metadata,
        output_dir=args.output_dir,
        task="AAC")
    
    # Process dataset
    processor.process_dataset(samples_per_tar=args.samples_per_tar)


if __name__ == "__main__":
    main()