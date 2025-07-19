#!/usr/bin/env python3
"""
Convert LASS dataset to WebDataset tar format.

Audio location: /scratch-shared/gwijngaard/laion/LASS/audio.tar.gz
Metadata: /gpfs/work4/0/einf6190/data-preparation/data/LASS/
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


class LASSProcessor(DatasetProcessor):
    """Processor for LASS dataset."""
    
    def __init__(self, audio_dir: str, metadata_path: str, output_dir: str):
        super().__init__(audio_dir, metadata_path, output_dir)
        self.audio_archive = self.audio_dir / "audio.tar.gz"
        self.extracted_audio = {}
        
    def load_metadata(self) -> pd.DataFrame:
        """Load LASS metadata from various CSV and JSON files."""
        print(f"Loading metadata from {self.metadata_path}")
        
        dfs = []
        
        # Load validation JSON
        json_path = self.metadata_path / "lass_validation.json"
        if json_path.exists():
            with open(json_path, 'r') as f:
                data = json.load(f)
            df = pd.DataFrame(data)
            if 'Captions' in df.columns:
                # Explode captions list
                df = df.explode('Captions')
                df.rename(columns={'Captions': 'caption', 'Index': 'file_name'}, inplace=True)
            df['split'] = 'valid'
            df['file_name'] = df['file_name'].apply(lambda x: f"lass_validation/{x}.wav")
            dfs.append(df)
            print(f"  Loaded {len(df)} validation entries")
            
        # Load synthetic validation CSV
        csv_path = self.metadata_path / "lass_synthetic_validation.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            df.rename(columns={'query': 'caption'}, inplace=True)
            df['file_name'] = df.apply(
                lambda x: f"synthetic_validation/{x['source']}_{x['noise']}_{x['snr']}.wav", 
                axis=1
            )
            df['split'] = 'valid'
            dfs.append(df)
            print(f"  Loaded {len(df)} synthetic validation entries")
            
        # Load synthetic evaluation CSV
        csv_path = self.metadata_path / "lass_synthetic_evaluation.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            df.rename(columns={'wav': 'file_name', 'query': 'caption'}, inplace=True)
            df['file_name'] = df['file_name'].apply(lambda x: f"lass_evaluation_synth/{x}")
            df['split'] = 'test'
            dfs.append(df)
            print(f"  Loaded {len(df)} synthetic evaluation entries")
            
        # Load real evaluation CSV
        csv_path = self.metadata_path / "lass_real_evaluation.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            df.rename(columns={'wav': 'file_name', 'query': 'caption'}, inplace=True)
            df['file_name'] = df['file_name'].apply(lambda x: f"lass_evaluation_real/{x}")
            df['split'] = 'test'
            dfs.append(df)
            print(f"  Loaded {len(df)} real evaluation entries")
            
        if dfs:
            combined_df = pd.concat(dfs, ignore_index=True)
            print(f"Total entries: {len(combined_df)}")
            return combined_df
        else:
            return pd.DataFrame()
        
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
        
        for _, row in metadata_df.iterrows():
            filename = row['file_name']
            
            # Try different path variations
            paths_to_try = [
                filename,
                filename.replace('lass_evaluation_synth/', ''),
                filename.replace('lass_evaluation_real/', ''),
                filename.replace('lass_validation/', ''),
                filename.replace('synthetic_validation/', ''),
                os.path.basename(filename)
            ]
            
            audio_found = False
            for path in paths_to_try:
                if path in self.extracted_audio:
                    audio_bytes = self.extracted_audio[path]
                    caption = row['caption']
                    metadata = {
                        'split': row['split'],
                        'original_filename': filename
                    }
                    
                    # Add additional metadata if available
                    for col in ['source', 'noise', 'snr']:
                        if col in row:
                            metadata[col] = row[col]
                            
                    matched.append((audio_bytes, caption, metadata))
                    audio_found = True
                    break
                    
            if not audio_found:
                missing_count += 1
                
        print(f"Matched {len(matched)} audio-text pairs")
        print(f"Missing audio files: {missing_count}")
        
        return matched


def main():
    parser = argparse.ArgumentParser(description="Convert LASS dataset to tar format")
    parser.add_argument("--audio-dir", type=str, 
                       default="/scratch-shared/gwijngaard/laion/LASS",
                       help="Path to directory containing audio.tar.gz")
    parser.add_argument("--metadata", type=str,
                       default="/gpfs/work4/0/einf6190/data-preparation/data/LASS",
                       help="Path to metadata directory")
    parser.add_argument("--output-dir", type=str,
                       default="/scratch-shared/gwijngaard/tar/lass",
                       help="Output directory for tar files")
    parser.add_argument("--samples-per-tar", type=int, default=2048,
                       help="Number of samples per tar file")
    
    args = parser.parse_args()
    
    # Create processor
    processor = LASSProcessor(
        audio_dir=args.audio_dir,
        metadata_path=args.metadata,
        output_dir=args.output_dir,
        task="AAC")
    
    # Process dataset
    processor.process_dataset(samples_per_tar=args.samples_per_tar)


if __name__ == "__main__":
    main()