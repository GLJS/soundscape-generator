#!/usr/bin/env python3
"""
Convert FAVDBench dataset to WebDataset tar format.

Audio location: /scratch-shared/gwijngaard/laion/FAVDBench/audios.tar.gz
Metadata: /gpfs/work4/0/einf6190/data-preparation/data/FAVDBench/FAVDBench_Audio_Updated.csv
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


class FAVDBenchProcessor(DatasetProcessor):
    """Processor for FAVDBench dataset."""
    
    def __init__(self, audio_dir: str, metadata_path: str, output_dir: str):
        super().__init__(audio_dir, metadata_path, output_dir)
        self.audio_archive = self.audio_dir / "audios.tar.gz"
        self.audios_directory = self.audio_dir / "audios"
        self.extracted_audio = {}
        
        # Determine if we're using archive or directory
        self.use_archive = self.audio_archive.exists()
        self.use_directory = self.audios_directory.exists() and self.audios_directory.is_dir()
        
        if self.use_directory:
            print(f"Using audio directory: {self.audios_directory}")
        elif self.use_archive:
            print(f"Using audio archive: {self.audio_archive}")
        else:
            raise ValueError(f"Neither audios.tar.gz nor audios directory found in {self.audio_dir}")
        
    def load_metadata(self) -> pd.DataFrame:
        """Load FAVDBench metadata CSV."""
        print(f"Loading metadata from {self.metadata_path}")
        
        csv_path = self.metadata_path / "FAVDBench_Audio_Updated.csv"
        if not csv_path.exists():
            csv_path = self.metadata_path / "FAVDBench_Audio.csv"
            
        df = pd.read_csv(csv_path)
        print(f"Loaded {len(df)} entries")
        
        # The CSV should have columns that include audio filename and text/caption
        return df
        
    def load_audio_files(self):
        """Load audio files from directory."""
        print(f"Loading audio files from {self.audios_directory}")
        
        audio_extensions = ['.wav', '.mp3', '.flac', '.ogg', '.m4a']
        
        for audio_file in self.audios_directory.rglob('*'):
            if audio_file.is_file() and any(audio_file.suffix.lower() == ext for ext in audio_extensions):
                try:
                    with open(audio_file, 'rb') as f:
                        # Store with relative path from audios directory
                        rel_path = str(audio_file.relative_to(self.audios_directory))
                        self.extracted_audio[rel_path] = f.read()
                except Exception as e:
                    print(f"Error reading {audio_file}: {e}")
                    
        print(f"Loaded {len(self.extracted_audio)} audio files")
        
    def extract_audio_files(self):
        """Extract audio files from tar.gz archive into memory."""
        print(f"Extracting audio from {self.audio_archive}")
        
        with tarfile.open(self.audio_archive, "r:gz") as tar:
            for member in tar.getmembers():
                if member.isfile() and any(member.name.endswith(ext) for ext in ['.wav', '.mp3', '.flac']):
                    f = tar.extractfile(member)
                    if f:
                        # Store with relative path
                        filename = member.name
                        self.extracted_audio[filename] = f.read()
                        
        print(f"Extracted {len(self.extracted_audio)} audio files")
        
    def match_audio_to_text(self, metadata_df: pd.DataFrame) -> List[Tuple[bytes, str, Dict]]:
        """Match audio files to their captions."""
        # First extract/load audio files if not already done
        if not self.extracted_audio:
            if self.use_directory:
                self.load_audio_files()
            else:
                self.extract_audio_files()
            
        matched = []
        missing_count = 0
        
        # Determine the audio filename column
        audio_col = None
        for col in ['file_name', 'audio_path', 'audio_file', 'filename']:
            if col in metadata_df.columns:
                audio_col = col
                break
                
        if not audio_col:
            print("Could not find audio filename column in metadata")
            return []
            
        # Determine caption column
        caption_col = None
        for col in ['caption', 'text', 'description', 'label']:
            if col in metadata_df.columns:
                caption_col = col
                break
                
        if not caption_col:
            print("Could not find caption column in metadata")
            return []
            
        for _, row in metadata_df.iterrows():
            filename = row[audio_col]
            
            # Try to find the audio file
            audio_found = False
            for audio_path, audio_bytes in self.extracted_audio.items():
                if filename in audio_path or os.path.basename(audio_path) == filename:
                    caption = str(row[caption_col])
                    metadata = {
                        'split': row.get('split', 'train'),
                        'original_filename': filename
                    }
                    
                    # Add any additional metadata columns
                    for col in metadata_df.columns:
                        if col not in [audio_col, caption_col, 'split']:
                            metadata[col] = row.get(col, '')
                            
                    matched.append((audio_bytes, caption, metadata))
                    audio_found = True
                    break
                    
            if not audio_found:
                missing_count += 1
                
        print(f"Matched {len(matched)} audio-text pairs")
        print(f"Missing audio files: {missing_count}")
        
        return matched


def main():
    parser = argparse.ArgumentParser(description="Convert FAVDBench dataset to tar format")
    parser.add_argument("--audio-dir", type=str, 
                       default="/scratch-shared/gwijngaard/laion/FAVDBench",
                       help="Path to directory containing audios.tar.gz")
    parser.add_argument("--metadata", type=str,
                       default="/gpfs/work4/0/einf6190/data-preparation/data/FAVDBench",
                       help="Path to metadata directory")
    parser.add_argument("--output-dir", type=str,
                       default="/scratch-shared/gwijngaard/tar/favdbench",
                       help="Output directory for tar files")
    parser.add_argument("--samples-per-tar", type=int, default=2048,
                       help="Number of samples per tar file")
    
    args = parser.parse_args()
    
    # Create processor
    processor = FAVDBenchProcessor(
        audio_dir=args.audio_dir,
        metadata_path=args.metadata,
        output_dir=args.output_dir)
    
    # Process dataset
    processor.process_dataset(samples_per_tar=args.samples_per_tar)


if __name__ == "__main__":
    main()