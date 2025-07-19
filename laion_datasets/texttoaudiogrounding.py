#!/usr/bin/env python3
"""
Convert TextToAudioGrounding dataset to WebDataset tar format.

Audio location: /scratch-shared/gwijngaard/laion/TextToAudioGrounding/audio.zip
Metadata: /gpfs/work4/0/einf6190/data-preparation/data/TextToAudioGrounding/
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import json
from pathlib import Path
from utils import DatasetProcessor, ArchiveExtractor
from typing import List, Tuple, Dict
import argparse
import zipfile
from dotenv import load_dotenv

load_dotenv()


class TextToAudioGroundingProcessor(DatasetProcessor):
    """Processor for TextToAudioGrounding dataset."""
    
    def __init__(self, audio_dir: str, metadata_path: str, output_dir: str):
        super().__init__(audio_dir, metadata_path, output_dir)
        self.audio_archive = self.audio_dir / "audio.zip"
        self.extracted_audio = {}
        
    def load_metadata(self) -> pd.DataFrame:
        """Load TextToAudioGrounding metadata JSON files."""
        print(f"Loading metadata from {self.metadata_path}")
        
        dfs = []
        
        # Load train, val, test JSON files
        for split, filename in [('train', 'train.json'), ('valid', 'val.json'), ('test', 'test.json')]:
            json_path = self.metadata_path / filename
            if json_path.exists():
                with open(json_path, 'r') as f:
                    data = json.load(f)
                    
                # Convert to dataframe
                records = []
                for item in data:
                    audio_id = item.get('audio_id', '')
                    tokens = item.get('tokens', '')
                    
                    # Handle different data structures
                    if isinstance(tokens, list):
                        caption = ' '.join(tokens)
                    else:
                        caption = str(tokens)
                        
                    records.append({
                        'file_name': audio_id,
                        'caption': caption,
                        'split': split
                    })
                    
                df = pd.DataFrame(records)
                dfs.append(df)
                print(f"  Loaded {len(df)} {split} entries")
                
        if dfs:
            combined_df = pd.concat(dfs, ignore_index=True)
            print(f"Total entries: {len(combined_df)}")
            return combined_df
        else:
            return pd.DataFrame()
        
    def extract_audio_files(self):
        """Extract audio files from zip archive into memory."""
        print(f"Extracting audio from {self.audio_archive}")
        
        if not self.audio_archive.exists():
            print(f"Audio archive {self.audio_archive} not found")
            return
            
        with zipfile.ZipFile(self.audio_archive, 'r') as zip_file:
            for name in zip_file.namelist():
                if any(name.endswith(ext) for ext in ['.wav', '.mp3', '.flac']):
                    audio_bytes = zip_file.read(name)
                    self.extracted_audio[name] = audio_bytes
                    
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
            
            # Try different filename patterns
            patterns = [
                filename,
                f"{filename}.wav",
                f"{filename}.mp3",
                f"{filename}.flac",
                f"audio/{filename}",
                f"audio/{filename}.wav"
            ]
            
            audio_found = False
            for pattern in patterns:
                for audio_path, audio_bytes in self.extracted_audio.items():
                    if pattern in audio_path or os.path.basename(audio_path) == pattern:
                        caption = row['caption']
                        metadata = {
                            'split': row['split'],
                            'original_filename': filename
                        }
                        matched.append((audio_bytes, caption, metadata))
                        audio_found = True
                        break
                if audio_found:
                    break
                    
            if not audio_found:
                missing_count += 1
                
        print(f"Matched {len(matched)} audio-text pairs")
        print(f"Missing audio files: {missing_count}")
        
        return matched


def main():
    parser = argparse.ArgumentParser(description="Convert TextToAudioGrounding dataset to tar format")
    parser.add_argument("--audio-dir", type=str, 
                       default="/scratch-shared/gwijngaard/laion/TextToAudioGrounding",
                       help="Path to directory containing audio.zip")
    parser.add_argument("--metadata", type=str,
                       default="/gpfs/work4/0/einf6190/data-preparation/data/TextToAudioGrounding",
                       help="Path to metadata directory")
    parser.add_argument("--output-dir", type=str,
                       default="/scratch-shared/gwijngaard/tar/texttoaudiogrounding",
                       help="Output directory for tar files")
    parser.add_argument("--samples-per-tar", type=int, default=2048,
                       help="Number of samples per tar file")
    
    args = parser.parse_args()
    
    # Create processor
    processor = TextToAudioGroundingProcessor(
        audio_dir=args.audio_dir,
        metadata_path=args.metadata,
        output_dir=args.output_dir,
        task="SED")
    
    # Process dataset
    processor.process_dataset(samples_per_tar=args.samples_per_tar)


if __name__ == "__main__":
    main()