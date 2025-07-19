#!/usr/bin/env python3
"""
Convert RichDetailAudioTextSimulation dataset to WebDataset tar format.

Audio location: /scratch-shared/gwijngaard/laion/RichDetailAudioTextSimulation/audio.tar.gz
Metadata: /gpfs/work4/0/einf6190/data-preparation/data/RichDetailAudioTextSimulation/caption_file.json
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


class RichDetailAudioProcessor(DatasetProcessor):
    """Processor for RichDetailAudioTextSimulation dataset."""
    
    def __init__(self, audio_dir: str, metadata_path: str, output_dir: str):
        super().__init__(audio_dir, metadata_path, output_dir)
        self.audio_archive = self.audio_dir / "audio.tar.gz"
        self.extracted_audio = {}
        
    def load_metadata(self) -> pd.DataFrame:
        """Load RichDetailAudio metadata JSON."""
        print(f"Loading metadata from {self.metadata_path}")
        
        json_path = self.metadata_path / "caption_file.json"
        
        # Read JSON as series and convert to dataframe
        series = pd.read_json(json_path, typ='series')
        df = pd.DataFrame({'file_name': series.index, 'caption': series.values})
        
        # Add .wav extension to filenames
        df['file_name'] = df['file_name'] + '.wav'
        df['split'] = 'train'
        
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
                    'original_filename': filename
                }
                matched.append((audio_bytes, caption, metadata))
            else:
                missing_count += 1
                
        print(f"Matched {len(matched)} audio-text pairs")
        print(f"Missing audio files: {missing_count}")
        
        return matched


def main():
    parser = argparse.ArgumentParser(description="Convert RichDetailAudioTextSimulation dataset to tar format")
    parser.add_argument("--audio-dir", type=str, 
                       default="/scratch-shared/gwijngaard/laion/RichDetailAudioTextSimulation",
                       help="Path to directory containing audio.tar.gz")
    parser.add_argument("--metadata", type=str,
                       default="/gpfs/work4/0/einf6190/data-preparation/data/RichDetailAudioTextSimulation",
                       help="Path to metadata directory")
    parser.add_argument("--output-dir", type=str,
                       default="/scratch-shared/gwijngaard/tar/richdetailaudio",
                       help="Output directory for tar files")
    parser.add_argument("--samples-per-tar", type=int, default=2048,
                       help="Number of samples per tar file")
    
    args = parser.parse_args()
    
    # Create processor
    processor = RichDetailAudioProcessor(
        audio_dir=args.audio_dir,
        metadata_path=args.metadata,
        output_dir=args.output_dir,
        task="AAC")
    
    # Process dataset
    processor.process_dataset(samples_per_tar=args.samples_per_tar)


if __name__ == "__main__":
    main()