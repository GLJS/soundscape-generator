#!/usr/bin/env python3
"""
Convert compa dataset to WebDataset tar format.

Audio location: /scratch-shared/gwijngaard/laion/compa/medleysolos.tar
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


class CompaProcessor(DatasetProcessor):
    """Processor for compa dataset."""
    
    def __init__(self, audio_dir: str, metadata_path: str, output_dir: str):
        super().__init__(audio_dir, metadata_path, output_dir)
        self.audio_archive = self.audio_dir / "medleysolos.tar"
        
    def load_metadata(self) -> pd.DataFrame:
        """Compa may not have separate metadata."""
        print("Compa: No separate metadata file expected")
        return pd.DataFrame()
        
    def match_audio_to_text(self, metadata_df: pd.DataFrame) -> List[Tuple[bytes, str, Dict]]:
        """Extract audio files and use filenames as captions."""
        matched = []
        
        print(f"Processing audio from {self.audio_archive}")
        
        if not self.audio_archive.exists():
            print(f"Audio archive {self.audio_archive} not found")
            return []
            
        with tarfile.open(self.audio_archive, "r") as tar:
            for member in tar.getmembers():
                if member.isfile() and any(member.name.endswith(ext) for ext in ['.wav', '.mp3', '.flac']):
                    f = tar.extractfile(member)
                    if f:
                        audio_bytes = f.read()
                        filename = os.path.basename(member.name)
                        
                        # Parse filename for instrument/caption info
                        # MedleySolos typically has instrument info in filename
                        caption = filename.replace('.wav', '').replace('.mp3', '').replace('.flac', '')
                        
                        # Extract instrument from path if available
                        parts = member.name.split('/')
                        if len(parts) > 1:
                            instrument = parts[-2]  # Usually instrument is in parent directory
                            caption = f"{instrument} solo: {caption}"
                        
                        metadata = {
                            'split': 'train',
                            'original_filename': filename,
                            'full_path': member.name
                        }
                        
                        matched.append((audio_bytes, caption, metadata))
                        
        print(f"Processed {len(matched)} audio files")
        return matched


def main():
    parser = argparse.ArgumentParser(description="Convert compa dataset to tar format")
    parser.add_argument("--audio-dir", type=str, 
                       default="/scratch-shared/gwijngaard/laion/compa",
                       help="Path to compa directory")
    parser.add_argument("--metadata", type=str,
                       default="/scratch-shared/gwijngaard/laion/compa",
                       help="Path to metadata (same as audio-dir)")
    parser.add_argument("--output-dir", type=str,
                       default="/scratch-shared/gwijngaard/tar/compa",
                       help="Output directory for tar files")
    parser.add_argument("--samples-per-tar", type=int, default=2048,
                       help="Number of samples per tar file")
    
    args = parser.parse_args()
    
    # Create processor
    processor = CompaProcessor(
        audio_dir=args.audio_dir,
        metadata_path=args.metadata,
        output_dir=args.output_dir,
        task="AAC")
    
    # Process dataset
    processor.process_dataset(samples_per_tar=args.samples_per_tar)


if __name__ == "__main__":
    main()