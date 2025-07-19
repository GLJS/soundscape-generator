#!/usr/bin/env python3
"""
Convert URBAN-SED dataset to WebDataset tar format.

Audio location: /scratch-shared/gwijngaard/laion/URBAN-SED_v2.0.0/audio.tar.gz
Annotations: /scratch-shared/gwijngaard/laion/URBAN-SED_v2.0.0/annotations.tar.gz
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


class URBANSEDProcessor(DatasetProcessor):
    """Processor for URBAN-SED dataset."""
    
    def __init__(self, audio_dir: str, metadata_path: str, output_dir: str):
        super().__init__(audio_dir, metadata_path, output_dir)
        self.audio_archive = self.audio_dir / "audio.tar.gz"
        self.annotations_archive = self.audio_dir / "annotations.tar.gz"
        self.extracted_audio = {}
        self.annotations = {}
        
    def load_metadata(self) -> pd.DataFrame:
        """Load URBAN-SED annotations."""
        print(f"Loading annotations from {self.annotations_archive}")
        
        if not self.annotations_archive.exists():
            print("annotations.tar.gz not found")
            return pd.DataFrame()
            
        # Extract annotation files
        records = []
        with tarfile.open(self.annotations_archive, "r:gz") as tar:
            for member in tar.getmembers():
                if member.isfile() and member.name.endswith('.txt'):
                    f = tar.extractfile(member)
                    if f:
                        content = f.read().decode('utf-8')
                        filename = os.path.basename(member.name).replace('.txt', '.wav')
                        
                        # Parse annotation content (adjust based on actual format)
                        # URBAN-SED typically has time-stamped sound events
                        lines = content.strip().split('\n')
                        events = []
                        for line in lines:
                            if line.strip():
                                # Example format: "start_time end_time event_label"
                                parts = line.split()
                                if len(parts) >= 3:
                                    events.append(parts[2])
                                    
                        caption = ', '.join(set(events)) if events else filename.replace('.wav', '')
                        
                        records.append({
                            'file_name': filename,
                            'caption': caption,
                            'split': 'train',
                            'raw_annotation': content
                        })
                        
        df = pd.DataFrame(records)
        print(f"Loaded {len(df)} annotation entries")
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
        """Match audio files to their annotations."""
        # First extract audio files if not already done
        if not self.extracted_audio:
            self.extract_audio_files()
            
        matched = []
        missing_count = 0
        
        # If we have annotations, match them
        if len(metadata_df) > 0:
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
        else:
            # No annotations, use audio files directly
            for filename, audio_bytes in self.extracted_audio.items():
                caption = filename.replace('.wav', '')
                metadata = {
                    'split': 'train',
                    'original_filename': filename
                }
                matched.append((audio_bytes, caption, metadata))
                
        print(f"Matched {len(matched)} audio-text pairs")
        if missing_count > 0:
            print(f"Missing audio files: {missing_count}")
        
        return matched


def main():
    parser = argparse.ArgumentParser(description="Convert URBAN-SED dataset to tar format")
    parser.add_argument("--audio-dir", type=str, 
                       default="/scratch-shared/gwijngaard/laion/URBAN-SED_v2.0.0",
                       help="Path to URBAN-SED directory")
    parser.add_argument("--metadata", type=str,
                       default="/scratch-shared/gwijngaard/laion/URBAN-SED_v2.0.0",
                       help="Path to metadata (same as audio-dir)")
    parser.add_argument("--output-dir", type=str,
                       default="/scratch-shared/gwijngaard/tar/urbansed",
                       help="Output directory for tar files")
    parser.add_argument("--samples-per-tar", type=int, default=2048,
                       help="Number of samples per tar file")
    
    args = parser.parse_args()
    
    # Create processor
    processor = URBANSEDProcessor(
        audio_dir=args.audio_dir,
        metadata_path=args.metadata,
        output_dir=args.output_dir,
        task="SED")
    
    # Process dataset
    processor.process_dataset(samples_per_tar=args.samples_per_tar)


if __name__ == "__main__":
    main()