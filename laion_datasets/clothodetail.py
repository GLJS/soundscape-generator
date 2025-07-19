#!/usr/bin/env python3
"""
Convert ClothoDetail dataset to WebDataset tar format.

Audio location: /scratch-shared/gwijngaard/laion/ClothoDetail/audio/ (extracted folders: development/, evaluation/)
Metadata: /gpfs/work4/0/einf6190/data-preparation/data/ClothoDetail/Clotho-detail-annotation.json
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
from dotenv import load_dotenv

load_dotenv()


class ClothoDetailProcessor(DatasetProcessor):
    """Processor for ClothoDetail dataset."""
    
    def __init__(self, audio_dir: str, metadata_path: str, output_dir: str):
        super().__init__(audio_dir, metadata_path, output_dir)
        self.dev_dir = self.audio_dir / "development"
        self.eval_dir = self.audio_dir / "evaluation"
        
    def load_metadata(self) -> pd.DataFrame:
        """Load ClothoDetail metadata JSON."""
        print(f"Loading metadata from {self.metadata_path}")
        
        json_path = self.metadata_path / "Clotho-detail-annotation.json"
        with open(json_path, 'r') as f:
            data = json.load(f)
            
        # Convert to dataframe
        df = pd.DataFrame(data['annotations'])
        
        # Add .wav extension if not present
        df['audio_id'] = df['audio_id'].apply(
            lambda x: x if x.endswith('.wav') else x + '.wav'
        )
        
        print(f"Loaded {len(df)} entries")
        return df
        
        
    def match_audio_to_text(self, metadata_df: pd.DataFrame) -> List[Tuple[Path, str, Dict]]:
        """Match audio files to their captions."""
        matched = []
        missing_count = 0
        
        for _, row in metadata_df.iterrows():
            audio_id = row['audio_id']
            
            # Try to find the audio file in development or evaluation directories
            audio_path = None
            split = None
            
            # Check development directory
            if self.dev_dir.exists():
                dev_path = self.dev_dir / audio_id
                if dev_path.exists():
                    audio_path = dev_path
                    split = 'train'
            
            # Check evaluation directory if not found in dev
            if not audio_path and self.eval_dir.exists():
                eval_path = self.eval_dir / audio_id
                if eval_path.exists():
                    audio_path = eval_path
                    split = 'valid'
            
            if audio_path:
                caption = row.get('caption', '')
                metadata = {
                    'split': split,
                    'original_filename': audio_id,
                    'task': 'AAC'
                }
                matched.append((audio_path, caption, metadata))
            else:
                missing_count += 1
                print(f"Missing audio file: {audio_id}")
                
        print(f"Matched {len(matched)} audio-text pairs")
        print(f"Missing audio files: {missing_count}")
        
        return matched
        


def main():
    parser = argparse.ArgumentParser(description="Convert ClothoDetail dataset to tar format")
    parser.add_argument("--audio-dir", type=str, 
                       default="/scratch-shared/gwijngaard/laion/ClothoDetail/",
                       help="Path to directory containing extracted audio files (with development/ and evaluation/ subdirs)")
    parser.add_argument("--metadata", type=str,
                       default="/gpfs/work4/0/einf6190/data-preparation/data/ClothoDetail",
                       help="Path to metadata directory")
    parser.add_argument("--output-dir", type=str,
                       default="/scratch-shared/gwijngaard/tar/clothodetail",
                       help="Output directory for tar files")
    parser.add_argument("--samples-per-tar", type=int, default=2048,
                       help="Number of samples per tar file")
    
    args = parser.parse_args()
    
    # Create processor
    processor = ClothoDetailProcessor(
        audio_dir=args.audio_dir,
        metadata_path=args.metadata,
        output_dir=args.output_dir)
    
    # Process dataset
    processor.process_dataset(samples_per_tar=args.samples_per_tar)


if __name__ == "__main__":
    main()