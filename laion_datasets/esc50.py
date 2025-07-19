#!/usr/bin/env python3
"""
Convert ESC50 dataset to WebDataset tar format.

Audio location: /scratch-shared/gwijngaard/laion/ESC50/audio.tar.gz
Metadata: /gpfs/work4/0/einf6190/data-preparation/data/ESC50/esc50.csv
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from pathlib import Path
from utils import DatasetProcessor, AudioProcessor, TarCreator
from typing import List, Tuple, Dict
import argparse
import tarfile
from dotenv import load_dotenv

load_dotenv()


class ESC50Processor(DatasetProcessor):
    """Processor for ESC50 dataset."""
    
    def __init__(self, audio_dir: str, metadata_path: str, output_dir: str):
        super().__init__(audio_dir, metadata_path, output_dir)
        self.audio_archive = self.audio_dir / "audio.tar.gz"
        self.extracted_audio = {}
        
    def load_metadata(self) -> pd.DataFrame:
        """Load ESC50 metadata CSV."""
        print(f"Loading metadata from {self.metadata_path}")
        df = pd.read_csv(self.metadata_path)
        print(f"Loaded {len(df)} entries")
        return df
        
    def extract_audio_files(self):
        """Extract audio files from tar.gz archive into memory."""
        print(f"Extracting audio from {self.audio_archive}")
        
        with tarfile.open(self.audio_archive, "r:gz") as tar:
            for member in tar.getmembers():
                if member.isfile() and member.name.endswith('.wav'):
                    f = tar.extractfile(member)
                    if f:
                        # Store just the filename without path
                        filename = os.path.basename(member.name)
                        self.extracted_audio[filename] = f.read()
                        
        print(f"Extracted {len(self.extracted_audio)} audio files")
        
    def match_audio_to_text(self, metadata_df: pd.DataFrame) -> List[Tuple[Path, str, Dict]]:
        """Match audio files to their captions."""
        # First extract audio files if not already done
        if not self.extracted_audio:
            self.extract_audio_files()
            
        matched = []
        missing_count = 0
        
        for _, row in metadata_df.iterrows():
            filename = row['filename']
            
            if filename in self.extracted_audio:
                # We'll pass the audio bytes directly
                audio_bytes = self.extracted_audio[filename]
                # ESC50 uses 'category' as the label/caption
                caption = row['category']
                metadata = {
                    'split': row.get('split', 'train'),
                    'original_filename': filename,
                    'target': row.get('target', ''),
                    'fold': row.get('fold', ''),
                    'esc10': row.get('esc10', False)
                }
                matched.append((audio_bytes, caption, metadata))
            else:
                missing_count += 1
                print(f"Missing audio file: {filename}")
                
        print(f"Matched {len(matched)} audio-text pairs")
        print(f"Missing audio files: {missing_count}")
        
        return matched
        
    def process_dataset(self, samples_per_tar: int = 256):
        """Process the entire dataset into tar files."""
        # Load metadata
        metadata_df = self.load_metadata()
        
        # Match audio to text
        matched_samples = self.match_audio_to_text(metadata_df)
        
        # Create tar files
        tar_creator = TarCreator(self.output_dir, prefix="esc50", 
                                 samples_per_tar=samples_per_tar)
        
        # Process in batches
        all_summaries = []
        for i in range(0, len(matched_samples), samples_per_tar):
            batch = matched_samples[i:i+samples_per_tar]
            samples = []
            
            for audio_data, text, metadata in batch:
                try:
                    # Process audio (audio_data is already bytes)
                    audio_bytes, audio_metadata = self.audio_processor.process_audio_file(audio_data)
                    samples.append({
                        'audio_bytes': audio_bytes,
                        'text': text,
                        'metadata': {**metadata, **audio_metadata, 'task': 'SED'}
                    })
                except Exception as e:
                    print(f"Failed to process audio: {e}")
                    
            if samples:
                summary = tar_creator.create_tar_from_samples(samples, i // samples_per_tar)
                all_summaries.append(summary)
                
        # Create size file
        tar_creator.create_size_file(all_summaries)
        
        # Summary
        total_successful = sum(s['successful'] for s in all_summaries)
        total_failed = sum(s['failed'] for s in all_summaries)
        
        print(f"\nProcessing complete!")
        print(f"Total samples: {len(matched_samples)}")
        print(f"Successfully processed: {total_successful}")
        print(f"Failed: {total_failed}")
        print(f"Created {len(all_summaries)} tar files in {self.output_dir}")
        
        return all_summaries


def main():
    parser = argparse.ArgumentParser(description="Convert ESC50 dataset to tar format")
    parser.add_argument("--audio-dir", type=str, 
                       default="/scratch-shared/gwijngaard/laion/ESC50",
                       help="Path to directory containing audio.tar.gz")
    parser.add_argument("--metadata", type=str,
                       default="/gpfs/work4/0/einf6190/data-preparation/data/ESC50/esc50.csv",
                       help="Path to metadata CSV")
    parser.add_argument("--output-dir", type=str,
                       default="/scratch-shared/gwijngaard/tar/esc50",
                       help="Output directory for tar files")
    parser.add_argument("--samples-per-tar", type=int, default=2048,
                       help="Number of samples per tar file")
    
    args = parser.parse_args()
    
    # Create processor
    processor = ESC50Processor(
        audio_dir=args.audio_dir,
        metadata_path=args.metadata,
        output_dir=args.output_dir
    )
    
    # Process dataset
    processor.process_dataset(samples_per_tar=args.samples_per_tar)


if __name__ == "__main__":
    main()