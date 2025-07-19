#!/usr/bin/env python3
"""
Convert ClothoChatGPTMixup dataset to WebDataset tar format.

Audio location: /scratch-shared/gwijngaard/laion/ClothoChatGPTMixup/audio/ (extracted folder)
Metadata: /gpfs/work4/0/einf6190/data-preparation/data/ClothoChatGPTMixup/mixed_audio_info.csv
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from pathlib import Path
from utils import DatasetProcessor, AudioProcessor, TarCreator, ArchiveExtractor
from typing import List, Tuple, Dict
import argparse
import tempfile
from dotenv import load_dotenv

load_dotenv()


class ClothoChatGPTMixupProcessor(DatasetProcessor):
    """Processor for ClothoChatGPTMixup dataset."""
    
    def __init__(self, audio_dir: str, metadata_path: str, output_dir: str):
        super().__init__(audio_dir, metadata_path, output_dir)
        
    def load_metadata(self) -> pd.DataFrame:
        """Load ClothoChatGPTMixup metadata CSV."""
        print(f"Loading metadata from {self.metadata_path}")
        df = pd.read_csv(self.metadata_path)
        print(f"Loaded {len(df)} entries")
        return df
        
        
    def match_audio_to_text(self, metadata_df: pd.DataFrame) -> List[Tuple[Path, str, Dict]]:
        """Match audio files to their captions."""
        matched = []
        missing_count = 0
        
        for _, row in metadata_df.iterrows():
            filename = row['filename']
            audio_path = self.audio_dir / filename
            
            if audio_path.exists():
                caption = row['combined_caption']
                metadata = {
                    'split': row.get('split', 'train'),
                    'original_filename': filename
                }
                # Pass the Path object
                matched.append((audio_path, caption, metadata))
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
        tar_creator = TarCreator(self.output_dir, prefix="clothochatgptmixup", 
                                 samples_per_tar=samples_per_tar)
        
        # Process in batches
        all_summaries = []
        for i in range(0, len(matched_samples), samples_per_tar):
            batch = matched_samples[i:i+samples_per_tar]
            samples = []
            
            for audio_path, text, metadata in batch:
                try:
                    # Process audio file from path
                    audio_bytes, audio_metadata = self.audio_processor.process_audio_file(audio_path)
                    samples.append({
                        'audio_bytes': audio_bytes,
                        'text': text,
                        'metadata': {**metadata, **audio_metadata, 'task': 'AAC'}
                    })
                except Exception as e:
                    print(f"Failed to process audio {audio_path}: {e}")
                    
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
    parser = argparse.ArgumentParser(description="Convert ClothoChatGPTMixup dataset to tar format")
    parser.add_argument("--audio-dir", type=str, 
                       default="/scratch-shared/gwijngaard/laion/ClothoChatGPTMixup/audio",
                       help="Path to directory containing extracted audio files")
    parser.add_argument("--metadata", type=str,
                       default="/gpfs/work4/0/einf6190/data-preparation/data/ClothoChatGPTMixup/mixed_audio_info.csv",
                       help="Path to metadata CSV")
    parser.add_argument("--output-dir", type=str,
                       default="/scratch-shared/gwijngaard/tar/clothochatgptmixup",
                       help="Output directory for tar files")
    parser.add_argument("--samples-per-tar", type=int, default=2048,
                       help="Number of samples per tar file")
    
    args = parser.parse_args()
    
    # Create processor
    processor = ClothoChatGPTMixupProcessor(
        audio_dir=args.audio_dir,
        metadata_path=args.metadata,
        output_dir=args.output_dir
    )
    
    # Process dataset
    processor.process_dataset(samples_per_tar=args.samples_per_tar)


if __name__ == "__main__":
    main()