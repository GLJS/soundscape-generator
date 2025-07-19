#!/usr/bin/env python3
"""
Convert AudioHallucination dataset to WebDataset tar format.

Audio location: /scratch-shared/gwijngaard/laion/AudioHallucination/audio/
Metadata: /gpfs/work4/0/einf6190/data-preparation/data/AudioHallucination/
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from pathlib import Path
from utils import DatasetProcessor
from typing import List, Tuple, Dict
import argparse
from dotenv import load_dotenv

load_dotenv()


class AudioHallucinationProcessor(DatasetProcessor):
    """Processor for AudioHallucination dataset."""
    
    def load_metadata(self) -> pd.DataFrame:
        """Load AudioHallucination metadata from parquet files."""
        print(f"Loading metadata from {self.metadata_path}")
        
        dfs = []
        
        # Load the three parquet files
        adversarial_path = self.metadata_path / "Adversarial/data/test-00000-of-00001.parquet"
        if adversarial_path.exists():
            df = pd.read_parquet(adversarial_path)
            df['subset'] = 'adversarial'
            dfs.append(df)
            print(f"  Loaded {len(df)} adversarial entries")
            
        popular_path = self.metadata_path / "Popular/data/test-00000-of-00001.parquet"
        if popular_path.exists():
            df = pd.read_parquet(popular_path)
            df['subset'] = 'popular'
            dfs.append(df)
            print(f"  Loaded {len(df)} popular entries")
            
        random_path = self.metadata_path / "Random/data/test-00000-of-00001.parquet"
        if random_path.exists():
            df = pd.read_parquet(random_path)
            df['subset'] = 'random'
            dfs.append(df)
            print(f"  Loaded {len(df)} random entries")
            
        if dfs:
            combined_df = pd.concat(dfs, ignore_index=True)
            print(f"Total entries: {len(combined_df)}")
            return combined_df
        else:
            print("No metadata files found!")
            return pd.DataFrame()
        
    def match_audio_to_text(self, metadata_df: pd.DataFrame) -> List[Tuple[Path, str, Dict]]:
        """Match audio files to their captions."""
        matched = []
        
        # Index available audio files
        print("Indexing audio files...")
        audio_files = {}
        
        # Check in subdirectories matching the subsets
        for subset in ['Adversarial', 'Popular', 'Random']:
            subset_dir = self.audio_dir / subset
            if subset_dir.exists():
                for audio_file in subset_dir.glob("*.wav"):
                    audio_files[audio_file.stem] = audio_file
                    
        # Also check root audio directory
        for audio_file in self.audio_dir.glob("*.wav"):
            audio_files[audio_file.stem] = audio_file
            
        print(f"Found {len(audio_files)} audio files")
        
        # Match with metadata
        missing_count = 0
        
        for _, row in metadata_df.iterrows():
            audio_index = row.get('audio_index', '')
            
            # Try different naming patterns
            audio_key = audio_index.replace('.wav', '')
            
            if audio_key in audio_files:
                audio_path = audio_files[audio_key]
                # Combine prompt_text and label for caption
                caption = f"{row.get('prompt_text', '')} Answer: {row.get('label', '')}"
                metadata = {
                    'split': 'test',  # All are test data
                    'original_filename': audio_index,
                    'subset': row.get('subset', ''),
                    'label': row.get('label', ''),
                    'prompt_text': row.get('prompt_text', ''),
                    "task": "AQA"
                }
                matched.append((audio_path, caption, metadata))
            else:
                missing_count += 1
                
        print(f"Matched {len(matched)} audio-text pairs")
        print(f"Missing audio files: {missing_count}")
        
        return matched


def main():
    parser = argparse.ArgumentParser(description="Convert AudioHallucination dataset to tar format")
    parser.add_argument("--audio-dir", type=str, 
                       default="/scratch-shared/gwijngaard/laion/AudioHallucination/audio",
                       help="Path to audio files")
    parser.add_argument("--metadata", type=str,
                       default="/gpfs/work4/0/einf6190/data-preparation/data/AudioHallucination",
                       help="Path to metadata directory")
    parser.add_argument("--output-dir", type=str,
                       default="/scratch-shared/gwijngaard/tar/audiohallucination",
                       help="Output directory for tar files")
    parser.add_argument("--samples-per-tar", type=int, default=2048,
                       help="Number of samples per tar file")
    
    args = parser.parse_args()
    
    # Create processor
    processor = AudioHallucinationProcessor(
        audio_dir=args.audio_dir,
        metadata_path=args.metadata,
        output_dir=args.output_dir,
        task="AQA")
    
    # Process dataset
    processor.process_dataset(samples_per_tar=args.samples_per_tar)


if __name__ == "__main__":
    main()