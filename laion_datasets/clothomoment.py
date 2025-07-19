#!/usr/bin/env python3
"""
Convert ClothoMoment dataset to WebDataset tar format.

Audio location: /scratch-shared/gwijngaard/laion/ClothoMoment/audio/ (extracted folder with train/, valid/, test/ subdirs)
Metadata: /gpfs/work4/0/einf6190/data-preparation/data/ClothoMoment/json/
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import json
from pathlib import Path
from utils import DatasetProcessor, TarCreator, AudioProcessor
from typing import List, Tuple, Dict
import argparse
from dotenv import load_dotenv

load_dotenv()


class ClothoMomentProcessor(DatasetProcessor):
    """Processor for ClothoMoment dataset."""
    
    def __init__(self, audio_dir: str, metadata_path: str, output_dir: str):
        super().__init__(audio_dir, metadata_path, output_dir)
        
    def load_metadata(self) -> pd.DataFrame:
        """Load ClothoMoment metadata JSON files."""
        print(f"Loading metadata from {self.metadata_path}")
        
        dfs = []
        json_dir = self.metadata_path / "json"
        
        # Load recipe JSON files for each split
        for split, filename in [
            ('train', 'recipe_train.json'),
            ('valid', 'recipe_valid.json'),
            ('test', 'recipe_test.json')
        ]:
            json_path = json_dir / filename
            if json_path.exists():
                with open(json_path, 'r') as f:
                    data = json.load(f)
                    
                # Convert to dataframe
                df = pd.json_normalize(data, sep='_')
                df['split'] = split
                
                # Drop rows where fg is empty
                df = df[df['fg'].map(lambda x: len(x) > 0)]
                
                # Explode foreground items
                df = df.explode('fg')
                
                # Flatten fg dict columns
                fg_df = pd.json_normalize(df['fg'].dropna().tolist(), sep='_')
                
                # Add flattened columns back
                df = df.drop('fg', axis=1)
                for col in fg_df.columns:
                    df[f'fg_{col}'] = fg_df[col].values
                    
                # Create caption with temporal information
                df['fg_end_time'] = df['fg_start_time'] + df['fg_duration']
                df['caption'] = df.apply(
                    lambda x: f"{x['fg_caption']} [{int(x['fg_start_time'])}s, {int(x['fg_end_time'])}s]",
                    axis=1
                )
                
                df['file_name'] = df['split'] + '/' + df['name'] + '.wav'
                
                dfs.append(df)
                print(f"  Loaded {len(df)} {split} entries")
                
        if dfs:
            combined_df = pd.concat(dfs, ignore_index=True)
            print(f"Total entries: {len(combined_df)}")
            return combined_df
        else:
            return pd.DataFrame()
        
        
    def match_audio_to_text(self, metadata_df: pd.DataFrame) -> List[Tuple[Path, str, Dict]]:
        """Match audio files to their captions."""
        matched = []
        missing_count = 0
        
        for _, row in metadata_df.iterrows():
            filename = row['file_name']
            
            # Try to find the audio file
            audio_path = self.audio_dir / filename
            
            if not audio_path.exists():
                # Try without the split prefix
                base_filename = filename.split('/', 1)[-1] if '/' in filename else filename
                audio_path = self.audio_dir / base_filename
                
            if audio_path.exists():
                caption = row['caption']
                metadata = {
                    'split': row['split'],
                    'original_filename': filename,
                    'start_time': row.get('fg_start_time', 0),
                    'end_time': row.get('fg_end_time', 0),
                    'duration': row.get('fg_duration', 0),
                    'fg_caption': row.get('fg_caption', '')
                }
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
        tar_creator = TarCreator(self.output_dir, prefix="clothomoment", 
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
    parser = argparse.ArgumentParser(description="Convert ClothoMoment dataset to tar format")
    parser.add_argument("--audio-dir", type=str, 
                       default="/scratch-shared/gwijngaard/laion/ClothoMoment/audio",
                       help="Path to directory containing extracted audio files (with train/, valid/, test/ subdirs)")
    parser.add_argument("--metadata", type=str,
                       default="/gpfs/work4/0/einf6190/data-preparation/data/ClothoMoment",
                       help="Path to metadata directory")
    parser.add_argument("--output-dir", type=str,
                       default="/scratch-shared/gwijngaard/tar/clothomoment",
                       help="Output directory for tar files")
    parser.add_argument("--samples-per-tar", type=int, default=2048,
                       help="Number of samples per tar file")
    
    args = parser.parse_args()
    
    # Create processor
    processor = ClothoMomentProcessor(
        audio_dir=args.audio_dir,
        metadata_path=args.metadata,
        output_dir=args.output_dir)
    
    # Process dataset
    processor.process_dataset(samples_per_tar=args.samples_per_tar)


if __name__ == "__main__":
    main()