#!/usr/bin/env python3
"""
Convert ClothoEntailment dataset to WebDataset tar format.

Audio location: /scratch-shared/gwijngaard/laion/ClothoEntailment/audio/ (extracted folder)
Metadata: /gpfs/work4/0/einf6190/data-preparation/data/ClothoEntailment/
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from pathlib import Path
from utils import DatasetProcessor, TarCreator, AudioProcessor
from typing import List, Tuple, Dict
import argparse
from dotenv import load_dotenv

load_dotenv()


class ClothoEntailmentProcessor(DatasetProcessor):
    """Processor for ClothoEntailment dataset."""
    
    def __init__(self, audio_dir: str, metadata_path: str, output_dir: str):
        super().__init__(audio_dir, metadata_path, output_dir)
        
    def load_metadata(self) -> pd.DataFrame:
        """Load ClothoEntailment metadata CSV files."""
        print(f"Loading metadata from {self.metadata_path}")
        
        dfs = []
        
        # Load development, evaluation, and validation CSVs
        for split, filename in [
            ('train', 'clotho_entailment_development.csv'),
            ('test', 'clotho_entailment_evaluation.csv'),
            ('valid', 'clotho_entailment_validation.csv')
        ]:
            csv_path = self.metadata_path / filename
            if csv_path.exists():
                df = pd.read_csv(csv_path)
                df['split'] = split
                dfs.append(df)
                print(f"  Loaded {len(df)} {split} entries")
                
        if dfs:
            combined_df = pd.concat(dfs, ignore_index=True)
            
            # Melt to create separate rows for each caption type
            melted_df = pd.melt(
                combined_df,
                id_vars=['Audio file', 'split'],
                value_vars=['Entailment', 'Neutral', 'Contradiction'],
                var_name='caption_type',
                value_name='caption_text'
            )
            
            # Combine caption type and text
            melted_df['caption'] = melted_df['caption_type'] + ': ' + melted_df['caption_text']
            melted_df.rename(columns={'Audio file': 'file_name'}, inplace=True)
            
            # Add split prefix to filename
            melted_df['file_name'] = melted_df.apply(
                lambda x: f"{x['split']}/{x['file_name']}", axis=1
            )
            
            print(f"Total entries after melting: {len(melted_df)}")
            return melted_df
        else:
            return pd.DataFrame()
        
        
    def match_audio_to_text(self, metadata_df: pd.DataFrame) -> List[Tuple[Path, str, Dict]]:
        """Match audio files to their captions."""
        matched = []
        missing_count = 0
        
        for _, row in metadata_df.iterrows():
            filename = row['file_name']
            
            # Try different paths to find the audio file
            audio_path = None
            
            # Try direct path
            possible_path = self.audio_dir / filename
            if possible_path.exists():
                audio_path = possible_path
            else:
                # Try without split prefix
                base_filename = filename.split('/', 1)[-1] if '/' in filename else filename
                possible_path = self.audio_dir / base_filename
                if possible_path.exists():
                    audio_path = possible_path
                else:
                    # Search in subdirectories
                    for subdir in self.audio_dir.iterdir():
                        if subdir.is_dir():
                            possible_path = subdir / base_filename
                            if possible_path.exists():
                                audio_path = possible_path
                                break
            
            if audio_path:
                caption = row['caption']
                metadata = {
                    'split': row['split'],
                    'original_filename': filename,
                    'caption_type': row.get('caption_type', '')
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
        tar_creator = TarCreator(self.output_dir, prefix="clothoentailment", 
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
                        'metadata': {**metadata, **audio_metadata, 'task': 'AQA'}
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
    parser = argparse.ArgumentParser(description="Convert ClothoEntailment dataset to tar format")
    parser.add_argument("--audio-dir", type=str, 
                       default="/scratch-shared/gwijngaard/laion/ClothoEntailment/audio",
                       help="Path to directory containing extracted audio files")
    parser.add_argument("--metadata", type=str,
                       default="/gpfs/work4/0/einf6190/data-preparation/data/ClothoEntailment",
                       help="Path to metadata directory")
    parser.add_argument("--output-dir", type=str,
                       default="/scratch-shared/gwijngaard/tar/clothoentailment",
                       help="Output directory for tar files")
    parser.add_argument("--samples-per-tar", type=int, default=2048,
                       help="Number of samples per tar file")
    
    args = parser.parse_args()
    
    # Create processor
    processor = ClothoEntailmentProcessor(
        audio_dir=args.audio_dir,
        metadata_path=args.metadata,
        output_dir=args.output_dir)
    
    # Process dataset
    processor.process_dataset(samples_per_tar=args.samples_per_tar)


if __name__ == "__main__":
    main()