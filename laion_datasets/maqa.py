#!/usr/bin/env python3
"""
Convert mAQA dataset to WebDataset tar format.

Audio location: /scratch-shared/gwijngaard/laion/mAQA/
Metadata: /gpfs/work4/0/einf6190/data-preparation/data/mAQA/
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from pathlib import Path
from utils import DatasetProcessor, TarCreator
from typing import List, Tuple, Dict
import argparse
from dotenv import load_dotenv

load_dotenv()


class MAQAProcessor(DatasetProcessor):
    """Processor for mAQA (multilingual Audio Question Answering) dataset."""
    
    def load_metadata(self) -> pd.DataFrame:
        """Load mAQA metadata CSV files."""
        print(f"Loading metadata from {self.metadata_path}")
        
        dfs = []
        
        # Load all CSV files in the directory
        for csv_file in self.metadata_path.glob("*.csv"):
            df = pd.read_csv(csv_file)
            
            # Extract split and language from filename
            filename = csv_file.stem
            parts = filename.split('_')
            
            if len(parts) >= 3:
                split = parts[1]  # train, test, val
                language = parts[2]  # eng, french, etc.
            else:
                split = 'train'
                language = 'unknown'
                
            df['split'] = split
            df['language'] = language
            df['source_file'] = filename
            
            dfs.append(df)
            print(f"  Loaded {len(df)} entries from {filename}")
            
        if dfs:
            combined_df = pd.concat(dfs, ignore_index=True)
            
            # Create caption from question and answer  
            combined_df['caption'] = combined_df.apply(
                lambda x: f"{x.get('QuestionText', '')} Answer: {x.get('answer', '')}", 
                axis=1
            )
            
            print(f"Total entries: {len(combined_df)}")
            return combined_df
        else:
            return pd.DataFrame()
        
    def match_audio_to_text(self, metadata_df: pd.DataFrame) -> List[Tuple[Path, str, Dict]]:
        """Match audio files to their captions."""
        matched = []
        missing_count = 0
        
        # Index available audio files
        print("Indexing audio files...")
        audio_files = {}
        
        # Look for audio files in the mAQA directory
        for ext in ['.wav', '.mp3', '.flac']:
            for audio_file in self.audio_dir.rglob(f"*{ext}"):
                audio_files[audio_file.name] = audio_file
                
        print(f"Found {len(audio_files)} audio files")
        
        # Match with metadata
        for _, row in metadata_df.iterrows():
            filename = row.get('file_name', '')
            
            if filename in audio_files:
                audio_path = audio_files[filename]
                caption = row['caption']
                # Determine task type based on source file
                task = 'MC' if 'multiple_choice' in str(row.get('source_file', '')).lower() else 'AQA'
                metadata = {
                    'split': row['split'],
                    'original_filename': filename,
                    'language': row.get('language', ''),
                    'question': row.get('QuestionText', ''),
                    'answer': row.get('answer', ''),
                    'task': task,
                    'source_file': row.get('source_file', '')
                }
                matched.append((audio_path, caption, metadata))
            else:
                missing_count += 1
                
        print(f"Matched {len(matched)} audio-text pairs")
        print(f"Missing audio files: {missing_count}")
        
        return matched
        


def main():
    parser = argparse.ArgumentParser(description="Convert mAQA dataset to tar format")
    parser.add_argument("--audio-dir", type=str, 
                       default="/scratch-shared/gwijngaard/laion/mAQA",
                       help="Path to audio files")
    parser.add_argument("--metadata", type=str,
                       default="/gpfs/work4/0/einf6190/data-preparation/data/mAQA",
                       help="Path to metadata directory")
    parser.add_argument("--output-dir", type=str,
                       default="/scratch-shared/gwijngaard/tar/maqa",
                       help="Output directory for tar files")
    parser.add_argument("--samples-per-tar", type=int, default=2048,
                       help="Number of samples per tar file")
    
    args = parser.parse_args()
    
    # Create processor
    processor = MAQAProcessor(
        audio_dir=args.audio_dir,
        metadata_path=args.metadata,
        output_dir=args.output_dir
    )
    
    # Process dataset
    processor.process_dataset(samples_per_tar=args.samples_per_tar)


if __name__ == "__main__":
    main()