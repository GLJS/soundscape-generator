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
import ast

load_dotenv()


class ClothoEntailmentProcessor(DatasetProcessor):
    """Processor for ClothoEntailment dataset."""
    
    def __init__(self, audio_dir: str, metadata_path: str, output_dir: str):
        super().__init__(audio_dir, metadata_path, output_dir)
        
    def load_metadata(self) -> pd.DataFrame:
        """Load ClothoEntailment metadata CSV files and create question-answer pairs."""
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
            
            # Question templates for each entailment type
            question_templates = {
                'entailment': "What follows this caption? Caption: ",
                'neutral': "What could be true about this caption? Caption: ",
                'contradiction': "What contradicts this caption? Caption: "
            }
            
            # Create rows for each caption and entailment type combination
            all_rows = []
            
            for _, row in combined_df.iterrows():
                audio_file = row['Audio file']
                split = row['split']
                
                # Parse the Caption column (it's a string representation of a list)
                try:
                    captions = ast.literal_eval(row['Caption'])
                except:
                    # If parsing fails, skip this row
                    print(f"Failed to parse captions for {audio_file}")
                    continue
                
                # For each caption, create 3 question-answer pairs
                for caption in captions:
                    # Entailment question
                    all_rows.append({
                        'file_name': f"{split}/{audio_file}",
                        'question': question_templates['entailment'] + caption,
                        'answer': row['Entailment'],
                        'caption': caption,
                        'question_type': 'entailment',
                        'split': split
                    })
                    
                    # Neutral question
                    all_rows.append({
                        'file_name': f"{split}/{audio_file}",
                        'question': question_templates['neutral'] + caption,
                        'answer': row['Neutral'],
                        'caption': caption,
                        'question_type': 'neutral',
                        'split': split
                    })
                    
                    # Contradiction question
                    all_rows.append({
                        'file_name': f"{split}/{audio_file}",
                        'question': question_templates['contradiction'] + caption,
                        'answer': row['Contradiction'],
                        'caption': caption,
                        'question_type': 'contradiction',
                        'split': split
                    })
            
            result_df = pd.DataFrame(all_rows)
            print(f"Total question-answer pairs created: {len(result_df)}")
            
            return result_df
        else:
            return pd.DataFrame()
        
        
    def match_audio_to_text(self, metadata_df: pd.DataFrame) -> List[Tuple[Path, str, Dict]]:
        """Match audio files to their question-answer pairs."""
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
                # Combine question and answer for the text field
                text = f"Question: {row['question']}\nAnswer: {row['answer']}"
                metadata = {
                    'split': row['split'],
                    'original_filename': filename,
                    'question_type': row['question_type'],
                    'caption': row['caption'],
                    'question': row['question'],
                    'answer': row['answer'],
                    'task': 'AQA'
                }
                matched.append((audio_path, text, metadata))
            else:
                missing_count += 1
                print(f"Missing audio file: {filename}")
                
        print(f"Matched {len(matched)} audio-question-answer triplets")
        print(f"Missing audio files: {missing_count}")
        
        return matched
        


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