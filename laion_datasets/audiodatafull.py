#!/usr/bin/env python3
"""
Convert AudioDataFull dataset to WebDataset tar format.

Audio location: /scratch-shared/gwijngaard/laion/AudioDataFull/
Note: This processes extracted audio files from subdirectories
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
import ast
from utils_llm import process_dataset_with_llm
from utils_gemma import process_dataset_with_gemma
from tqdm import tqdm
import json
load_dotenv()


class AudioDataFullProcessor(DatasetProcessor):
    """Processor for AudioDataFull dataset."""
    
    def __init__(self, audio_dir: str, metadata_path: str, output_dir: str, 
                 use_recaption: bool = True, batch_size: int = 128, num_workers: int = 16,
                 model_type: str = "gemma"):
        super().__init__(audio_dir, metadata_path, output_dir)
        # Process extracted directories instead of tar archive
        self.use_recaption = use_recaption
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.model_type = model_type
        
    def load_metadata(self) -> pd.DataFrame:
        """Load metadata from df_lemm_srl_path.csv."""
        print(f"Loading metadata from {self.audio_dir}")
        
        # Use the specific CSV file
        csv_path = self.audio_dir / "df_lemm_srl_path.csv"
        
        if csv_path.exists():
            try:
                df = pd.read_csv(csv_path)
                print(f"  Loaded {len(df)} entries from {csv_path.name}")
                
                # Convert paths from /root/share/AudioData/ to actual location
                if 'path' in df.columns:
                    df['local_path'] = df['path'].str.replace(
                        '/root/share/AudioData/', 
                        str(self.audio_dir) + '/',
                        regex=False
                    )
                
                return df
            except Exception as e:
                print(f"  Error loading {csv_path.name}: {e}")
        
        # No metadata, will process audio files directly
        print("No metadata CSV file found")
        return pd.DataFrame()
        
    def match_audio_to_text(self, metadata_df: pd.DataFrame) -> List[Tuple[Path, str, Dict]]:
        """Process audio files using metadata from CSV."""
        matched = []
        
        print(f"Processing audio files from {self.audio_dir}")
        
        if len(metadata_df) > 0 and 'local_path' in metadata_df.columns:
            # Process files based on metadata
            print(f"  Processing {len(metadata_df)} files from metadata")
            
            # First collect all valid audio paths and metadata
            audio_paths = []
            metadata_list = []
            
            for idx, row in tqdm(metadata_df.iterrows(), total=len(metadata_df)):
                try:
                    if idx > 1000:
                        break
                    audio_path = Path(row['local_path'])
                    
                    if not audio_path.exists():
                        print(f"    File not found: {audio_path}")
                        continue
                    
                    metadata = {
                        'split': 'train',
                        'original_filename': row['filename'],
                        'full_path': row.get('path', '').replace('/root/share/AudioData/', ''),
                        'original_description': row.get('description', '')
                    }

                    
                    audio_paths.append(audio_path)
                    metadata_list.append(metadata)
                                            
                except Exception as e:
                    print(f"    Error processing row {idx}: {e}")
                    continue
            
            if self.model_type == "gemma":
                print(f"\n  Starting Gemma audio processing for {len(audio_paths)} files...")
                print(f"  Using {self.batch_size} batch size and {self.num_workers} workers")
                # Prepare text descriptions from metadata for context
                texts = []
                for metadata in metadata_list:
                    # Create descriptive text from metadata
                    desc_parts = []
                    if metadata.get('original_filename'):
                        desc_parts.append(f"Filename: {metadata['original_filename']}")
                    if metadata.get('original_description'):
                        desc_parts.append(f"Original Description: {metadata['original_description']}")
                    texts.append(" | ".join(desc_parts) if desc_parts else "")

                
                # Sort by text length for sequence bucketing (improves throughput ~2x)
                print("  Sorting by text length for sequence bucketing...")
                combined = list(zip(audio_paths, texts, metadata_list))
                combined_sorted = sorted(combined, key=lambda x: len(x[1]), reverse=True)
                audio_paths, texts, metadata_list = zip(*combined_sorted)
                audio_paths, texts, metadata_list = list(audio_paths), list(texts), list(metadata_list)
                
                # Process with Gemma using actual audio files
                gemma_results = process_dataset_with_gemma(
                    audio_paths=audio_paths,
                    texts=texts,  # Provide context texts
                    metadata=metadata_list,
                    system_prompt="You are a helpful assistant.",
                    user_prompt_template="Describe in detail what you hear in the audio in max. 100 words. For context, use the following information, but don't quote it directly: {text}.",
                    batch_size=self.batch_size,
                    num_workers=self.num_workers,
                    generation_config={
                        "max_new_tokens": 200,
                        "temperature": 0.7,
                        "top_p": 0.9,
                    }
                )
                
                # Results already in correct format (audio_path, generated_caption, metadata)
                matched.extend(gemma_results)
            
            else:  # audioflamingo
                print(f"\n  Starting AudioFlamingo recaptioning for {len(audio_paths)} files...")
                
                # Use the CSV path for label context
                labels_csv = str(self.audio_dir / "df_lemm_srl_path.csv")
                
                # Process all audio files with LLM
                llm_results = process_dataset_with_llm(
                    audio_paths=audio_paths,
                    metadata=metadata_list,
                    batch_size=self.batch_size,
                    num_workers=self.num_workers,
                    labels_csv=labels_csv if Path(labels_csv).exists() else None
                )
                
                matched.extend(llm_results)
                            
        print(f"Total processed: {len(matched)} audio-text pairs")
        # save to json as backup (convert Path objects to strings)
        matched_serializable = []
        for item in matched:
            # Convert Path objects to strings
            audio_path = str(item[0]) if hasattr(item[0], '__fspath__') else item[0]
            text = item[1]
            metadata = item[2]
            matched_serializable.append([audio_path, text, metadata])
        
        with open("matched_audiodatafull.json", "w") as f:
            json.dump(matched_serializable, f)
        print(f"Saved {len(matched)} audio-text pairs to matched_audiodatafull.json")
        return matched


def main():
    parser = argparse.ArgumentParser(description="Convert AudioDataFull dataset to tar format")
    parser.add_argument("--audio-dir", type=str, 
                       default="/scratch-shared/gwijngaard/laion/AudioDataFull",
                       help="Path to AudioDataFull directory")
    parser.add_argument("--metadata", type=str,
                       default="/scratch-shared/gwijngaard/laion/AudioDataFull",
                       help="Path to metadata (same as audio-dir)")
    parser.add_argument("--output-dir", type=str,
                       default="/scratch-shared/gwijngaard/tar/audiodatafull",
                       help="Output directory for tar files")
    parser.add_argument("--samples-per-tar", type=int, default=2048,
                       help="Number of samples per tar file")
    parser.add_argument("--batch-size", type=int, default=96,
                       help="Batch size for LLM processing")
    parser.add_argument("--num-workers", type=int, default=12,
                       help="Number of workers for data loading")
    parser.add_argument("--no-recaption", action="store_true",
                       help="Disable LLM recaptioning (use original captions)")
    parser.add_argument("--model", type=str, default="gemma", 
                       choices=["gemma", "audioflamingo"],
                       help="Model to use for recaptioning (default: gemma)")
    
    args = parser.parse_args()
    
    # Create processor
    processor = AudioDataFullProcessor(
        audio_dir=args.audio_dir,
        metadata_path=args.metadata,
        output_dir=args.output_dir,
        use_recaption=not args.no_recaption,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        model_type=args.model)
    
    # Process dataset
    processor.process_dataset(samples_per_tar=args.samples_per_tar)


if __name__ == "__main__":
    main()