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

load_dotenv()


class AudioDataFullProcessor(DatasetProcessor):
    """Processor for AudioDataFull dataset."""
    
    def __init__(self, audio_dir: str, metadata_path: str, output_dir: str, 
                 use_recaption: bool = True, batch_size: int = 8, num_workers: int = 8):
        super().__init__(audio_dir, metadata_path, output_dir)
        # Process extracted directories instead of tar archive
        self.use_recaption = use_recaption
        self.batch_size = batch_size
        self.num_workers = num_workers
        
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
        processed_count = 0
        
        print(f"Processing audio files from {self.audio_dir}")
        
        if len(metadata_df) > 0 and 'local_path' in metadata_df.columns:
            # Process files based on metadata
            print(f"  Processing {len(metadata_df)} files from metadata")
            
            # First collect all valid audio paths and metadata
            audio_paths = []
            metadata_list = []
            
            for idx, row in metadata_df.iterrows():
                try:
                    audio_path = Path(row['local_path'])
                    
                    if not audio_path.exists():
                        print(f"    File not found: {audio_path}")
                        continue
                    
                    metadata = {
                        'split': 'train',
                        'original_filename': row['filename'],
                        'dataset_root': row.get('dataset_root', ''),
                        'full_path': row.get('path', ''),
                        'local_path': str(audio_path),
                        'original_description': row.get('description', '')
                    }
                    
                    audio_paths.append(audio_path)
                    metadata_list.append(metadata)
                    processed_count += 1
                    
                    # Limit for testing/memory management
                    if processed_count >= 1000:
                        print(f"  Collected {processed_count} files (limited for testing)")
                        break
                        
                except Exception as e:
                    print(f"    Error processing row {idx}: {e}")
                    continue
            
            # Process with LLM recaptioning if enabled
            if self.use_recaption and audio_paths:
                print(f"\n  Starting LLM recaptioning for {len(audio_paths)} files...")
                
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
            else:
                # Fallback to original caption extraction
                print("  Using original captions from metadata...")
                for audio_path, metadata in zip(audio_paths, metadata_list):
                    caption = metadata.get('original_description', '')
                    if isinstance(caption, str) and caption.startswith('['):
                        try:
                            desc_list = ast.literal_eval(caption)
                            if desc_list and isinstance(desc_list[0], str):
                                caption = desc_list[0].replace('comment=', '').split('\\;')[0]
                        except:
                            pass
                    
                    if not caption or caption == '[]':
                        caption = metadata['original_filename'].replace('.wav', '').replace('.mp3', '').replace('.flac', '')
                    
                    matched.append((audio_path, caption, metadata))
                    
        else:
            # Fallback: process directories without metadata
            print("  No metadata available, processing directories directly")
            
            # Define subdirectories to process
            subdirs = [
                'Digiffects2448',
                'GeneralHardDriveCombo',
                'HollywoodEdge-2448',
                'MikeMcDonoughSpecialty-2448',
                'Serafine-2448',
                'Soundstorm-2448',
                'Ultimate-2448'
            ]
            
            audio_paths = []
            metadata_list = []
            
            for subdir in subdirs:
                subdir_path = self.audio_dir / subdir
                if not subdir_path.exists():
                    continue
                    
                print(f"  Collecting files from {subdir}...")
                
                # Find all audio files in subdirectory
                audio_files = []
                for ext in ['.wav', '.mp3', '.flac']:
                    audio_files.extend(subdir_path.rglob(f'*{ext}'))
                
                for audio_path in audio_files:
                    try:
                        filename = audio_path.name
                        relative_path = str(audio_path.relative_to(self.audio_dir))
                        
                        metadata = {
                            'split': 'train',
                            'original_filename': filename,
                            'subdirectory': subdir,
                            'full_path': relative_path
                        }
                        
                        audio_paths.append(audio_path)
                        metadata_list.append(metadata)
                        processed_count += 1
                        
                        if processed_count >= 1000:
                            print(f"  Collected {processed_count} files (limited for testing)")
                            break
                            
                    except Exception as e:
                        print(f"    Error processing {audio_path}: {e}")
                        continue
                
                if processed_count >= 1000:
                    break
            
            # Process with LLM recaptioning if enabled
            if self.use_recaption and audio_paths:
                print(f"\n  Starting LLM recaptioning for {len(audio_paths)} files...")
                
                llm_results = process_dataset_with_llm(
                    audio_paths=audio_paths,
                    metadata=metadata_list,
                    batch_size=self.batch_size,
                    num_workers=self.num_workers,
                    labels_csv=None  # No labels CSV for directory processing
                )
                
                matched.extend(llm_results)
            else:
                # Fallback to filename-based captions
                print("  Using filename-based captions...")
                for audio_path, metadata in zip(audio_paths, metadata_list):
                    caption = metadata['original_filename'].replace('.wav', '').replace('.mp3', '').replace('.flac', '')
                    matched.append((audio_path, caption, metadata))
                            
        print(f"Total processed: {len(matched)} audio-text pairs")
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
    parser.add_argument("--batch-size", type=int, default=8,
                       help="Batch size for LLM processing")
    parser.add_argument("--num-workers", type=int, default=8,
                       help="Number of workers for data loading")
    parser.add_argument("--no-recaption", action="store_true",
                       help="Disable LLM recaptioning (use original captions)")
    
    args = parser.parse_args()
    
    # Create processor
    processor = AudioDataFullProcessor(
        audio_dir=args.audio_dir,
        metadata_path=args.metadata,
        output_dir=args.output_dir,
        use_recaption=not args.no_recaption,
        batch_size=args.batch_size,
        num_workers=args.num_workers)
    
    # Process dataset
    processor.process_dataset(samples_per_tar=args.samples_per_tar)


if __name__ == "__main__":
    main()