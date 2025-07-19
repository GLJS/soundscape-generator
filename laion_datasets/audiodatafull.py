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

load_dotenv()


class AudioDataFullProcessor(DatasetProcessor):
    """Processor for AudioDataFull dataset."""
    
    def __init__(self, audio_dir: str, metadata_path: str, output_dir: str):
        super().__init__(audio_dir, metadata_path, output_dir)
        # Process extracted directories instead of tar archive
        
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
            
            for idx, row in metadata_df.iterrows():
                try:
                    audio_path = Path(row['local_path'])
                    
                    if not audio_path.exists():
                        print(f"    File not found: {audio_path}")
                        continue
                    
                    # Extract caption from description
                    caption = row.get('description', '')
                    if isinstance(caption, str) and caption.startswith('['):
                        # Parse the description field
                        try:
                            desc_list = ast.literal_eval(caption)
                            if desc_list and isinstance(desc_list[0], str):
                                # Extract meaningful part from comment
                                caption = desc_list[0].replace('comment=', '').split('\\;')[0]
                        except:
                            pass
                    
                    # If no good caption, use filename
                    if not caption or caption == '[]':
                        caption = row['filename'].replace('.wav', '').replace('.mp3', '').replace('.flac', '')
                    
                    metadata = {
                        'split': 'train',
                        'original_filename': row['filename'],
                        'dataset_root': row.get('dataset_root', ''),
                        'full_path': row.get('path', ''),
                        'local_path': str(audio_path)
                    }
                    
                    matched.append((audio_path, caption, metadata))
                    processed_count += 1
                    
                    # Limit for testing/memory management
                    if processed_count >= 1000:
                        print(f"  Processed {processed_count} files (limited for testing)")
                        break
                        
                    if processed_count % 100 == 0:
                        print(f"  Processed {processed_count} files...")
                        
                except Exception as e:
                    print(f"    Error processing row {idx}: {e}")
                    continue
                    
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
            
            for subdir in subdirs:
                subdir_path = self.audio_dir / subdir
                if not subdir_path.exists():
                    continue
                    
                print(f"  Processing {subdir}...")
                
                # Find all audio files in subdirectory
                audio_files = []
                for ext in ['.wav', '.mp3', '.flac']:
                    audio_files.extend(subdir_path.rglob(f'*{ext}'))
                
                for audio_path in audio_files:
                    try:
                        filename = audio_path.name
                        relative_path = str(audio_path.relative_to(self.audio_dir))
                        caption = filename.replace('.wav', '').replace('.mp3', '').replace('.flac', '')
                        
                        metadata = {
                            'split': 'train',
                            'original_filename': filename,
                            'subdirectory': subdir,
                            'full_path': relative_path
                        }
                        
                        matched.append((audio_path, caption, metadata))
                        processed_count += 1
                        
                        if processed_count >= 1000:
                            print(f"  Processed {processed_count} files (limited for testing)")
                            break
                            
                        if processed_count % 100 == 0:
                            print(f"    Processed {processed_count} files...")
                            
                    except Exception as e:
                        print(f"    Error processing {audio_path}: {e}")
                        continue
                
                if processed_count >= 1000:
                    break
                            
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
    
    args = parser.parse_args()
    
    # Create processor
    processor = AudioDataFullProcessor(
        audio_dir=args.audio_dir,
        metadata_path=args.metadata,
        output_dir=args.output_dir)
    
    # Process dataset
    processor.process_dataset(samples_per_tar=args.samples_per_tar)


if __name__ == "__main__":
    main()