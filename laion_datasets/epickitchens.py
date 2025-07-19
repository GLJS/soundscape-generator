#!/usr/bin/env python3
"""
Convert epic-kitchens dataset to WebDataset tar format.

Audio location: /scratch-shared/gwijngaard/laion/epic-kitchens/
Note: This dataset uses HDF5 format
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import h5py
from pathlib import Path
from utils import DatasetProcessor
from typing import List, Tuple, Dict
import argparse
import numpy as np
from dotenv import load_dotenv

load_dotenv()


class EpicKitchensProcessor(DatasetProcessor):
    """Processor for epic-kitchens dataset."""
    
    def load_metadata(self) -> pd.DataFrame:
        """Epic-kitchens metadata might be in HDF5 files."""
        print(f"Checking for metadata in {self.audio_dir}")
        
        # Look for CSV or metadata files
        csv_files = list(self.audio_dir.glob("*.csv"))
        if csv_files:
            dfs = []
            for csv_file in csv_files:
                df = pd.read_csv(csv_file)
                dfs.append(df)
                print(f"  Loaded {len(df)} entries from {csv_file.name}")
            return pd.concat(dfs, ignore_index=True)
        else:
            print("No metadata CSV files found")
            return pd.DataFrame()
        
    def match_audio_to_text(self, metadata_df: pd.DataFrame) -> List[Tuple[bytes, str, Dict]]:
        """Process HDF5 audio files."""
        matched = []
        
        # Find HDF5 files
        h5_files = list(self.audio_dir.rglob("*.hdf5")) + list(self.audio_dir.rglob("*.h5"))
        
        print(f"Found {len(h5_files)} HDF5 files")
        
        for h5_path in h5_files:
            try:
                with h5py.File(h5_path, 'r') as f:
                    # Explore structure
                    print(f"\nProcessing {h5_path.name}")
                    print(f"  Keys: {list(f.keys())}")
                    
                    # Common patterns for audio data in HDF5
                    audio_keys = ['audio', 'waveform', 'data', 'samples']
                    caption_keys = ['caption', 'text', 'label', 'narration', 'action']
                    
                    audio_data = None
                    caption = None
                    
                    # Find audio data
                    for key in audio_keys:
                        if key in f:
                            audio_data = f[key][:]
                            break
                            
                    # Find caption
                    for key in caption_keys:
                        if key in f:
                            caption_data = f[key]
                            if isinstance(caption_data, h5py.Dataset):
                                if caption_data.dtype.kind == 'S':  # String type
                                    caption = caption_data[()].decode('utf-8')
                                else:
                                    caption = str(caption_data[()])
                            break
                            
                    if audio_data is not None:
                        # Convert numpy array to bytes (assuming float32 audio)
                        if isinstance(audio_data, np.ndarray):
                            # Normalize if needed
                            if audio_data.dtype == np.float32 or audio_data.dtype == np.float64:
                                audio_bytes = (audio_data * 32767).astype(np.int16).tobytes()
                            else:
                                audio_bytes = audio_data.tobytes()
                                
                            if caption is None:
                                caption = h5_path.stem
                                
                            metadata = {
                                'split': 'train',
                                'original_filename': h5_path.name,
                                'format': 'hdf5'
                            }
                            
                            matched.append((audio_bytes, caption, metadata))
                            
            except Exception as e:
                print(f"  Error processing {h5_path.name}: {e}")
                
        # Also check for regular audio files
        for ext in ['.wav', '.mp3', '.flac']:
            for audio_file in self.audio_dir.rglob(f"*{ext}"):
                caption = audio_file.stem.replace('_', ' ')
                metadata = {
                    'split': 'train',
                    'original_filename': audio_file.name
                }
                matched.append((audio_file, caption, metadata))
                
        print(f"\nTotal matched: {len(matched)} audio-text pairs")
        return matched


def main():
    parser = argparse.ArgumentParser(description="Convert epic-kitchens dataset to tar format")
    parser.add_argument("--audio-dir", type=str, 
                       default="/scratch-shared/gwijngaard/laion/epic-kitchens",
                       help="Path to epic-kitchens directory")
    parser.add_argument("--metadata", type=str,
                       default="/scratch-shared/gwijngaard/laion/epic-kitchens",
                       help="Path to metadata (same as audio-dir)")
    parser.add_argument("--output-dir", type=str,
                       default="/scratch-shared/gwijngaard/tar/epickitchens",
                       help="Output directory for tar files")
    parser.add_argument("--samples-per-tar", type=int, default=2048,
                       help="Number of samples per tar file")
    
    args = parser.parse_args()
    
    # Create processor
    processor = EpicKitchensProcessor(
        audio_dir=args.audio_dir,
        metadata_path=args.metadata,
        output_dir=args.output_dir,
        task="AAC")
    
    # Process dataset
    processor.process_dataset(samples_per_tar=args.samples_per_tar)


if __name__ == "__main__":
    main()