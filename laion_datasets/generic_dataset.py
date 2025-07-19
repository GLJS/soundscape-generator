#!/usr/bin/env python3
"""
Generic dataset converter for WebDataset tar format.
Use this for datasets without specific handling requirements.

Usage:
python generic_dataset.py --audio-dir /path/to/audio --output-dir /path/to/output --dataset-name mydataset
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from pathlib import Path
from utils import DatasetProcessor, find_audio_files
from typing import List, Tuple, Dict
import argparse
import tarfile
import zipfile
from dotenv import load_dotenv

load_dotenv()


class GenericDatasetProcessor(DatasetProcessor):
    """Generic processor for datasets without specific requirements."""
    
    def __init__(self, audio_dir: str, metadata_path: str, output_dir: str, dataset_name: str = "dataset"):
        super().__init__(audio_dir, metadata_path, output_dir)
        self.dataset_name = dataset_name
        
    def load_metadata(self) -> pd.DataFrame:
        """Try to load any metadata files found."""
        print(f"Looking for metadata in {self.metadata_path}")
        
        # Try CSV files
        csv_files = list(Path(self.metadata_path).glob("*.csv"))
        if csv_files:
            print(f"Found {len(csv_files)} CSV files")
            return pd.read_csv(csv_files[0])
            
        # Try JSON files
        json_files = list(Path(self.metadata_path).glob("*.json"))
        if json_files:
            print(f"Found {len(json_files)} JSON files")
            return pd.read_json(json_files[0])
            
        print("No metadata files found")
        return pd.DataFrame()
        
    def match_audio_to_text(self, metadata_df: pd.DataFrame) -> List[Tuple[Path, str, Dict]]:
        """Match audio files, handling both archives and directories."""
        matched = []
        
        # Check for archives
        archives = list(self.audio_dir.glob("*.tar.gz")) + \
                  list(self.audio_dir.glob("*.tar")) + \
                  list(self.audio_dir.glob("*.zip"))
                  
        if archives:
            print(f"Found {len(archives)} archives")
            for archive in archives:
                matched.extend(self._process_archive(archive, metadata_df))
        else:
            # Look for audio files directly
            audio_files = find_audio_files(self.audio_dir)
            print(f"Found {len(audio_files)} audio files")
            
            for audio_file in audio_files:
                caption = audio_file.stem.replace('_', ' ').replace('-', ' ')
                
                # Try to match with metadata if available
                if len(metadata_df) > 0 and 'file_name' in metadata_df.columns:
                    meta_match = metadata_df[metadata_df['file_name'] == audio_file.name]
                    if not meta_match.empty:
                        caption = meta_match.iloc[0].get('caption', meta_match.iloc[0].get('text', caption))
                        
                metadata = {
                    'split': 'train',
                    'original_filename': audio_file.name,
                    'dataset': self.dataset_name
                }
                
                matched.append((audio_file, caption, metadata))
                
        print(f"Total matched: {len(matched)} audio-text pairs")
        return matched
        
    def _process_archive(self, archive_path: Path, metadata_df: pd.DataFrame) -> List[Tuple[bytes, str, Dict]]:
        """Process files from an archive."""
        matched = []
        
        if archive_path.suffix == '.zip':
            with zipfile.ZipFile(archive_path, 'r') as zf:
                for name in zf.namelist():
                    if any(name.endswith(ext) for ext in ['.wav', '.mp3', '.flac']):
                        audio_bytes = zf.read(name)
                        filename = os.path.basename(name)
                        caption = filename.rsplit('.', 1)[0].replace('_', ' ')
                        
                        metadata = {
                            'split': 'train',
                            'original_filename': filename,
                            'dataset': self.dataset_name,
                            'archive': archive_path.name
                        }
                        
                        matched.append((audio_bytes, caption, metadata))
                        
        elif archive_path.suffix in ['.gz', '.tar']:
            mode = 'r:gz' if archive_path.suffix == '.gz' else 'r'
            with tarfile.open(archive_path, mode) as tf:
                for member in tf.getmembers():
                    if member.isfile() and any(member.name.endswith(ext) for ext in ['.wav', '.mp3', '.flac']):
                        f = tf.extractfile(member)
                        if f:
                            audio_bytes = f.read()
                            filename = os.path.basename(member.name)
                            caption = filename.rsplit('.', 1)[0].replace('_', ' ')
                            
                            metadata = {
                                'split': 'train',
                                'original_filename': filename,
                                'dataset': self.dataset_name,
                                'archive': archive_path.name
                            }
                            
                            matched.append((audio_bytes, caption, metadata))
                            
        return matched


def main():
    parser = argparse.ArgumentParser(description="Generic dataset converter to tar format")
    parser.add_argument("--audio-dir", type=str, required=True,
                       help="Path to audio files or archives")
    parser.add_argument("--metadata", type=str,
                       help="Path to metadata (defaults to audio-dir)")
    parser.add_argument("--output-dir", type=str, required=True,
                       help="Output directory for tar files")
    parser.add_argument("--dataset-name", type=str, default="dataset",
                       help="Name of the dataset")
    parser.add_argument("--samples-per-tar", type=int, default=256,
                       help="Number of samples per tar file")
    
    args = parser.parse_args()
    
    if not args.metadata:
        args.metadata = args.audio_dir
    
    # Create processor
    processor = GenericDatasetProcessor(
        audio_dir=args.audio_dir,
        metadata_path=args.metadata,
        output_dir=args.output_dir,
        dataset_name=args.dataset_name
    ,
        task="GENERIC")
    
    # Process dataset
    processor.process_dataset(samples_per_tar=args.samples_per_tar)


if __name__ == "__main__":
    main()