#!/usr/bin/env python3
"""
Convert TUT Acoustic Scenes 2016 dataset to WebDataset tar format.

Audio location: /scratch-shared/gwijngaard/laion/TUT2016/
- TUT-acoustic-scenes-2016-development/
- TUT-acoustic-scenes-2016-evaluation/
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from pathlib import Path
from utils import DatasetProcessor, AudioProcessor, TarCreator
from typing import List, Tuple, Dict
import argparse
from dotenv import load_dotenv

load_dotenv()


class TUT2016Processor(DatasetProcessor):
    """Processor for TUT Acoustic Scenes 2016 dataset."""
    
    def __init__(self, audio_dir: str, output_dir: str):
        super().__init__(audio_dir, audio_dir, output_dir)
        self.dev_dir = self.audio_dir / "TUT-acoustic-scenes-2016-development"
        self.eval_dir = self.audio_dir / "TUT-acoustic-scenes-2016-evaluation"
        
    def load_metadata(self, split: str = 'development') -> pd.DataFrame:
        """Load TUT2016 metadata from meta.txt files."""
        if split == 'development':
            meta_path = self.dev_dir / "meta.txt"
            error_path = self.dev_dir / "error.txt"
        else:
            meta_path = self.eval_dir / "meta.txt"
            error_path = self.eval_dir / "error.txt"
            
        print(f"Loading metadata from {meta_path}")
        
        # Load main metadata
        df = pd.read_csv(meta_path, sep='\t', header=None, names=['file_path', 'scene'])
        
        # Load error annotations if they exist
        errors = {}
        if error_path.exists():
            with open(error_path, 'r') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 3:
                        file_path = parts[0]
                        error_type = parts[1]
                        description = parts[2] if len(parts) > 2 else ""
                        errors[file_path] = {
                            'error_type': error_type,
                            'error_description': description
                        }
        
        # Add error info to dataframe
        df['has_error'] = df['file_path'].isin(errors)
        df['error_info'] = df['file_path'].map(errors)
        
        # Add split info
        df['split'] = 'train' if split == 'development' else 'test'
        
        print(f"Loaded {len(df)} entries for {split} set")
        if errors:
            print(f"Found {len(errors)} files with errors")
            
        return df
        
    def create_caption(self, scene: str, has_error: bool = False) -> str:
        """Create a natural language caption from scene label."""
        # Clean up scene name
        scene_readable = scene.replace('_', ' ')
        
        # Create base caption
        caption = f"Acoustic scene of {scene_readable}"
        
        # Add context based on scene type
        scene_descriptions = {
            'bus': 'inside a city bus during travel',
            'cafe/restaurant': 'in a small cafe or restaurant with ambient sounds',
            'car': 'inside a car driving through the city',
            'city_center': 'in an urban city center with traffic and people',
            'forest_path': 'on a quiet forest path with natural sounds',
            'grocery_store': 'inside a medium-sized grocery store',
            'home': 'in a residential home environment',
            'lakeside_beach': 'at a lakeside beach with water sounds',
            'library': 'in a quiet library environment',
            'metro_station': 'in an underground metro station',
            'office': 'in a busy office with multiple people working',
            'residential_area': 'in a residential neighborhood area',
            'train': 'inside a train during travel',
            'tram': 'inside a city tram during travel',
            'urban_park': 'in an urban park with city and nature sounds'
        }
        
        if scene in scene_descriptions:
            caption = f"Acoustic scene recorded {scene_descriptions[scene]}"
            
        return caption
        
        
    def load_metadata(self) -> pd.DataFrame:
        """Override to load both development and evaluation sets."""
        dfs = []
        
        # Load development set
        dev_df = self.load_metadata_split('development')
        if not dev_df.empty:
            dfs.append(dev_df)
            
        # Load evaluation set  
        eval_df = self.load_metadata_split('evaluation')
        if not eval_df.empty:
            dfs.append(eval_df)
            
        if dfs:
            return pd.concat(dfs, ignore_index=True)
        return pd.DataFrame()
        
    def load_metadata_split(self, split: str) -> pd.DataFrame:
        """Load metadata for a specific split - original load_metadata logic."""
        if split == 'development':
            meta_path = self.dev_dir / "meta.txt"
            error_path = self.dev_dir / "error.txt"
        else:
            meta_path = self.eval_dir / "meta.txt"
            error_path = self.eval_dir / "error.txt"
            
        print(f"Loading metadata from {meta_path}")
        
        # Load main metadata
        df = pd.read_csv(meta_path, sep='\t', header=None, names=['file_path', 'scene'])
        
        # Load error annotations if they exist
        errors = {}
        if error_path.exists():
            with open(error_path, 'r') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 3:
                        file_path = parts[0]
                        error_type = parts[1]
                        description = parts[2] if len(parts) > 2 else ""
                        errors[file_path] = {
                            'error_type': error_type,
                            'error_description': description
                        }
        
        # Add error info to dataframe
        df['has_error'] = df['file_path'].isin(errors)
        df['error_info'] = df['file_path'].map(errors)
        
        # Add split info
        df['split'] = 'train' if split == 'development' else 'test'
        
        print(f"Loaded {len(df)} entries for {split} set")
        if errors:
            print(f"  Found {len(errors)} files with errors")
        
        return df
        
    def match_audio_to_text(self, metadata_df: pd.DataFrame) -> List[Tuple[Path, str, Dict]]:
        """Override to handle both dev and eval directories."""
        matched = []
        
        for _, row in metadata_df.iterrows():
            # Determine which directory based on split
            if row['split'] == 'train':
                base_dir = self.dev_dir
            else:
                base_dir = self.eval_dir
                
            # Construct full audio path
            audio_path = base_dir / row['file_path']
            
            if audio_path.exists():
                # Create caption
                caption = self.create_caption(row['scene'], row['has_error'])
                
                # Create metadata
                metadata = {
                    'split': row['split'],
                    'original_filename': os.path.basename(row['file_path']),
                    'scene': row['scene'],
                    'has_error': row['has_error'],
                    'task': 'ASC'
                }
                
                # Add error info if present
                if row['has_error'] and row['error_info']:
                    metadata['error_type'] = row['error_info']['error_type']
                    metadata['error_description'] = row['error_info']['error_description']
                
                matched.append((audio_path, caption, metadata))
            else:
                print(f"Missing audio file: {audio_path}")
                
        print(f"Matched {len(matched)} audio-text pairs")
        return matched


def main():
    parser = argparse.ArgumentParser(description="Convert TUT Acoustic Scenes 2016 dataset to tar format")
    parser.add_argument("--audio-dir", type=str, 
                       default="/scratch-shared/gwijngaard/laion/TUT2016",
                       help="Path to TUT2016 root directory")
    parser.add_argument("--output-dir", type=str,
                       default="/scratch-shared/gwijngaard/tar/tut2016",
                       help="Output directory for tar files")
    parser.add_argument("--samples-per-tar", type=int, default=256,
                       help="Number of samples per tar file")
    
    args = parser.parse_args()
    
    # Create processor
    processor = TUT2016Processor(
        audio_dir=args.audio_dir,
        output_dir=args.output_dir
    )
    
    # Process dataset
    processor.process_dataset(samples_per_tar=args.samples_per_tar)


if __name__ == "__main__":
    main()