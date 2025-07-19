#!/usr/bin/env python3
"""
Convert TUT Acoustic Scenes 2017 dataset to WebDataset tar format.

Audio location: /scratch-shared/gwijngaard/laion/TUT2017/
- Development: TUT-acoustic-scenes-2017-development.audio.*.zip (10 files)
- Evaluation: TUT-acoustic-scenes-2017-evaluation.audio.*.zip (4 files)
- Or extracted directories if available
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from pathlib import Path
from utils import DatasetProcessor, AudioProcessor, TarCreator
from typing import List, Tuple, Dict, Optional
import argparse
import zipfile
import io
from dotenv import load_dotenv

load_dotenv()


class TUT2017Processor(DatasetProcessor):
    """Processor for TUT Acoustic Scenes 2017 dataset."""
    
    def __init__(self, audio_dir: str, output_dir: str):
        super().__init__(audio_dir, audio_dir, output_dir)
        
        # Check for extracted directories first
        self.dev_dir = self.audio_dir / "TUT-acoustic-scenes-2017-development"
        self.eval_dir = self.audio_dir / "TUT-acoustic-scenes-2017-evaluation"
        
        # ZIP file patterns
        self.dev_zips = sorted(self.audio_dir.glob("TUT-acoustic-scenes-2017-development.audio.*.zip"))
        self.eval_zips = sorted(self.audio_dir.glob("TUT-acoustic-scenes-2017-evaluation.audio.*.zip"))
        
        # Determine if we're using extracted directories or ZIP files
        self.use_extracted = self.dev_dir.exists() and (self.dev_dir / "audio").exists()
        
        if self.use_extracted:
            print("Using extracted directories")
        else:
            print(f"Using ZIP archives: {len(self.dev_zips)} development, {len(self.eval_zips)} evaluation")
            
    def load_metadata(self, split: str = 'development') -> pd.DataFrame:
        """Load TUT2017 metadata from meta.txt files."""
        # Try to find metadata in extracted directory first
        if split == 'development':
            if self.use_extracted:
                meta_path = self.dev_dir / "meta.txt"
                error_path = self.dev_dir / "error.txt"
            else:
                # Look for doc zip
                doc_zip = self.audio_dir / "TUT-acoustic-scenes-2017-development.doc.zip"
                if doc_zip.exists():
                    return self.load_metadata_from_zip(doc_zip, split)
                else:
                    raise FileNotFoundError(f"Cannot find metadata for {split}")
        else:
            if self.use_extracted:
                meta_path = self.eval_dir / "meta.txt"
                error_path = self.eval_dir / "error.txt"
            else:
                # Look for doc zip
                doc_zip = self.audio_dir / "TUT-acoustic-scenes-2017-evaluation.doc.zip"
                if doc_zip.exists():
                    return self.load_metadata_from_zip(doc_zip, split)
                else:
                    raise FileNotFoundError(f"Cannot find metadata for {split}")
        
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
        
    def load_metadata_from_zip(self, doc_zip_path: Path, split: str) -> pd.DataFrame:
        """Load metadata from doc zip file."""
        print(f"Loading metadata from {doc_zip_path}")
        
        with zipfile.ZipFile(doc_zip_path, 'r') as zf:
            # Find meta.txt
            meta_content = None
            error_content = None
            
            for name in zf.namelist():
                if name.endswith('meta.txt'):
                    meta_content = zf.read(name).decode('utf-8')
                elif name.endswith('error.txt'):
                    error_content = zf.read(name).decode('utf-8')
                    
        if not meta_content:
            raise ValueError(f"No meta.txt found in {doc_zip_path}")
            
        # Parse metadata
        lines = meta_content.strip().split('\n')
        data = []
        for line in lines:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                data.append({'file_path': parts[0], 'scene': parts[1]})
                
        df = pd.DataFrame(data)
        
        # Parse errors if available
        errors = {}
        if error_content:
            for line in error_content.strip().split('\n'):
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
        return df
        
    def create_caption(self, scene: str, has_error: bool = False) -> str:
        """Create a natural language caption from scene label."""
        # Clean up scene name
        scene_readable = scene.replace('_', ' ')
        
        # Create base caption
        caption = f"Acoustic scene of {scene_readable}"
        
        # Scene descriptions (same as TUT2016)
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
        
    def get_audio_from_zip(self, audio_filename: str, zip_files: List[Path]) -> Optional[bytes]:
        """Extract audio from ZIP files."""
        # Try each ZIP file
        for zip_path in zip_files:
            try:
                with zipfile.ZipFile(zip_path, 'r') as zf:
                    # Try to find the audio file
                    for name in zf.namelist():
                        if name.endswith(audio_filename) or os.path.basename(name) == audio_filename:
                            return zf.read(name)
            except Exception as e:
                continue
                
        return None
        
    def match_audio_to_text(self, metadata_df: pd.DataFrame, split: str) -> List[Tuple[bytes, str, Dict]]:
        """Match audio files to their captions."""
        matched = []
        missing_count = 0
        
        # Determine which ZIP files to use
        zip_files = self.dev_zips if split == 'development' else self.eval_zips
        base_dir = self.dev_dir if split == 'development' else self.eval_dir
        
        for _, row in metadata_df.iterrows():
            audio_filename = os.path.basename(row['file_path'])
            
            # Try to get audio either from extracted directory or ZIP
            audio_bytes = None
            
            if self.use_extracted:
                # Try extracted directory
                audio_path = base_dir / row['file_path']
                if audio_path.exists():
                    try:
                        with open(audio_path, 'rb') as f:
                            audio_bytes = f.read()
                    except Exception as e:
                        print(f"Failed to read {audio_path}: {e}")
            else:
                # Try ZIP files
                audio_bytes = self.get_audio_from_zip(audio_filename, zip_files)
                
            if audio_bytes:
                # Create caption
                caption = self.create_caption(row['scene'], row['has_error'])
                
                # Create metadata
                metadata = {
                    'split': row['split'],
                    'original_filename': audio_filename,
                    'scene': row['scene'],
                    'has_error': row['has_error'],
                    'segment_duration': 10,  # TUT2017 uses 10-second segments
                    'task': 'ASC'
                }
                
                # Add error info if present
                if row['has_error'] and row['error_info']:
                    metadata['error_type'] = row['error_info']['error_type']
                    metadata['error_description'] = row['error_info']['error_description']
                
                matched.append((audio_bytes, caption, metadata))
            else:
                missing_count += 1
                if missing_count <= 10:  # Only print first 10
                    print(f"Missing audio file: {audio_filename}")
                    
        print(f"Matched {len(matched)} audio-text pairs for {split}")
        if missing_count > 0:
            print(f"Missing audio files: {missing_count}")
        
        return matched
        
    # NOTE: This file needs a custom process_dataset due to complex ZIP handling
    def process_dataset(self, samples_per_tar: int = 256):
        """Process the entire dataset into tar files."""
        all_matched = []
        
        # Process development set
        print("\nProcessing development set...")
        try:
            dev_metadata = self.load_metadata('development')
            dev_matched = self.match_audio_to_text(dev_metadata, 'development')
            all_matched.extend(dev_matched)
        except Exception as e:
            print(f"Failed to process development set: {e}")
        
        # Process evaluation set
        print("\nProcessing evaluation set...")
        try:
            eval_metadata = self.load_metadata('evaluation')
            eval_matched = self.match_audio_to_text(eval_metadata, 'evaluation')
            all_matched.extend(eval_matched)
        except Exception as e:
            print(f"Failed to process evaluation set: {e}")
        
        if not all_matched:
            print("No audio files matched. Exiting.")
            return []
        
        # Create tar files
        tar_creator = TarCreator(self.output_dir, prefix="tut2017", 
                                 samples_per_tar=samples_per_tar)
        
        # Process in batches
        all_summaries = []
        for i in range(0, len(all_matched), samples_per_tar):
            batch = all_matched[i:i+samples_per_tar]
            samples = []
            
            for audio_bytes, text, metadata in batch:
                try:
                    # Process audio
                    processed_audio, audio_metadata = self.audio_processor.process_audio_file(audio_bytes)
                    samples.append({
                        'audio_bytes': processed_audio,
                        'text': text,
                        'metadata': {**metadata, **audio_metadata, 'task': 'ASC'}
                    })
                except Exception as e:
                    print(f"Failed to process audio: {e}")
                    
            if samples:
                summary = tar_creator.create_tar_from_samples(samples, i // samples_per_tar)
                all_summaries.append(summary)
                
        # Create size file
        tar_creator.create_size_file(all_summaries)
        
        # Summary
        total_successful = sum(s['successful'] for s in all_summaries)
        total_failed = sum(s['failed'] for s in all_summaries)
        
        print(f"\nProcessing complete!")
        print(f"Total samples: {len(all_matched)}")
        print(f"Successfully processed: {total_successful}")
        print(f"Failed: {total_failed}")
        print(f"Created {len(all_summaries)} tar files in {self.output_dir}")
        
        return all_summaries


def main():
    parser = argparse.ArgumentParser(description="Convert TUT Acoustic Scenes 2017 dataset to tar format")
    parser.add_argument("--audio-dir", type=str, 
                       default="/scratch-shared/gwijngaard/laion/TUT2017",
                       help="Path to TUT2017 root directory")
    parser.add_argument("--output-dir", type=str,
                       default="/scratch-shared/gwijngaard/tar/tut2017",
                       help="Output directory for tar files")
    parser.add_argument("--samples-per-tar", type=int, default=256,
                       help="Number of samples per tar file")
    
    args = parser.parse_args()
    
    # Create processor
    processor = TUT2017Processor(
        audio_dir=args.audio_dir,
        output_dir=args.output_dir
    )
    
    # Process dataset
    processor.process_dataset(samples_per_tar=args.samples_per_tar)


if __name__ == "__main__":
    main()