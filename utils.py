"""
Common utilities for processing LAION audio datasets into tar files.
"""

import os
import json
import tarfile
import zipfile
import librosa
import soundfile as sf
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from tqdm import tqdm
import io
import tempfile
from dotenv import load_dotenv

load_dotenv()


class ProcessingError(Exception):
    """Custom exception for processing errors."""
    pass


class AudioProcessor:
    """Handles audio loading, conversion and processing."""
    
    def __init__(self, target_sr: int = 48000, output_format: str = "flac", bit_depth: int = 16):
        self.target_sr = target_sr
        self.output_format = output_format
        self.bit_depth = bit_depth
        
    def process_audio_file(self, audio_path: Union[str, bytes, io.BytesIO]) -> Tuple[bytes, Dict]:
        """
        Process an audio file and return bytes in target format.
        
        Args:
            audio_path: Path to audio file or audio bytes
            
        Returns:
            Tuple of (audio_bytes, metadata_dict)
        """
        try:
            # Load audio
            if isinstance(audio_path, (bytes, io.BytesIO)):
                if isinstance(audio_path, bytes):
                    audio_path = io.BytesIO(audio_path)
                audio, sr = librosa.load(audio_path, sr=None, mono=False)
            else:
                audio, sr = librosa.load(str(audio_path), sr=None, mono=False)
            
            # Resample if needed
            if sr != self.target_sr:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=self.target_sr)
                sr = self.target_sr
                
            # Convert to mono if stereo
            if len(audio.shape) > 1:
                audio = librosa.to_mono(audio)
                
            # Save to bytes with specified bit depth
            output_buffer = io.BytesIO()
            sf.write(output_buffer, audio, sr, format=self.output_format.upper(), 
                    subtype='PCM_16' if self.bit_depth == 16 else 'PCM_24')
            output_buffer.seek(0)
            
            metadata = {
                "sample_rate": sr,
                "duration": len(audio) / sr,
                "channels": 1,
                "format": self.output_format
            }
            
            return output_buffer.read(), metadata
            
        except Exception as e:
            raise ProcessingError(f"Failed to process audio: {e}")


class TarCreator:
    """Creates WebDataset-compatible tar files."""
    
    def __init__(self, output_dir: str, prefix: str = "data", samples_per_tar: int = 2048, split: Optional[str] = None):
        self.output_dir = Path(output_dir)
        self.prefix = prefix
        self.samples_per_tar = samples_per_tar
        self.split = split
        
        # If split is provided, create subdirectory
        if self.split:
            self.output_dir = self.output_dir / self.split
            
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def create_tar_from_samples(self, samples: List[Dict], tar_index: int) -> Dict:
        """
        Create a tar file from a list of samples.
        
        Args:
            samples: List of dicts with 'audio_bytes', 'text', and optionally 'metadata'
            tar_index: Index for tar file naming
            
        Returns:
            Summary dict with statistics
        """
        tar_path = self.output_dir / f"{tar_index:04d}.tar"
        successful = 0
        failed = 0
        
        with tarfile.open(tar_path, "w") as tar:
            for idx, sample in enumerate(tqdm(samples, desc=f"Creating {tar_path.name}")):
                try:
                    # Create key for this sample
                    key = f"{tar_index:06d}_{idx:06d}"
                    
                    # Add audio file
                    audio_info = tarfile.TarInfo(name=f"{key}.flac")
                    audio_info.size = len(sample['audio_bytes'])
                    tar.addfile(audio_info, io.BytesIO(sample['audio_bytes']))
                    
                    # Create JSON with text and metadata
                    json_data = {
                        "text": sample.get('caption', sample.get('text', '')),
                    }
                    
                    if 'metadata' in sample:
                        json_data.update(sample['metadata'])
                        
                    json_bytes = json.dumps(json_data, ensure_ascii=False).encode('utf-8')
                    
                    # Add JSON file
                    json_info = tarfile.TarInfo(name=f"{key}.json")
                    json_info.size = len(json_bytes)
                    tar.addfile(json_info, io.BytesIO(json_bytes))
                    
                    successful += 1
                    
                except Exception as e:
                    print(f"Failed to process sample {idx}: {e}")
                    failed += 1
                    
        return {
            "tar_file": tar_path.name,
            "successful": successful,
            "failed": failed,
            "total": len(samples)
        }
        
    def create_size_file(self, tar_summaries: List[Dict]):
        """Create sizes.json file for WebDataset."""
        sizes = {s['tar_file']: s['successful'] for s in tar_summaries}
        
        with open(self.output_dir / "sizes.json", "w") as f:
            json.dump(sizes, f, indent=2)
            


class ArchiveExtractor:
    """Extracts files from various archive formats."""
    
    @staticmethod
    def extract_from_tar_gz(archive_path: Path, file_pattern: Optional[str] = None) -> Dict[str, bytes]:
        """Extract files from tar.gz archive."""
        files = {}
        
        with tarfile.open(archive_path, "r:gz") as tar:
            for member in tar.getmembers():
                if member.isfile():
                    if file_pattern is None or file_pattern in member.name:
                        f = tar.extractfile(member)
                        if f:
                            files[member.name] = f.read()
                            
        return files
        
    @staticmethod
    def extract_from_tar_bz2(archive_path: Path, file_pattern: Optional[str] = None) -> Dict[str, bytes]:
        """Extract files from tar.bz2 archive."""
        files = {}
        
        with tarfile.open(archive_path, "r:bz2") as tar:
            for member in tar.getmembers():
                if member.isfile():
                    if file_pattern is None or file_pattern in member.name:
                        f = tar.extractfile(member)
                        if f:
                            files[member.name] = f.read()
                            
        return files
        
    @staticmethod
    def extract_from_zip(archive_path: Path, file_pattern: Optional[str] = None) -> Dict[str, bytes]:
        """Extract files from zip archive."""
        files = {}
        
        with zipfile.ZipFile(archive_path, "r") as zip_file:
            for name in zip_file.namelist():
                if file_pattern is None or file_pattern in name:
                    files[name] = zip_file.read(name)
                    
        return files


class DatasetProcessor:
    """Base class for dataset-specific processors."""
    
    def __init__(self, audio_dir: str, metadata_path: Optional[str], output_dir: str, task: str = None):
        self.audio_dir = Path(audio_dir)
        self.metadata_path = Path(metadata_path) if metadata_path else None
        self.output_dir = Path(output_dir)
        self.audio_processor = AudioProcessor()
        self.task = task
        
    def load_metadata(self) -> pd.DataFrame:
        """Load metadata/captions for the dataset."""
        raise NotImplementedError("Subclasses must implement load_metadata")
        
    def match_audio_to_text(self, metadata_df: pd.DataFrame) -> List[Tuple[Path, str, Dict]]:
        """Match audio files to their text captions."""
        raise NotImplementedError("Subclasses must implement match_audio_to_text")
        
    def process_dataset(self, samples_per_tar: int = 2048):
        """Process the entire dataset into tar files."""
        # Load metadata
        metadata_df = self.load_metadata()
        
        # Match audio to text
        matched_samples = self.match_audio_to_text(metadata_df)
        
        # Validate splits
        valid_splits = {'train', 'test', 'valid', 'val'}
        missing_split_count = 0
        invalid_split_values = set()
        
        # Group samples by split
        samples_by_split = {}
        for audio_path, text, metadata in matched_samples:
            split = metadata.get('split')
            
            # Default to 'train' if split is missing
            if split is None:
                missing_split_count += 1
                split = 'train'
                metadata['split'] = 'train'
                if missing_split_count == 1:
                    print(f"WARNING: No split information found, defaulting all samples to 'train'")
            
            # Normalize 'val' to 'valid'
            if split == 'val':
                split = 'valid'
                metadata['split'] = 'valid'
            
            # Check if split is valid
            if split not in valid_splits:
                invalid_split_values.add(split)
                print(f"WARNING: Sample {audio_path} has invalid split '{split}', defaulting to 'train'")
                split = 'train'
                metadata['split'] = 'train'
            
            if split not in samples_by_split:
                samples_by_split[split] = []
            samples_by_split[split].append((audio_path, text, metadata))
        
        # Process each split separately
        all_summaries = []
        for split, split_samples in samples_by_split.items():
            print(f"\nProcessing {split} split with {len(split_samples)} samples...")
            
            # Create tar files with split subdirectory
            tar_creator = TarCreator(self.output_dir, prefix=self.__class__.__name__.lower().replace('processor', ''), 
                                     samples_per_tar=samples_per_tar, split=split)
            
            # Process in batches
            for i in range(0, len(split_samples), samples_per_tar):
                batch = split_samples[i:i+samples_per_tar]
                samples = []
                
                for audio_path, text, metadata in batch:
                    try:
                        audio_bytes, audio_metadata = self.audio_processor.process_audio_file(audio_path)
                        sample_metadata = {**metadata, **audio_metadata}
                        if self.task:
                            sample_metadata['task'] = self.task
                        samples.append({
                            'audio_bytes': audio_bytes,
                            'text': text,
                            'metadata': sample_metadata
                        })
                    except Exception as e:
                        print(f"Failed to process {audio_path}: {e}")
                        
                if samples:
                    summary = tar_creator.create_tar_from_samples(samples, i // samples_per_tar)
                    all_summaries.append(summary)
                    
            # Create size file for this split
            tar_creator.create_size_file([s for s in all_summaries if split in s.get('tar_file', '')])
        
        # Summary
        total_successful = sum(s['successful'] for s in all_summaries)
        total_failed = sum(s['failed'] for s in all_summaries)
        
        print(f"\nProcessing complete!")
        print(f"Total samples: {len(matched_samples)}")
        print(f"Successfully processed: {total_successful}")
        print(f"Failed: {total_failed}")
        print(f"Created {len(all_summaries)} tar files in {self.output_dir}")
        
        return all_summaries


def find_audio_files(directory: Path, extensions: List[str] = None) -> List[Path]:
    """
    Recursively find all audio files in a directory.
    
    Args:
        directory: Directory to search
        extensions: List of file extensions to look for (default: common audio formats)
        
    Returns:
        List of Path objects for found audio files
    """
    if extensions is None:
        extensions = ['.wav', '.mp3', '.flac', '.m4a', '.ogg', '.opus']
        
    audio_files = []
    for ext in extensions:
        audio_files.extend(directory.rglob(f"*{ext}"))
        
    return sorted(audio_files)


def create_dataset_processor(dataset_name: str, audio_dir: str, 
                           metadata_path: Optional[str], output_dir: str) -> DatasetProcessor:
    """
    Factory function to create appropriate processor for a dataset.
    
    Args:
        dataset_name: Name of the dataset
        audio_dir: Path to audio files
        metadata_path: Path to metadata/captions
        output_dir: Output directory for tar files
        
    Returns:
        DatasetProcessor instance
    """
    # Import dataset-specific processors dynamically
    # This will be expanded as we create individual dataset scripts
    processors = {
        # Add mappings as we create dataset-specific scripts
    }
    
    if dataset_name in processors:
        return processors[dataset_name](audio_dir, metadata_path, output_dir)
    else:
        raise ValueError(f"No processor found for dataset: {dataset_name}")