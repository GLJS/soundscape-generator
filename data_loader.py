#!/usr/bin/env python3
"""
Data loader for LAION audio datasets.
Handles loading and preprocessing of audio files from various tar archives.
"""

import os
import json
import tarfile
import logging
import multiprocessing
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import tqdm

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LAIONDataLoader:
    """
    Data loader for LAION audio datasets stored in tar archives.
    Supports multiple dataset types including generated sound events, audiosnippets, etc.
    """
    
    def __init__(self, data_root: str = "/scratch-shared/gwijngaard/laion/", 
                 cache_dir: str = "/scratch-shared/gwijngaard/laion/extracted/cache", 
                 extract_dir: str = "/scratch-shared/gwijngaard/laion/extracted/data"):
        """
        Initialize the data loader.
        
        Args:
            data_root: Root directory containing the LAION datasets
            cache_dir: Directory to cache file indices and metadata
            extract_dir: Directory to extract tar files
        """
        self.data_root = Path(data_root)
        self.cache_dir = Path(cache_dir)
        self.extract_dir = Path(extract_dir)
        
        # Create directories if they don't exist
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.extract_dir.mkdir(parents=True, exist_ok=True)
        
        # Available dataset types
        self.dataset_types = {
            'generated-sound-events': 'generated-sound-events',
            'audiosnippets': 'audiosnippets_long_2_8M',
            'freesound': 'freesound-commercially-permissive-subset-with-captions',
            'audioset': 'audioset-with-grounded-captions',
            'music': 'captioned-ai-music-snippets',
            'wild-events': 'in-the-wild-sound-events'
        }
        
        self.file_indices = {}
        
    def discover_datasets(self) -> Dict[str, List[str]]:
        """
        Discover all available datasets and their tar files.
        
        Returns:
            Dictionary mapping dataset names to lists of tar file paths
        """
        datasets = {}
        
        for dataset_name, dataset_dir in self.dataset_types.items():
            dataset_path = self.data_root / dataset_dir
            if dataset_path.exists():
                tar_files = list(dataset_path.glob("*.tar"))
                if tar_files:
                    datasets[dataset_name] = [str(f) for f in tar_files]
                    logger.info(f"Found {len(tar_files)} tar files for {dataset_name}")
                else:
                    logger.warning(f"No tar files found for {dataset_name}")
            else:
                logger.warning(f"Dataset directory not found: {dataset_path}")
        
        return datasets
    
    def extract_tar_worker(self, tar_path: str) -> bool:
        """
        Worker function to extract a single TAR file.
        
        Args:
            tar_path: Path to the tar file to extract
            
        Returns:
            True if extraction successful, False otherwise
        """
        try:
            with tarfile.open(tar_path, 'r') as archive:
                archive.extractall(path=self.extract_dir)
            logger.info(f"Successfully extracted {tar_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to extract {tar_path}: {e}")
            return False
    
    def extract_dataset(self, dataset_name: str, max_workers: Optional[int] = None) -> bool:
        """
        Extract all tar files for a given dataset.
        
        Args:
            dataset_name: Name of the dataset to extract
            max_workers: Number of parallel extraction workers
            
        Returns:
            True if all extractions successful, False otherwise
        """
        if dataset_name not in self.dataset_types:
            logger.error(f"Unknown dataset: {dataset_name}")
            return False
        
        datasets = self.discover_datasets()
        if dataset_name not in datasets:
            logger.error(f"No tar files found for dataset: {dataset_name}")
            return False
        
        tar_files = datasets[dataset_name]
        
        # Check if already extracted
        index_file = self.cache_dir / f"{dataset_name}_index.json"
        if index_file.exists():
            logger.info(f"Dataset {dataset_name} already extracted and indexed")
            return True
        
        logger.info(f"Extracting {len(tar_files)} tar files for {dataset_name}")
        
        if max_workers is None:
            max_workers = min(len(tar_files), os.cpu_count())
        
        with multiprocessing.Pool(processes=max_workers) as pool:
            results = list(tqdm.tqdm(
                pool.imap(self.extract_tar_worker, tar_files),
                total=len(tar_files),
                desc=f"Extracting {dataset_name}"
            ))
        
        success_count = sum(results)
        logger.info(f"Successfully extracted {success_count}/{len(tar_files)} tar files")
        
        # Create file index after extraction
        self.create_file_index(dataset_name)
        
        return success_count == len(tar_files)
    
    def create_file_index(self, dataset_name: str) -> List[Dict]:
        """
        Create an index of all extracted audio files and their metadata.
        
        Args:
            dataset_name: Name of the dataset to index
            
        Returns:
            List of file information dictionaries
        """
        logger.info(f"Creating file index for {dataset_name}")
        
        # Find all audio files (mp3, wav, etc.)
        audio_extensions = ['.mp3', '.wav', '.flac', '.ogg', '.m4a']
        audio_files = []
        
        for ext in audio_extensions:
            audio_files.extend(list(self.extract_dir.rglob(f"*{ext}")))
        
        logger.info(f"Found {len(audio_files)} audio files")
        
        # Create index with metadata
        file_index = []
        for audio_file in tqdm.tqdm(audio_files, desc="Indexing files"):
            # Look for corresponding JSON metadata file
            json_file = audio_file.with_suffix('.json')
            
            file_info = {
                'audio_path': str(audio_file),
                'metadata_path': str(json_file) if json_file.exists() else None,
                'filename': audio_file.name,
                'dataset': dataset_name
            }
            
            # Load metadata if available
            if json_file.exists():
                try:
                    with open(json_file, 'r') as f:
                        metadata = json.load(f)
                        file_info['metadata'] = metadata
                except Exception as e:
                    logger.warning(f"Failed to load metadata for {audio_file}: {e}")
            
            file_index.append(file_info)
        
        # Save index to cache
        index_file = self.cache_dir / f"{dataset_name}_index.json"
        with open(index_file, 'w') as f:
            json.dump(file_index, f, indent=2)
        
        logger.info(f"Created index with {len(file_index)} files, saved to {index_file}")
        
        self.file_indices[dataset_name] = file_index
        return file_index
    
    def load_file_index(self, dataset_name: str) -> Optional[List[Dict]]:
        """
        Load a previously created file index.
        
        Args:
            dataset_name: Name of the dataset
            
        Returns:
            List of file information dictionaries or None if not found
        """
        if dataset_name in self.file_indices:
            return self.file_indices[dataset_name]
        
        index_file = self.cache_dir / f"{dataset_name}_index.json"
        if index_file.exists():
            try:
                with open(index_file, 'r') as f:
                    file_index = json.load(f)
                    self.file_indices[dataset_name] = file_index
                    logger.info(f"Loaded index for {dataset_name} with {len(file_index)} files")
                    return file_index
            except Exception as e:
                logger.error(f"Failed to load index for {dataset_name}: {e}")
        
        return None
    
    def get_files_by_criteria(self, dataset_name: str, 
                            min_duration: Optional[float] = None,
                            max_duration: Optional[float] = None,
                            has_caption: bool = False) -> List[Dict]:
        """
        Get files that match specific criteria.
        
        Args:
            dataset_name: Name of the dataset
            min_duration: Minimum duration in seconds
            max_duration: Maximum duration in seconds
            has_caption: Whether the file must have a caption
            
        Returns:
            List of filtered file information dictionaries
        """
        file_index = self.load_file_index(dataset_name)
        if not file_index:
            logger.error(f"No file index found for {dataset_name}")
            return []
        
        filtered_files = []
        for file_info in file_index:
            metadata = file_info.get('metadata', {})
            
            # Check duration criteria
            if min_duration is not None or max_duration is not None:
                duration = metadata.get('duration_ms', 0) / 1000.0  # Convert to seconds
                if min_duration is not None and duration < min_duration:
                    continue
                if max_duration is not None and duration > max_duration:
                    continue
            
            # Check caption criteria
            if has_caption:
                caption = metadata.get('comprehensive_caption') or metadata.get('caption')
                if not caption:
                    continue
            
            filtered_files.append(file_info)
        
        logger.info(f"Found {len(filtered_files)} files matching criteria")
        return filtered_files
    
    def get_dataset_stats(self, dataset_name: str) -> Dict:
        """
        Get statistics about a dataset.
        
        Args:
            dataset_name: Name of the dataset
            
        Returns:
            Dictionary with dataset statistics
        """
        file_index = self.load_file_index(dataset_name)
        if not file_index:
            return {}
        
        stats = {
            'total_files': len(file_index),
            'files_with_metadata': sum(1 for f in file_index if f.get('metadata')),
            'files_with_captions': 0,
            'total_duration': 0,
            'avg_duration': 0,
            'min_duration': float('inf'),
            'max_duration': 0
        }
        
        durations = []
        for file_info in file_index:
            metadata = file_info.get('metadata', {})
            
            # Check for captions
            if metadata.get('comprehensive_caption') or metadata.get('caption'):
                stats['files_with_captions'] += 1
            
            # Duration statistics
            duration_ms = metadata.get('duration_ms', 0)
            if duration_ms > 0:
                duration_s = duration_ms / 1000.0
                durations.append(duration_s)
                stats['total_duration'] += duration_s
                stats['min_duration'] = min(stats['min_duration'], duration_s)
                stats['max_duration'] = max(stats['max_duration'], duration_s)
        
        if durations:
            stats['avg_duration'] = sum(durations) / len(durations)
            stats['min_duration'] = min(durations)
        else:
            stats['min_duration'] = 0
        
        return stats


def main():
    """Example usage of the data loader."""
    loader = LAIONDataLoader()
    
    # Discover available datasets
    datasets = loader.discover_datasets()
    print("Available datasets:")
    for name, files in datasets.items():
        print(f"  {name}: {len(files)} tar files")
    
    # Extract and index the generated sound events dataset
    if 'generated-sound-events' in datasets:
        print("\nExtracting generated sound events...")
        success = loader.extract_dataset('generated-sound-events')
        if success:
            print("Extraction successful!")
            
            # Get some statistics
            stats = loader.get_dataset_stats('generated-sound-events')
            print(f"Dataset statistics: {stats}")
            
            # Get files suitable for soundscape generation
            suitable_files = loader.get_files_by_criteria(
                'generated-sound-events',
                min_duration=2.0,  # At least 2 seconds
                max_duration=30.0,  # At most 30 seconds
                has_caption=True
            )
            print(f"Found {len(suitable_files)} files suitable for soundscape generation")


if __name__ == "__main__":
    main()