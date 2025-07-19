#!/usr/bin/env python3
"""
Convert VGGSound dataset to WebDataset tar format.

Audio location: /scratch-shared/gwijngaard/laion/VGGSound/vggsound_*.tar.gz (20 files)
Metadata: /gpfs/work4/0/einf6190/data-preparation/data/VGGSound/vggsound.csv
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from pathlib import Path
from utils import DatasetProcessor, AudioProcessor, TarCreator
from typing import List, Tuple, Dict
import argparse
import tarfile
import concurrent.futures
from dotenv import load_dotenv

load_dotenv()


class VGGSoundProcessor(DatasetProcessor):
    """Processor for VGGSound dataset - handles multiple tar.gz archives."""
    
    def __init__(self, audio_dir: str, metadata_path: str, output_dir: str):
        super().__init__(audio_dir, metadata_path, output_dir)
        # VGGSound has 20 tar.gz files
        self.audio_archives = sorted(self.audio_dir.glob("vggsound_*.tar.gz"))
        self.audio_index = {}  # Maps filename to (archive_index, member_name)
        
    def load_metadata(self) -> pd.DataFrame:
        """Load VGGSound metadata CSV."""
        print(f"Loading metadata from {self.metadata_path}")
        # vggsound.csv has columns: file_name, start_sec, caption, split
        df = pd.read_csv(self.metadata_path, header=None, 
                        names=['youtube_id', 'start_sec', 'caption', 'split'])
        
        # Create expected filename format
        df['file_name'] = df.apply(
            lambda x: f"{x['youtube_id']}_{str(x['start_sec']).zfill(6)}.mp4",
            axis=1
        )
        
        print(f"Loaded {len(df)} entries")
        return df
        
    def index_audio_archives(self):
        """Create an index of all audio files across all archives."""
        print(f"Indexing {len(self.audio_archives)} audio archives...")
        
        for idx, archive_path in enumerate(self.audio_archives):
            print(f"  Indexing {archive_path.name}...")
            try:
                with tarfile.open(archive_path, "r:gz") as tar:
                    for member in tar.getmembers():
                        if member.isfile() and member.name.endswith(('.mp4', '.wav', '.mp3')):
                            filename = os.path.basename(member.name)
                            self.audio_index[filename] = (idx, member.name)
            except Exception as e:
                print(f"    Error indexing {archive_path.name}: {e}")
                
        print(f"Indexed {len(self.audio_index)} audio files total")
        
    def extract_audio_from_archive(self, archive_idx: int, member_name: str) -> bytes:
        """Extract a specific audio file from a specific archive."""
        archive_path = self.audio_archives[archive_idx]
        
        with tarfile.open(archive_path, "r:gz") as tar:
            member = tar.getmember(member_name)
            f = tar.extractfile(member)
            if f:
                return f.read()
        return None
        
    def match_audio_to_text(self, metadata_df: pd.DataFrame) -> List[Tuple[bytes, str, Dict]]:
        """Match audio files to their captions."""
        # First index all archives if not done
        if not self.audio_index:
            self.index_audio_archives()
            
        matched = []
        missing_count = 0
        
        for _, row in metadata_df.iterrows():
            filename = row['file_name']
            
            # Also check without .mp4 extension and with .wav
            filenames_to_check = [
                filename,
                filename.replace('.mp4', '.wav'),
                filename.replace('.mp4', ''),
                os.path.basename(filename)
            ]
            
            audio_found = False
            for check_filename in filenames_to_check:
                if check_filename in self.audio_index:
                    archive_idx, member_name = self.audio_index[check_filename]
                    
                    # We'll store the info needed to extract later
                    caption = row['caption']
                    metadata = {
                        'split': row['split'],
                        'original_filename': filename,
                        'youtube_id': row['youtube_id'],
                        'start_sec': row['start_sec'],
                        'archive_idx': archive_idx,
                        'member_name': member_name
                    }
                    matched.append((None, caption, metadata))  # None for audio, will extract later
                    audio_found = True
                    break
                    
            if not audio_found:
                missing_count += 1
                if missing_count <= 10:  # Only print first 10
                    print(f"Missing audio file: {filename}")
                    
        print(f"Matched {len(matched)} audio-text pairs")
        print(f"Missing audio files: {missing_count}")
        
        return matched
        
    def process_dataset(self, samples_per_tar: int = 256):
        """Process the entire dataset into tar files."""
        # Load metadata
        metadata_df = self.load_metadata()
        
        # Match audio to text (gets metadata only)
        matched_samples = self.match_audio_to_text(metadata_df)
        
        # Create tar files
        tar_creator = TarCreator(self.output_dir, prefix="vggsound", 
                                 samples_per_tar=samples_per_tar)
        
        # Process in batches
        all_summaries = []
        for i in range(0, len(matched_samples), samples_per_tar):
            batch = matched_samples[i:i+samples_per_tar]
            samples = []
            
            for _, text, metadata in batch:
                try:
                    # Extract audio on demand
                    archive_idx = metadata.pop('archive_idx')
                    member_name = metadata.pop('member_name')
                    
                    audio_bytes = self.extract_audio_from_archive(archive_idx, member_name)
                    if audio_bytes:
                        # Process audio
                        processed_audio, audio_metadata = self.audio_processor.process_audio_file(audio_bytes)
                        samples.append({
                            'audio_bytes': processed_audio,
                            'text': text,
                            'metadata': {**metadata, **audio_metadata}
                        })
                except Exception as e:
                    print(f"Failed to process {metadata.get('original_filename', 'unknown')}: {e}")
                    
            if samples:
                summary = tar_creator.create_tar_from_samples(samples, i // samples_per_tar)
                all_summaries.append(summary)
                
        # Create size file
        tar_creator.create_size_file(all_summaries)
        
        # Summary
        total_successful = sum(s['successful'] for s in all_summaries)
        total_failed = sum(s['failed'] for s in all_summaries)
        
        print(f"\nProcessing complete!")
        print(f"Total samples: {len(matched_samples)}")
        print(f"Successfully processed: {total_successful}")
        print(f"Failed: {total_failed}")
        print(f"Created {len(all_summaries)} tar files in {self.output_dir}")
        
        return all_summaries


def main():
    parser = argparse.ArgumentParser(description="Convert VGGSound dataset to tar format")
    parser.add_argument("--audio-dir", type=str, 
                       default="/scratch-shared/gwijngaard/laion/VGGSound",
                       help="Path to directory containing vggsound_*.tar.gz files")
    parser.add_argument("--metadata", type=str,
                       default="/gpfs/work4/0/einf6190/data-preparation/data/VGGSound/vggsound.csv",
                       help="Path to metadata CSV")
    parser.add_argument("--output-dir", type=str,
                       default="/scratch-shared/gwijngaard/tar/vggsound",
                       help="Output directory for tar files")
    parser.add_argument("--samples-per-tar", type=int, default=2048,
                       help="Number of samples per tar file")
    
    args = parser.parse_args()
    
    # Create processor
    processor = VGGSoundProcessor(
        audio_dir=args.audio_dir,
        metadata_path=args.metadata,
        output_dir=args.output_dir,
        task="AAC")
    
    # Process dataset
    processor.process_dataset(samples_per_tar=args.samples_per_tar)


if __name__ == "__main__":
    main()