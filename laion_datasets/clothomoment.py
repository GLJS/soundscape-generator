#!/usr/bin/env python3
"""
Convert ClothoMoment dataset tar files to WebDataset tar format with proper audio conversion.

Input: /scratch-shared/gwijngaard/laion/ClothoMoment/{train,valid,test}/*.tar (with WAV files)
Output: Tar files with FLAC audio (48kHz mono 16bit) and exactly 2048 files per tar
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tarfile
import json
from pathlib import Path
from utils import AudioProcessor, TarCreator
from typing import List, Dict, Iterator, Tuple
import argparse
from tqdm import tqdm
import io
from dotenv import load_dotenv

load_dotenv()


class ClothoMomentProcessor:
    """Processor for ClothoMoment dataset - converts existing tar files."""
    
    def __init__(self, input_dir: str, output_dir: str):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.audio_processor = AudioProcessor(target_sr=48000, output_format="flac", bit_depth=16)
        
    def process_tar_file(self, tar_path: Path) -> Iterator[Dict]:
        """Process a single tar file and yield samples."""
        
        try:
            with tarfile.open(tar_path, 'r') as tar:
                # Get list of members
                members = tar.getmembers()
                
                # Group files by base name (without extension)
                file_groups = {}
                for member in members:
                    if member.isfile():
                        # Extract base name without extension
                        base_name = member.name.rsplit('.', 1)[0]
                        if base_name not in file_groups:
                            file_groups[base_name] = {}
                        
                        # Store by extension
                        ext = member.name.rsplit('.', 1)[-1]
                        file_groups[base_name][ext] = member
                
                # Process each file group
                for base_name, files in file_groups.items():
                    # We need both audio and json files
                    if 'wav' not in files and 'flac' not in files:
                        continue
                    if 'json' not in files:
                        continue
                    
                    try:
                        # Get audio file (prefer wav over flac for input)
                        audio_member = files.get('wav', files.get('flac'))
                        json_member = files['json']
                        
                        # Extract audio data
                        audio_file = tar.extractfile(audio_member)
                        if audio_file is None:
                            continue
                        audio_data = audio_file.read()
                        
                        # Extract json data
                        json_file = tar.extractfile(json_member)
                        if json_file is None:
                            continue
                        json_data = json.loads(json_file.read().decode('utf-8'))
                        
                        # Process audio to correct format
                        processed_audio, audio_metadata = self.audio_processor.process_audio_file(io.BytesIO(audio_data))
                        
                        # Merge metadata
                        if 'metadata' in json_data:
                            json_data['metadata'].update(audio_metadata)
                        else:
                            json_data['metadata'] = audio_metadata
                        
                        yield {
                            'audio_bytes': processed_audio,
                            'text': json_data.get('text', ''),
                            'metadata': json_data.get('metadata', {})
                        }
                        
                    except Exception as e:
                        print(f"Error processing {base_name}: {e}")
                        continue
                        
        except Exception as e:
            print(f"Error reading tar file {tar_path}: {e}")
            
    def process_split(self, split: str, samples_per_tar: int = 2048):
        """Process all tar files for a given split."""
        split_dir = self.input_dir / split
        
        if not split_dir.exists():
            print(f"Split directory not found: {split_dir}")
            return []
            
        # Get all tar files for this split
        tar_files = sorted(split_dir.glob("*.tar"))
        print(f"Found {len(tar_files)} tar files for {split} split")
        
        # Create output directory for this split
        output_split_dir = self.output_dir / split
        output_split_dir.mkdir(parents=True, exist_ok=True)
        
        # Process all tar files
        all_samples = []
        global_tar_index = 0
        current_batch = []
        summaries = []
        pbar = tqdm(tar_files, desc=f"Processing {split} tar files")
        
        for tar_file in pbar:
            # Process samples from this tar file
            for sample in self.process_tar_file(tar_file):
                current_batch.append(sample)

                pbar.set_postfix(samples_processed=len(current_batch), tar_file=tar_file.name)
                
                # Write tar when we have exactly samples_per_tar samples
                if len(current_batch) >= samples_per_tar:
                    # Take exactly samples_per_tar samples
                    batch_to_write = current_batch[:samples_per_tar]
                    current_batch = current_batch[samples_per_tar:]
                    
                    # Write tar file with proper naming
                    tar_path = output_split_dir / f"{global_tar_index:04d}.tar"
                    summary = self.write_tar_file(batch_to_write, tar_path, global_tar_index)
                    summaries.append(summary)
                    global_tar_index += 1
        
        # Write remaining samples if any
        if current_batch:
            tar_path = output_split_dir / f"{global_tar_index:04d}.tar"
            summary = self.write_tar_file(current_batch, tar_path, global_tar_index)
            summaries.append(summary)
            
        # Create sizes.json for this split
        self.create_size_file(output_split_dir, summaries)
        
        return summaries
        
    def write_tar_file(self, samples: List[Dict], tar_path: Path, tar_index: int) -> Dict:
        """Write samples to a tar file."""
        successful = 0
        failed = 0
        
        with tarfile.open(tar_path, "w") as tar:
            for idx, sample in enumerate(samples):
                try:
                    # Create key for this sample
                    key = f"{tar_index:06d}_{idx:06d}"
                    
                    # Add audio file (as FLAC)
                    audio_info = tarfile.TarInfo(name=f"{key}.flac")
                    audio_info.size = len(sample['audio_bytes'])
                    tar.addfile(audio_info, io.BytesIO(sample['audio_bytes']))
                    
                    # Create JSON with text and metadata
                    json_data = {
                        "text": sample['text'],
                    }
                    
                    if 'metadata' in sample:
                        json_data['metadata'] = sample['metadata']
                        
                    json_bytes = json.dumps(json_data, ensure_ascii=False).encode('utf-8')
                    
                    # Add JSON file
                    json_info = tarfile.TarInfo(name=f"{key}.json")
                    json_info.size = len(json_bytes)
                    tar.addfile(json_info, io.BytesIO(json_bytes))
                    
                    successful += 1
                    
                except Exception as e:
                    print(f"Failed to write sample {idx} to {tar_path}: {e}")
                    failed += 1
                    
        return {
            "tar_file": tar_path.name,
            "successful": successful,
            "failed": failed,
            "total": len(samples)
        }
        
    def create_size_file(self, output_dir: Path, summaries: List[Dict]):
        """Create sizes.json file for WebDataset."""
        sizes = {s['tar_file']: s['successful'] for s in summaries}
        
        with open(output_dir / "sizes.json", "w") as f:
            json.dump(sizes, f, indent=2)
            
    def process_dataset(self, samples_per_tar: int = 2048):
        """Process the entire dataset."""
        all_summaries = []
        
        # Process each split
        for split in ['train', 'valid', 'test']:
            print(f"\nProcessing {split} split...")
            summaries = self.process_split(split, samples_per_tar)
            all_summaries.extend(summaries)
            
        # Summary
        total_successful = sum(s['successful'] for s in all_summaries)
        total_failed = sum(s['failed'] for s in all_summaries)
        
        print(f"\nProcessing complete!")
        print(f"Successfully processed: {total_successful}")
        print(f"Failed: {total_failed}")
        print(f"Created {len(all_summaries)} tar files in {self.output_dir}")
        
        return all_summaries


def main():
    parser = argparse.ArgumentParser(description="Convert ClothoMoment tar files to proper format")
    parser.add_argument("--input-dir", type=str, 
                       default="/scratch-shared/gwijngaard/laion/ClothoMoment",
                       help="Path to directory containing existing tar files")
    parser.add_argument("--output-dir", type=str,
                       default="/scratch-shared/gwijngaard/tar/clothomoment",
                       help="Output directory for converted tar files")
    parser.add_argument("--samples-per-tar", type=int, default=2048,
                       help="Number of samples per tar file")
    
    args = parser.parse_args()
    
    # Create processor
    processor = ClothoMomentProcessor(
        input_dir=args.input_dir,
        output_dir=args.output_dir
    )
    
    # Process dataset
    processor.process_dataset(samples_per_tar=args.samples_per_tar)


if __name__ == "__main__":
    main()