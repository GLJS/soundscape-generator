#!/usr/bin/env python3
"""
Convert epic-kitchens dataset to WebDataset tar format.

Audio location: /scratch-shared/gwijngaard/laion/epic-kitchens/
Note: This dataset uses HDF5 format with full audio tracks that we segment into 30-second chunks
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import h5py
from pathlib import Path
from utils import DatasetProcessor
from typing import List, Tuple, Dict, Optional
import argparse
import numpy as np
from dotenv import load_dotenv
from tqdm import tqdm
import json
import random

load_dotenv()


class EpicKitchensProcessor(DatasetProcessor):
    """Processor for epic-kitchens dataset with 30-second segmentation."""
    
    def __init__(self, audio_dir: str, metadata_path: str, output_dir: str,
                 segment_duration: int = 30):
        super().__init__(audio_dir, metadata_path, output_dir)
        self.segment_duration = segment_duration  # in seconds
        self.sample_rate = 24000  # Epic Kitchens audio is 24kHz
        self.samples_per_segment = self.segment_duration * self.sample_rate
        
    def timestamp_to_seconds(self, timestamp: str) -> float:
        """Convert HH:mm:ss.SSS timestamp to seconds."""
        try:
            # Parse timestamp in format HH:mm:ss.SSS
            if '.' in timestamp:
                time_part, ms_part = timestamp.split('.')
                h, m, s = map(int, time_part.split(':'))
                ms = int(ms_part)
                return h * 3600 + m * 60 + s + ms / 1000
            else:
                h, m, s = map(int, timestamp.split(':'))
                return h * 3600 + m * 60 + s
        except:
            return 0.0
    
    def load_metadata(self) -> pd.DataFrame:
        """Load Epic-kitchens CSV metadata files."""
        print(f"Loading metadata from {self.metadata_path}")
        
        # Look for CSV files
        csv_files = []
        for pattern in ['*train*.csv', '*validation*.csv', '*test*.csv', '*.csv']:
            csv_files.extend(list(self.metadata_path.glob(pattern)))
        
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {self.metadata_path}")
        
        dfs = []
        for csv_file in csv_files:
            # Skip files that might not be event annotations
            if 'class' in csv_file.name.lower() or 'mapping' in csv_file.name.lower():
                continue
                
            df = pd.read_csv(csv_file)
            
            # Check if this looks like an event annotation file
            required_cols = ['video_id', 'start_timestamp', 'stop_timestamp']
            if all(col in df.columns for col in required_cols):
                # Add source file info
                df['source_file'] = csv_file.name
                dfs.append(df)
                print(f"  Loaded {len(df)} entries from {csv_file.name}")
        
        if not dfs:
            raise ValueError("No valid event annotation CSV files found")
            
        combined_df = pd.concat(dfs, ignore_index=True)
        
        # Convert timestamps to seconds for easier processing
        if 'start_timestamp' in combined_df.columns:
            combined_df['start_seconds'] = combined_df['start_timestamp'].apply(self.timestamp_to_seconds)
        if 'stop_timestamp' in combined_df.columns:
            combined_df['stop_seconds'] = combined_df['stop_timestamp'].apply(self.timestamp_to_seconds)
            
        print(f"Total loaded: {len(combined_df)} event annotations")
        return combined_df
        
    def segment_audio(self, audio_data: np.ndarray, video_id: str) -> List[Tuple[np.ndarray, int, int]]:
        """Segment audio into fixed-duration chunks."""
        segments = []
        total_samples = len(audio_data)
        
        # Create segments
        for start_idx in range(0, total_samples, self.samples_per_segment):
            end_idx = min(start_idx + self.samples_per_segment, total_samples)
            segment = audio_data[start_idx:end_idx]
            
            # Only include segments that are at least 1 second long
            if len(segment) >= self.sample_rate:
                segments.append((segment, start_idx, end_idx))
                
        return segments
    
    def map_events_to_segment(self, events_df: pd.DataFrame, segment_start_sec: float, 
                             segment_end_sec: float) -> List[Dict]:
        """Find all events that overlap with a given segment."""
        overlapping_events = []
        
        for _, event in events_df.iterrows():
            event_start = event.get('start_seconds', 0)
            event_stop = event.get('stop_seconds', event_start)
            
            # Check if event overlaps with segment
            if event_start < segment_end_sec and event_stop > segment_start_sec:
                # Calculate relative position within segment
                relative_start = max(0, event_start - segment_start_sec)
                relative_end = min(self.segment_duration, event_stop - segment_start_sec)
                
                event_info = {
                    'description': event.get('description', ''),
                    'start_time': relative_start,
                    'end_time': relative_end,
                    'original_start': event_start,
                    'original_end': event_stop
                }
                
                # Only include if description exists
                if event_info['description']:
                    overlapping_events.append(event_info)
        
        # Sort by start time
        overlapping_events.sort(key=lambda x: x['start_time'])
        
        return overlapping_events
    
    def create_qa_caption(self, events: List[Dict]) -> str:
        """Create a question-answer caption from temporal events."""
        if not events:
            # Handle empty segments
            questions = [
                "What events occur in this audio?",
                "What can be heard in this recording?",
                "Describe the sounds in this audio segment."
            ]
            question = random.choice(questions)
            answer = "This audio segment contains no annotated events."
            return f"Question: {question} Answer: {answer}"
        
        # Question templates
        question_templates = [
            "What events occur in this audio?",
            "What happens first in this audio?",
            "Describe the sequence of events in this audio.",
            "What sounds can be heard and when?",
            "List the events in chronological order.",
            "What activities take place in this recording?",
            "Summarize the temporal sequence of sounds.",
            "What kitchen sounds are present in this audio?"
        ]
        
        question = random.choice(question_templates)
        
        # Generate different answer formats based on question
        if "first" in question.lower():
            # Focus on the first event
            first_event = events[0]
            if first_event['end_time'] - first_event['start_time'] < 0.5:
                answer = f"First, {first_event['description']} occurs at {first_event['start_time']:.1f} seconds."
            else:
                answer = f"First, {first_event['description']} occurs from {first_event['start_time']:.1f} to {first_event['end_time']:.1f} seconds."
        
        elif "sequence" in question.lower() or "chronological" in question.lower():
            # List events in order with transition words
            event_descriptions = []
            for i, event in enumerate(events):
                desc = event['description']
                start = event['start_time']
                end = event['end_time']
                
                if i == 0:
                    prefix = "First, "
                elif i == len(events) - 1:
                    prefix = "finally, "
                else:
                    prefix = "then "
                
                if end - start < 0.5:
                    time_str = f"at {start:.1f}s"
                else:
                    time_str = f"from {start:.1f}s to {end:.1f}s"
                
                event_descriptions.append(f"{prefix}{desc} {time_str}")
            
            answer = "The sequence is: " + ", ".join(event_descriptions) + "."
        
        else:
            # General format listing all events
            if len(events) == 1:
                event = events[0]
                if event['end_time'] - event['start_time'] < 0.5:
                    answer = f"This audio contains {event['description']} at {event['start_time']:.1f} seconds."
                else:
                    answer = f"This audio contains {event['description']} from {event['start_time']:.1f} to {event['end_time']:.1f} seconds."
            else:
                event_descriptions = []
                for event in events:
                    desc = event['description']
                    start = event['start_time']
                    end = event['end_time']
                    
                    if end - start < 0.5:
                        time_str = f"at {start:.1f}s"
                    else:
                        time_str = f"from {start:.1f}s to {end:.1f}s"
                    
                    event_descriptions.append(f"{desc} {time_str}")
                
                answer = f"This audio contains {len(events)} events: " + "; ".join(event_descriptions) + "."
        
        return f"Question: {question} Answer: {answer}"
        
    def process_dataset(self, samples_per_tar: int = 2048):
        """Custom process_dataset for HDF5-based Epic Kitchens dataset."""
        # Load metadata
        metadata_df = self.load_metadata()
        
        # Find HDF5 file
        h5_files = list(self.audio_dir.glob("*.hdf5")) + list(self.audio_dir.glob("*.h5"))
        
        if not h5_files:
            raise FileNotFoundError(f"No HDF5 files found in {self.audio_dir}")
        
        # Use the first (or only) HDF5 file
        h5_path = h5_files[0]
        print(f"\nProcessing {h5_path}")
        
        # Create tar creator
        from utils import TarCreator
        tar_creator = TarCreator(
            self.output_dir, 
            prefix='epickitchens', 
            samples_per_tar=samples_per_tar,
            split='train'  # Default to train split
        )
        
        current_batch = []
        tar_index = 0
        all_summaries = []
        total_segments = 0
        
        with h5py.File(h5_path, 'r') as f:
            video_ids = list(f.keys())
            print(f"Found {len(video_ids)} videos in HDF5")
            
            # Process each video
            for video_id in tqdm(video_ids, desc="Processing videos"):
                try:
                    # Load audio data
                    audio_data = f[video_id][:]
                    
                    # Get events for this video
                    video_events = metadata_df[metadata_df['video_id'] == video_id]
                    
                    # Segment audio
                    segments = self.segment_audio(audio_data, video_id)
                    
                    # Process each segment
                    for seg_idx, (segment_audio, start_sample, end_sample) in enumerate(segments):
                        # Calculate time boundaries
                        segment_start_sec = start_sample / self.sample_rate
                        segment_end_sec = end_sample / self.sample_rate
                        
                        # Find overlapping events
                        events = self.map_events_to_segment(
                            video_events, segment_start_sec, segment_end_sec
                        )
                        
                        # Create segment ID
                        segment_id = f"{video_id}_seg_{seg_idx:04d}"
                        
                        # Generate Q&A caption
                        qa_caption = self.create_qa_caption(events)
                        
                        # Create metadata
                        metadata = {
                            'split': 'train',  # Could determine from source_file if needed
                            'video_id': video_id,
                            'segment_id': segment_id,
                            'segment_index': seg_idx,
                            'start_time': segment_start_sec,
                            'end_time': segment_end_sec,
                            'duration': (end_sample - start_sample) / self.sample_rate,
                            'num_events': len(events),
                            'task': 'AQA'  # Audio Question Answering
                        }
                        
                        # Convert audio to bytes for storage
                        # Normalize if needed
                        if segment_audio.dtype == np.float32 or segment_audio.dtype == np.float64:
                            audio_bytes = (segment_audio * 32767).astype(np.int16).tobytes()
                        else:
                            audio_bytes = segment_audio.tobytes()
                        
                        # Add to current batch
                        sample = {
                            'audio_bytes': audio_bytes,
                            'text': qa_caption,
                            'metadata': metadata
                        }
                        current_batch.append(sample)
                        total_segments += 1
                        
                        # Create tar when batch is full
                        if len(current_batch) >= samples_per_tar:
                            summary = tar_creator.create_tar_from_samples(current_batch, tar_index)
                            all_summaries.append(summary)
                            tar_index += 1
                            current_batch = []
                        
                except Exception as e:
                    print(f"Error processing video {video_id}: {e}")
                    continue
        
        # Process remaining samples
        if current_batch:
            summary = tar_creator.create_tar_from_samples(current_batch, tar_index)
            all_summaries.append(summary)
        
        # Create size file
        tar_creator.create_size_file(all_summaries)
        
        # Summary
        total_successful = sum(s['successful'] for s in all_summaries)
        total_failed = sum(s['failed'] for s in all_summaries)
        
        print(f"\nDataset processing complete!")
        print(f"Total segments processed: {total_segments}")
        print(f"Successfully written: {total_successful}")
        print(f"Failed: {total_failed}")
        print(f"Created {len(all_summaries)} tar files")
        
        # Save to json as backup
        print(f"\nSaving segment metadata to matched_epickitchens_qa.json...")
        with open("matched_epickitchens_qa.json", "w") as f:
            json.dump({
                'total_segments': total_segments,
                'total_videos': len(video_ids),
                'summaries': all_summaries
            }, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Convert epic-kitchens dataset to tar format with 30-second segmentation and Q&A captions")
    parser.add_argument("--audio-dir", type=str, 
                       default="/scratch-shared/gwijngaard/laion/epic-kitchens",
                       help="Path to epic-kitchens directory with HDF5 file")
    parser.add_argument("--metadata", type=str,
                       default="/scratch-shared/gwijngaard/laion/epic-kitchens",
                       help="Path to metadata CSV files")
    parser.add_argument("--output-dir", type=str,
                       default="/scratch-shared/gwijngaard/tar/epickitchens",
                       help="Output directory for tar files")
    parser.add_argument("--samples-per-tar", type=int, default=2048,
                       help="Number of samples per tar file")
    parser.add_argument("--segment-duration", type=int, default=30,
                       help="Duration of each audio segment in seconds")
    
    args = parser.parse_args()
    
    # Create processor
    processor = EpicKitchensProcessor(
        audio_dir=args.audio_dir,
        metadata_path=args.metadata,
        output_dir=args.output_dir,
        segment_duration=args.segment_duration
    )
    
    # Process dataset
    processor.process_dataset(samples_per_tar=args.samples_per_tar)


if __name__ == "__main__":
    main()