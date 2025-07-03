#!/usr/bin/env python3
"""
Soundscape generator - definitive final version from provided code.
"""

import os
import json
import random
import logging
import tempfile
from pathlib import Path
import numpy as np
from typing import List, Dict, Optional

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SoundscapeGenerator:
    """Soundscape generator with augmentations and descriptions."""
    
    def __init__(self, output_dir: str = "/scratch-shared/gwijngaard/laion/extracted/output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Configuration from the original code
        self.NUM_SAMPLES_TO_GENERATE = 10
        self.MIN_SAMPLES_PER_SOUNDSCAPE = 2
        self.MAX_SAMPLES_PER_SOUNDSCAPE = 5
        self.MIN_EVENT_DURATION_S = 2
        self.MAX_TOTAL_DURATION_S = 30
        self.TARGET_BITRATE = "96k"
        
        # Overlap/Transition Configuration
        self.ALLOW_OVERLAPS = True
        self.MAX_OVERLAP_PERCENTAGE = 0.4
        self.ENABLE_TRANSITIONS = True
        self.TRANSITION_PROBABILITY = 0.7
        self.MIN_TRANSITION_MS = 500
        self.MAX_TRANSITION_MS = 3000
        
        # Augmentation Configuration
        self.ENABLE_AUGMENTATIONS = True
        self.AUGMENTATION_PROBABILITY = 0.25
        
        self._init_augmentations()
        self._init_templates()
    
    def _init_augmentations(self):
        """Initialize augmentation configurations."""
        self.AUGMENTATION_CONFIG = [
            {
                "name": "Clipping Distortion",
                "function": self.apply_clipping_distortion,
                "params": {
                    "mild": {"threshold": 0.5},
                    "normal": {"threshold": 0.2},
                    "intense": {"threshold": 0.1}
                },
                "descriptions": {
                    "mild": ["A mild clipping gives the loudest parts a slightly edgy quality."],
                    "normal": ["Moderate clipping provides a noticeable crunchy and compressed sound."],
                    "intense": ["Heavy clipping makes the audio sound very harsh and noisy."]
                }
            },
            {
                "name": "Tube Distortion",
                "function": self.apply_tube_distortion,
                "params": {
                    "mild": {"gain": 4.0},
                    "normal": {"gain": 8.0},
                    "intense": {"gain": 12.0}
                },
                "descriptions": {
                    "mild": ["A mild tube distortion adds a subtle warmth and saturation."],
                    "normal": ["A moderate tube distortion provides a classic, warm, and overdriven sound."],
                    "intense": ["Heavy tube distortion makes the sound very saturated and harmonically rich."]
                }
            },
            {
                "name": "Reverse",
                "function": self.apply_reverse,
                "params": {"mild": {}, "normal": {}, "intense": {}},
                "descriptions": {
                    "mild": ["The audio clip has been played in reverse, creating a surreal effect."],
                    "normal": ["This sound event is presented backwards from its original form."],
                    "intense": ["The clip is reversed, making for an unusual listening experience."]
                }
            },
            {
                "name": "Gain Change",
                "function": self.apply_gain_change,
                "params": {
                    "mild": {"gain_db": 6},
                    "normal": {"gain_db": 12},
                    "intense": {"gain_db": 20}
                },
                "descriptions": {
                    "mild": ["The volume is moderately boosted by 6 dB, making it more prominent."],
                    "normal": ["A significant 12 dB gain boost makes the sound very loud."],
                    "intense": ["A dramatic 20 dB gain boost is applied, pushing the audio into loud, clipped distortion."]
                }
            }
        ]
    
    def _init_templates(self):
        """Initialize description templates."""
        self.VOLUME_TEMPLATES = {
            "high_dynamic": [
                "The sound is highly dynamic, with loud peaks at {peak_dbfs:.1f} dB but a much quieter average level of {avg_dbfs:.1f} dB.",
                "It features a large dynamic range; its sharp peaks are significantly louder than its average volume.",
                "This is a percussive sound with a high crest factor; its peaks are sharp and loud compared to the rest of the sound."
            ],
            "mid_dynamic": [
                "The audio has a moderate dynamic range, with some clear variation between loud and quiet parts.",
                "It has a balanced dynamic character, neither flat nor overly peaky.",
                "The volume shows a healthy amount of variation throughout the event."
            ],
            "low_dynamic": [
                "The sound is dynamically compressed, with little difference between its average ({avg_dbfs:.1f} dB) and peak ({peak_dbfs:.1f} dB) volume.",
                "This is a sustained and consistent sound with a very steady volume.",
                "The audio has a very low dynamic range, with a consistent level throughout."
            ]
        }
        
        self.TRANSITION_TEMPLATES = {
            "abrupt_in": "The event begins abruptly",
            "fade_in": "The sound fades in smoothly over {duration:.1f} seconds",
            "abrupt_out": "and ends abruptly.",
            "fade_out": "and fades out gently over {duration:.1f} seconds."
        }
        
        self.SILENCE_TEMPLATES = [
            "It is followed by a {duration:.1f}-second pause.",
            "After this, there is a silence of {duration:.1f} seconds."
        ]
    
    # Augmentation functions (simplified versions without external dependencies)
    def apply_clipping_distortion(self, segment, params):
        """Apply clipping distortion."""
        try:
            from pydub import AudioSegment
            samples = np.array(segment.get_array_of_samples(), dtype=np.float32)
            clipped = np.clip(samples, -params['threshold']*32767, params['threshold']*32767)
            return segment._spawn(clipped.astype(np.int16).tobytes())
        except ImportError:
            return segment
    
    def apply_tube_distortion(self, segment, params):
        """Apply tube distortion."""
        try:
            from pydub import AudioSegment
            samples = np.array(segment.get_array_of_samples(), dtype=np.float32) / 32767.0
            distorted = np.tanh(params['gain'] * samples)
            return segment._spawn((distorted * 32767).astype(np.int16).tobytes())
        except ImportError:
            return segment
    
    def apply_reverse(self, segment, params):
        """Apply reverse effect."""
        try:
            return segment.reverse()
        except:
            return segment
    
    def apply_gain_change(self, segment, params):
        """Apply gain change."""
        try:
            return segment.apply_gain(params['gain_db'])
        except:
            return segment
    
    def get_volume_description(self, segment):
        """Get volume description for a segment."""
        try:
            avg_dbfs = segment.dBFS
            peak_dbfs = segment.max_dBFS
            
            # Calculate dynamic range
            if len(segment) > 100:
                std_dev_db = np.std([c.dBFS for c in segment[::100] if c.dBFS != -np.inf])
            else:
                std_dev_db = 0
            
            if std_dev_db > 5.5:
                key = "high_dynamic"
            elif std_dev_db > 2.5:
                key = "mid_dynamic"
            else:
                key = "low_dynamic"
            
            return random.choice(self.VOLUME_TEMPLATES[key]).format(
                avg_dbfs=avg_dbfs, peak_dbfs=peak_dbfs
            )
        except:
            return "The audio has moderate volume levels."
    
    def generate_soundscape(self, file_index, output_index):
        """Generate a single soundscape."""
        try:
            from pydub import AudioSegment
            
            # Sample files for this soundscape
            num_to_sample = random.randint(self.MIN_SAMPLES_PER_SOUNDSCAPE, self.MAX_SAMPLES_PER_SOUNDSCAPE)
            source_files = []
            checked_indices = set()
            max_attempts = 500
            
            while len(source_files) < num_to_sample and max_attempts > 0 and len(checked_indices) < len(file_index):
                max_attempts -= 1
                random_index = random.randint(0, len(file_index) - 1)
                
                if random_index in checked_indices:
                    continue
                
                checked_indices.add(random_index)
                file_pair = file_index[random_index]
                
                try:
                    # Load metadata
                    metadata_path = file_pair.get("metadata_path")
                    if metadata_path and os.path.exists(metadata_path):
                        with open(metadata_path, 'r') as f:
                            metadata = json.load(f)
                    else:
                        metadata = file_pair.get("metadata", {})
                    
                    # Check duration
                    audio_path = file_pair["audio_path"]
                    if not os.path.exists(audio_path):
                        continue
                    
                    duration_ms = metadata.get("duration_ms")
                    if duration_ms is None:
                        try:
                            audio = AudioSegment.from_file(audio_path)
                            duration_ms = len(audio)
                        except:
                            continue
                    
                    actual_duration_ms = duration_ms - metadata.get("silence_at_beginning_ms", 0) - metadata.get("silence_at_end_ms", 0)
                    
                    if actual_duration_ms >= (self.MIN_EVENT_DURATION_S * 1000):
                        source_files.append(file_pair)
                        
                except Exception as e:
                    logger.warning(f"Error processing file: {e}")
                    continue
            
            if not source_files:
                logger.warning(f"No suitable files found for soundscape {output_index}")
                return False
            
            # Create timeline
            timeline = []
            cursor_ms = 0
            
            for i, file_pair in enumerate(source_files):
                try:
                    # Load metadata
                    metadata_path = file_pair.get("metadata_path")
                    if metadata_path and os.path.exists(metadata_path):
                        with open(metadata_path, 'r') as f:
                            metadata = json.load(f)
                    else:
                        metadata = file_pair.get("metadata", {})
                    
                    # Load audio
                    audio = AudioSegment.from_file(file_pair["audio_path"])
                    
                    # Trim silence
                    start_silence = metadata.get("silence_at_beginning_ms", 0)
                    end_silence = metadata.get("silence_at_end_ms", 0)
                    trimmed_audio = audio[start_silence:len(audio) - end_silence]
                    
                    # Calculate positioning
                    start_ms = cursor_ms
                    if self.ALLOW_OVERLAPS and i > 0:
                        last_event_end_ms = timeline[-1]['start_ms'] + timeline[-1]['duration_ms']
                        max_overlap = min(
                            timeline[-1]['duration_ms'] * self.MAX_OVERLAP_PERCENTAGE,
                            len(trimmed_audio) * self.MAX_OVERLAP_PERCENTAGE
                        )
                        start_ms = random.uniform(last_event_end_ms - max_overlap, last_event_end_ms)
                    
                    # Check duration limit
                    if start_ms + len(trimmed_audio) > self.MAX_TOTAL_DURATION_S * 1000:
                        break
                    
                    timeline.append({
                        "audio": trimmed_audio,
                        "start_ms": start_ms,
                        "duration_ms": len(trimmed_audio),
                        "caption": metadata.get("comprehensive_caption", "Audio event"),
                        "transition_in": {"type": "abrupt", "duration": 0},
                        "transition_out": {"type": "abrupt", "duration": 0}
                    })
                    
                    if not self.ALLOW_OVERLAPS:
                        cursor_ms = start_ms + len(trimmed_audio) + random.randint(0, 3000)
                        
                except Exception as e:
                    logger.warning(f"Error processing audio: {e}")
                    continue
            
            if not timeline:
                return False
            
            # Create soundscape
            soundscape = AudioSegment.silent(duration=self.MAX_TOTAL_DURATION_S * 1000)
            final_metadata_list = []
            final_duration_ms = 0
            
            for i, event in enumerate(timeline):
                audio = event['audio']
                augmentation_desc = ""
                
                # Apply augmentations
                if self.ENABLE_AUGMENTATIONS and random.random() < self.AUGMENTATION_PROBABILITY:
                    try:
                        aug_choice = random.choice(self.AUGMENTATION_CONFIG)
                        intensity = random.choice(['mild', 'normal', 'intense'])
                        audio = aug_choice['function'](audio, aug_choice['params'][intensity])
                        augmentation_desc = random.choice(aug_choice['descriptions'][intensity])
                    except Exception as e:
                        logger.warning(f"Could not apply augmentation: {e}")
                
                # Apply transitions
                percentage_cap_ms = event['duration_ms'] * 0.25
                effective_max_ms = min(self.MAX_TRANSITION_MS, percentage_cap_ms)
                td = 0
                
                if effective_max_ms >= self.MIN_TRANSITION_MS:
                    td = random.randint(self.MIN_TRANSITION_MS, int(effective_max_ms))
                
                if self.ENABLE_TRANSITIONS and random.random() < self.TRANSITION_PROBABILITY and td > 0:
                    try:
                        audio = audio.fade_in(td)
                        event['transition_in'] = {"type": "fade", "duration": td}
                        
                        if random.choice([True, False]):
                            audio = audio.fade_out(td)
                            event['transition_out'] = {"type": "fade", "duration": td}
                    except Exception as e:
                        logger.warning(f"Could not apply transitions: {e}")
                
                # Overlay audio
                try:
                    soundscape = soundscape.overlay(audio, position=event['start_ms'])
                except Exception as e:
                    logger.warning(f"Could not overlay audio: {e}")
                    continue
                
                # Calculate final duration
                event_end_ms = event['start_ms'] + event['duration_ms']
                final_duration_ms = max(final_duration_ms, event_end_ms)
                
                # Generate descriptions
                volume_desc = self.get_volume_description(audio)
                tin, tout = event['transition_in'], event['transition_out']
                
                desc_in = self.TRANSITION_TEMPLATES[f"{tin['type']}_in"].format(
                    duration=tin['duration']/1000
                )
                desc_out = self.TRANSITION_TEMPLATES[f"{tout['type']}_out"].format(
                    duration=tout['duration']/1000
                )
                transition_desc = f"{desc_in} {desc_out}"
                
                # Add silence description
                silence_desc = ""
                if i < len(timeline) - 1:
                    gap = timeline[i+1]['start_ms'] - event_end_ms
                    if gap > 200:
                        silence_desc = " " + random.choice(self.SILENCE_TEMPLATES).format(
                            duration=gap/1000
                        )
                
                final_caption = ". ".join(filter(None, [
                    event['caption'],
                    volume_desc,
                    augmentation_desc,
                    transition_desc
                ])) + silence_desc
                
                final_metadata_list.append({
                    "start_time": round(event['start_ms']/1000, 2),
                    "end_time": round(event_end_ms/1000, 2),
                    "caption": final_caption,
                    "transition_in": tin,
                    "transition_out": tout
                })
            
            # Export files
            final_soundscape = soundscape[:final_duration_ms]
            
            output_mp3 = self.output_dir / f"{output_index}.mp3"
            output_json = self.output_dir / f"{output_index}.json"
            
            try:
                final_soundscape.export(str(output_mp3), format="mp3", bitrate=self.TARGET_BITRATE)
                
                with open(output_json, 'w') as f:
                    json.dump({"events": final_metadata_list}, f, indent=4)
                
                logger.info(f"Generated soundscape {output_index} with {len(final_metadata_list)} events")
                return True
                
            except Exception as e:
                logger.error(f"Failed to save soundscape {output_index}: {e}")
                return False
                
        except ImportError:
            logger.error("pydub not available - cannot generate soundscapes")
            return False
        except Exception as e:
            logger.error(f"Failed to generate soundscape {output_index}: {e}")
            return False
    
    def generate_soundscapes(self, file_list, start_index=0):
        """Generate multiple soundscapes."""
        logger.info(f"Starting generation of {self.NUM_SAMPLES_TO_GENERATE} soundscapes from index {start_index}")
        
        success_count = 0
        try:
            import tqdm
            iterator = tqdm.tqdm(range(self.NUM_SAMPLES_TO_GENERATE), desc="Generating Soundscapes")
        except ImportError:
            iterator = range(self.NUM_SAMPLES_TO_GENERATE)
        
        for i in iterator:
            if self.generate_soundscape(file_list, start_index + i):
                success_count += 1
        
        logger.info(f"Successfully generated {success_count}/{self.NUM_SAMPLES_TO_GENERATE} soundscapes")
        return success_count


if __name__ == "__main__":
    generator = SoundscapeGenerator()
    print("Soundscape generator initialized.")
    print("Use this with file data from the data loader.")