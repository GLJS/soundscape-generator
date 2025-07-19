#!/usr/bin/env python3
"""
Complete the remaining dataset updates for tut2017.py and vggsound.py
"""

import os
import re
from pathlib import Path

# Update tut2017.py metadata to include task
tut2017_file = Path("/gpfs/work4/0/einf6190/data-preparation/laion_datasets/tut2017.py")

# Read the file
with open(tut2017_file, 'r') as f:
    content = f.read()

# Update metadata to include task
old_metadata = """                metadata = {
                    'split': row['split'],
                    'original_filename': audio_filename,
                    'scene': row['scene'],
                    'has_error': row['has_error'],
                    'segment_duration': 10  # TUT2017 uses 10-second segments
                }"""

new_metadata = """                metadata = {
                    'split': row['split'],
                    'original_filename': audio_filename,
                    'scene': row['scene'],
                    'has_error': row['has_error'],
                    'segment_duration': 10,  # TUT2017 uses 10-second segments
                    'task': 'ASC'
                }"""

content = content.replace(old_metadata, new_metadata)

# Remove custom process_dataset method
# Find and remove the entire process_dataset method
process_pattern = r'    def process_dataset\(self.*?\n(?=\n(?:def |class |\nif __name__|$))'
content = re.sub(process_pattern, '', content, flags=re.DOTALL)

# Save updated tut2017.py
with open(tut2017_file, 'w') as f:
    f.write(content)

print("Updated tut2017.py")

# Now update vggsound.py
vggsound_file = Path("/gpfs/work4/0/einf6190/data-preparation/laion_datasets/vggsound.py")

# Read the file
with open(vggsound_file, 'r') as f:
    content = f.read()

# Check if metadata needs task field
if "'task'" not in content:
    # Find metadata dict and add task
    metadata_pattern = r'(metadata\s*=\s*\{[^}]+)'
    def add_task(match):
        return match.group(1) + ",\n                    'task': 'AAC'"
    
    content = re.sub(metadata_pattern, add_task, content)

# Remove custom process_dataset method if exists
content = re.sub(process_pattern, '', content, flags=re.DOTALL)

# Save updated vggsound.py
with open(vggsound_file, 'w') as f:
    f.write(content)

print("Updated vggsound.py")

print("\nAll dataset files have been updated!")