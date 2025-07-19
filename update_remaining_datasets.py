#!/usr/bin/env python3
"""
Script to update remaining dataset files that override process_dataset
"""

import os
import re
from pathlib import Path

# Files that still need to be updated
remaining_files = [
    "epidemic.py",
    "esc50.py", 
    "maestro.py",
    "maqa.py",
    "soundingearth.py",
    "tut2016.py",
    "tut2017.py",
    "vggsound.py"
]

laion_dir = Path("/gpfs/work4/0/einf6190/data-preparation/laion_datasets")

for filename in remaining_files:
    filepath = laion_dir / filename
    
    print(f"\nProcessing {filename}...")
    
    # Read the file
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Find the process_dataset method
    process_pattern = r'    def process_dataset\(self.*?\n(?=\n(?:    def|\nclass|\nif __name__|$))'
    
    # Remove the process_dataset method if found
    if re.search(process_pattern, content, re.DOTALL):
        content = re.sub(process_pattern, '', content, flags=re.DOTALL)
        print(f"  Removed process_dataset method from {filename}")
    
    # Check if metadata dict in match_audio_to_text includes 'split'
    # Look for metadata = { ... } patterns
    metadata_pattern = r'metadata\s*=\s*\{([^}]+)\}'
    
    matches = list(re.finditer(metadata_pattern, content, re.DOTALL))
    
    for match in matches:
        metadata_content = match.group(1)
        
        # Check if 'split' is already in metadata
        if "'split'" not in metadata_content and '"split"' not in metadata_content:
            print(f"  WARNING: Found metadata dict without 'split' field in {filename}")
            print(f"    Location: line {content[:match.start()].count(chr(10)) + 1}")
            
    # Write the updated content back
    with open(filepath, 'w') as f:
        f.write(content)
        
print("\nDone! Please manually check the files marked with WARNING to ensure they set 'split' properly.")