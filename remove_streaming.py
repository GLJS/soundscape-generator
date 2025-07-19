#!/usr/bin/env python3
"""Script to remove streaming functionality from utils.py"""

with open('utils.py', 'r') as f:
    lines = f.readlines()

# Remove create_tar_streaming method (lines 154-227)
# Remove process_dataset_streaming method (lines 357-402)

# Create new lines excluding the methods
new_lines = []
skip = False
i = 0

while i < len(lines):
    # Check if we're starting create_tar_streaming
    if i == 153 and lines[i].strip().startswith("def create_tar_streaming"):
        skip = True
        i = 227  # Skip to line after the method
        continue
    
    # Check if we're starting process_dataset_streaming  
    if i == 356 and lines[i].strip().startswith("def process_dataset_streaming"):
        skip = True
        i = 402  # Skip to line after the method
        continue
        
    new_lines.append(lines[i])
    i += 1

# Write back the modified content
with open('utils.py', 'w') as f:
    f.writelines(new_lines)

print("Removed streaming methods from utils.py")