import os
import tarfile
import argparse

def create_webdataset(input_dir, output_dir, target_size=2e9):
    def get_file_paths(directory):
        for root, dirs, files in os.walk(directory):
            for file in files:
                yield os.path.join(root, file)

    def create_tarfile(batch, batch_num, split_dir):
        tar_path = os.path.join(split_dir, f'{batch_num:04d}.tar')
        with tarfile.open(tar_path, 'w') as tar:
            for file_path in batch:
                arcname = os.path.relpath(file_path, input_dir)
                tar.add(file_path, arcname=arcname)

    os.makedirs(output_dir, exist_ok=True)

    # Detect splits from input directory structure
    splits = []
    for item in os.listdir(input_dir):
        item_path = os.path.join(input_dir, item)
        if os.path.isdir(item_path):
            splits.append(item)
    
    # If no subdirectories found, treat the whole directory as a single split
    if not splits:
        splits = ['']
    
    # Process each split separately
    for split in splits:
        split_input_dir = os.path.join(input_dir, split) if split else input_dir
        split_output_dir = os.path.join(output_dir, split) if split else output_dir
        
        os.makedirs(split_output_dir, exist_ok=True)
        
        current_batch = []
        current_size = 0
        batch_num = 0

        for file_path in get_file_paths(split_input_dir):
            file_size = os.path.getsize(file_path)
            if current_size + file_size > target_size and current_batch:
                create_tarfile(current_batch, batch_num, split_output_dir)
                batch_num += 1
                current_batch = []
                current_size = 0

            current_batch.append(file_path)
            current_size += file_size

        if current_batch:
            create_tarfile(current_batch, batch_num, split_output_dir)

# Usage
# input_directory = 'Clotho-AQA/data'
# output_directory = 'tar/clotho-aqa'



parser = argparse.ArgumentParser(description='Process input and output directories for sound descriptions.')

# Define arguments
parser.add_argument('--input_directory', required=True,  
                    help='Path to the input directory containing sound descriptions.')
parser.add_argument('--output_directory', required=True,  
                    help='Path to the output directory where processed files will be saved.')

# Parse arguments
args = parser.parse_args()

# Use the parsed arguments
input_directory = args.input_directory
output_directory = args.output_directory

print(f"Input Directory: {input_directory}")
print(f"Output Directory: {output_directory}")


create_webdataset(input_directory, output_directory)
