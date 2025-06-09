import argparse
import os


def list_images_in_folders(base_dir, folders, output_file):
    output_list = []

    for folder in folders:
        folder_path = os.path.join(base_dir, folder)
        for filename in os.listdir(folder_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                abs_path = os.path.abspath(os.path.join(folder_path, filename))
                no_ext = os.path.splitext(abs_path)[0]
                output_list.append(no_ext)

    with open(output_file, 'w') as f:
        for line in output_list:
            f.write(line + '\n')

    print(f"Saved {len(output_list)} entries to {output_file}")

def parse_args():
    parser = argparse.ArgumentParser(description='List images in folders and save to text files for train/val.')
    parser.add_argument('--base_dir', type=str, default='.', help='Base directory containing the folders.')
    parser.add_argument('--train_folders', nargs='+', default=['sa_000024', 'sa_000020', 'sa_000028', 'sa_000021', 'sa_000022', 'sa_000027', 'sa_000026', 'sa_000025', 'sa_000023'], help='List of training folders to search for images.')
    parser.add_argument('--val_folders', nargs='+', default=['sa_000029'], help='List of validation folders to search for images.')
    parser.add_argument('--train_output_file', type=str, default='file_list_train.txt', help='Name of the training output text file.')
    parser.add_argument('--val_output_file', type=str, default='file_list_val.txt', help='Name of the validation output text file.')
    return parser.parse_args()

args = parse_args()

# Process training folders
print("Processing training folders...")
list_images_in_folders(args.base_dir, args.train_folders, args.train_output_file)

# Process validation folders
print("Processing validation folders...")
list_images_in_folders(args.base_dir, args.val_folders, args.val_output_file)
