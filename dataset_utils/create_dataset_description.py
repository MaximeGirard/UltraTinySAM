import os

# Base directory
base_dir = '.'

# List all subfolders, exclude the validation one '_only_4000'
folders = ['sa_000024', 'sa_000020', 'sa_000028', 'sa_000021', 'sa_000022', 'sa_000027', 'sa_000026', 'sa_000025', 'sa_000023']
#folders = ['sa_000029'] #for validation

#print(f"Found folders: {folders}")

# Output list
output_list = []

# Loop through each folder
for folder in folders:
    folder_path = os.path.join(base_dir, folder)
    for filename in os.listdir(folder_path):
        if filename.endswith('.png') or filename.endswith('.jpg') or filename.endswith('.jpeg'):
            abs_path = os.path.abspath(os.path.join(folder_path, filename))
            no_ext = os.path.splitext(abs_path)[0]
            output_list.append(no_ext)

# Write to file_list.txt
with open('file_list_train.txt', 'w') as f:
    for line in output_list:
        f.write(line + '\n')

print(f"Saved {len(output_list)} entries to file_list.txt")