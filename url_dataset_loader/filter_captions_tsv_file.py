import os

# Define file paths
tsv_file = "Train_GCC-training.tsv"  # Path to your original TSV file
image_folder = "loadedimages"  # Folder where images are stored
output_tsv = "filtered_captions.tsv"  # Output TSV file

# Get a set of existing image filenames (without extensions)
available_images = {os.path.splitext(f)[0] for f in os.listdir(image_folder) if
                    os.path.isfile(os.path.join(image_folder, f))}

# Get a set of existing image filenames (without extensions)
available_images = {os.path.splitext(f)[0] for f in os.listdir(image_folder) if
                    os.path.isfile(os.path.join(image_folder, f))}

# Process the TSV file
with open(tsv_file, "r", encoding="utf-8") as infile, open(output_tsv, "w", encoding="utf-8") as outfile:
    for i, line in enumerate(infile, start=0):
        parts = line.strip().split("\t")
        if len(parts) != 2:
            continue  # Skip malformed lines
        caption, _ = parts  # Extract caption, ignore URL
        image_name = str(i)  # Image names are based on line numbers

        if image_name in available_images:
            outfile.write(f"{image_name}\t{caption}\n")

print(f"Filtered captions saved to {output_tsv}")
