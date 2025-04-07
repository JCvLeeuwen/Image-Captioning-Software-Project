import os
import sys
import argparse
import shutil
import time
import concurrent.futures
from tqdm import tqdm
from pathlib import Path

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.config import IMAGE_DIR, DATA_DIR


def parse_args():
    """
    command line arguments
    """
    parser = argparse.ArgumentParser(description="Validate images and remove broken ones")
    
    parser.add_argument("--image-dir", type=str, default=IMAGE_DIR,
                       help="Directory containing images to validate")
    parser.add_argument("--output-dir", type=str, default=os.path.join(DATA_DIR, "broken_images"),
                       help="Directory to move broken images to")
    parser.add_argument("--delete", action="store_true",
                       help="Delete broken images instead of moving them")
    parser.add_argument("--min-size", type=int, default=1000,
                       help="Minimum file size in bytes")
    parser.add_argument("--workers", type=int, default=4,
                       help="Number of worker threads")
    parser.add_argument("--batch-size", type=int, default=100,
                       help="Batch size for processing")
    parser.add_argument("--extensions", type=str, default=".jpg,.jpeg,.png,.gif,.bmp,.webp",
                       help="Comma-separated list of image extensions")
    
    return parser.parse_args()


def is_broken_file(file_path, min_size=1000):
    """
    check if a file is too small to be a valid image
    
    Args:
        file_path: path to the file
        min_size: minimum file size in bytes
        
    Returns:
        tuple of (is_broken, reason)
    """
    try:
        # file size
        size = os.path.getsize(file_path)
        if size < min_size:
            return True, f"File too small: {size} bytes"

        # quick header check for slightly larger files (optional)
        if size < 5000:  # only check for suspicious small files
            try:
                with open(file_path, 'rb') as f:
                    header = f.read(16)

                # common image file headers
                valid_headers = [
                    b'\xff\xd8\xff',  # JPEG
                    b'\x89PNG\r\n\x1a\n',  # PNG
                    b'GIF8',  # GIF
                    b'BM',  # BMP
                    b'RIFF'  # WEBP
                ]

                # matches any valid image header?
                if not any(header.startswith(h) for h in valid_headers):
                    return True, "invalid image header"
            except:
                # if we can't read the file, consider it broken
                return True, "can't read file"

        # open and verify the image
        try:
            from PIL import Image
            img = Image.open(file_path)
            img.verify()  # valid image?
        except Exception as e:
            return True, f"failed image verification: {str(e)}"

        return False, "OK"
    except Exception as e:
        return True, str(e)


def process_batch(files, min_size):
    """
    process a batch of files, return list of broken files
    
    Args:
        files: list of file paths
        min_size: min file size in bytes
        
    Returns:
        list of tuples (file_path, reason) for broken files
    """
    broken = []
    for file_path in files:
        is_broken, reason = is_broken_file(file_path, min_size)
        if is_broken:
            broken.append((file_path, reason))
    return broken


def main():
    """
    main function for validating images
    """
    args = parse_args()
    
    start_time = time.time()
    
    # extensions
    extensions = [ext.strip() for ext in args.extensions.split(",")]
    
    # make sure output directory exists if we're moving files
    if not args.delete and not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # all image files
    print(f"Scanning directory: {args.image_dir}")
    image_files = []
    
    for ext in extensions:
        image_files.extend(list(Path(args.image_dir).glob(f"*{ext}")))
    
    print(f"Found {len(image_files)} image files to check")
    
    # batches for processing
    batches = [image_files[i:i+args.batch_size] for i in range(0, len(image_files), args.batch_size)]
    
    # process all batches
    all_broken_files = []
    files_processed = 0
    
    print(f"Processing in {len(batches)} batches with {args.workers} workers")
    
    for batch_idx, batch in enumerate(batches):
        batch_start = time.time()
        
        # process batch
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
            # smaller sub-batches for each worker
            sub_batch_size = max(1, len(batch) // args.workers)
            sub_batches = [batch[i:i+sub_batch_size] for i in range(0, len(batch), sub_batch_size)]
            
            # process sub-batches in parallel
            futures = [executor.submit(process_batch, sub_batch, args.min_size) for sub_batch in sub_batches]
            broken_batches = [future.result() for future in concurrent.futures.as_completed(futures)]
            
            batch_broken = [item for sublist in broken_batches for item in sublist]
        
        # handle broken files
        for file_path, reason in batch_broken:
            all_broken_files.append((file_path, reason))
            
            # remove or move the file
            if args.delete:
                try:
                    os.remove(file_path)
                except Exception as e:
                    print(f"error deleting {file_path}: {e}")
            else:
                try:
                    filename = os.path.basename(file_path)
                    dest_path = os.path.join(args.output_dir, filename)
                    
                    # handle filename collisions
                    if os.path.exists(dest_path):
                        base, ext = os.path.splitext(filename)
                        dest_path = os.path.join(args.output_dir, f"{base}_{int(time.time()*1000) % 10000}{ext}")
                    
                    shutil.move(file_path, dest_path)
                except Exception as e:
                    print(f"error moving {file_path}: {e}")
        
        # progress
        files_processed += len(batch)
        progress_percent = files_processed / len(image_files) * 100
        
        # timing metrics
        batch_time = time.time() - batch_start
        files_per_second = len(batch) / max(0.001, batch_time)
        elapsed_time = time.time() - start_time
        
        # progress
        print(f"Batch {batch_idx+1}/{len(batches)} | "
              f"Progress: {progress_percent:.1f}% | "
              f"Found {len(batch_broken)} broken files in this batch | "
              f"Speed: {files_per_second:.1f} files/sec")
    
    # make report
    report_path = os.path.join(args.output_dir if not args.delete else os.path.dirname(args.image_dir),
                              "broken_images_report.txt")
    
    with open(report_path, 'w') as f:
        f.write("# Empty/broken files report\n\n")
        f.write(f"Total files processed: {len(image_files)}\n")
        f.write(f"Empty/broken files found: {len(all_broken_files)}\n\n")
        f.write("## List of empty/broken files\n\n")
        
        for file_path, reason in all_broken_files:
            f.write(f"{file_path}\t{reason}\n")
    
    # final stats
    total_time = time.time() - start_time
    print("\n" + "="*50)
    print("PROCESSING COMPLETE")
    print("="*50)
    print(f"processed {len(image_files)} files in {total_time:.1f} seconds")
    print(f"found {len(all_broken_files)} empty or broken files")
    print(f"average speed: {len(image_files)/total_time:.1f} files per second")
    print(f"report saved to: {report_path}")


if __name__ == "__main__":
    main()