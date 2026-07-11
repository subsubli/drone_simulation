"""Merges all per-episode CSVs from a shape_dataset.py collection folder into one CSV.

Adds an `episode_id` column (0-indexed, one per source file, in sorted filename order) so
episode boundaries survive the merge even without relying on the `done` column or the
per-episode metadata columns. The original per-episode CSVs are only read, never modified.

Example
-------
In a terminal, run as:

    $ python merge_shape_dataset.py --input_folder dataset_1M/shape_dataset --output_file dataset_1M/merged.csv

"""
import os
import csv
import glob
import argparse

DEFAULT_OUTPUT_FILE = 'merged.csv'


def merge(input_folder, output_file):
    files = sorted(glob.glob(os.path.join(input_folder, '*.csv')))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {input_folder}")

    with open(files[0], newline='') as f:
        header = next(csv.reader(f))

    total_rows = 0
    with open(output_file, 'w', newline='') as out_f:
        writer = csv.writer(out_f)
        writer.writerow(['episode_id'] + header)
        for episode_id, path in enumerate(files):
            with open(path, newline='') as in_f:
                reader = csv.reader(in_f)
                next(reader)  # skip this file's own header
                for row in reader:
                    writer.writerow([episode_id] + row)
                    total_rows += 1

    print(f"[INFO] merged {len(files)} episodes, {total_rows} rows -> {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Merge shape_dataset.py per-episode CSVs into one CSV')
    parser.add_argument('--input_folder', required=True, type=str, help='Folder containing per-episode CSVs (e.g. dataset_1M/shape_dataset)', metavar='')
    parser.add_argument('--output_file',  default=DEFAULT_OUTPUT_FILE, type=str, help=f'Path of the merged CSV to write (default: "{DEFAULT_OUTPUT_FILE}")', metavar='')
    ARGS = parser.parse_args()

    merge(ARGS.input_folder, ARGS.output_file)
