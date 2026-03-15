"""
json_to_csv.py
--------------
Walks a root directory, finds all JSON files inside date-time named
sub-folders, and merges them into a single CSV file.

Folder structure expected:
  root_dir/
    2024-01-15_10-30-00/
      data.json          ← { "key": "value", ... }
    2024-01-16_11-45-22/
      data.json
    ...

Usage:
  python json_to_csv.py                         # uses current directory
  python json_to_csv.py /path/to/root_dir       # specify root directory
  python json_to_csv.py /path/to/root output.csv  # specify root + output file
"""

import os
import json
import csv
import sys
from pathlib import Path


def find_json_files(root_dir: Path) -> list[tuple[str, dict]]:
    """
    Recursively searches root_dir for folders that contain a JSON file.
    Returns a list of (folder_name, json_data) tuples, sorted by folder name.
    """
    records = []

    for entry in sorted(root_dir.iterdir()):
        if not entry.is_dir():
            continue

        # Find any .json file inside this folder (takes the first one found)
        json_files = sorted(entry.glob("*.json"))
        if not json_files:
            print(f"  [skip] No JSON file found in: {entry.name}")
            continue

        json_path = json_files[0]
        if len(json_files) > 1:
            print(f"  [warn] Multiple JSON files in '{entry.name}', using: {json_path.name}")

        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            print(f"  [error] Could not parse {json_path}: {e}")
            continue
        except OSError as e:
            print(f"  [error] Could not read {json_path}: {e}")
            continue

        if not isinstance(data, dict):
            print(f"  [skip] JSON in '{entry.name}' is not a flat object (got {type(data).__name__})")
            continue

        records.append((entry.name, data))

    return records


def flatten_record(folder_name: str, data: dict) -> dict:
    """
    Prepends the folder name as 'datetime' column, then spreads all JSON keys.
    If any value is itself a dict/list it's serialised to a JSON string so the
    CSV stays readable.
    """
    row = {"datetime": folder_name}
    for key, value in data.items():
        if isinstance(value, (dict, list)):
            row[key] = json.dumps(value, ensure_ascii=False)
        else:
            row[key] = value
    return row


def write_csv(records: list[tuple[str, dict]], output_path: Path) -> None:
    rows = [flatten_record(folder, data) for folder, data in records]

    # Collect all unique column names, keeping 'datetime' first
    all_keys = ["datetime"]
    seen = {"datetime"}
    for row in rows:
        for k in row:
            if k not in seen:
                all_keys.append(k)
                seen.add(k)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"\n✓ Wrote {len(rows)} rows × {len(all_keys)} columns → {output_path}")


def main():
    # --- Argument handling ---
    args = sys.argv[1:]
    root_dir   = Path(args[0]) if len(args) >= 1 else Path(".")
    output_csv = Path(args[1]) if len(args) >= 2 else root_dir / "output.csv"

    if not root_dir.is_dir():
        print(f"Error: '{root_dir}' is not a directory.")
        sys.exit(1)

    print(f"Scanning: {root_dir.resolve()}")
    records = find_json_files(root_dir)

    if not records:
        print("No valid JSON records found. Nothing to write.")
        sys.exit(0)

    write_csv(records, output_csv)


if __name__ == "__main__":
    main()