#!/usr/bin/env python3
"""Replace every T with U in the cds_sequence column of a CSV.

Usage: python3 t2u.py input.csv output.csv
"""

import argparse
import csv
import sys


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input_file")
    ap.add_argument("output_file")
    args = ap.parse_args()

    with open(args.input_file, newline="", encoding="utf-8") as fin, open(
        args.output_file, "w", newline="", encoding="utf-8"
    ) as fout:
        reader = csv.DictReader(fin)
        if not reader.fieldnames or "cds_sequence" not in reader.fieldnames:
            print("Error: input file is missing the cds_sequence column", file=sys.stderr)
            sys.exit(1)
        writer = csv.DictWriter(fout, fieldnames=reader.fieldnames)
        writer.writeheader()
        for row in reader:
            if row.get("cds_sequence"):
                row["cds_sequence"] = row["cds_sequence"].replace("T", "U")
            writer.writerow(row)

    print(f"[t2u] Done: {args.input_file} -> {args.output_file}")


if __name__ == "__main__":
    main()
