#!/bin/bash

# Usage information and argument handling
usage() {
    echo "Usage: \$0 [-d delimiter] input_file output_file"
    echo "Examples: "
    echo "  TSV input: \$0 -d $'\t' input.tsv output.tsv"
    echo "  CSV input: \$0 -d ',' input.csv output.csv"
    exit 1
}

# Create the temp directory and the cleanup function
# Temp directory lives under ./tmp in the working directory and is deleted when done
mkdir -p ./tmp
TEMP_DIR=$(mktemp -d ./tmp/cdhit.XXXXXX)
cleanup() {
    rm -rf "$TEMP_DIR"
}
trap cleanup EXIT

# Default delimiter is a comma
DELIM=","

# Parse command-line options
while getopts "d:" opt; do
    case $opt in
        d) DELIM=$OPTARG ;;
        *) usage ;;
    esac
done
shift $((OPTIND-1))

# Check the number of remaining arguments
if [ $# -ne 2 ]; then
    usage
fi

input_file=$1
output_file=$2

# Generate unique temporary file names
fasta_file="$TEMP_DIR/seq_$$.fasta"
out_fasta_file="$TEMP_DIR/seq-out_$$.fasta"
kept_lines="$TEMP_DIR/kept_lines_$$.txt"
cdhit_log="$TEMP_DIR/cdhit_$$.log"

# Safe method to check that the column exists
header=$(head -1 "$input_file")
if ! echo "$header" | tr "$DELIM" '\n' | grep -qx "protein_sequence"; then
    echo "Error: input file is missing the 'protein_sequence' column"
    exit 1
fi

# Get the column index (portable across shells)
cds_col=$(echo "$header" | awk -v delim="$DELIM" 'BEGIN {FS=delim} {for(i=1;i<=NF;i++) if($i=="protein_sequence") print i}')
#cds_col=$(echo "$header" | awk -v delim="$DELIM" 'BEGIN {FS=delim} {for(i=1;i<=NF;i++) if($i=="protein_binder") print i}')
if [ -z "$cds_col" ]; then
    echo "Error: cannot locate the 'protein_sequence' column"
    exit 1
fi

# Generate the FASTA file (handles fields containing special characters)
awk -v delim="$DELIM" -v col="$cds_col" '
BEGIN {FS=delim}
NR>1 {
    gsub(/[[:space:]]+/, "", $col)  # Strip any whitespace
    printf ">%d\n%s\n", NR, $col
}' "$input_file" > "$fasta_file"

# CD-HIT clustering
cd-hit -i "$fasta_file" -o "$out_fasta_file" -c 0.9 -n 5 -T 16 2>&1 | tee "$cdhit_log"

# Extract the line numbers to keep (for performance)
grep '^>' "$out_fasta_file" | sed 's/^>//g' > "$kept_lines"

# Generate the final result file (optimized - reads the input only once)
{
    head -1 "$input_file"  # Write the header line
    awk 'BEGIN {
            # Load the line numbers to keep into an array
            while (getline line_num < "'$kept_lines'") {
                keep[line_num] = 1
            }
            close("'$kept_lines'")
        }
        FNR > 1 && keep[FNR] {print}' "$input_file"
} > "$output_file"


echo "Processing complete, results saved to $output_file"
echo "Records kept:  $(wc -l < "$kept_lines")"
# Cleanup is handled automatically by the trap; no need to delete temp files manually

