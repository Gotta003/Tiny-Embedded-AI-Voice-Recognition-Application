#!/bin/bash

# Check if folder path was provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <wav_folder_path>"
    exit 1
fi

wav_folder="$1"
output_file="${wav_folder}/combined_results.txt"
temp_file="${wav_folder}/temp_results.txt"

gcc src/*.c -o ndp_model -I./include -I/opt/homebrew/include -L/opt/homebrew/lib -lportaudio -lfftw3 -lm

# Initialize counters
hello_matteo_count=0
user_not_enrolled_count=0

# Clear previous output files
> "$output_file"
> "$temp_file"

# Process each WAV file in the folder
for wav_file in "$wav_folder"/*.wav; do
    if [ -f "$wav_file" ]; then
        # Get the base filename
        base_name=$(basename "$wav_file")
        
        echo "Processing: $base_name" | tee -a "$temp_file"
        
        # Run the model and capture output
        ./ndp_model 1 "$wav_file" | tee -a "$temp_file"
        
        # Check the exit status
        if [ $? -eq 0 ]; then
            echo "----- SUCCESS -----" | tee -a "$temp_file"
        else
            echo "----- ERROR -----" | tee -a "$temp_file"
        fi
        
        echo -e "\n" | tee -a "$temp_file"
    fi
done

# Process the results and count occurrences
while read -r line; do
    # Write all lines to output file
    echo "$line" >> "$output_file"
    
    # Count patterns (case insensitive)
    if [[ "$line" =~ [Hh][Ee][Ll][Ll][Oo].*[Mm][Aa][Tt][Tt][Ee][Oo] ]]; then
        ((hello_matteo_count++))
    elif [[ "$line" =~ [Uu][Ss][Ee][Rr].*[Nn][Oo][Tt].*[Ee][Nn][Rr][Oo][Ll][Ll][Ee][Dd] ]]; then
        ((user_not_enrolled_count++))
    fi
done < "$temp_file"

# Append summary statistics
echo -e "\n\n=== SUMMARY STATISTICS ===" >> "$output_file"
echo "HELLO MATTEO occurrences: $hello_matteo_count" >> "$output_file"
echo "USER NOT ENROLLED occurrences: $user_not_enrolled_count" >> "$output_file"
echo "Total files processed: $(ls "$wav_folder"/*.wav 2>/dev/null | wc -l)" >> "$output_file"

# Clean up temporary file
rm "$temp_file"

echo "Processing complete. Results saved to $output_file"
echo "HELLO MATTEO: $hello_matteo_count"
echo "USER NOT ENROLLED: $user_not_enrolled_count"
