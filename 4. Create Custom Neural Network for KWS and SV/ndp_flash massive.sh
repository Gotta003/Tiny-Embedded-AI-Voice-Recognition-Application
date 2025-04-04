#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <wav_folder_path>"
    exit 1
fi

wav_folder="$1"
output_file="${wav_folder%/}/combined_results.txt"
temp_file="$(mktemp)"

hello_matteo_count=0
user_not_enrolled_count=0
not_sheila_word_recognized_count=0

echo "Compiling model..."
    gcc src/*.c -o ndp_model -I./include -I/opt/homebrew/include -L/opt/homebrew/lib -lportaudio -lfftw3 -lm || {
        echo "Compilation failed"
        exit 1
    }

: > "$output_file"

shopt -s nullglob
for wav_file in "$wav_folder"/*.wav; do
    base_name=$(basename "$wav_file")
    
    {
        echo "Processing: $base_name"
        if ./ndp_model 1 "$wav_file"; then
            echo "----- SUCCESS -----"
        else
            echo "----- ERROR -----" >&2
        fi
        
        echo -e "\n"
    } | tee -a "$temp_file"
done

while IFS= read -r line; do
    echo "$line" >> "$output_file"
    line_lower=$(echo "$line" | tr '[:upper:]' '[:lower:]')
    
    # Count patterns
    if [[ "$line_lower" == *"hello"* ]] && [[ "$line_lower" == *"matteo"* ]]; then
        hello_matteo_count=$((hello_matteo_count + 1))
    elif [[ "$line_lower" == *"user not enrolled"* ]]; then
        user_not_enrolled_count=$((user_not_enrolled_count + 1))
    elif [[ "$line_lower" == *"not sheila word recognized"* ]]; then
        not_sheila_word_recognized_count=$((not_sheila_word_recognized_count + 1))
    fi
done < "$temp_file"

sheila_word_recognized_count=$((hello_matteo_count + user_not_enrolled_count))
total_files_processed=$(ls "$wav_folder"/*.wav 2>/dev/null | wc -l | tr -d ' ')

{
    echo ""
    echo "=== SUMMARY STATISTICS ==="
    echo "HELLO MATTEO occurrences: $hello_matteo_count"
    echo "USER NOT ENROLLED occurrences: $user_not_enrolled_count"
    echo "NOT SHEILA WORD RECOGNIZED: $not_sheila_word_recognized_count"
    echo "SHEILA WORD RECOGNIZED: $sheila_word_recognized_count"
    echo "Total files processed: $total_files_processed"
} >> "$output_file"

rm -f "$temp_file"

echo "Processing complete. Results saved to $output_file"
echo ""
echo "=== FINAL RESULTS ==="
echo "HELLO MATTEO: $hello_matteo_count"
echo "USER NOT ENROLLED: $user_not_enrolled_count"
echo "NOT SHEILA WORD RECOGNIZED: $not_sheila_word_recognized_count"
echo "SHEILA WORD RECOGNIZED: $sheila_word_recognized_count"
echo "Total files processed: $total_files_processed"
