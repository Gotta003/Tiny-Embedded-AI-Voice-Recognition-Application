#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <wav_folder_path>"
    exit 1
fi

wav_folder="$1"
output_file="${wav_folder%/}/combined_results.txt"
temp_file="$(mktemp)"

hello_matteo_conv_count=0
hello_matteo_dense_count=0
user_not_enrolled_conv_count=0
user_not_enrolled_dense_count=0
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
    trimmed_line=$(echo "$line" | tr -d '\r' | sed 's/^[ \t]*//;s/[ \t]*$//')
    line_lower=$(echo "$trimmed_line" | tr '[:upper:]' '[:lower:]')

    if [[ "$line_lower" == *"hello matteo - conv"* ]]; then
        hello_matteo_conv_count=$((hello_matteo_conv_count + 1))
    elif [[ "$line_lower" == *"hello matteo - dense"* ]]; then
        hello_matteo_dense_count=$((hello_matteo_dense_count + 1))
    elif [[ "$line_lower" == *"user not enrolled - conv"* ]]; then
        user_not_enrolled_conv_count=$((user_not_enrolled_conv_count + 1))
    elif [[ "$line_lower" == *"user not enrolled - dense"* ]]; then
        user_not_enrolled_dense_count=$((user_not_enrolled_dense_count + 1))
    elif [[ "$line_lower" == *"not sheila word recognized"* ]]; then
        not_sheila_word_recognized_count=$((not_sheila_word_recognized_count + 1))
    fi
done < "$temp_file"

sheila_word_recognized_count=$((hello_matteo_conv_count + user_not_enrolled_conv_count))
total_files_processed=$(ls "$wav_folder"/*.wav 2>/dev/null | wc -l | tr -d ' ')

{
    echo ""
    echo "=== SUMMARY STATISTICS ==="
    echo "HELLO MATTEO CONV occurrences: $hello_matteo_conv_count"
    echo "HELLO MATTEO DENSE occurrences: $hello_matteo_dense_count"
    echo "USER NOT ENROLLED CONV occurrences: $user_not_enrolled_conv_count"
    echo "USER NOT ENROLLED DENSE occurrences: $user_not_enrolled_dense_count"
    echo "NOT SHEILA WORD RECOGNIZED: $not_sheila_word_recognized_count"
    echo "SHEILA WORD RECOGNIZED: $sheila_word_recognized_count"
    echo "Total files processed: $total_files_processed"
    echo ""
    echo "KWS Model ($total_files_processed samples):"
    echo "Found Sheila Rate: $((sheila_word_recognized_count*100/total_files_processed))% [$sheila_word_recognized_count]"
    echo "Sheila Not Found Rate: $((not_sheila_word_recognized_count*100/total_files_processed))% [$not_sheila_word_recognized_count]"
    echo ""
    echo "SV Model CONV ($sheila_word_recognized_count samples):"
    echo "Matteo Recognized Rate: $((hello_matteo_conv_count*100/sheila_word_recognized_count))% [$hello_matteo_conv_count]"
    echo "Other User Rate: $((user_not_enrolled_conv_count*100/sheila_word_recognized_count))% [$user_not_enrolled_conv_count]"
    echo ""
    echo "SV Model DENSE ($sheila_word_recognized_count samples):"
    echo "Matteo Recognized Rate: $((hello_matteo_dense_count*100/sheila_word_recognized_count))% [$hello_matteo_dense_count]"
    echo "Other User Rate: $((user_not_enrolled_dense_count*100/sheila_word_recognized_count))% [$user_not_enrolled_dense_count]"
} >> "$output_file"

rm -f "$temp_file"

echo "Processing complete. Results saved to $output_file"
echo ""
echo "=== SUMMARY STATISTICS ==="
echo "HELLO MATTEO CONV occurrences: $hello_matteo_conv_count"
echo "HELLO MATTEO DENSE occurrences: $hello_matteo_dense_count"
echo "USER NOT ENROLLED CONV occurrences: $user_not_enrolled_conv_count"
echo "USER NOT ENROLLED DENSE occurrences: $user_not_enrolled_dense_count"
echo "NOT SHEILA WORD RECOGNIZED: $not_sheila_word_recognized_count"
echo "SHEILA WORD RECOGNIZED: $sheila_word_recognized_count"
echo "Total files processed: $total_files_processed"
echo ""
echo "KWS Model ($total_files_processed samples):"
echo "Found Sheila Rate: $((sheila_word_recognized_count*100/total_files_processed))% [$sheila_word_recognized_count]"
echo "Sheila Not Found Rate: $((not_sheila_word_recognized_count*100/total_files_processed))% [$not_sheila_word_recognized_count]"
echo ""
echo "SV Model CONV ($sheila_word_recognized_count samples):"
echo "Matteo Recognized Rate: $((hello_matteo_conv_count*100/sheila_word_recognized_count))% [$hello_matteo_conv_count]"
echo "Other User Rate: $((user_not_enrolled_conv_count*100/sheila_word_recognized_count))% [$user_not_enrolled_conv_count]"
echo ""
echo "SV Model DENSE ($sheila_word_recognized_count samples):"
echo "Matteo Recognized Rate: $((hello_matteo_dense_count*100/sheila_word_recognized_count))% [$hello_matteo_dense_count]"
echo "Other User Rate: $((user_not_enrolled_dense_count*100/sheila_word_recognized_count))% [$user_not_enrolled_dense_count]"
