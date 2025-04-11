#!/bin/bash

# Usage: ./rename_files.sh /path/to/master_folder

master_folder="$1"

if [ -z "$master_folder" ]; then
    echo "Error: Please provide the master folder path."
    echo "Usage: $0 /path/to/master_folder"
    exit 1
fi

if [ ! -d "$master_folder" ]; then
    echo "Error: Directory '$master_folder' does not exist."
    exit 1
fi

# Process each subfolder
find "$master_folder" -mindepth 1 -maxdepth 1 -type d | while read -r subfolder; do
    subfolder_name=$(basename "$subfolder")
    
    echo "Processing files in: $subfolder_name"
    
    # Process each file in the subfolder
    find "$subfolder" -maxdepth 1 -type f | while read -r file; do
        filename=$(basename "$file")
        dirpath=$(dirname "$file")
        
        new_name="$dirpath/$subfolder_name.$filename"
        
        # Rename the file
        mv "$file" "$new_name"
        echo "Renamed: $filename → $subfolder_name.$filename"
    done
done

echo "Renaming complete!"
