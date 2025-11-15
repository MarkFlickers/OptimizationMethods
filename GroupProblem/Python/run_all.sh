#!/bin/bash

for file in BIRDS*.txt; do
    logfile="log_${file}.log"
    python3 birdparser.py "$file" > "$logfile" 2>&1
    echo "Processed $file -> $logfile"
done
