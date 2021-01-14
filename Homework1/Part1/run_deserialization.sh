#!/usr/bin/env bash

## Path to data files
data_path="/Users/nguyjust/CS662_NLP/Homework1/Part1/GW-cna_eng/*"


## Loop through data files and run deserialization.py script
for xmlFile in $data_path; do
    python3 deserialization.py $xmlFile

    ## WC to check appending after processing each file
    echo "Processed $xmlFile line count $(cat /Users/nguyjust/CS662_NLP/Homework1/Part1/deserialized.txt | wc -l)"
done
