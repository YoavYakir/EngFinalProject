#!/bin/bash

# Outer loop from 1 to 3
for i in {1..3}
do
    # Middle loop from 1 to 3
    for j in {1..4}
    do
        # Inner loop from 1 to 3
        for k in {1..3}
        do
            echo "Killing python scripts"
            ps aux | grep python | grep -v "grep python" | awk '{print $2}' | xargs kill -9
            
            # Call the Python script with i, j, and k as arguments (converted to strings)
            echo "Calling python script with i=$i, j=$j, k=$k"
            python -m EngFinalProject.tests.clean_python.run_all_tests "$i" "$j" "$k"
        done
    done
done