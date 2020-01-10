#!/bin/bash

# Generate Results for Part3 PTSD
for j in {0,1}
do
	for i in {0..34}
	do 
		python main.py ageMatched PTSD $i 10 $j
	done
done
