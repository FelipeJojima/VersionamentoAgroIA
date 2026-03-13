#!/bin/bash

for i in $(seq 1 10)
do
    python3 model_for_tests_novo.py $i
done
