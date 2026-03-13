#!/bin/bash

for i in $(seq 1 10)
do
nohup ./scascacscasc.sh > log_new_execs/log_tests_bertscore_llama_antigo_exec_$i.txt 2>&1
done

for i in $(seq 1 10)
do
nohup ./safasca.sh > log_new_execs/log_tests_bertscore_llama_novo_exec_$i.txt 2>&1
done