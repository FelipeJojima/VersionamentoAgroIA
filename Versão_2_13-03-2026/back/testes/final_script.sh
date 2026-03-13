#!/bin/bash

for i in $(seq 1 30)
do
    nohup ./script.sh $i > logs_testes/log_exec_ind_${i}.txt 2>&1 
done
