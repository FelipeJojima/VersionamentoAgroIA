#!/bin/bash

# echo "----------------------INICIANDO AVALIAÇÃO ROUGE-L ------------------------------"
# for i in $(seq 1 30)
# do
#     python3 rouge_l.py $i
# done


echo "----------------------INICIANDO AVALIAÇÃO ROUGE-1 ------------------------------"
for i in $(seq 1 30)
do
    python3 rouge_1.py $i
done
