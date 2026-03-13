#!/bin/bash

echo "-----------------------------Iniciando execução sequêncial:----------------------------------"
for i in $(seq 1 2)
do
    for j in $(seq 1 2)
    do
        VALOR_CALCULADO=$((2 * i))
        python3 testes.py 1 $VALOR_CALCULADO $j 30 $1  
    done
done
echo "-----------------------------Iniciando execução individual:----------------------------------"
echo "------- Pergunta 01 ------------"
for i in $(seq 1 2)
do
    for j in $(seq 1 2)
    do
        VALOR_CALCULADO=$((2 * i))
        python3 testes.py 0 $VALOR_CALCULADO $j 30 $1 1
    done
done
echo "------- Pergunta 02 ------------"
for i in $(seq 1 2)
do
    for j in $(seq 1 2)
    do
        VALOR_CALCULADO=$((2 * i))
        python3 testes.py 0 $VALOR_CALCULADO $j 30 $1 2
    done
done
echo "------- Pergunta 03 ------------"
for i in $(seq 1 2)
do
    for j in $(seq 1 2)
    do
        VALOR_CALCULADO=$((2 * i))
        python3 testes.py 0 $VALOR_CALCULADO $j 30 $1 3
    done
done
echo "------- Pergunta 05 ------------"
for i in $(seq 1 2)
do
    for j in $(seq 1 2)
    do
        VALOR_CALCULADO=$((2 * i))
        python3 testes.py 0 $VALOR_CALCULADO $j 30 $1 4
    done
done
echo "------- Pergunta 08 ------------"
for i in $(seq 1 2)
do
    for j in $(seq 1 2)
    do
        VALOR_CALCULADO=$((2 * i))
        python3 testes.py 0 $VALOR_CALCULADO $j 30 $1 5
    done
done
echo "------- Pergunta 09 ------------"
for i in $(seq 1 2)
do
    for j in $(seq 1 2)
    do
        VALOR_CALCULADO=$((2 * i))
        python3 testes.py 0 $VALOR_CALCULADO $j 30 $1 6
    done
done
echo "------- Pergunta 10 ------------"
for i in $(seq 1 2)
do
    for j in $(seq 1 2)
    do
        VALOR_CALCULADO=$((2 * i))
        python3 testes.py 0 $VALOR_CALCULADO $j 30 $1 7
    done
done
echo "------- Pergunta 11 ------------"
for i in $(seq 1 2)
do
    for j in $(seq 1 2)
    do
        VALOR_CALCULADO=$((2 * i))
        python3 testes.py 0 $VALOR_CALCULADO $j 30 $1 8
    done
done
echo "------- Pergunta 12 ------------"
for i in $(seq 1 2)
do
    for j in $(seq 1 2)
    do
        VALOR_CALCULADO=$((2 * i))
        python3 testes.py 0 $VALOR_CALCULADO $j 30 $1 9
    done
done
echo "------- Pergunta 14 ------------"
for i in $(seq 1 2)
do
    for j in $(seq 1 2)
    do
        VALOR_CALCULADO=$((2 * i))
        python3 testes.py 0 $VALOR_CALCULADO $j 30 $1 10
    done
done
