#!/bin/bash

for i in $(seq 2 4 22); do
    ./start-experiment.sh "rag_benchmark_$i" 10 20 false
done