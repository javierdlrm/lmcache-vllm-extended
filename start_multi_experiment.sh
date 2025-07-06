#!/bin/bash

for i in {1..4}; do
    ./start-experiment.sh "task3_question1_$i" 10 3 true
done