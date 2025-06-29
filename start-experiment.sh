#!/bin/bash

source ~/venv/bin/activate

cd frontend

rm -rf reports/*

python -m run_experiment.py "$@"