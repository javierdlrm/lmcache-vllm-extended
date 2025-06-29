#!/bin/bash

source ~/venv/bin/activate

cd frontend

rm -rf reports/*

python -m run_experiment "$@" 2>&1 | tee reports/output.log