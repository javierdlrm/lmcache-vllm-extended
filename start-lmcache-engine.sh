#!/bin/bash

cd ..

cd lmcache-vllm-extended
 LMCACHE_CONFIG_FILE="configuration.yaml" \
 CUDA_VISIBLE_DEVICES=0 \
 python lmcache_vllm/script.py serve \
 Qwen/Qwen2.5-1.5B-Instruct \--gpu-memory-utilization 0.8 \--dtype half \--port 8000