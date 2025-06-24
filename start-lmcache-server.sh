#!/bin/bash

source ~/venv/bin/activate

cd ..

cd lmcache-server/
 python -m lmcache_server.server \
 192.168.2.27 64321 ./