#!/bin/bash

cd ..

cd lmcache-server/
 python -m lmcache_server.server \
 192.168.2.29 65432 ./
