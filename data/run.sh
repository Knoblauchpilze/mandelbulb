#!/bin/sh

CURR_DIR=$(dirname $0)
echo $(cat data/config/local)
./bin/mandelbulb $(cat data/config/local)
