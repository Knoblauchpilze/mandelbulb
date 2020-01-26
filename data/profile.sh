#!/bin/sh

CURR_DIR=$(dirname $0)
echo $(cat data/config/local)
nvprof --analysis-metrics --profile-from-start off -o /home/knoblauch/IAmDeveloper/mandelbulb/mandelbulb_%p.nvprof ./bin/mandelbulb $(cat data/config/local)
