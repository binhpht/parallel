#!/bin/bash

OUT=parallel_stdout.txt
APP=$1
DUR=$2

make $APP
echo "Running for ${DUR} seconds."
timeout ${DUR} unbuffer ./${APP} > ${OUT}
python3 print_average.py
