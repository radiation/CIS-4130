#!/bin/sh

DATA_HOME=/home/bryan_choate/data

for data_file in `ls ${DATA_HOME}/*.json`
do
    echo "Processing ${data_file}"
    python data_cleaning_refactored.py ${data_file}
done