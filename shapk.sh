#!/usr/bin/bash

for i in `seq 0 4`
do
{
    echo $(expr $i );
    python parameter_metrix_shapk.py $i
} &
done
wait