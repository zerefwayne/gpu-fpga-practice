#!/bin/bash

for i in {10..90..10}; do
    python3 generate.py "$i"
done

for i in {100..1000..100}; do
    python3 generate.py "$i"
done
