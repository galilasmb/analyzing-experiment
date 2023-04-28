#!/bin/bash

rm -r resultTime*
rm -r results_*
python3 scripts/generate_time.py
python3 scripts/generate_result_time.py
