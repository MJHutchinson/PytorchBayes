#!/usr/bin/env bash

python parameter_sweep_regression.py -c ./config/single_run/wine-quality-red.yaml -ds wine-quality-red -ld ./results/ -dd ./data_dir/ -cm MLP-test &