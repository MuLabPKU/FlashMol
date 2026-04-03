#!/bin/bash
NPROC=${1:-$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)}
torchrun --standalone --nproc_per_node=$NPROC main_geom_dmd.py "${@:2}"
