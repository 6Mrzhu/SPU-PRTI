#!/usr/bin/env bash
# script_path=$(cd `dirname $0`; pwd)
# cd $script_path

in_data_dir='../data/PU1K/test/input_512/input_512'
gt_data_dir='../data/PU1K/test/input_2048/gt_8192'

num_shape_point=2048

Model="../network/model/demo/release/demo/model_99.pth"

python main.py --phase test --ckpt ${Model}  --num_shape_point ${num_shape_point} --test_data  ${in_data_dir}


