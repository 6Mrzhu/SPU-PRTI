#!/usr/bin/env bash
script_path=$(cd `dirname $0`; pwd)
cd $script_path

in_data_dir='../data/PU1K/test/input_2048/input_2048'
gt_data_dir='../data/PU1K/test/input_2048/gt_8192'

num_shape_point=2048

Model="../network/model/demo/release3/demo/model_99.pth"

cd ../evaluation_code
bash eval_pu1k.sh
cd $script_path

python main.py --phase eval --ckpt ${Model} --num_shape_point ${num_shape_point} --test_data  ${in_data_dir} --gt_path  ${gt_data_dir}
