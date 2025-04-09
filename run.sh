#!/bin/bash
python ./multi_task/train_multi_task.py --param_file=./sample.json
python ./multi_task/train_multi_task.py --param_file=./sampleVR.json
python ./multi_task/train_multi_task.py --param_file=./sampleVRM.json
python ./multi_task/train_multi_task.py --param_file=./sampleVRP.json
python ./multi_task/train_multi_task.py --param_file=./sampleVRMP.json
python ./multi_task/train_multi_task.py --param_file=./sampleMOCO.json
python ./multi_task/train_multi_task.py --param_file=./sampleSGDA.json




