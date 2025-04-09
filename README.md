# Variance Reduction Method in Multi-objective Optimization Algorithm

This document provides an overview of the setup and requirements for running a multi-objective optimization algorithm that employs a variance reduction method.
sampleVR.json corresponds to the STIMULUS algorithm, sampleVRM.json to STIMULUS-M.
## Requirements

To successfully run the code, specific versions of Python and some key packages are required:

Python==3.7
pillow==6.2.2
scipy==1.2.2
numpy==1.16.4
pandas==0.24.2
autograd==1.3
torch==1.3.0
scikit-learn==0.21.2
cvxopt==1.2.3


To train models using the multi-task optimization approach, execute the provided `run.sh` script. This script sequentially trains models using different parameter files, as outlined below:

```bash
#!/bin/bash
python ./multi_task/train_multi_task.py --param_file=./sample.json
python ./multi_task/train_multi_task.py --param_file=./sampleVR.json
python ./multi_task/train_multi_task.py --param_file=./sampleVRM.json
python ./multi_task/train_multi_task.py --param_file=./sampleSGDA.json
