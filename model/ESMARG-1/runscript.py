import subprocess
import os
'''Different combinations of negative samples were used for training, and each combination randomly selected six different negative sample data'''
f_combinations = [(5, 0), (7, 0), (10, 0), (20, 0)]


for i in range(1, 7):
    print(f"Running for iteration {i}")

    for f1, f2 in f_combinations:
        print(f"Running traincuda.py - Iteration {i} with f1={f1} and f2={f2}")
        subprocess.run(f"CUDA_VISIBLE_DEVICES=1 python traincuda.py --num {i} --f1 {f1} --f2 {f2}", shell=True)


