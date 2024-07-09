#!/bin/bash

#SBATCH --job-name=jupyter
#SBATCH --partition=a100        # 계산노드 종류 선택: a6000 or a100
#SBATCH --gres=gpu:1             # Use 1 GPU
#SBATCH --time=10-04:30:00        # d-hh:mm:ss 형식, 본인 job의 max time limit 지정
#SBATCH --mem=48000              # cpu memory size
#SBATCH --cpus-per-task=8        # cpu 개수
#SBATCH --output=jupyter_test.txt

ml cuda/11.3                # 필요한 쿠다 버전 로드
eval "$(conda shell.bash hook)"  # Initialize Conda Environment
conda activate 23_Research             # Activate your conda environment


srun jupyter notebook --no-browser --port=1822