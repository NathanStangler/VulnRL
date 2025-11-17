#!/bin/bash        

#SBATCH --job-name=VulnRL
#SBATCH -p msigpu
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=20G
#SBATCH --time=1:00:00
#SBATCH --output=./logs/%j.log

module load python3/3.10.9_anaconda2023.03_libmamba cuda/12.0
cd ~/VulnRL
if [ ! -d .venv ]; then
    python -m venv .venv
fi
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
export PYTORCH_ALLOC_CONF=expandable_segments:True
python finetuned.py