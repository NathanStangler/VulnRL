# VulnRL

## Install

1. Clone this repository:

```bash
git clone git@github.com:NathanStangler/VulnRL.git
```

2. Install packages:

```bash
pip install --upgrade pip
pip install -r requirements.txt
sudo apt update && sudo apt install -y clang clang-tidy lld clang-tools
```

## MSI

### Setting resouces
Modify in script.sbatch:
```
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=20G
#SBATCH --time=1:00:00
```

### Queue default finetune job

```bash
sbatch script.sbatch
```

### Configure job

Use --export to pass variables

Variables:
- RUN_MODE: finetune | evaluate | performance
- MODEL_NAME: HuggingFace model
- OUTPUT_DIR: output path
- LOG_DIR: log path
- EPOCHS, TRAIN_BATCH_SIZE, EVAL_BATCH_SIZE, LR
- USE_LORA: true | false
- LOAD_IN_4BIT: true | false
- REPORT_TO: wandb | none

### Examples

Finetune custom config:
```bash
sbatch --export=RUN_MODE=finetune,MODEL_NAME=Qwen/Qwen2.5-Coder-1.5B-Instruct,EPOCHS=3,TRAIN_BATCH_SIZE=8,EVAL_BATCH_SIZE=8,LR=1e-5,USE_LORA=true,LOAD_IN_4BIT=true script.sbatch
```

Evaluate model:
```bash
sbatch --export=RUN_MODE=evaluate,OUTPUT_DIR=./finetuned_model script.sbatch
```

Performance evaluation:
```bash
sbatch --export=RUN_MODE=performance,OUTPUT_DIR=./finetuned_model script.sbatch
```

### Manage jobs

Check queue status

```bash
squeue --me
```

Cancel queued job

```bash
scancel <JOBID>
```

## API Server

1. Launch the FastAPI vulnerability analysis endpoint:

```bash
python api.py
```

2. Then send a .cpp file for analysis:

```bash
curl -X POST -F "file=@test.cpp" http://localhost:8080/analyze/
```

## Test

Run test cases:

```bash
pytest
```