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

## Test

Run test cases:

```bash
pytest
```

## MSI

Modify `script.sbatch` to have needed gpu, memory, and time.

- Queue job

```bash
sbatch script.sbatch
```

- Check queue status

```bash
squeue --me
```

- Cancel queued job

```bash
scancel <JOBID>
```

## Usage

Run the full training + evaluation pipeline:

```bash
python train_and_evaluate.py
```

This will:
- Load and tokenize the vulnerability dataset
- Fine-tune a code language model (default: microsoft/phi-2)
- Evaluate test accuracy
- Run compiler feedback on example C++ code
- Log results to ./logs/summary.json and feedback_logs.jsonl

## API Server

1. Launch the FastAPI vulnerability analysis endpoint:

```bash
python api.py
```

2. Then send a .cpp file for analysis:

```bash
curl -X POST -F "file=@test.cpp" http://localhost:8080/analyze/
```