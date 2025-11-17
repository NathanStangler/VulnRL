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