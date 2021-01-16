# Prescriptor Phase

Repo for developing prescriptors based on an RL agent.

## Setup environment:

### Base environment

Analogous commands with python `venv`:

```bash
conda create --name xprize_rl python=3.8
conda activate xprize_rl
cd $HERE
conda install -c conda-forge --file requirements.txt
```

### Installing transatlantic RL library

```bash
cd $HERE
pip install .
```

### Installing standard predictor from XPrize repo

```bash
cd $HERE
python ./update_xprize-repo.py
cd covid-xprize-uptodate
pip install .
```

### (Optional) Fix environment inconsistencies:

```bash
conda update --all
```