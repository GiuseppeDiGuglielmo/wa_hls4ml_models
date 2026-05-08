# wa-hls4ml Surrogate Models

This repository contains surrogate models and utilities for the wa-hls4ml dataset.

## **Repository Structure**

### **Directories**
- `transformer/`: Contains transformer based model and utilities.
  - `data.py`: Data preprocessing scripts.
  - `model.py`: Transformer model definitions.
  - `train.py`: Training scripts for the transformer model.
  - `requirements.txt`: Dependencies for transformer workflow.

- `GNN/`: Contains Graph Neural Network (GNN) model and utilities.
  - `DatasetMay29Complete.py`: Dataset preparation for GNN model.
  - `Models.py`: GNN model definitions.
  - `training_scripts/`: Scripts for training the GNN model.
  - `requirements.txt`: Dependencies for GNN workflows.

- `dataset/`: Contains dataset preparation scripts.
  - `Dataset_to_csvs6_with_ii.py`: Converts datasets to CSV format.

### **Files**
- `5_26_requirements.txt`: Dependencies for the project.
- `requirements_perlmutter.txt`: Extra packages needed on Perlmutter (on top of the NERSC pytorch module).
- `.gitignore`: Specifies files and directories to ignore.

## **Perlmutter Setup (NERSC)**

The NERSC `pytorch` module already provides torch, torch-geometric, numpy, pandas, scikit-learn, matplotlib, and tqdm. Only `plotly` needs to be added.

Create a thin virtual environment on `$SCRATCH` that inherits the module's packages:

```bash
module load pytorch/2.8.0
python -m venv --system-site-packages $SCRATCH/venv_wa_hls4ml_models
source $SCRATCH/venv_wa_hls4ml_models/bin/activate
pip install -r requirements_perlmutter.txt
```

Activate before every training or evaluation session:

```bash
module load pytorch/2.8.0
source $SCRATCH/venv_wa_hls4ml_models/bin/activate
```

Training on a GPU node (interactive):

```bash
salloc -N 1 -C gpu -q interactive -t 01:00:00 --gpus-per-node=1 -A <account>
module load pytorch/2.8.0
source $SCRATCH/venv_wa_hls4ml_models/bin/activate
cd transformer
python run.py --arch transformer --epochs 200 --lr 1e-5 --batch-size 512 --base-dir <path/to/split>
```
