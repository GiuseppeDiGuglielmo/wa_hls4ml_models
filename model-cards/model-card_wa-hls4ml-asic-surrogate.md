---
language:
- en
tags:
- project:genesis
- project:wa-hls4ml-asic
- type:model
- science:ASIC design, high-level synthesis, machine learning
- risk:general
license: CC-BY-NC-4.0
datasets:
    - https://github.com/GiuseppeDiGuglielmo/wa-hls4ml-search  # wa-hls4ml ASIC Synthesis Dataset (Catapult-ASIC-dev branch)
metrics:
    - r2_score
    - smape
    - rmse
---

# wa-hls4ml ASIC Surrogate Models

wa-hls4ml ASIC Surrogate is a family of regression models (Transformer and Graph
Neural Networks) that predict Siemens Catapult HLS synthesis outcomes — latency,
throughput (initiation interval), and ASIC cell area — for quantized dense neural
network accelerators generated via hls4ml, targeting Nangate 45 nm and
GlobalFoundries 22FDX FD-SOI technology nodes. The models enable rapid
design-space exploration without invoking the full HLS synthesis flow.

*Last Updated*: **2026-06-25**

## Developed by

[Arghya Ranjan Das](mailto:das214@purdue.edu), Purdue University

[Jason Weitz](mailto:jdweitz@ucsd.edu), University of California San Diego

[Giuseppe Di Guglielmo](https://orcid.org/0000-0002-5749-1432),
Fermi National Accelerator Laboratory — [gdg@fnal.gov](mailto:gdg@fnal.gov)

## Contributed by

[Benjamin Hawks](https://orcid.org/0000-0001-5700-0288),
Fermi National Accelerator Laboratory — [bhawks@fnal.gov](mailto:bhawks@fnal.gov)

## Model Changelog

+ **2026-06-25** Initial version (Catapult-ASIC-dev branch)

## Model short description

A family of Transformer and GNN-based surrogate models for ASIC synthesis
latency and area estimation of quantized dense neural networks generated via
hls4ml and Siemens Catapult HLS, with a sum-decomposition readout strategy
enabling depth extrapolation.

## Model description

Four surrogate model variants are provided, all predicting **latency** (clock
cycles) and **cell area** (µm²). **Throughput** (initiation interval) is computed
exactly as II = max_i II_i (the slowest streaming stage) rather than learned,
achieving exact R² = 1.000.

Each network layer is encoded as a 33-dimensional feature vector (input/output
widths, bitwidth, reuse factor, convolutional placeholders — standardised; layer
type, activation, padding — one-hot encoded). Two design-global directives
(hls4ml strategy, I/O type) are concatenated at the readout head.

### Model variants

| Variant | Architecture | Readout | Params | Latency R² | Area R² |
|---|---|---|---|---|---|
| **Transformer** | Standard encoder | CLS token → linear | 3.2 M | 0.9935 | 0.9905 |
| **GraphSAGE** | SAGEConv encoder | sum-pool → MLP | 0.59 M | 0.9925 | 0.9919 |
| **GATv2 (pooled)** | GATv2 encoder (54 M) | multi-pool → MLP | 54.47 M | **0.9997** | **0.9991** |
| **GATv2 (sum-decomp)** | GATv2 encoder (1.70 M) | exp → Σ per layer | 1.70 M | 0.9954† | 0.9797† |

† on depth-extrapolation task {1,2}→3 (deeper than training); pooled GATv2 drops
to 0.634 / 0.789 on the same task.

The **sum-decomposition** readout applies a weight-shared MLP head to each
per-layer node embedding and sums the exp-transformed outputs: ŷ = Σ exp(h(z_i)).
This constrains predictions to accumulate additively across layers, enabling
reliable depth extrapolation to networks much deeper than seen during training.

## Finetuned from model (optional)

Cross-node fine-tuning: a model pre-trained on Nangate 45 nm can be fine-tuned to
GF22FDX by freezing the encoder and training only the readout head (2.5% of
parameters). With a 45 nm prior trained on depths {1–5}, head-only fine-tuning on
~43K GF22FDX designs achieves latency R² = 0.995 and area R² = 0.970 on a
held-out GF22FDX 3-layer test set — matching or exceeding training from scratch.

See `--freeze-encoder` flag in `transformer/run.py` and `make finetune`.

## Model Type

Transformer encoder and Graph Neural Network (GraphSAGE, GATv2) regression
models implemented in PyTorch.

## Inputs and outputs

**Input**: A variable-length sequence of dense (QDense) layers, each described
by a 33-dimensional feature vector:
- Numerical (standardised): input width, output width, bitwidth b ∈ {4,6,8,10,12,14},
  ReuseFactor ∈ {1,4,8,16}, convolutional placeholders
- Categorical (one-hot): layer type, activation (relu/sigmoid/tanh), padding

Two design-global directives (strategy: Latency/Resource; I/O type: parallel/stream)
are concatenated at the readout head.

**Output**:
- **Latency** (clock cycles, input-valid to output-valid)
- **Cell area** (µm², total standard-cell area)
- **Throughput / II** computed exactly from ReuseFactor and clock period

## Compute Infrastructure

Trained on NERSC Perlmutter GPU nodes (SLURM, QoS `gpu_shared`, AmSC project
account `amsc011`).

### Hardware

[NERSC Perlmutter](https://www.nersc.gov/systems/perlmutter/) — NVIDIA A100 GPU
nodes.

### Software

PyTorch 2.8.0 (NERSC module `pytorch/2.8.0`). Training orchestrated via
`make train` / `sbatch slurm/train_surrogate.sh`.

Full requirements: `wa-hls4ml-models/requirements_perlmutter.txt`

## Papers and Scientific Outputs

*ASIC-specific paper in preparation — this card will be updated with the
corresponding reference and DOI upon publication.*

The following paper describes the predecessor FPGA/Vivado surrogate models and
the wa-hls4ml benchmark; the ASIC work presented here extends that approach to
ASIC technology nodes.

```bibtex
@misc{hawks2025wahls4mlbenchmarksurrogatemodels,
      title={wa-hls4ml: A Benchmark and Surrogate Models for hls4ml Resource and Latency Estimation},
      note={FPGA/Vivado surrogate — predecessor to this ASIC work},
      author={Benjamin Hawks and Jason Weitz and Dmitri Demler and Karla Tame-Narvaez and Dennis Plotnikov and Mohammad Mehdi Rahimifar and Hamza Ezzaoui Rahali and Audrey C. Therrien and Donovan Sproule and Elham E Khoda and Keegan A. Smith and Russell Marroquin and Giuseppe Di Guglielmo and Nhan Tran and Javier Duarte and Vladimir Loncar},
      year={2025},
      eprint={2511.05615},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2511.05615},
}
```

## Model License

The model and training code are available under the
[Creative Commons Attribution Non Commercial 4.0 International (CC-BY-NC-4.0)](https://creativecommons.org/licenses/by-nc/4.0/)
license. Note: the training dataset (synthesis outputs) is subject to separate
access restrictions — see the dataset card for details.

## Contact Info and Model Card Authors

[Giuseppe Di Guglielmo](https://orcid.org/0000-0002-5749-1432),
Fermi National Accelerator Laboratory — [gdg@fnal.gov](mailto:gdg@fnal.gov)


# Intended Uses

## Intended Use

Rapid ASIC synthesis latency and area estimation for quantized dense neural
networks generated via hls4ml and Siemens Catapult HLS, targeting Nangate 45 nm
and GlobalFoundries 22FDX FD-SOI. Intended to drive agentic hardware co-design
loops within the wa-hls4ml and Genesis projects, reducing design evaluation from
hours of HLS synthesis to milliseconds of inference.

### Primary Intended Users

Researchers and engineers within the wa-hls4ml and Genesis projects at Fermilab
and collaborating institutions performing neural network architecture search for
ASIC accelerators, including in extreme scientific environments (cryogenic qubit
readout, high-radiation).

### Mission Relevance

Part of the Genesis program effort to enable rapid custom ASIC design for
DOE-critical applications. Enables a path from trained neural networks to
cryogenic ASIC implementations via hls4ml and Catapult HLS, accelerating
design cycles for superconducting qubit readout and other extreme-environment
applications. Cross-node transfer from Nangate 45 nm to GF22FDX demonstrates
scalability to nodes whose PDKs are not yet ready for large synthesis campaigns.

## Out-of-Scope Use Cases

- Synthesis prediction for FPGA targets (use the FPGA/Vivado wa-hls4ml model).
- Non-dense layer types (convolutional, recurrent, etc.).
- Timing closure at block level or full-chip — these models predict HLS-level
  metrics only.


# How to use

## Install Instructions

```bash
git clone --recurse-submodules https://github.com/fastmachinelearning/wa-hls4ml-paper.git
cd wa-hls4ml-paper
git checkout Catapult-ASIC-dev
cd wa-hls4ml-models
make env   # creates venv and installs requirements_perlmutter.txt
```

## Training configuration

```bash
# Prepare data from the ASIC synthesis archive
# For dataset access, provenance, and coverage details see the complementary
# dataset card: wa-hls4ml-search/data-cards/genesis_datacard_wa_hls4ml_asic.yaml
make data split ARCHIVE=/global/cfs/cdirs/amsc011/shared/wa-hls4ml-catapult

# Train (default: 500 epochs, lr=1e-4, batch=32)
make train EPOCHS=500

# Resume from checkpoint
make train EPOCHS=200 TRAIN_ARGS="--resume /path/to/model.pt"

# Fine-tune encoder-frozen on a new technology node
make finetune RESUME=/path/to/45nm_model.pt EPOCHS=100

# Via SLURM on Perlmutter
sbatch slurm/train_surrogate.sh
```

Key hyperparameters (see `Makefile`):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `EPOCHS`  | 500     | Training epochs |
| `LR`      | 1e-4    | AdamW learning rate |
| `BATCH`   | 32      | Batch size |

## Inference configuration

```bash
cd transformer
python run.py \
    --arch transformer \
    --base-dir /path/to/split \
    --model-path /path/to/best_model/model.pt \
    --eval-only \
    --drop-throughput \
    --thruput-lookup ../dataset/thruput_lookup.pkl
```

# Code snippets of how to use the model

```python
# Run from inside wa-hls4ml-models/transformer/
import torch
from model import TransformerRegressor

model = TransformerRegressor(
    feature_dim=33,
    embed_dim=512,
    num_heads=8,
    ff_dim=512,
    num_layers=2,
    output_dim=2,   # latency + area (throughput derived analytically)
    max_layers=51,
    dropout=0.1,
)
model.load_state_dict(torch.load("best_model/model.pt", map_location="cpu"))
model.eval()

# x: (1, L, 33) — L layers, each with 33 normalised features
# pad_mask: (1, L) — False for real layers, True for padding
with torch.no_grad():
    preds = model(x, pad_mask)  # (1, 2) — [latency, area]
```


# Limitations

## Risks

No national security risks identified. The model predicts ASIC synthesis outcomes
for small quantized neural networks and has no dual-use concern.

## Limitations

- Models predict HLS-level metrics; post-routing timing and power are not
  estimated.


# Training details

## Training data

The models are trained on the **wa-hls4ml ASIC Synthesis Dataset**: ~527K
Siemens Catapult HLS synthesis results for quantized dense MLPs on Nangate 45 nm
(~484K designs, L=1–10) and GF22FDX (~43.5K designs, L=1–3).

Architecture coverage: depth L ∈ {1,...,10}; widths {4,8,16,32,64};
bitwidths {4,6,8,10,12,14} bits; activations relu/sigmoid/tanh;
ReuseFactor {1,4,8,16}. L≤3 exhaustive Cartesian; L≥4 Latin Hypercube Sampling
(N=2,500 unique architectures per depth × 4 RF).

Dataset card: `wa-hls4ml-search/data-cards/genesis_datacard_wa_hls4ml_asic.yaml`

Data location (NERSC CFS): `/global/cfs/cdirs/amsc011/shared/wa-hls4ml-catapult/`

## Training Procedure

1. JSON synthesis reports parsed and converted to NumPy arrays
   (`dataset/catapult_asic_to_numpy.py`, `make data`).
2. Random 70/15/15 train/val/test split applied to L=1,2,3 designs
   (303,660 training / 65,070 test for in-distribution evaluation).
3. Features z-score normalised using training-set statistics.
4. Throughput dropped from regression targets and derived analytically via
   `thruput_lookup.pkl` (exact, R²=1.000).
5. Models trained with AdamW, MSE loss on normalised log-space targets,
   with early stopping on validation loss.

### Reproducibility Information

- Optimizer: AdamW, lr=1e-4, default betas.
- Loss: MSE on normalised log-space targets.
- Reported results: mean over 4 random seeds (GNN variants).
- Training script: `slurm/train_surrogate.sh`.
- Machine: NERSC Perlmutter A100 GPU node, PyTorch 2.8.0.

## Pre-training information

- No external pre-training. Models trained from scratch on the ASIC dataset,
  except for cross-node fine-tuning experiments (45 nm → GF22FDX).
- For fine-tuning: encoder frozen, only readout head trained (2.5% of params).
- Stopping criterion: best validation loss (checkpoint saved per epoch).
- Batch size: 32. Dropout: 0.1 (Transformer encoder layers).


# Evaluation details

## Evaluation data

In-distribution: random 15% hold-out of L=1,2,3 designs (65,070 test designs).

Depth-extrapolation: trained on depths {1,2} or {1,2,3}, evaluated on held-out
deeper designs up to L=10.

Cross-node transfer: held-out GF22FDX 3-layer test set (8,000 designs), with
training on GF22FDX {1,2} only.

## Evaluation Procedure

R², SMAPE, and RMSE per target variable (latency, area) computed after
inverse-normalising predictions. Throughput always exact (R²=1.000).
GNN results reported as mean over 4 seeds.

## Uncertainty Quantification

Relative percentage error (RPE) distribution per target variable.
No formal uncertainty quantification (ensembles/Bayesian) implemented.

## Evaluation results

### In-distribution (random 70/15/15 split, L=1,2,3)

| Model | Params | Latency R² | Area R² |
|---|---|---|---|
| Transformer | 3.2 M | 0.9935 | 0.9905 |
| GraphSAGE (sum-pool) | 0.59 M | 0.9925 | 0.9919 |
| GATv2 (multi-pool) | 54.47 M | **0.9997** | **0.9991** |

Throughput: R² = 1.000 (exact analytical formula) for all models.

### Depth extrapolation ({1,2}→L, mean over 4 seeds)

| Train→Test | Pooled GATv2 54M Lat. | Pooled GATv2 54M Area | Sum-Decomp 1.70M Lat. | Sum-Decomp 1.70M Area |
|---|---|---|---|---|
| {1,2}→3  | 0.634  | 0.789  | **0.9954** | **0.9797** |
| {1,2}→6  | -0.448 | 0.239  | 0.983 | 0.949 |
| {1,2}→10 | -1.152 | -0.181 | 0.989 | 0.884 |
| {1,2,3}→10 | -0.615 | 0.273 | **1.000** | 0.962 |

### Cross-node transfer to GF22FDX ({1,2}→3, head-only fine-tuning, mean 4 seeds)

| Initialization | Sum-Decomp 1.70M Lat. | Sum-Decomp 1.70M Area |
|---|---|---|
| Random (scratch) | 0.978 | 0.950 |
| 45 nm {1,2} prior | 0.956 | 0.945 |
| 45 nm {1–5} prior | **0.995** | **0.970** |


# More Information

- Training and inference code: `wa-hls4ml-models/` (branch `Catapult-ASIC-dev`)
  at [https://github.com/fastmachinelearning/wa-hls4ml-paper](https://github.com/fastmachinelearning/wa-hls4ml-paper)
- Dataset generation code: `wa-hls4ml-search/` (branch `Catapult-ASIC-dev`)
  at [https://github.com/fastmachinelearning/wa-hls4ml-paper](https://github.com/fastmachinelearning/wa-hls4ml-paper)
- Synthesis dataset: NERSC CFS `/global/cfs/cdirs/amsc011/shared/wa-hls4ml-catapult/`
- Funded by the U.S. Department of Energy, Genesis program.
