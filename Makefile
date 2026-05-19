SHELL := /bin/bash
.DEFAULT_GOAL := help

# ── Configurable paths ────────────────────────────────────────────────────────
VENV      ?= $(SCRATCH)/venv_wa_hls4ml_models
PYTHON    := $(VENV)/bin/python
ARCHIVE   ?=
EXCLUDE   ?= run_20260504 run_20260505 run_20260506 run_20260507_143427
RAW_DATA  ?=
DATA_OUT  ?= $(SCRATCH)/catapult_asic_data
SPLIT_DIR ?= $(DATA_OUT)/split

# ── Training hyper-parameters (tuned for Catapult ASIC dataset, 1728 samples) ─
EPOCHS     ?= 500
LR         ?= 1e-4
BATCH      ?= 32
TRAIN_ARGS ?=

# ── Sentinel files ────────────────────────────────────────────────────────────
FEATURES_FILE  := $(DATA_OUT)/catapult_asic_features.npy
SPLIT_SENTINEL := $(SPLIT_DIR)/train_features.npy

# ── Targets ───────────────────────────────────────────────────────────────────
# Set REQUIRE_GPU=0 to allow training on CPU (e.g. for a quick smoke-test)
REQUIRE_GPU ?= 1

.PHONY: help check-env check-gpu env data split train all clean

help:
	@echo ""
	@echo "Usage: make <target> [VAR=value ...]"
	@echo ""
	@echo "Targets:"
	@echo "  env      Create venv and install extras  (run 'module load pytorch/2.8.0' first)"
	@echo "  data     Convert raw Catapult JSON reports → numpy arrays"
	@echo "  split    Split numpy arrays into train / val / test"
	@echo "  train    Train the Transformer surrogate model"
	@echo "  all      env + data + split + train"
	@echo "  clean    Remove generated numpy arrays and split files"
	@echo ""
	@echo "Variables (override with VAR=value on the command line):"
	@printf "  %-12s  Archive root with run_*/reports/ layout (preferred)\n" "ARCHIVE"
	@printf "  %-12s  Runs to skip when using ARCHIVE  [%s]\n" "EXCLUDE"   "$(EXCLUDE)"
	@printf "  %-12s  Flat dir of JSON reports (alternative to ARCHIVE)\n" "RAW_DATA"
	@printf "  %-12s  Output dir for numpy arrays    [%s]\n" "DATA_OUT"  "$(DATA_OUT)"
	@printf "  %-12s  Output dir for train/val/test  [%s]\n" "SPLIT_DIR" "$(SPLIT_DIR)"
	@printf "  %-12s  Path to virtual environment    [%s]\n" "VENV"      "$(VENV)"
	@printf "  %-12s  Training epochs                [%s]\n" "EPOCHS"    "$(EPOCHS)"
	@printf "  %-12s  Learning rate                  [%s]\n" "LR"        "$(LR)"
	@printf "  %-12s  Batch size                     [%s]\n" "BATCH"       "$(BATCH)"
	@printf "  %-12s  Require GPU for train (0=off)  [%s]\n" "REQUIRE_GPU" "$(REQUIRE_GPU)"
	@echo ""
	@echo "Example (GPU node):"
	@echo "  salloc -N 1 -C gpu -q interactive -t 01:00:00 --gpus-per-node=1 -A amsc011"
	@echo "  module load pytorch/2.8.0"
	@echo "  make env"
	@echo "  make data split ARCHIVE=/global/cfs/cdirs/amsc011/shared/wa-hls4ml-catapult"
	@echo "  make train EPOCHS=200"
	@echo ""
	@echo "Smoke-test on CPU (login node, no GPU required):"
	@echo "  make train EPOCHS=5 REQUIRE_GPU=0"

# Verify the venv's python can import torch (i.e. module is loaded and env exists)
check-env:
	@test -f $(PYTHON) || \
		(echo "ERROR: venv not found at $(VENV). Run 'module load pytorch/2.8.0' then 'make env'" && exit 1)
	@$(PYTHON) -c "import torch" 2>/dev/null || \
		(echo "ERROR: torch not importable in $(VENV). Run 'module load pytorch/2.8.0' then 'make env'" && exit 1)

# Check that a GPU is visible (skipped when REQUIRE_GPU=0)
check-gpu:
ifeq ($(REQUIRE_GPU),1)
	@$(PYTHON) -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" || \
		(echo "ERROR: no GPU visible. Request a GPU node with salloc, or set REQUIRE_GPU=0 to train on CPU." && exit 1)
	@$(PYTHON) -c "import torch; print('GPU:', torch.cuda.get_device_name(0))"
endif

# ── env ───────────────────────────────────────────────────────────────────────
env: $(PYTHON)

$(PYTHON):
	@test -n "$$SCRATCH" || (echo "ERROR: \$$SCRATCH is not set" && exit 1)
	python -m venv --system-site-packages $(VENV)
	$(PYTHON) -m pip install -q -r requirements_perlmutter.txt
	@echo "Venv ready: $(VENV)"

# ── data ──────────────────────────────────────────────────────────────────────
data: check-env $(FEATURES_FILE)

$(FEATURES_FILE):
ifeq ($(ARCHIVE),)
	@test -n "$(RAW_DATA)" || \
		(echo "ERROR: Set ARCHIVE=<archive-root> or RAW_DATA=<flat-dir>" && exit 1)
	$(PYTHON) dataset/catapult_asic_to_numpy.py \
		--input  $(RAW_DATA) \
		--output $(DATA_OUT) \
		--prefix catapult_asic
else
	$(PYTHON) dataset/catapult_asic_to_numpy.py \
		--archive $(ARCHIVE) \
		--exclude $(EXCLUDE) \
		--output  $(DATA_OUT) \
		--prefix  catapult_asic
endif

# ── split ─────────────────────────────────────────────────────────────────────
split: check-env $(SPLIT_SENTINEL)

$(SPLIT_SENTINEL): $(FEATURES_FILE)
	$(PYTHON) dataset/split_data.py \
		--features $(DATA_OUT)/catapult_asic_features.npy \
		--labels   $(DATA_OUT)/catapult_asic_labels.npy \
		--output   $(SPLIT_DIR)

# ── train ─────────────────────────────────────────────────────────────────────
train: check-env check-gpu $(SPLIT_SENTINEL)
	cd transformer && $(PYTHON) run.py \
		--arch transformer \
		--epochs $(EPOCHS) \
		--lr $(LR) \
		--batch-size $(BATCH) \
		--base-dir $(SPLIT_DIR) \
		$(TRAIN_ARGS)

# ── all ───────────────────────────────────────────────────────────────────────
all: env data split train

# ── clean ─────────────────────────────────────────────────────────────────────
clean:
	rm -rf $(DATA_OUT)
	@echo "Removed $(DATA_OUT)"
