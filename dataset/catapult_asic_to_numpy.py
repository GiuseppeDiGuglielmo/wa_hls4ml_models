import argparse
import glob
import json
import logging
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Reuse the same feature schema as Dataset_to_csvs6_with_ii.py
FEATURES = {
    "d_in1": 0, "d_in2": 1, "d_in3": 2,
    "d_out1": 3, "d_out2": 4, "d_out3": 5,
    "prec": 6, "rf": 7, "strategy": 8,
    "layer_type": 9, "activation_type": 10,
    "filters": 11, "kernel_size": 12, "stride": 13,
    "padding": 14, "pooling": 15, "batchnorm": 16, "io_type": 17,
}

LAYER_TYPE_MAPPING = {
    'Dense': 1,
    # activation layer classes from Catapult LayerSummary
    'relu': 0, 'sigmoid': 0, 'tanh': 0,
    'hard_sigmoid': 0, 'hard_tanh': 0, 'HardActivation': 0,
    'linear': 0,
}

ACTIVATION_MAPPING = {
    'relu': 2,
    'tanh': 3, 'hard_tanh': 3,
    'sigmoid': 4, 'hard_sigmoid': 4,
    'linear': 1,
}

STRATEGY_MAPPING = {'latency': 0, 'resource': 1}

IO_TYPE_MAPPING = {'io_parallel': 0, 'io_stream': 1}

# Layer classes to include (skip ApplyAlpha, input layers)
INCLUDED_LAYER_CLASSES = {'Dense', 'relu', 'sigmoid', 'tanh', 'hard_sigmoid', 'hard_tanh', 'HardActivation'}


def parse_ac_fixed(type_str: str) -> Optional[int]:
    """Parse 'ac_fixed<4,3,true>' or 'fixed<4,3,TRN,WRAP,0>' → total bitwidth (4)."""
    if not type_str:
        return None
    m = re.search(r'<\s*(\d+)', type_str)
    if m:
        return int(m.group(1))
    return None


def parse_shape(shape_str: str) -> int:
    """Parse '[4]' or '[None, 4]' → first non-None integer dimension."""
    if not shape_str:
        return 0
    nums = re.findall(r'\d+', shape_str)
    return int(nums[0]) if nums else 0


def get_activation_type(layer_class: str, hls_class: Optional[str] = None) -> int:
    """Map layer class (and optionally the HLSConfig class name) to activation_type int."""
    # Prefer HLSConfig class name when available (resolves HardActivation ambiguity)
    name = hls_class if hls_class else layer_class
    return ACTIVATION_MAPPING.get(name, 0)


class CatapultASICProcessor:
    """Convert Catapult ASIC synthesis JSON reports to numpy arrays for ML training."""

    label_columns = ["latency_cycles", "total_area", "thruput_cycles"]

    def __init__(self):
        self.feature_columns = list(FEATURES.keys())
        self.feature_index = {name: i for i, name in enumerate(self.feature_columns)}

    # ------------------------------------------------------------------
    # JSON parsing helpers
    # ------------------------------------------------------------------

    def _build_hls_activation_map(self, config: Dict) -> Dict[str, str]:
        """Return {layer_name: LayerClassName} from Config.HLSConfig.LayerName."""
        return {
            name: entry.get('LayerClassName', '')
            for name, entry in config.get('HLSConfig', {}).get('LayerName', {}).items()
        }

    def _get_labels(self, data: Dict) -> Dict[str, float]:
        qofr = data.get('QOFRSummary', {})
        return {
            'latency_cycles': float(qofr.get('latency_cycles', 0)),
            'total_area': float(qofr.get('total_area', 0.0)),
            'thruput_cycles': float(qofr.get('thruput_cycles', 0)),
        }

    def _is_valid(self, data: Dict) -> bool:
        qofr = data.get('QOFRSummary', {})
        return bool(qofr) and qofr.get('total_area', 0) > 0

    # ------------------------------------------------------------------
    # Per-file processing
    # ------------------------------------------------------------------

    def process_json(self, file_path: str) -> Tuple[Optional[pd.DataFrame], Optional[Dict]]:
        """Parse one Catapult ASIC JSON report → (features_df, labels_dict) or (None, None)."""
        try:
            with open(file_path) as f:
                data = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load {file_path}: {e}")
            return None, None

        if not self._is_valid(data):
            return None, None

        config = data.get('Config', {})
        hls_config = config.get('HLSConfig', {})
        model_cfg = hls_config.get('Model', {})
        layer_summary = data.get('LayerSummary', [])

        # Global params
        global_rf = int(model_cfg.get('ReuseFactor', 1))
        strategy_str = str(model_cfg.get('Strategy', 'latency')).lower()
        strategy_val = STRATEGY_MAPPING.get(strategy_str, 0)
        io_type_str = str(config.get('IOType', 'io_parallel'))
        io_type_val = IO_TYPE_MAPPING.get(io_type_str, 0)

        # Build activation map for HardActivation disambiguation
        act_map = self._build_hls_activation_map(config)

        rows = []
        last_prec = None  # inherit precision into activation rows

        for entry in layer_summary:
            layer_class = entry.get('Layer Class', '')

            if layer_class not in INCLUDED_LAYER_CLASSES:
                continue

            layer_name = entry.get('Layer Name', '')
            hls_class = act_map.get(layer_name, None)

            row = [0.0] * len(self.feature_columns)

            # Shapes
            d_in = parse_shape(entry.get('Input Shape', ''))
            d_out = parse_shape(entry.get('Output Shape', ''))
            row[FEATURES['d_in1']] = d_in
            row[FEATURES['d_in2']] = 0
            row[FEATURES['d_in3']] = 0
            row[FEATURES['d_out1']] = d_out
            row[FEATURES['d_out2']] = 0
            row[FEATURES['d_out3']] = 0

            # Precision
            if layer_class == 'Dense':
                prec = parse_ac_fixed(entry.get('Weight Type', ''))
                if prec is None:
                    prec = parse_ac_fixed(
                        hls_config.get('Model', {}).get('Precision', {}).get('default', '')
                    )
                last_prec = prec
            else:
                prec = last_prec  # activation inherits from previous dense

            row[FEATURES['prec']] = prec if prec is not None else 0

            # Reuse factor
            rf_str = entry.get('Reuse', '')
            try:
                rf = int(rf_str)
            except (ValueError, TypeError):
                rf = global_rf
            row[FEATURES['rf']] = rf

            # Strategy / io_type (global)
            row[FEATURES['strategy']] = strategy_val
            row[FEATURES['io_type']] = io_type_val

            # Layer type and activation type
            if layer_class == 'Dense':
                row[FEATURES['layer_type']] = 1
                row[FEATURES['activation_type']] = 0
            else:
                row[FEATURES['layer_type']] = 0
                row[FEATURES['activation_type']] = get_activation_type(layer_class, hls_class)

            # Unused spatial features (always 0 for dense-only)
            for f in ('filters', 'kernel_size', 'stride', 'padding', 'pooling', 'batchnorm'):
                row[FEATURES[f]] = 0

            rows.append(row)

        if not rows:
            return None, None

        features_df = pd.DataFrame(rows, columns=self.feature_columns)
        labels = self._get_labels(data)
        return features_df, labels

    # ------------------------------------------------------------------
    # Batch processing
    # ------------------------------------------------------------------

    def process_files_to_numpy(self, file_paths: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        all_features = []
        all_labels = []

        for fp in tqdm(file_paths, desc="Processing", unit="file"):
            feat_df, labels = self.process_json(fp)
            if feat_df is None:
                continue
            all_features.append(feat_df)
            all_labels.append(labels)

        if not all_features:
            logger.error("No valid files processed.")
            return np.array([]), np.array([])

        max_layers = max(len(df) for df in all_features)
        N = len(all_features)
        nf = len(self.feature_columns)
        nl = len(self.label_columns)

        X = np.full((N, max_layers, nf), -1, dtype=float)
        y = np.zeros((N, nl), dtype=float)

        for i, (feat_df, labels) in enumerate(zip(all_features, all_labels)):
            n_layers = len(feat_df)
            for j, col in enumerate(self.feature_columns):
                X[i, :n_layers, j] = feat_df[col].values
            for j, col in enumerate(self.label_columns):
                y[i, j] = labels[col]

        return X, y

    def save_numpy_arrays(self, X: np.ndarray, y: np.ndarray,
                          output_dir: str, prefix: str = '') -> Tuple[str, str]:
        os.makedirs(output_dir, exist_ok=True)
        feat_name = f"{prefix}_features.npy" if prefix else "features.npy"
        lbl_name = f"{prefix}_labels.npy" if prefix else "labels.npy"
        X_path = os.path.join(output_dir, feat_name)
        y_path = os.path.join(output_dir, lbl_name)
        np.save(X_path, X)
        np.save(y_path, y)
        print(f"Saved features {X.shape} → {X_path}")
        print(f"Saved labels  {y.shape} → {y_path}")
        return X_path, y_path

    def process_folder(self, input_dir: str, output_dir: str, prefix: str = '') -> Tuple[str, str]:
        json_files = sorted(glob.glob(os.path.join(input_dir, "*.json")))
        print(f"Found {len(json_files)} JSON files in {input_dir}")
        X, y = self.process_files_to_numpy(json_files)
        if X.size == 0:
            print("No data produced.")
            return '', ''
        return self.save_numpy_arrays(X, y, output_dir, prefix)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Catapult ASIC JSON reports to numpy arrays.")
    parser.add_argument("--input", required=True, help="Directory containing raw JSON reports")
    parser.add_argument("--output", required=True, help="Output directory for .npy files")
    parser.add_argument("--prefix", default="catapult_asic", help="Filename prefix (default: catapult_asic)")
    args = parser.parse_args()

    processor = CatapultASICProcessor()
    processor.process_folder(args.input, args.output, args.prefix)
