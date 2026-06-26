# run.py

import torch # type: ignore
# from data import load_and_preprocess_data, get_first_non_padded_layer
# from GNN.Dataset import *
from GNN.Dataset2 import * # updated to load the split data
from model import TransformerRegressor
from train import train_model, test_model, calculate_metrics, calculate_metrics_per_feature, calculate_metrics_by_layer_count
from plot import plot_loss, plot_box_plots_symlog, plot_results_simplified
import numpy as np # type: ignore
import os
import argparse
from datetime import datetime

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", choices=["gnn", "transformer"], default="transformer", help="Architecture type")
    parser.add_argument("--eval-only", action="store_true", help="Only evaluate and plot, no training.")
    parser.add_argument("--model-path", type=str, default=None, help="Path to saved model checkpoint (for eval-only mode).")
    parser.add_argument("--epochs", type=int, default=200, help="Number of epochs to train")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=512, help="Batch size")
    parser.add_argument("--base-dir", type=str, default=None, help="Directory with train/val/test split .npy files (overrides hardcoded default)")
    parser.add_argument("--drop-throughput", action="store_true",
                        help="Train on latency+area only; derive throughput analytically at inference via 1-layer lookup")
    parser.add_argument("--thruput-lookup", type=str, default=None,
                        help="Path to thruput_lookup.pkl (required with --drop-throughput)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to model.pt checkpoint to warm-start training from")
    parser.add_argument("--freeze-encoder", action="store_true",
                        help="Freeze tokenizer+transformer; only train the regression head (for fine-tuning on new process)")
    args = parser.parse_args()

    if args.drop_throughput and not args.thruput_lookup:
        raise ValueError("--thruput-lookup is required when --drop-throughput is set")

    timestamp = datetime.now().strftime("%m_%d_%H_%M")

    num_epochs = args.epochs
    learning_rate = args.lr
    BATCH_SIZE = args.batch_size
    USE_LOG_TRANSFORM = True  # Set to True to enable log transformation
    LOG_EPSILON = 1e-6       # Small value to add before log transform

    if args.eval_only:
        outdir = f"new_results_plots/testing_only_{timestamp}_{num_epochs}epochs_{learning_rate}lr_{BATCH_SIZE}bs"
    else:
        outdir = f"new_results_plots/{timestamp}_{num_epochs}epochs_{learning_rate}lr_{BATCH_SIZE}bs"

    # Config (output_features resolved after dataloaders are created)
    # outdir = "results_and_plots/5_31_results_2_epochs_TESTING"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # JOB
    # # image_data_path = '../../app/'
    # image_data_path = '/app/dataset/May_29_complete/'
    # FEATURES_PATH = os.path.join(image_data_path, FEATURES_PATH)
    # LABELS_PATH = os.path.join(image_data_path, LABELS_PATH)
    # STATS_PATH = os.path.join(image_data_path, STATS_PATH)
    # SPLIT_MAPPING_PATH = os.path.join(image_data_path, SPLIT_MAPPING_PATH)

    # train_loader, val_loader, test_loader, dataset, node_feature_dim, num_targets = create_dataloaders(
    #         feature_path=FEATURES_PATH,
    #         labels_path=LABELS_PATH,
    #         stats_load_path=STATS_PATH,
    #         stats_save_path=STATS_PATH,
    #         split_mapping_path=SPLIT_MAPPING_PATH,  # NEW: Add this parameter
    #         batch_size=BATCH_SIZE,
    #         train_val_test_split=(0.7, 0.15, 0.15),
    #         random_seed=42,
    #         num_workers=4, # dropped from 5
    #         pin_memory=True if device.type == 'cuda' else False
    #     )

## CHANGE BEFORE PUSH
    if args.base_dir:
        base_dir = args.base_dir
    else:
        base_dir = "../dataset/output/split_dataset/result/result/"  # UPDATE FOR THE JOB WITH NEW PATH
    # base_dir = "/jason-pvc/june_wa-hls4ml/result/" # IN THE PVC

    best_model_dir = os.path.join(outdir, "best_model")
    os.makedirs(best_model_dir, exist_ok=True)

    # Stats live next to the model checkpoint so they always match.
    # eval-only: load from the checkpoint's directory.
    # training:  compute from training data, save to best_model_dir.
    if args.eval_only:
        _model_path = args.model_path or os.path.join(best_model_dir, "model.pt")
        _stats_load = os.path.join(os.path.dirname(_model_path), "lognormalization_stats.npy")
        _stats_save = None
    else:
        _stats_load = os.path.join(base_dir, "lognormalization_stats.npy")
        _stats_save = os.path.join(best_model_dir, "lognormalization_stats.npy")

    _label_cols = [0, 1] if args.drop_throughput else None

    train_loader, val_loader, test_loader, node_feature_dim, num_targets = create_dataloaders_from_split_data(
            train_features_path=os.path.join(base_dir, "train_features.npy"),
            train_labels_path=os.path.join(base_dir, "train_labels.npy"),
            val_features_path=os.path.join(base_dir, "val_features.npy"),
            val_labels_path=os.path.join(base_dir, "val_labels.npy"),
            test_features_path=os.path.join(base_dir, "test_features.npy"),
            test_labels_path=os.path.join(base_dir, "test_labels.npy"),
            stats_load_path=_stats_load,
            stats_save_path=_stats_save,
            batch_size=BATCH_SIZE,
            num_workers=4,
            pin_memory=True if device.type == 'cuda' else False,
            mode=args.arch,
            use_log_transform=USE_LOG_TRANSFORM,
            log_epsilon=LOG_EPSILON,
            label_cols=_label_cols,
        )

    # Set mode on datasets!
    for loader in [train_loader, val_loader, test_loader]:
        loader.dataset.mode = args.arch

    # Derive output feature names from num_targets (fall back to generic names)
    default_output_features = ['CYCLES', 'FF', 'LUT', 'BRAM', 'DSP', 'II']
    asic_output_features    = ['LATENCY', 'AREA', 'THROUGHPUT']
    if args.drop_throughput:
        output_features = ['LATENCY', 'AREA']
    elif num_targets == len(asic_output_features):
        output_features = asic_output_features
    elif num_targets <= len(default_output_features):
        output_features = default_output_features[:num_targets]
    else:
        output_features = [f'TARGET_{i}' for i in range(num_targets)]

    model = TransformerRegressor(output_dim=num_targets).to(device)
    loss_fn = torch.nn.MSELoss()

    if args.eval_only:
        model_path = args.model_path or os.path.join(best_model_dir, "model.pt")
        print(f"Loading model weights from: {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        if args.resume:
            print(f"Warm-starting from checkpoint: {args.resume}")
            model.load_state_dict(torch.load(args.resume, map_location=device))
        if args.freeze_encoder:
            for param in model.tokenizer.parameters():
                param.requires_grad_(False)
            for param in model.transformer.parameters():
                param.requires_grad_(False)
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"Encoder frozen — training head only ({trainable:,} trainable params)")
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate
        )
        # Training
        train_losses, val_losses = train_model(
            model, train_loader, val_loader, optimizer, loss_fn, device, num_epochs=num_epochs, verbose=True, checkpoint_path=best_model_dir
        )
        # Plot Loss
        plot_loss(train_losses, val_losses, outdir=os.path.join(outdir, "plots"))
        # Load best checkpoint for evaluation
        model_path = os.path.join(best_model_dir, "model.pt")
        model.load_state_dict(torch.load(model_path, map_location=device))

    # Test Evaluation
    y_true, y_pred = test_model(model, test_loader, device)
    # For denormalization, use the dataset, not the class
    y_true_denorm = test_loader.dataset.denormalize_labels(torch.tensor(y_true)).numpy()
    y_pred_denorm = test_loader.dataset.denormalize_labels(torch.tensor(y_pred)).numpy()

    # Load test input features (raw, un-normalized) for model-type coloring and thruput derivation
    test_features_np = np.load(os.path.join(base_dir, "test_features.npy"))  # (N, max_layers, 18)

    # When throughput was excluded from training, derive it analytically from the
    # 1-layer lookup and append as the third column so downstream metrics and plots
    # see the full LATENCY / AREA / THROUGHPUT triple.
    if args.drop_throughput:
        import pickle
        with open(args.thruput_lookup, 'rb') as f:
            lut = pickle.load(f)

        thruput_derived = []
        for design in test_features_np:
            rows = [row for row in design if not np.all(row == -1)]
            layer_thruputs = []
            for row in rows:
                if int(row[9]) != 1:   # feature index 9 = layer_type; 1 = Dense
                    continue
                key = (int(row[0]), int(row[3]), int(row[6]), int(row[7]))
                t = lut.get(key)
                if t is not None:
                    layer_thruputs.append(t)
            thruput_derived.append(float(max(layer_thruputs)) if layer_thruputs else 0.0)

        thruput_derived = np.array(thruput_derived).reshape(-1, 1)
        y_pred_denorm = np.hstack([y_pred_denorm, thruput_derived])

        # Actual throughput from the raw labels file (column index 2)
        all_labels_raw = np.load(os.path.join(base_dir, "test_labels.npy"))
        y_true_denorm = np.hstack([y_true_denorm, all_labels_raw[:, 2:3]])

        output_features = ['LATENCY', 'AREA', 'THROUGHPUT']

    # Use for metrics and plotting:
    calculate_metrics(y_true_denorm, y_pred_denorm)

    # After obtaining y_true_denorm and y_pred_denorm:
    metrics_per_feature = calculate_metrics_per_feature(y_true_denorm, y_pred_denorm, output_features)

    # Breakdown by number of Dense layers
    print("\n\n--- Metrics by layer count ---")
    calculate_metrics_by_layer_count(y_true_denorm, y_pred_denorm, test_features_np, output_features)

    # # Extract strategies for each model (0: latency, 1: resource)
    # def get_strategies(features_np):
    #     strategies = []
    #     for model in features_np:
    #         for layer in model:
    #             if not np.all(layer == -1):
    #                 strategies.append(int(layer[8]))  # 0=latency, 1=resource
    #                 break
    #         else:
    #             strategies.append(None)  # Handle fully padded model if ever
    #     return strategies

    # test_strategies = get_strategies(test_features_np)

    def get_model_types(features_np):
        model_types = []
        for model in features_np:
            valid_layers = [layer for layer in model if not np.all(layer == -1)]
            layer_types = [int(layer[9]) for layer in valid_layers] # model_type is the 9th index
            if any(lt == 2 for lt in layer_types):
                model_types.append('Conv1D')
            elif any(lt == 3 for lt in layer_types):
                model_types.append('Conv2D')
            else:
                model_types.append('Dense')
        return model_types

    test_model_types = get_model_types(test_features_np)
    # END ADDITION

    plot_box_plots_symlog(y_pred_denorm, y_true_denorm, folder_name=outdir, output_features=output_features)
    plot_results_simplified(
        name="run1",
        mpl_plots=True,
        y_test=y_true_denorm,
        y_pred=y_pred_denorm,
        output_features=output_features,
        folder_name=outdir,
        model_types=test_model_types # , ADDED
        # strategies=test_strategies # ADDED
    )

if __name__ == "__main__":
    main()

# nohup python run.py --arch transformer > new_logs/output1.log 2>&1 &

# # example of inference only:
# python run.py --arch transformer --eval-only --model-path new_results_plots/06_02_17_45_20epochs_0.0001lr_1024bs/best_model/model.pt

# nohup python run.py --arch transformer > 6_3_logs/test1.log 2>&1 &