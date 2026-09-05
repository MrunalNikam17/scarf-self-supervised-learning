import os
import sys
import copy
import numpy as np
import pandas as pd
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scarf"))

from scripts.run_benchmark import load_dataset_auto
from scarf.data import preprocess_dataset, make_semi_supervised
from scarf.trainer import pretrain_scarf, finetune_classifier, evaluate_accuracy

def run_experiment(n_trials=10):
    X_veh, y_veh, cat_veh, dname_veh = load_dataset_auto(54)
    input_dim = X_veh.shape[1]

    records = []

    print("=======================================================================")
    print(f" VEHICLE (ID 54) WARM-UP FLOOR COMPARISON: min_epochs=0 vs min_epochs=15")
    print(f" Trials: 0 to {n_trials - 1} | Settings: 100% Labeled and 25% Labeled")
    print("=======================================================================\n")

    for setting in ["100% Labeled", "25% Labeled"]:
        print(f"\n==================== SETTING: {setting} ====================")
        for trial in range(n_trials):
            splits = preprocess_dataset(X_veh, y_veh, cat_veh, scale="zscore", random_state=trial)

            if setting == "100% Labeled":
                x_train_ft = splits.x_train
                y_train_ft = splits.y_train
                x_pt = splits.x_train
            else:
                lab_idx, unlab_idx = make_semi_supervised(splits, labeled_frac=0.25, random_state=trial)
                x_train_ft = splits.x_train[lab_idx]
                y_train_ft = splits.y_train[lab_idx]
                x_pt = splits.x_train  # SCARF pre-trains on all available train data

            # Pretrain SCARF once per trial
            torch.manual_seed(1000 + trial)
            scarf_enc, pt_hist = pretrain_scarf(
                x_pt, splits.x_val, input_dim,
                max_epochs=300, patience=3, device="cpu"
            )

            # Test configurations:
            # 1. Control min_epochs=0
            # 2. Control min_epochs=15
            # 3. Control+SCARF min_epochs=0
            # 4. Control+SCARF min_epochs=15

            configs = [
                ("Control", False, 0),
                ("Control", False, 15),
                ("Control+SCARF", True, 0),
                ("Control+SCARF", True, 15),
            ]

            trial_res = {}

            for method_name, use_scarf, min_ep in configs:
                seed = 2000 + trial
                torch.manual_seed(seed)
                np.random.seed(seed)

                enc_init = copy.deepcopy(scarf_enc) if use_scarf else None

                enc, head, hist = finetune_classifier(
                    x_train=x_train_ft,
                    y_train=y_train_ft,
                    x_val=splits.x_val,
                    y_val=splits.y_val,
                    n_classes=splits.n_classes,
                    input_dim=input_dim,
                    encoder=enc_init,
                    max_epochs=150,
                    patience=3,
                    min_epochs=min_ep,
                    device="cpu"
                )

                test_acc = evaluate_accuracy(enc, head, splits.x_test, splits.y_test)
                val_acc = evaluate_accuracy(enc, head, splits.x_val, splits.y_val)
                total_epochs = len(hist.val_loss)
                best_epoch = hist.best_epoch

                key = f"{method_name} (min_ep={min_ep})"
                trial_res[key] = {
                    "acc": test_acc,
                    "total_epochs": total_epochs,
                    "best_epoch": best_epoch,
                }

                records.append({
                    "setting": setting,
                    "trial": trial,
                    "method": method_name,
                    "min_epochs": min_ep,
                    "test_acc": test_acc,
                    "val_acc": val_acc,
                    "total_epochs": total_epochs,
                    "best_epoch": best_epoch,
                })

            ctrl_0 = trial_res["Control (min_ep=0)"]
            ctrl_15 = trial_res["Control (min_ep=15)"]
            sc_0 = trial_res["Control+SCARF (min_ep=0)"]
            sc_15 = trial_res["Control+SCARF (min_ep=15)"]

            print(f"Trial {trial:2d}:")
            print(f"  min_epochs=0:  Ctrl={ctrl_0['acc']:.4f} (ep={ctrl_0['total_epochs']:2d}, best={ctrl_0['best_epoch']:2d}) | "
                  f"SCARF={sc_0['acc']:.4f} (ep={sc_0['total_epochs']:2d}, best={sc_0['best_epoch']:2d}) | "
                  f"Diff={sc_0['acc'] - ctrl_0['acc']:+.4f}")
            print(f"  min_epochs=15: Ctrl={ctrl_15['acc']:.4f} (ep={ctrl_15['total_epochs']:2d}, best={ctrl_15['best_epoch']:2d}) | "
                  f"SCARF={sc_15['acc']:.4f} (ep={sc_15['total_epochs']:2d}, best={sc_15['best_epoch']:2d}) | "
                  f"Diff={sc_15['acc'] - ctrl_15['acc']:+.4f}")
            print(f"  Delta SCARF (15 vs 0): {sc_15['acc'] - sc_0['acc']:+.4f} (epochs: {sc_0['total_epochs']} -> {sc_15['total_epochs']})")

    df = pd.DataFrame(records)
    out_csv = os.path.join(PROJECT_ROOT, "scratch", "vehicle_min_epochs_comparison.csv")
    df.to_csv(out_csv, index=False)
    print(f"\nSaved raw results to {out_csv}")

    # Summary table
    print("\n" + "="*80)
    print("SUMMARY COMPARISON TABLE")
    print("="*80)
    summary = df.groupby(["setting", "method", "min_epochs"]).agg(
        mean_acc=("test_acc", "mean"),
        std_acc=("test_acc", "std"),
        avg_total_epochs=("total_epochs", "mean"),
        avg_best_epoch=("best_epoch", "mean"),
    ).reset_index()
    print(summary.to_string(index=False))

    return df

if __name__ == "__main__":
    run_experiment(n_trials=10)
