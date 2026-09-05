import os
import sys
import copy
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scarf"))

from scripts.run_benchmark import load_dataset_auto
from scarf.data import preprocess_dataset
from scarf.trainer import pretrain_scarf, finetune_classifier, evaluate_accuracy

def test_fully_supervised_vehicle(n_trials=10):
    X_veh, y_veh, cat_veh, dname_veh = load_dataset_auto(54)
    input_dim = X_veh.shape[1]

    print("=================================================================")
    print(f" FULLY-SUPERVISED (100% LABELED) VEHICLE: TRIALS 0 TO {n_trials-1}")
    print(" Running with default patience=3 (same as results_openml6)")
    print("=================================================================")

    results = []

    for trial in range(n_trials):
        splits = preprocess_dataset(X_veh, y_veh, cat_veh, scale="zscore", random_state=trial)
        # 100% labeled setting: x_train and y_train are full splits
        x_train = splits.x_train
        y_train = splits.y_train

        # 1. SCARF Pre-training
        scarf_enc, pt_hist = pretrain_scarf(
            x_train, splits.x_val, input_dim,
            max_epochs=300, patience=3, device="cpu"
        )
        pt_total = len(pt_hist.val_loss)
        pt_best = pt_hist.best_epoch

        # 2. Control fine-tuning (random init, encoder=None)
        ctrl_enc, ctrl_head, ctrl_hist = finetune_classifier(
            x_train=x_train, y_train=y_train, x_val=splits.x_val, y_val=splits.y_val,
            n_classes=splits.n_classes, input_dim=input_dim, encoder=None,
            max_epochs=150, patience=3, device="cpu"
        )
        ctrl_total = len(ctrl_hist.val_loss)
        ctrl_best = ctrl_hist.best_epoch
        ctrl_test = evaluate_accuracy(ctrl_enc, ctrl_head, splits.x_test, splits.y_test)
        ctrl_val = evaluate_accuracy(ctrl_enc, ctrl_head, splits.x_val, splits.y_val)

        # 3. Control+SCARF fine-tuning (pretrained init)
        sc_enc_copy = copy.deepcopy(scarf_enc)
        sc_enc, sc_head, sc_hist = finetune_classifier(
            x_train=x_train, y_train=y_train, x_val=splits.x_val, y_val=splits.y_val,
            n_classes=splits.n_classes, input_dim=input_dim, encoder=sc_enc_copy,
            max_epochs=150, patience=3, device="cpu"
        )
        sc_total = len(sc_hist.val_loss)
        sc_best = sc_hist.best_epoch
        sc_test = evaluate_accuracy(sc_enc, sc_head, splits.x_test, splits.y_test)
        sc_val = evaluate_accuracy(sc_enc, sc_head, splits.x_val, splits.y_val)

        results.append({
            "trial": trial,
            "pt_total": pt_total,
            "pt_best": pt_best,
            "ctrl_total": ctrl_total,
            "ctrl_best": ctrl_best,
            "ctrl_test": ctrl_test,
            "sc_total": sc_total,
            "sc_best": sc_best,
            "sc_test": sc_test,
            "ctrl_val_errs": [round(x, 3) for x in ctrl_hist.val_loss],
            "sc_val_errs": [round(x, 3) for x in sc_hist.val_loss],
        })

        print(f"Trial {trial:2d}:")
        print(f"  SCARF PT:      total_epochs={pt_total:2d} (best={pt_best:2d})")
        print(f"  Control:       total_epochs={ctrl_total:2d} (best={ctrl_best:2d}), test_acc={ctrl_test:.4f}, val_errs={results[-1]['ctrl_val_errs']}")
        print(f"  Control+SCARF: total_epochs={sc_total:2d} (best={sc_best:2d}), test_acc={sc_test:.4f}, val_errs={results[-1]['sc_val_errs']}")
        print(f"  Diff (SC - Ctrl): {sc_test - ctrl_test:+.4f}")

    ctrl_accs = [r["ctrl_test"] for r in results]
    sc_accs = [r["sc_test"] for r in results]
    print("\n=================================================================")
    print(f"SUMMARY (Trials 0..{n_trials-1}):")
    print(f"Control:       mean={np.mean(ctrl_accs):.4f} +/- {np.std(ctrl_accs):.4f}, avg_epochs={np.mean([r['ctrl_total'] for r in results]):.1f}, avg_best={np.mean([r['ctrl_best'] for r in results]):.1f}")
    print(f"Control+SCARF: mean={np.mean(sc_accs):.4f} +/- {np.std(sc_accs):.4f}, avg_epochs={np.mean([r['sc_total'] for r in results]):.1f}, avg_best={np.mean([r['sc_best'] for r in results]):.1f}")
    print(f"Mean Difference: {np.mean(sc_accs) - np.mean(ctrl_accs):+.4f}")
    print("=================================================================")

if __name__ == "__main__":
    test_fully_supervised_vehicle(n_trials=10)
