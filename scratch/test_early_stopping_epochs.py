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

def test_epochs():
    X_veh, y_veh, cat_veh, dname_veh = load_dataset_auto(54)
    input_dim = 18

    print("=================================================================")
    print(" TESTING EARLY-STOPPING HYPOTHESIS ON VEHICLE (TRIALS 0 TO 4)")
    print(" Running exactly as benchmark: patience=3 for both pretrain and finetune")
    print("=================================================================")

    results = []

    for trial in range(5):
        splits = preprocess_dataset(X_veh, y_veh, cat_veh, scale="zscore", random_state=trial)
        lab_idx, unlab_idx = make_semi_supervised(splits, labeled_frac=0.25, random_state=trial)
        x_lab = splits.x_train[lab_idx]
        y_lab = splits.y_train[lab_idx]

        # 1. SCARF Pre-training
        scarf_enc, pt_hist = pretrain_scarf(
            splits.x_train, splits.x_val, input_dim,
            max_epochs=200, patience=3, device="cpu"
        )
        pt_epochs = len(pt_hist.val_loss)
        pt_best = pt_hist.best_epoch

        # 2. Control fine-tuning (random init, encoder=None)
        ctrl_enc, ctrl_head, ctrl_hist = finetune_classifier(
            x_train=x_lab, y_train=y_lab, x_val=splits.x_val, y_val=splits.y_val,
            n_classes=splits.n_classes, input_dim=input_dim, encoder=None,
            max_epochs=100, patience=3, device="cpu"
        )
        ctrl_total = len(ctrl_hist.val_loss)
        ctrl_best = ctrl_hist.best_epoch
        ctrl_test = evaluate_accuracy(ctrl_enc, ctrl_head, splits.x_test, splits.y_test)
        ctrl_val = evaluate_accuracy(ctrl_enc, ctrl_head, splits.x_val, splits.y_val)
        ctrl_train = evaluate_accuracy(ctrl_enc, ctrl_head, x_lab, y_lab)

        # 3. Control+SCARF fine-tuning (pretrained init)
        sc_enc_copy = copy.deepcopy(scarf_enc)
        sc_enc, sc_head, sc_hist = finetune_classifier(
            x_train=x_lab, y_train=y_lab, x_val=splits.x_val, y_val=splits.y_val,
            n_classes=splits.n_classes, input_dim=input_dim, encoder=sc_enc_copy,
            max_epochs=100, patience=3, device="cpu"
        )
        sc_total = len(sc_hist.val_loss)
        sc_best = sc_hist.best_epoch
        sc_test = evaluate_accuracy(sc_enc, sc_head, splits.x_test, splits.y_test)
        sc_val = evaluate_accuracy(sc_enc, sc_head, splits.x_val, splits.y_val)
        sc_train = evaluate_accuracy(sc_enc, sc_head, x_lab, y_lab)

        # 4. Mixup vs Mixup+SCARF
        mx_enc, mx_head, mx_hist = finetune_classifier(
            x_train=x_lab, y_train=y_lab, x_val=splits.x_val, y_val=splits.y_val,
            n_classes=splits.n_classes, input_dim=input_dim, encoder=None,
            mixup_alpha=0.2, max_epochs=100, patience=3, device="cpu"
        )
        sc_mx_enc, sc_mx_head, sc_mx_hist = finetune_classifier(
            x_train=x_lab, y_train=y_lab, x_val=splits.x_val, y_val=splits.y_val,
            n_classes=splits.n_classes, input_dim=input_dim, encoder=copy.deepcopy(scarf_enc),
            mixup_alpha=0.2, max_epochs=100, patience=3, device="cpu"
        )

        results.append({
            "trial": trial,
            "ctrl_total_epochs": ctrl_total,
            "ctrl_best_epoch": ctrl_best,
            "ctrl_test_acc": ctrl_test,
            "sc_total_epochs": sc_total,
            "sc_best_epoch": sc_best,
            "sc_test_acc": sc_test,
            "ctrl_val_err_hist": [round(x, 3) for x in ctrl_hist.val_loss],
            "sc_val_err_hist": [round(x, 3) for x in sc_hist.val_loss],
            "mixup_total_epochs": len(mx_hist.val_loss),
            "sc_mixup_total_epochs": len(sc_mx_hist.val_loss),
        })

        print(f"Trial {trial}:")
        print(f"  Control:      total_epochs={ctrl_total:2d} (best={ctrl_best:2d}), test_acc={ctrl_test:.3f}, val_errs={results[-1]['ctrl_val_err_hist']}")
        print(f"  Control+SCARF: total_epochs={sc_total:2d} (best={sc_best:2d}), test_acc={sc_test:.3f}, val_errs={results[-1]['sc_val_err_hist']}")
        print(f"  Mixup total={len(mx_hist.val_loss):2d} vs Mixup+SCARF total={len(sc_mx_hist.val_loss):2d}")

if __name__ == "__main__":
    test_epochs()
