import sys
import os
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
from scarf.trainer import finetune_classifier, evaluate_accuracy

def probe_tri_training_churn(dataset_id: int, n_trials: int = 3, n_iterations: int = 10, max_epochs: int = 150):
    X, y, cat_mask, dname = load_dataset_auto(dataset_id)
    print(f"\n=======================================================")
    print(f" PROBING TRI-TRAINING CONVERGENCE ON {dname} (ID {dataset_id})")
    print(f" {X.shape[0]} rows x {X.shape[1]} features, {y.nunique()} classes")
    print(f"=======================================================")

    churn_records = []

    for trial in range(n_trials):
        splits = preprocess_dataset(X, y, cat_mask, scale="zscore", random_state=trial)
        input_dim = splits.x_train.shape[1]
        lab_idx, unlab_idx = make_semi_supervised(splits, labeled_frac=0.25, random_state=trial)

        x_labeled = splits.x_train[lab_idx]
        y_labeled = splits.y_train[lab_idx]
        x_unlabeled = splits.x_train[unlab_idx]
        n_unlabeled = len(x_unlabeled)
        n_labeled = len(x_labeled)

        rng = np.random.RandomState(trial)
        l_sets = []
        for _ in range(3):
            boot_idx = rng.choice(n_labeled, size=n_labeled, replace=True)
            present_classes = np.unique(y_labeled)
            boot_classes = np.unique(y_labeled[boot_idx])
            missing = set(present_classes) - set(boot_classes)
            for m in missing:
                replace_pos = rng.randint(0, n_labeled)
                boot_idx[replace_pos] = np.where(y_labeled == m)[0][0]
            l_sets.append((x_labeled[boot_idx].copy(), y_labeled[boot_idx].copy()))

        prev_agreed_preds = [None, None, None]

        print(f"\n--- Trial {trial + 1}/{n_trials} (N_labeled={n_labeled}, N_unlabeled={n_unlabeled}) ---")
        models = [None, None, None]

        for it in range(n_iterations):
            # Train all 3 models
            for i in range(3):
                m_enc, m_head, _ = finetune_classifier(
                    l_sets[i][0], l_sets[i][1], splits.x_val, splits.y_val,
                    n_classes=splits.n_classes, encoder=None, input_dim=input_dim,
                    max_epochs=max_epochs, patience=3, device="cpu",
                )
                models[i] = (m_enc, m_head)

            # Predictions on unlabeled pool
            preds = []
            u_t = torch.as_tensor(x_unlabeled, dtype=torch.float32)
            for m_enc, m_head in models:
                m_enc.eval()
                m_head.eval()
                with torch.no_grad():
                    preds.append(m_head(m_enc(u_t)).argmax(dim=-1).cpu().numpy())

            # Evaluate agreement and churn for each model
            iter_churn = []
            iter_agreed_counts = []
            new_l_sets = []
            new_points_added = False

            for i in range(3):
                j, k = [idx for idx in range(3) if idx != i]
                agree_mask = (preds[j] == preds[k])
                n_agree = int(agree_mask.sum())
                iter_agreed_counts.append(n_agree)

                # Churn: how many predictions in the agreed mask changed from previous iteration?
                current_agreed_pred = np.where(agree_mask, preds[j], -1)
                if prev_agreed_preds[i] is not None:
                    # Differences in agreement status or predicted label
                    churn = int((current_agreed_pred != prev_agreed_preds[i]).sum())
                else:
                    churn = n_agree
                iter_churn.append(churn)
                prev_agreed_preds[i] = current_agreed_pred

                if n_agree > 0:
                    add_x = x_unlabeled[agree_mask]
                    add_y = preds[j][agree_mask]
                    cur_x = np.concatenate([l_sets[i][0], add_x], axis=0)
                    cur_y = np.concatenate([l_sets[i][1], add_y], axis=0)
                    new_l_sets.append((cur_x, cur_y))
                    new_points_added = True
                else:
                    new_l_sets.append(l_sets[i])

            l_sets = new_l_sets

            mean_agree = float(np.mean(iter_agreed_counts))
            mean_churn = float(np.mean(iter_churn))
            pct_churn = (mean_churn / n_unlabeled) * 100.0

            churn_records.append({
                "dataset": dname,
                "trial": trial + 1,
                "iteration": it + 1,
                "mean_agreed_points": mean_agree,
                "mean_churn": mean_churn,
                "churn_pct_of_unlabeled": pct_churn,
            })

            print(f"  Iteration {it + 1:2d}/10: Agreed={mean_agree:4.0f}/{n_unlabeled} | "
                  f"Churn={mean_churn:4.0f} ({pct_churn:5.1f}% of U)")

            if not new_points_added or mean_churn == 0:
                print(f"  --> Plateau reached at iteration {it + 1} (churn=0)")
                break

    df = pd.DataFrame(churn_records)
    return df

if __name__ == "__main__":
    # Test on balance-scale (id 11)
    df_small = probe_tri_training_churn(11, n_trials=2, n_iterations=10, max_epochs=100)
    # Test on vehicle (id 54) or qsar-biodeg (id 1494)
    df_large = probe_tri_training_churn(1494, n_trials=2, n_iterations=10, max_epochs=100)
    
    combined = pd.concat([df_small, df_large], ignore_index=True)
    combined.to_csv(os.path.join(HERE, "tri_training_convergence_probe.csv"), index=False)
    print("\nConvergence probe completed and saved to scratch/tri_training_convergence_probe.csv")
