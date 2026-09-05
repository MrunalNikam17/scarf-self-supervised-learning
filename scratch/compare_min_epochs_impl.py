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
from scarf.data import preprocess_dataset, make_semi_supervised
from scarf.model import Encoder, ClassificationHead
from scarf.trainer import pretrain_scarf, finetune_classifier, evaluate_accuracy, TrainHistory
import torch.nn.functional as F

def finetune_with_min_epochs(
    x_train, y_train, x_val, y_val, n_classes,
    encoder=None, input_dim=None, patience=3, min_epochs=15,
    suppress_counter=True, max_epochs=150, device="cpu"
):
    """
    suppress_counter=True (Option B): epochs_no_improve starts counting after min_epochs
    suppress_counter=False (Option A): epochs_no_improve counts always, break checked after min_epochs
    """
    x_train_t = torch.as_tensor(x_train, dtype=torch.float32, device=device)
    y_train_t = torch.as_tensor(y_train, dtype=torch.long, device=device)
    x_val_t = torch.as_tensor(x_val, dtype=torch.float32, device=device)
    y_val_t = torch.as_tensor(y_val, dtype=torch.long, device=device)

    if encoder is None:
        encoder = Encoder(input_dim, 256, 4).to(device)
    else:
        encoder = encoder.to(device)
    head = ClassificationHead(encoder.output_dim, 256, n_classes, 2).to(device)
    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(head.parameters()), lr=1e-3)

    n = x_train_t.shape[0]
    history = TrainHistory()
    best_val_err = float("inf")
    best_state = None
    epochs_no_improve = 0

    for epoch in range(max_epochs):
        encoder.train()
        head.train()
        perm = torch.randperm(n, device=device)
        epoch_losses = []
        for start in range(0, n, 128):
            batch_idx = perm[start:start + 128]
            xb = x_train_t[batch_idx]
            yb = y_train_t[batch_idx]
            logits = head(encoder(xb))
            loss = F.cross_entropy(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())

        train_loss = float(np.mean(epoch_losses)) if epoch_losses else float("nan")
        history.train_loss.append(train_loss)

        encoder.eval()
        head.eval()
        with torch.no_grad():
            logits = head(encoder(x_val_t))
            preds = logits.argmax(dim=-1)
            val_err = (preds != y_val_t).float().mean().item()
        history.val_loss.append(val_err)

        if val_err < best_val_err - 1e-6:
            best_val_err = val_err
            best_state = {
                "encoder": {k: v.detach().clone() for k, v in encoder.state_dict().items()},
                "head": {k: v.detach().clone() for k, v in head.state_dict().items()},
            }
            history.best_epoch = epoch
            epochs_no_improve = 0
        else:
            if suppress_counter:
                if (epoch + 1) >= min_epochs:
                    epochs_no_improve += 1
                    if epochs_no_improve >= patience:
                        break
                else:
                    epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if (epoch + 1) >= min_epochs and epochs_no_improve >= patience:
                    break

    if best_state is not None:
        encoder.load_state_dict(best_state["encoder"])
        head.load_state_dict(best_state["head"])
    return encoder, head, history

def compare_options():
    X_veh, y_veh, cat_veh, dname_veh = load_dataset_auto(54)
    input_dim = 18

    print("=== TESTING ON TRIAL 0 (25% LABELED) ===")
    splits = preprocess_dataset(X_veh, y_veh, cat_veh, scale="zscore", random_state=0)
    lab_idx, unlab_idx = make_semi_supervised(splits, labeled_frac=0.25, random_state=0)
    x_lab = splits.x_train[lab_idx]
    y_lab = splits.y_train[lab_idx]

    torch.manual_seed(0)
    np.random.seed(0)
    sc_base, _ = pretrain_scarf(splits.x_train, splits.x_val, input_dim, max_epochs=200, patience=3)

    for name, supp in [("Option A (break after 15 if patience met)", False),
                       ("Option B (early-stop active only after 15)", True)]:
        print(f"\n--- {name} ---")
        # Control
        torch.manual_seed(42)
        c_enc, c_head, c_hist = finetune_with_min_epochs(
            x_lab, y_lab, splits.x_val, splits.y_val, splits.n_classes,
            input_dim=input_dim, patience=3, min_epochs=15, suppress_counter=supp
        )
        c_test = evaluate_accuracy(c_enc, c_head, splits.x_test, splits.y_test)
        print(f"Control:       total={len(c_hist.val_loss):2d}, best={c_hist.best_epoch:2d}, test_acc={c_test:.4f}")

        # SCARF
        torch.manual_seed(42)
        s_enc, s_head, s_hist = finetune_with_min_epochs(
            x_lab, y_lab, splits.x_val, splits.y_val, splits.n_classes,
            encoder=copy.deepcopy(sc_base), patience=3, min_epochs=15, suppress_counter=supp
        )
        s_test = evaluate_accuracy(s_enc, s_head, splits.x_test, splits.y_test)
        print(f"Control+SCARF: total={len(s_hist.val_loss):2d}, best={s_hist.best_epoch:2d}, test_acc={s_test:.4f}")

if __name__ == "__main__":
    compare_options()
