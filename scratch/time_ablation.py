import time
import sys
import os

sys.path.insert(0, os.path.abspath("scarf"))
from scarf.data import load_openml_dataset, preprocess_dataset
from scarf.ablations import pretrain_ablation
from scarf.trainer import finetune_classifier, evaluate_accuracy

print("Loading Phonemes dataset (1489)...")
t0 = time.time()
X, y, cat_mask, feat_names = load_openml_dataset(1489)
splits = preprocess_dataset(X, y, cat_mask, scale="zscore", random_state=0)
inp_d = splits.x_train.shape[1]
print(f"Loaded and preprocessed in {time.time() - t0:.2f}s. Shape: {splits.x_train.shape}")

print("Testing 1 training of pretrain_ablation + finetune_classifier...")
t1 = time.time()
enc, hist_pt = pretrain_ablation(
    splits.x_train, splits.x_val, inp_d,
    corruption_strategy="marginal", max_epochs=150, device="cpu"
)
t_pt = time.time() - t1
print(f"Pretrain took {t_pt:.2f}s (stopped at epoch {hist_pt.best_epoch}, total epochs {len(hist_pt.train_loss)})")

t2 = time.time()
enc, head, hist_ft = finetune_classifier(
    splits.x_train, splits.y_train, splits.x_val, splits.y_val,
    n_classes=splits.n_classes, encoder=enc, input_dim=inp_d,
    max_epochs=100, device="cpu"
)
t_ft = time.time() - t2
acc = evaluate_accuracy(enc, head, splits.x_test, splits.y_test, device="cpu")
print(f"Finetune took {t_ft:.2f}s (stopped at epoch {hist_ft.best_epoch}, total epochs {len(hist_ft.train_loss)})")
print(f"Total time for 1 run: {t_pt + t_ft:.2f}s, test acc: {acc:.4f}")
