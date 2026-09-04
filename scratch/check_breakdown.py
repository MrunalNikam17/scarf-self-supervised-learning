import json
import numpy as np
from scipy import stats

with open('results_local6/raw_results.json') as f:
    raw = json.load(f)

# raw is {method: {dataset: [acc_trial_0, ...]}}
pretrain_methods = ['scarf', 'scarf_ae']
reference_methods = ['control', 'mixup', 'label_smooth']

for pt in pretrain_methods:
    for ref in reference_methods:
        combo = f"{ref}+{pt}"
        print(f"============================================================")
        print(f"Pair: {combo} vs {ref}")
        gains = []
        for d in sorted(raw[ref].keys()):
            acc_m = np.array(raw[combo][d])
            acc_r = np.array(raw[ref][d])
            t, p = stats.ttest_ind(acc_m, acc_r, equal_var=False)
            diff = 100.0 * (acc_m.mean() - acc_r.mean()) / acc_r.mean()
            status = "INCLUDED (p < 0.20)" if p < 0.20 else "EXCLUDED (p >= 0.20)"
            if p < 0.20:
                gains.append(diff)
            print(f"  {d:15s}: ref={acc_r.mean()*100:5.2f}%, combo={acc_m.mean()*100:5.2f}%, diff={diff:+6.2f}%, p={p:6.4f} -> {status}")
        avg_gain = np.mean(gains) if gains else float('nan')
        print(f"Average Relative Gain (Table 1): {avg_gain:.2f}% (from {len(gains)}/{len(raw[ref])} datasets)\n")

