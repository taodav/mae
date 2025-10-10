import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np
import pandas as pd

rc('font', **{'family': 'serif', 'serif': ['cmr10'], 'size': 18})
rc('axes', unicode_minus=False)


if __name__ == "__main__":
    experiment_dir = Path("/Users/ruoyutao/Documents/mae/output/old_linprobe")
    max_key = 'test_acc1'

    all_res = {}
    max_res_for_each_hparam = {}

    for ckpt_dir in experiment_dir.iterdir():
        name = ckpt_dir.name
        log_fpath = ckpt_dir / "log.txt"
        with open(log_fpath, 'r') as f:
            all_epoch_res = []
            max_val, max_val_idx = -float('inf'), None
            for i, line in enumerate(f):
                res = json.loads(line)
                all_epoch_res.append(res)

                if res[max_key] > max_val:
                    max_val = res[max_key]
                    max_val_idx = i
            all_res[name] = all_epoch_res

            exp_arr_id, model_str, blr_str, crop_min_str, color_jitter = name.split('__')

            arr_id = int(exp_arr_id.split('_')[-1])
            model = model_str.replace('model_', '')
            blr = float(blr_str.split('_')[-1])
            crop_min = float(crop_min_str.split('_')[-1])
            color_jitter = True if color_jitter[0] == '+' else False

            max_res_for_each_hparam[arr_id] = {
                'model': model,
                'blr': blr,
                'crop_scale_min': crop_min,
                'color_jitter': color_jitter,
                **all_epoch_res[max_val_idx]
            }

    indices = list(max_res_for_each_hparam.keys())
    for_df = [max_res_for_each_hparam[i] for i in indices]
    df = pd.DataFrame(for_df, index=indices)

    csm_mean_df = df.groupby('crop_scale_min').mean(numeric_only=True)
    csm_sem_df = df.groupby('crop_scale_min').sem(numeric_only=True)
    csm_max_df = df.groupby('crop_scale_min').max(numeric_only=True)

    blr_mean_df = df.groupby('blr').mean(numeric_only=True)
    blr_sem_df = df.groupby('blr').sem(numeric_only=True)
    blr_max_df = df.groupby('blr').max(numeric_only=True)

    cj_mean_df = df.groupby('color_jitter').mean(numeric_only=True)
    cj_sem_df = df.groupby('color_jitter').sem(numeric_only=True)
    cj_max_df = df.groupby('color_jitter').max(numeric_only=True)

    all_keys = ['crop_scale_min', 'blr', 'color_jitter']
    all_means = [csm_mean_df, blr_mean_df, cj_mean_df]
    all_sems = [csm_sem_df, blr_sem_df, cj_sem_df]
    all_max_dfs = [csm_max_df, blr_max_df, cj_max_df]

    fig = plt.figure(figsize=(12, 8))  # Increased figure size
    subfigs = fig.subfigures(2, 1, height_ratios=[1, 1])  # Added hspace
    axes_mean = subfigs[0].subplots(1, 3)
    subfigs[0].suptitle('Mean')

    for ax, mean_df, sem_df in zip(axes_mean, all_means, all_sems):
        x = mean_df.index
        key = x.name
        if key == 'blr':
            x = np.log10(x)
        y = mean_df[max_key]
        ax.plot(x, y)
        ax.fill_between(x, y - sem_df[max_key], y + sem_df[max_key], alpha=0.2)
        ax.set_title(key)
        if key == 'blr':
            ax.set_xlabel('log ' + key)
        else:
            ax.set_xlabel(key)
        ax.set_ylabel(max_key)

    subfigs[1].suptitle('Max')
    axes_max = subfigs[1].subplots(1, 3)

    for ax, max_df in zip(axes_max, all_max_dfs):
        x = max_df.index
        key = x.name
        if key == 'blr':
            x = np.log10(x)
        y = max_df[max_key]
        ax.plot(x, y)
        ax.set_title(key)
        if key == 'blr':
            ax.set_xlabel('log ' + key)
        else:
            ax.set_xlabel(key)
        ax.set_ylabel(max_key)

    # fig.tight_layout()
    plt.subplots_adjust(top=0.8, hspace=0.3, wspace=0.3, bottom=0.2)
    plt.show()
    fig.savefig('linprobe_results.pdf')
