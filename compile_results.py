import os
import numpy as np
import pandas as pd

pick_key = 'val_F1-macro_all'
# pick_key = 'test_F1-macro_all'
# pick_key = 'val_epoch'

# epoch   acc_avg  recall-macro_all  F1-macro_all
exp_dirs = [
    "/vision/u/chpatel/dstest/dann_orig",
    "/vision/u/chpatel/dstest/dann",
    "/vision/u/chpatel/dstest/dann_bsp",
    "/vision/u/chpatel/dstest/dann_nwd",
    "/vision/u/chpatel/dstest/dann_nwd_avoiddann",
    "/vision/u/chpatel/dstest/dann_nwd_avoiddann_neg",
    "/vision/u/chpatel/dstest/dann_nwd_avoiddann_w1",
    "/vision/u/chpatel/dstest/dann_nwd_avoiddann_wneg1",
    "/vision/u/chpatel/dstest/erm",
    "/vision/u/chpatel/dstest/ermnoaug_bnm",
    "/vision/u/chpatel/dstest/ermnoaug_bnm_wneg1",
    "/vision/u/chpatel/dstest/cdan",
    "/vision/u/chpatel/dstest/cdane",
    "/vision/u/chpatel/dstest/dann_bsp_randaug",
]

def multi100(df):
    df['acc_avg'] = df['acc_avg'].apply(lambda x: x*100)
    df['recall-macro_all'] = df['recall-macro_all'].apply(lambda x: x*100)
    df['F1-macro_all'] = df['F1-macro_all'].apply(lambda x: x*100)
    return df

all_exp_dfs = []
for expd in exp_dirs:
    if expd.endswith('/'): expd = expd[:-1]
    name = os.path.basename(expd)
    train_eval = pd.read_csv(os.path.join(expd, 'train_eval.csv'))
    val_eval = pd.read_csv(os.path.join(expd, 'val_eval.csv'))
    test_eval = pd.read_csv(os.path.join(expd, 'test_eval.csv'))

    train_eval = multi100(train_eval)
    val_eval = multi100(val_eval)
    test_eval = multi100(test_eval)

    df = pd.concat([
        train_eval.drop(['epoch'], axis=1).add_prefix('train_'),
        val_eval.add_prefix('val_'), 
        test_eval.drop(['epoch'], axis=1).add_prefix('test_'), 
    ], axis=1).round(1)


    best_idx = df[pick_key].idxmax()
    selected_df = df.loc[[best_idx]]
    selected_df.insert(0, "name", [name,])
    all_exp_dfs.append(selected_df)

final_df = pd.concat(all_exp_dfs)
print(final_df)