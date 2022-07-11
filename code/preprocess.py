import json
from pathlib import Path
import numpy as np
import pandas as pd
#from scipy import sparse
from tqdm import tqdm
import os
import argparse
from sklearn.model_selection import GroupShuffleSplit

def parse_args():
    parser = argparse.ArgumentParser(description='Data Proprecessing for CodeBERT fine-tuning')
    
    parser.add_argument('--data_dir', type=str, default='/workspace/jpx/ai4code/')
    parser.add_argument('--out_data_dir', type=str, default='./data_debug/')
    parser.add_argument('--keep_code_len', type=int, default=200)
    parser.add_argument('--keep_code_cell_num', type=int, default=20)
    parser.add_argument('--json_file_num', type=int, default=-1)
    parser.add_argument('--valid_ratio', type=float, default=0.1)

    args = parser.parse_args()
    return args

def read_files(args):
    data_dir = Path(args.data_dir)
    out_data_dir = Path(args.out_data_dir)
    if not os.path.exists(out_data_dir):
        os.mkdir(out_data_dir)

    def read_notebook(path):
        return (
            pd.read_json(
                path,
                dtype={'cell_type': 'category', 'source': 'str'}
                ).assign(id=path.stem).rename_axis('cell_id')
        )

    paths_train = list((data_dir / 'train').glob('*.json')) # 139,256 files
    alen = len(paths_train) if args.json_file_num == -1 else args.json_file_num
    print('reading {} json files:'.format(alen))
    notebooks_train = [
        read_notebook(path) for path in tqdm(paths_train[:alen], desc='Train NBs') 
    ]

    df = (
        pd.concat(notebooks_train)
            .set_index('id', append=True)
            .swaplevel()
            .sort_index(level='id', sort_remaining=False)
    ) # e.g., df.shape=[3203, 2], columns = (id, cell_id, cell_type, source)
    # 'id' and 'cell_id' are indices, not data columns
    
    print('reading train orders:')
    df_orders = pd.read_csv(
        data_dir / 'train_orders.csv',
        index_col='id',
        squeeze=True,
    ).str.split()  # Split the string representation of cell_ids into a list

    def get_ranks(base, derived):
        return [base.index(d) for d in derived]

    # combine df_orders and df
    df_orders_ = df_orders.to_frame().join(
        df.reset_index('cell_id').groupby('id')['cell_id'].apply(list),
        how='right',
    ) # (100, 2), 3 columns, (id, cell_order=参考答案, cell_id=json里面的顺序)
    #                                                       cell_order                                            cell_id
    #id
    #07dc688848f594  [a3aeb78d, 829e2acf, 6bbfc969, b4ef346c, ecb12...  [6bbfc969, ecb128cb, 8ef84d81, 6e127a6f, 318e2...

    ranks = {}
    for id_, cell_order, cell_id in df_orders_.itertuples():
        ranks[id_] = {'cell_id': cell_id, 'rank': get_ranks(cell_order, cell_id)}

    df_ranks = (
        pd.DataFrame
            .from_dict(ranks, orient='index')
            .rename_axis('id')
            .apply(pd.Series.explode)
            .set_index('cell_id', append=True)
    )
    #                        rank
    #id             cell_id
    #07dc688848f594 6bbfc969    2
    #               ecb128cb    4
    #[4203 rows x 1 columns]
    
    print('reading ancestor relations:')
    df_ancestors = pd.read_csv(data_dir / 'train_ancestors.csv', index_col='id')
    #(139256, 2)
    #               ancestor_id       parent_id
    #id
    #00015c83e2717b    aa2da37e  317b65d12af9df

    df = df.reset_index().merge(df_ranks, on=["id", "cell_id"]).merge(df_ancestors, on=["id"])
    #(4203, 7)
    #ipdb> df.head(2)
    #               id   cell_id cell_type  ... rank ancestor_id       parent_id
    #0  07dc688848f594  6bbfc969      code  ...    2    7ed25a40  d540db9ace3eb7

    df["pct_rank"] = df["rank"] / df.groupby("id")["cell_id"].transform("count")

    NVALID = args.valid_ratio #0.1  # size of validation set
    splitter = GroupShuffleSplit(n_splits=1, test_size=NVALID, random_state=0)
    # GroupShuffleSplit(n_splits=1, random_state=0, test_size=0.1, train_size=None)

    train_ind, val_ind = next(splitter.split(df, groups=df["ancestor_id"]))
    #ipdb> train_ind = array([   0,    1,    2, ..., 4200, 4201, 4202])
    #ipdb> val_ind = array([ 435,  436,  437,  438,  439,  440,  441,  442,  443,  444,  445,
    print('split train={} and valid={} sets'.format(len(train_ind), len(val_ind)))

    train_df = df.loc[train_ind].reset_index(drop=True)
    val_df = df.loc[val_ind].reset_index(drop=True)

    # Base markdown dataframes
    train_df_mark = train_df[train_df["cell_type"] == "markdown"].reset_index(drop=True)
    val_df_mark = val_df[val_df["cell_type"] == "markdown"].reset_index(drop=True)

    train_df_mark.to_csv(os.path.join(out_data_dir, "train_mark.csv"), index=False)
    val_df_mark.to_csv(os.path.join(out_data_dir, "val_mark.csv"), index=False)

    val_df.to_csv(os.path.join(out_data_dir, "val.csv"), index=False)
    train_df.to_csv(os.path.join(out_data_dir, "train.csv"), index=False)

    return train_df, val_df

# Additional code cells
def clean_code(cell):
    return str(cell).replace("\\n", "\n")

def sample_cells(cells, n, keep_code_len):
    cells = [clean_code(cell) for cell in cells]
    if n >= len(cells):
        #parser.add_argument('--keep_code_len', type=int, default=200)
        return [cell[:keep_code_len] for cell in cells] 
        # 200, 代表的是一行代码的前200个char! 可以微调的！！！ TODO
    else:
        results = []
        step = len(cells) / n
        idx = 0
        while int(np.round(idx)) < len(cells):
            results.append(cells[int(np.round(idx))])
            idx += step
        assert cells[0] in results
        if cells[-1] not in results:
            results[-1] = cells[-1]
        return results

def get_features(df, args):
    features = dict()
    df = df.sort_values("rank").reset_index(drop=True) # 这是根据reference，重新排序，得到的df是reference

    for idx, sub_df in tqdm(df.groupby("id")):
        features[idx] = dict()
        total_md = sub_df[sub_df.cell_type == "markdown"].shape[0] # e.g., total_md = 1
        code_sub_df = sub_df[sub_df.cell_type == "code"] # (13, 8), since there is one markdown, thus from (14,8) to (13,8)
        total_code = code_sub_df.shape[0] # total_code=13

        #parser.add_argument('--keep_code_cell_num', type=int, default=20)
        #parser.add_argument('--keep_code_len', type=int, default=200)
        codes = sample_cells(code_sub_df.source.values, args.keep_code_cell_num, args.keep_code_len) 

        features[idx]["total_code"] = total_code
        features[idx]["total_md"] = total_md
        features[idx]["codes"] = codes
    return features

def main():
    args = parse_args()
    print(args)
    train_df, val_df = read_files(args)
    
    print('getting features for valid set:')
    val_fts = get_features(val_df, args)
    json.dump(val_fts, open(os.path.join(args.out_data_dir, "val_fts.json"), "wt"))
    # 'wt'模式下，Python写文件时会用\r\n来表示换行。

    print('getting features for train set:')
    train_fts = get_features(train_df, args)
    json.dump(train_fts, open(os.path.join(args.out_data_dir, "train_fts.json"), "wt"))

if __name__ == '__main__':
    main()

