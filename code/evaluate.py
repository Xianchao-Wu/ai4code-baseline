import json
from pathlib import Path
from dataset import *
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from model import *
from tqdm import tqdm
import sys, os
from metrics import *
import torch
import argparse
import time

start_time = time.time()

parser = argparse.ArgumentParser(description='Process some arguments')
parser.add_argument('--model_name_or_path', type=str, default='microsoft/codebert-base')
parser.add_argument('--train_mark_path', type=str, default='./data/train_mark.csv')
parser.add_argument('--train_features_path', type=str, default='./data/train_fts.json')
parser.add_argument('--val_mark_path', type=str, default='./data/val_mark.csv')
#parser.add_argument('--val_features_path', type=str, default='./data/val_fts.csv')
parser.add_argument('--val_features_path', type=str, default='./data/val_fts.json')
parser.add_argument('--val_path', type=str, default="./data/val.csv")

parser.add_argument('--md_max_len', type=int, default=64)
parser.add_argument('--total_max_len', type=int, default=512)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--accumulation_steps', type=int, default=4)
parser.add_argument('--epochs', type=int, default=5)
parser.add_argument('--n_workers', type=int, default=8)

parser.add_argument('--data-path', type=str, default='/workspace/jpx/ai4code/', help='data path')
parser.add_argument('--out-model-path', type=str, default='./outputs.mgpu/', help='output checkpointpath')

# initial weight path
parser.add_argument('--weights', type=str, default='', help='initial weights (existing checkpoint) path')
parser.add_argument('--device', default='cuda:0', help='device id (i.e., 0 or 0,1 or cpu)')

args = parser.parse_args()
data_dir = Path(args.data_path)

#import ipdb; ipdb.set_trace()

print('after parse args = {}'.format(time.time() - start_time))


val_df_mark = pd.read_csv(args.val_mark_path).drop("parent_id", axis=1).dropna().reset_index(drop=True)
val_fts = json.load(open(args.val_features_path))
val_df = pd.read_csv(args.val_path)

#order_df = pd.read_csv(os.path.join(data_dir, "train_orders.csv")).set_index("id")
df_orders = pd.read_csv(
    data_dir / 'train_orders.csv',
    index_col='id',
    squeeze=True,
).str.split()

print('after load data files = {}'.format(time.time() - start_time))

val_ds = MarkdownDataset(val_df_mark, model_name_or_path=args.model_name_or_path, md_max_len=args.md_max_len,
                         total_max_len=args.total_max_len, fts=val_fts)
val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.n_workers,
                        pin_memory=False, drop_last=False)

print('after init dataset and data-loader = {}'.format(time.time() - start_time))

def read_data(data):
    return tuple(d.cuda() for d in data[:-1]), data[-1].cuda()


def validate(model, val_loader):
    model.eval()
    tbar = tqdm(val_loader, file=sys.stdout)

    preds, labels = [], []

    with torch.no_grad():
        for idx, data in enumerate(tbar):
            inputs, target = read_data(data)

            with torch.cuda.amp.autocast():
                pred = model(*inputs)

            preds.append(pred.detach().cpu().numpy().ravel())
            labels.append(target.detach().cpu().numpy().ravel())
    return np.concatenate(labels), np.concatenate(preds)

def evaluate_only(ckpt_name, model, val_loader):
    np.random.seed(0)

    y_val, y_pred = validate(model, val_loader)
    val_df["pred"] = val_df.groupby(["id", "cell_type"])["rank"].rank(pct=True)
    val_df.loc[val_df["cell_type"] == "markdown", "pred"] = y_pred
    y_dummy = val_df.sort_values("pred").groupby('id')['cell_id'].apply(list)

    ktau = kendall_tau(df_orders.loc[y_dummy.index], y_dummy)
    print("Preds score, ktau={}, ckpt={}".format(ktau, ckpt_name)) 
    #kendall_tau(df_orders.loc[y_dummy.index], y_dummy))

model = MarkdownModel(args.model_name_or_path)
print('after init model = {}'.format(time.time() - start_time))

# TODO loop all the existing checkpoints here to speed up:

model_dir = '/workspace/jpx/ai4code/ai4code-baseline/code/outputs.mgpu'
for weights_path_part in os.listdir(model_dir):
    if 'model_0_' in weights_path_part or 'model_1_' in weights_path_part:
        continue

    weights_path = os.path.join(model_dir, weights_path_part) 
    print(weights_path)

    if os.path.exists(weights_path):
        device = torch.device(args.device)
        weights_dict = torch.load(weights_path, map_location=device)
        load_weights_dict = {k:v for k, v in weights_dict.items()
                if model.state_dict()[k].numel() == v.numel()}
        model.load_state_dict(load_weights_dict, strict=False)
        print('loaded checkpoint from: {}'.format(weights_path))
        print('after load checkpoint file = {}'.format(time.time() - start_time))
    else:
        print('error: checkpoint {} is null or not found'.format(weights_path))
        #exit(1)
        continue

    model = model.cuda()
    print('after put model to gpu cuda = {}'.format(time.time() - start_time))

    evaluate_only(weights_path, model, val_loader)
    print('after evaluate = {}'.format(time.time() - start_time))

    print('-'*30)

