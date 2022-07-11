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

parser = argparse.ArgumentParser(description='Process some arguments')
parser.add_argument('--model_name_or_path', type=str, default='microsoft/codebert-base')


parser.add_argument('--train_mark_path', type=str, default='train_mark.csv')
parser.add_argument('--train_features_path', type=str, default='train_fts.json')
parser.add_argument('--val_mark_path', type=str, default='val_mark.csv')
#parser.add_argument('--val_features_path', type=str, default='val_fts.csv')
parser.add_argument('--val_features_path', type=str, default='val_fts.json')
parser.add_argument('--val_path', type=str, default="val.csv")


parser.add_argument('--md_max_len', type=int, default=64)
parser.add_argument('--total_max_len', type=int, default=512)
parser.add_argument('--num_code_cell', type=int, default=20, 
        help='num of code cells for current markdown as context')

parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--accumulation_steps', type=int, default=4)
parser.add_argument('--epochs', type=int, default=5)
parser.add_argument('--n_workers', type=int, default=8)

parser.add_argument('--data-path', type=str, default='/workspace/jpx/ai4code/', help='data path of csv')
parser.add_argument('--feature-file-path', type=str, default='./data', help='path of feature files')
parser.add_argument('--out-model-path', type=str, default='./outputs.mgpu/', help='output checkpointpath')

parser.add_argument('--loss', type=str, default='L1', help='L1 or MSE loss')

# initial weight path
parser.add_argument('--weights', type=str, default='', help='initial weights (existing checkpoint) path')
parser.add_argument('--device', default='cuda:0', help='device id (i.e., 0 or 0,1 or cpu)')

args = parser.parse_args()
print(args)
#os.makedirs("./outputs", exist_ok=True)
#data_dir = Path('..//input/')
#data_dir = Path('/workspace/jpx/ai4code/')
data_dir = Path(args.data_path)

if os.path.exists(args.out_model_path) is False:
    os.makedirs(args.out_model_path)

#import ipdb; ipdb.set_trace()

# 1
train_mark_path = os.path.join(args.feature_file_path, args.train_mark_path)
print('read train mark file = {}'.format(train_mark_path))
train_df_mark = pd.read_csv(train_mark_path).drop("parent_id", axis=1).dropna().reset_index(drop=True)

# 2
train_features_path = os.path.join(args.feature_file_path, args.train_features_path)
print('read train features file = {}'.format(train_features_path))
train_fts = json.load(open(train_features_path))

# 3
val_mark_path = os.path.join(args.feature_file_path, args.val_mark_path)
print('read val mark file = {}'.format(val_mark_path))
val_df_mark = pd.read_csv(val_mark_path).drop("parent_id", axis=1).dropna().reset_index(drop=True)

# 4
val_features_path = os.path.join(args.feature_file_path, args.val_features_path)
print('read val features file = {}'.format(val_features_path))
val_fts = json.load(open(val_features_path))

# 5
val_path = os.path.join(args.feature_file_path, args.val_path)
print('read val path file = {}'.format(val_path))
val_df = pd.read_csv(val_path)

val_df["pred"] = val_df.groupby(["id", "cell_type"])["rank"].rank(pct=True)

#order_df = pd.read_csv(os.path.join(data_dir, "train_orders.csv")).set_index("id")
df_orders = pd.read_csv(
    data_dir / 'train_orders.csv',
    index_col='id',
    squeeze=True,
).str.split()

train_ds = MarkdownDataset(train_df_mark, 
        model_name_or_path=args.model_name_or_path, 
        md_max_len=args.md_max_len,
        total_max_len=args.total_max_len, 
        num_code_cell=args.num_code_cell,
        fts=train_fts)

val_ds = MarkdownDataset(val_df_mark, 
        model_name_or_path=args.model_name_or_path, 
        md_max_len=args.md_max_len,
        total_max_len=args.total_max_len, 
        num_code_cell=args.num_code_cell,
        fts=val_fts)

train_loader = DataLoader(train_ds, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.n_workers,
        pin_memory=False, 
        drop_last=True)

val_loader = DataLoader(val_ds, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.n_workers,
        pin_memory=False, 
        drop_last=False)

def read_data(data, device):
    #return tuple(d.cuda() for d in data[:-1]), data[-1].cuda()
    # case 1: ids, mask, fts, target (pct_rank) 
    # case 2: ids, mask, fts, target (pct_rank), json_ids, cell_ids 
    target_idx = -1 if (len(data) == 4) else -3
    return tuple(d.to(device) for d in data[:target_idx]), data[target_idx].to(device)

def validate(model, val_loader, device):
    model.eval()

    tbar = tqdm(val_loader, file=sys.stdout)

    preds, labels = [], []
    json_id_list, cell_id_list = [], []

    with torch.no_grad():
        for idx, data in enumerate(tbar):
            inputs, target = read_data(data, device)
            json_ids, cell_ids = None, None
            if len(data) == 6:
                json_ids = data[-2] # e.g., ('000b8e6d58544b', '000b8e6d58544b') 
                cell_ids = data[-1] # e.g., ('5287437b', 'e8672233')

            with torch.cuda.amp.autocast():
                pred = model(*inputs)

            preds.append(pred.detach().cpu().numpy().ravel())
            labels.append(target.detach().cpu().numpy().ravel())
            
            if json_ids is not None and cell_ids is not None:
                json_id_list.extend(list(json_ids))
                cell_id_list.extend(list(cell_ids))

            #break # TODO for multi-gpu debug only!
    #import ipdb; ipdb.set_trace()
    return np.concatenate(labels), np.concatenate(preds), np.array(json_id_list), np.array(cell_id_list)


def train(model, train_loader, val_loader, epochs, device):
    np.random.seed(0)
    # Creating optimizer and lr schedulers
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    num_train_optimization_steps = int(args.epochs * len(train_loader) / args.accumulation_steps)
    optimizer = AdamW(optimizer_grouped_parameters, lr=3e-5,
                      correct_bias=False)  # To reproduce BertAdam specific behavior set correct_bias=False
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0.05 * num_train_optimization_steps,
                                                num_training_steps=num_train_optimization_steps)  # PyTorch scheduler

    criterion = torch.nn.L1Loss() if args.loss == 'L1' else torch.nn.MSELoss()
    # TODO torch.nn.MSELoss() # can try other types of loss functions!
    print('loss=', criterion)

    scaler = torch.cuda.amp.GradScaler()
    
    #import ipdb; ipdb.set_trace()
    eva_first = True
    if eva_first:
        y_val, y_pred, json_ids, cell_ids = validate(model, val_loader, device)
        #import ipdb; ipdb.set_trace()

        #val_df["pred"] = val_df.groupby(["id", "cell_type"])["rank"].rank(pct=True)
        val_df.loc[val_df["cell_type"] == "markdown", "pred"] = y_pred # TODO not same size for multi-gpu
        y_dummy = val_df.sort_values("pred").groupby('id')['cell_id'].apply(list)

        ktau = kendall_tau(df_orders.loc[y_dummy.index], y_dummy)
        print("Preds score", ktau) #kendall_tau(df_orders.loc[y_dummy.index], y_dummy))

    #import ipdb; ipdb.set_trace()
    for e in range(epochs):
        print('start training, epoch={}'.format(e))
        model.train()
        tbar = tqdm(train_loader, file=sys.stdout)
        loss_list = []
        preds = []
        labels = []

        for idx, data in enumerate(tbar):
            inputs, target = read_data(data, device)

            with torch.cuda.amp.autocast():
                pred = model(*inputs)
                loss = criterion(pred, target)
            scaler.scale(loss).backward()
            if idx % args.accumulation_steps == 0 or idx == len(tbar) - 1:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

            loss_list.append(loss.detach().cpu().item())
            preds.append(pred.detach().cpu().numpy().ravel())
            labels.append(target.detach().cpu().numpy().ravel())

            avg_loss = np.round(np.mean(loss_list), 4)

            tbar.set_description(f"Epoch {e} Loss: {avg_loss} lr: {scheduler.get_last_lr()}")

        y_val, y_pred, json_ids, cell_ids = validate(model, val_loader, device)

        #val_df["pred"] = val_df.groupby(["id", "cell_type"])["rank"].rank(pct=True)
        val_df.loc[val_df["cell_type"] == "markdown", "pred"] = y_pred
        y_dummy = val_df.sort_values("pred").groupby('id')['cell_id'].apply(list)

        ktau = kendall_tau(df_orders.loc[y_dummy.index], y_dummy)
        print("Preds score", ktau) #kendall_tau(df_orders.loc[y_dummy.index], y_dummy))

        out_model = os.path.join(args.out_model_path, "model_{}_ktau{}.bin".format(e, ktau))
        print('save checkpoint to {}'.format(out_model))
        torch.save(model.state_dict(), out_model) 

    return model, y_pred

model = MarkdownModel(args.model_name_or_path) # parameter size = 124,645,632 = 124.6M

# TODO
weights_path = args.weights

if os.path.exists(weights_path):
    device = torch.device(args.device)
    weights_dict = torch.load(weights_path, map_location=device)
    load_weights_dict = {k:v for k, v in weights_dict.items()
            if model.state_dict()[k].numel() == v.numel()}
    model.load_state_dict(load_weights_dict, strict=False)
    print('loaded checkpoint from: {}'.format(weights_path))

#model = model.cuda()
model = model.to(args.device)
model, y_pred = train(model, train_loader, val_loader, epochs=args.epochs, device=args.device)

