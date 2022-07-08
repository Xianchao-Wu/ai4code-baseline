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

from multi_train_utils.distributed_utils import init_distributed_mode, dist, cleanup # TODO for multi-gpu
from multi_train_utils.train_eval_utils import train_one_epoch, evaluate

def parse_args():
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
    parser.add_argument('--lr', type=float, default=3e-5)

    parser.add_argument('--accumulation_steps', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--n_workers', type=int, default=8)

    parser.add_argument('--data-path', type=str, default='/workspace/jpx/ai4code/', help='data path')
    parser.add_argument('--out-model-path', type=str, default='./outputs.mgpu/', help='output checkpoint path')

    # initial weight path
    parser.add_argument('--weights', type=str, default='', help='initial weights (existing checkpoint) path')

    # TODO for m-gpu
    parser.add_argument('--device', default='cuda', help='device id (i.e., 0 or 0,1 or cpu)')
    parser.add_argument('--world-size', default=4, type=int, help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')

    args = parser.parse_args()
    return args

def train_evaluate(model, train_loader, train_sampler, val_loader,  
        args, device, val_df, df_orders):
    #np.random.seed(0)
    # Creating optimizer and lr schedulers
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    num_train_optimization_steps = int(args.epochs * len(train_loader) / args.accumulation_steps)

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr,
        correct_bias=False)  # To reproduce BertAdam specific behavior set correct_bias=False
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0.05 * num_train_optimization_steps,
        num_training_steps=num_train_optimization_steps)  # PyTorch scheduler
    
    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)

        # train one epoch
        mean_loss = train_one_epoch(model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                data_loader=train_loader,
                device=device,
                epoch=epoch,
                accumulation_steps=args.accumulation_steps)

        scheduler.step()
        
        # evaluate
        ktau = evaluate(model=model,
                data_loader=val_loader,
                device=device, 
                val_df=val_df,
                df_orders=df_orders)

        if rank == 0:
            print('epoch {}, k-tau={}'.format(epoch, round(ktau, 4)))
            torch.save(model.module.state_dict(), 
                    os.path.join(args.out_model_path, 'model_{}_ktau{}.bin'.format(epoch, round(ktau,4))))

def main(args):
    if torch.cuda.is_available() is False:
        raise EnvironmentError('not find GPU device for training.')
    init_distributed_mode(args=args) # TODO for init m-gpu env

    rank = args.rank
    device = torch.device(args.device)
    batch_size = args.batch_size
    weights_path = args.weights # existing checkpoint path, if exists
    args.lr *= args.world_size # TODO
    checkpoint_path = ""

    if rank == 0:
        print(args)
        if os.path.exists(args.out_model_path) is False:
            os.makedirs(args.out_model_path)

    data_dir = Path(args.data_path)
    train_df_mark = pd.read_csv(args.train_mark_path).drop("parent_id", axis=1).dropna().reset_index(drop=True)
    train_fts = json.load(open(args.train_features_path))
    val_df_mark = pd.read_csv(args.val_mark_path).drop("parent_id", axis=1).dropna().reset_index(drop=True)
    val_fts = json.load(open(args.val_features_path))
    val_df = pd.read_csv(args.val_path)

    #order_df = pd.read_csv(os.path.join(data_dir, "train_orders.csv")).set_index("id")
    df_orders = pd.read_csv(
        data_dir / 'train_orders.csv',
        index_col='id',
        squeeze=True,
    ).str.split()

    train_ds = MarkdownDataset(train_df_mark, 
            model_name_or_path=args.model_name_or_path, md_max_len=args.md_max_len,
            total_max_len=args.total_max_len, fts=train_fts)
    val_ds = MarkdownDataset(val_df_mark, 
            model_name_or_path=args.model_name_or_path, md_max_len=args.md_max_len,
            total_max_len=args.total_max_len, fts=val_fts)

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_ds)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_ds)

    train_batch_sampler = torch.utils.data.BatchSampler(
            train_sampler, batch_size, drop_last=True)

    if rank == 0:
        print('using {} dataloader workers per process'.format(args.n_workers))

    #train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.n_workers,
    #                          pin_memory=False, drop_last=True)
    #val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.n_workers,
    #                        pin_memory=False, drop_last=False)

    train_loader = DataLoader(train_ds, 
            batch_sampler=train_batch_sampler,
            pin_memory=True,
            num_workers=args.n_workers) #,
            #collate_fn=train_ds.collate_fn) # TODO

    val_loader = DataLoader(val_ds,
            batch_size=args.batch_size,
            sampler=val_sampler,
            pin_memory=True,
            num_workers=args.n_workers)#,
            #collate_fn=val_ds.collate_fn) # TODO

    model = MarkdownModel(args.model_name_or_path).to(device)
    #model = model.cuda()
    #device = 'cuda:1'
    #model = model.to(device)
    # TODO
    ###model, y_pred = train(model, train_loader, val_loader, epochs=args.epochs)

    if os.path.exists(weights_path):
        weights_dict = torch.load(weights_path, map_location=device)
        load_weights_dict = {k:v for k, v in weights_dict.items() 
                if model.state_dict()[k].numel() == v.numel()}
        model.load_state_dict(load_weights_dict, strict=False)
    else:
        checkpoint_path = os.path.join(args.out_model_path, 'init_weights.pt')
        if rank == 0:
            torch.save(model.state_dict(), checkpoint_path)
        dist.barrier()
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    
    # to ddp model
    model = torch.nn.parallel.DistributedDataParallel(model, 
            device_ids=[args.gpu],
            find_unused_parameters=True)


    #pg = [p for p in model.parameters() if p.requires_grad]
    #optimizer = optim.SGD(pg, lr=args.lr, momentum=0.9, weight_decay=0.005)
    
    train_evaluate(model, train_loader, train_sampler, 
            val_loader, args, device, val_df, df_orders)

    if rank == 0:
        # delete 'init_weights.pt'
        if os.path.exists(checkpoint_path) is True:
            os.remove(checkpoint_path)
    cleanup()

if __name__ == '__main__':
    args = parse_args()
    main(args)
