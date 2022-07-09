import sys

from tqdm import tqdm
import torch
import numpy as np

from multi_train_utils.distributed_utils import reduce_value, is_main_process

sys.path.append('../')
from metrics import * # load kendall_tau method 

def read_data(data, device):
    return tuple(d.to(device) for d in data[:-1]), data[-1].to(device)

def train_one_epoch(model, optimizer, scheduler, data_loader, device, epoch, accumulation_steps):
    model.train()
    # mean absolute error (MAE) |x(n) - y(n)|, reduction='mean' by default
    criterion = torch.nn.L1Loss() # CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler()

    mean_loss = torch.zeros(1).to(device)
    optimizer.zero_grad()
    loss_list = [] #, preds, labels = [], [], []

    # 在进程0中打印训练进度
    if is_main_process():
        data_loader = tqdm(data_loader, file=sys.stdout)

    for step, data in enumerate(data_loader):
        inputs, target = read_data(data, device)
        
        with torch.cuda.amp.autocast():
            pred = model(*inputs)
            loss = criterion(pred, target)

        scaler.scale(loss).backward()
        
        if step % accumulation_steps == 0 or step == len(data_loader) - 1:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()
        
        loss = reduce_value(loss, average=True) # TODO 为什么先backward()，然后reduce_value? 不该是反过来吗?
        mean_loss = (mean_loss * step + loss.detach()) / (step + 1)

        loss_list.append(loss.detach().cpu().item())
        #preds.append(pred.detach().cpu().numpy().ravel())
        #labels.append(target.detach().cpu().numpy().ravel())

        avg_loss = np.round(np.mean(loss_list), 4)
        if is_main_process():
            data_loader.set_description('Epoch={}, loss={}, avg_Loss={}, mean_loss={}, LR={}'.format(
                epoch, round(loss.detach().cpu().item(), 4), avg_loss, mean_loss.item(), 
                scheduler.get_last_lr()))
        #break # for debug only TODO

    if device != torch.device('cpu'):
        torch.cuda.synchronize(device)

    return mean_loss.item()

@torch.no_grad()
def evaluate(model, data_loader, device, val_df, df_orders):
    model.eval()

    # 用于存储预测正确的样本个数
    sum_num = torch.zeros(1).to(device)

    preds, labels = [], [] # TODO 这个如何在多个gpu之间搞事情呢???
    # 或者，只计算均值k-tau即可???

    # 在进程0中打印验证进度
    if is_main_process(): # TODO
        data_loader = tqdm(data_loader, file=sys.stdout)

    for step, data in enumerate(data_loader):
        inputs, target = read_data(data, device)

        with torch.cuda.amp.autocast():
            #pred = model(images.to(device))
            pred = model(*inputs)

        preds.append(pred.detach().cpu().numpy().ravel()) # ravel() -> 数组多维度拉成一维数组
        labels.append(target.detach().cpu().numpy().ravel())
        #break # TODO debug only

    y_pred = np.concatenate(preds)
    y_val = np.concatenate(labels)
   
    # TODO need to check this code: [run in cpu!]
    #val_df['pred'] = val_df.groupby(['id', 'cell_type'])['rank'].rank(pct=True)
    #val_df.loc[val_df['cell_type'] == 'markdown', 'pred'] = y_pred
    #y_dummy = val_df.sort_values('pred').groupby('id')['cell_id'].apply(list)

    val_df["pred"] = val_df.groupby(["id", "cell_type"])["rank"].rank(pct=True)
    if is_main_process():
        temp = val_df.loc[val_df["cell_type"] == "markdown", "pred"] 
        print('temp.shape={}'.format(temp.shape))
        print('y_pred.shape={}'.format(y_pred.shape))

    val_df.loc[val_df["cell_type"] == "markdown", "pred"] = y_pred
    y_dummy = val_df.sort_values("pred").groupby('id')['cell_id'].apply(list)


    ktau = kendall_tau(df_orders.loc[y_dummy.index], y_dummy)
    #sum_num += ktau # TODO should put ktau to torch.Tensor...
    sum_num += torch.Tensor([ktau]).to(device) # TODO should put ktau to torch.Tensor...
    
    # 等待所有进程计算完毕
    if device != torch.device("cpu"):
        torch.cuda.synchronize(device) # TODO

    sum_num = reduce_value(sum_num, average=True) # TODO sum up all GPU's results?
    # 每个gpu都会计算，自己负责的那部分”评测集合“数据的，预测正确的个数！所以，所有gpu的结果相加，即可！！！
    return sum_num.item()

