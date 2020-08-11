from dataset import TaxiBJDataset
from STResNet import STResNet
from torch.utils.data import DataLoader
from torch import optim
from utils.mertics import RMSELoss, MAPELoss
from utils.earlystopping import EarlyStopping
import torch
import numpy as np


"""
parameter setting
"""
l_c = 3
l_p = 1
l_t = 1
# flow_dim = 2
# map_height = 32
# map_width = 32

T = 48
days_test = 28
len_test = T * days_test
validate_ratio = 0.1
early_stop_patience = 30
batch_size = 32
learning_rate = 0.0002
epoch_nums = 500
num_workers = 4
random_seed = 1
gpu_id = 0

rmse = RMSELoss().to(gpu_id)
mape = MAPELoss().to(gpu_id)


def set_seed(seed):
    torch.manual_seed(seed)  # cpu
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def rescale_loss(mmn, y, y_pred):
    y_pred = mmn.inverse_transform(y_pred)
    y = mmn.inverse_transform(y)
    rescale_rmse = rmse(y, y_pred)
    rescale_mape = mape(y, y_pred)
    return rescale_rmse, rescale_mape


def validate(model, val_data_loader):
    model.eval()
    val_loss = 0
    for batch_idx, (X_c, X_p, X_t, X_meta, Y_batch) in enumerate(val_data_loader):
        X_c = X_c.type(torch.FloatTensor).to(gpu_id)
        X_p = X_p.type(torch.FloatTensor).to(gpu_id)
        X_t = X_t.type(torch.FloatTensor).to(gpu_id)
        X_meta = X_meta.type(torch.FloatTensor).to(gpu_id)
        Y_batch = Y_batch.type(torch.FloatTensor).to(gpu_id)

        outputs = model(X_c, X_p, X_t, X_meta)
        rmse_loss = rmse(y=Y_batch, y_pred=outputs)
        val_loss += rmse_loss.item()
    print('Validate RMSE Loss: ', val_loss)
    return val_loss


def train():
    set_seed(random_seed)
    stresnet = STResNet(lc=l_c, lp=l_p, lt=l_t).to(gpu_id)
    train_dataset = TaxiBJDataset(mode='train', len_c=l_c, len_p=l_p, len_t=l_t, len_test=len_test)

    validate_size = int(validate_ratio * len(train_dataset))
    train_size = len(train_dataset) - validate_size
    train_data, validate_data = torch.utils.data.random_split(train_dataset, [train_size, validate_size])

    train_data_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    validate_data_loader = DataLoader(validate_data, batch_size=batch_size)

    es = EarlyStopping(patience=early_stop_patience)
    optimizer = optim.Adam(stresnet.parameters(), lr=learning_rate)

    for e in range(epoch_nums):
        epoch_rmse = []
        for batch_idx, (X_c, X_p, X_t, X_meta, Y_batch) in enumerate(train_data_loader):
            X_c = X_c.type(torch.FloatTensor).to(gpu_id)
            X_p = X_p.type(torch.FloatTensor).to(gpu_id)
            X_t = X_t.type(torch.FloatTensor).to(gpu_id)
            X_meta = X_meta.type(torch.FloatTensor).to(gpu_id)
            Y_batch = Y_batch.type(torch.FloatTensor).to(gpu_id)

            optimizer.zero_grad()
            outputs = stresnet(X_c, X_p, X_t, X_meta)

            rmse_loss = rmse(y=Y_batch, y_pred=outputs)

            rmse_loss.backward()
            optimizer.step()
            epoch_rmse.append(rmse_loss.item())

        print('Train Epoch Mean RMSELoss: ', np.mean(epoch_rmse))

        val_loss = validate(stresnet, validate_data_loader)
        if es.step(val_loss):
            print('Early stopped! With val loss:', val_loss)
            stresnet.save(stop_epoch=e)
            break


def evaluate(model_file):
    model = STResNet(lc=l_c, lp=l_p, lt=l_t).to(gpu_id)
    model.load(model_file)
    model.eval()
    test_dataset = TaxiBJDataset(mode='test', len_c=l_c, len_p=l_p, len_t=l_t, len_test=len_test)
    mmn = test_dataset.mmn
    test_data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    test_rmse = []
    test_mape = []
    for batch_idx, (X_c, X_p, X_t, X_meta, Y_batch) in enumerate(test_data_loader):
        X_c = X_c.type(torch.FloatTensor).to(gpu_id)
        X_p = X_p.type(torch.FloatTensor).to(gpu_id)
        X_t = X_t.type(torch.FloatTensor).to(gpu_id)
        X_meta = X_meta.type(torch.FloatTensor).to(gpu_id)
        Y_batch = Y_batch.type(torch.FloatTensor).to(gpu_id)
        outputs = model(X_c, X_p, X_t, X_meta)
        rmse_loss, mape_loss = rescale_loss(mmn=mmn, y=Y_batch, y_pred=outputs)
        test_rmse.append(rmse_loss.item())
        test_mape.append(mape_loss.item())

    print('[Test] RMSE: ', np.mean(test_rmse), ' MAPE: ', np.mean(test_mape))


if __name__ == '__main__':
    # train()
    evaluate('checkpoints/STResNet_0811_14_26_164.pth')