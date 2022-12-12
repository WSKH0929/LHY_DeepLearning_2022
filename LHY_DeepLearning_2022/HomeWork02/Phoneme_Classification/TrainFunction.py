# -*-coding:utf-8-*-
# Author: WSKH
# Blog: wskh0929.blog.csdn.net
# Time: 2022/12/12 11:05
# 定义训练循环
import torch


def train(train_loader, valid_loader, model, config, device):
    """ DNN training """

    n_epochs = config['n_epochs']  # Maximum number of epochs

    # Setup optimizer
    optimizer = getattr(torch.optim, config['optimizer'])(
        model.parameters(), **config['optim_hyper_paras'])

    best_acc = -1  # 最佳模型的识别正确数量
    loss_record = {'train': [valid(train_loader, model, config['device'])[0]],
                   'val': [valid(valid_loader, model, config['device'])[0]]}  # for recording training loss
    early_stop_cnt = 0
    epoch = 0
    while epoch < n_epochs:
        train_acc = 0  # 训练数据中预测正确的数量
        train_loss = 0
        model.train()  # set model to training mode
        for x, y in train_loader:  # iterate through the dataloader
            optimizer.zero_grad()  # set gradient to zero
            x, y = x.to(device), y.to(device)  # move data to device (cpu/cuda)
            pred = model(x)  # forward pass (compute output)
            loss = model.cal_loss(pred, y)  # compute loss
            train_loss += loss.item()
            _, train_pred = torch.max(pred, 1)
            train_acc += (train_pred.cpu() == y.cpu()).sum().item()
            loss.backward()  # compute gradient (backpropagation)
            optimizer.step()  # update model with optimizer
            loss_record['train'].append(loss.detach().cpu().item())
        train_loss = train_loss / len(train_loader)
        # After each epoch, test your model on the validation (development) set.
        val_loss, val_acc = valid(valid_loader, model, device)
        if val_acc > best_acc:
            # Save model if your model improved
            best_acc = val_acc
            print('[{:03d}/{:03d}] Train Acc: {:3.6f} Loss: {:3.6f} | Val Acc: {:3.6f} loss: {:3.6f}'.format(
                epoch + 1, n_epochs, train_acc / len(train_loader.dataset), train_loss,
                val_acc / len(valid_loader.dataset), val_loss
            ))
            torch.save(model.state_dict(), config['model_save_dir'] + "model.pth")  # Save model to specified path
            early_stop_cnt = 0
        else:
            early_stop_cnt += 1

        epoch += 1
        loss_record['val'].append(val_loss)
        if early_stop_cnt > config['early_stop']:
            # Stop training if your model stops improving for "config['early_stop']" epochs.
            break

    print('Finished training after {} epochs'.format(epoch))
    return loss_record


def valid(valid_loader, model, device):
    model.eval()  # set model to evalutation mode
    total_loss = 0
    val_acc = 0
    for x, y in valid_loader:  # iterate through the dataloader
        x, y = x.to(device), y.to(device)  # move data to device (cpu/cuda)
        with torch.no_grad():  # disable gradient calculation
            pred = model(x)  # forward pass (compute output)
            loss = model.cal_loss(pred, y)  # compute loss
            _, val_pred = torch.max(pred, 1)
            # get the index of the class with the highest probability
            val_acc += (val_pred.cpu() == y.cpu()).sum().item()
        total_loss += loss.item()  # accumulate loss
    total_loss = total_loss / len(valid_loader)  # compute averaged loss
    return total_loss, val_acc
