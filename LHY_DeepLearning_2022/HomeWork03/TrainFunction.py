# -*-coding:utf-8-*-
# Author: WSKH
# Blog: wskh0929.blog.csdn.net
# Time: 2022/12/12 11:05
# 定义训练循环
import torch
from torch import nn


def train(train_loader, valid_loader, model, config, device):
    """ CNN training """

    n_epochs = config['n_epochs']  # Maximum number of epochs

    # Setup optimizer
    optimizer = getattr(torch.optim, config['optimizer'])(
        model.parameters(), **config['optim_hyper_paras'])

    best_acc = -1  # 最佳模型的识别正确数量
    loss_record = {'train': [valid(train_loader, model, config['device'])[0]],
                   'val': [valid(valid_loader, model, config['device'])[0]]}  # for recording training loss
    early_stop_cnt = 0
    for epoch in range(n_epochs):
        # ---------- Training ----------
        # Make sure the model is in train mode before training.
        model.train()
        # These are used to record information in training.
        train_loss = 0
        train_acc = 0
        for batch in train_loader:
            # A batch consists of image data and corresponding labels.
            imgs, labels = batch
            # Forward the data. (Make sure data and model are on the same device.)
            logits = model(imgs.to(device))
            # Calculate the cross-entropy loss.
            # We don't need to apply softmax before computing cross-entropy as it is done automatically.
            loss = model.cal_loss(logits, labels.to(device))
            # Gradients stored in the parameters in the previous step should be cleared out first.
            optimizer.zero_grad()
            # Compute the gradients for parameters.
            loss.backward()
            # Clip the gradient norms for stable training.
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)
            # Update the parameters with computed gradients.
            optimizer.step()
            # Compute the accuracy for current batch.
            acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean().item()
            # Record the loss and accuracy.
            loss_record['train'].append(loss.item())
            train_loss += loss.item()
            train_acc += acc
        train_loss = train_loss / len(train_loader)
        train_acc = train_acc / len(train_loader)

        # Print the information.
        print(f"[ Train | {epoch + 1:03d}/{n_epochs:03d} ] loss = {train_loss:.5f}, acc = {train_acc:.5f}", end="\t")

        # ---------- Validation ----------
        valid_loss, valid_acc = valid(valid_loader, model, config['device'])
        loss_record['val'].append(valid_loss)

        # save models
        if valid_acc > best_acc:
            print(
                f"[ Valid | {epoch + 1:03d}/{n_epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f} -> best")
            torch.save(model.state_dict(),
                       config['model_save_dir'] + "model.pth")  # only save best to prevent output memory exceed error
            best_acc = valid_acc
            early_stop_cnt = 0
        else:
            print(f"[ Valid | {epoch + 1:03d}/{n_epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f}")
            early_stop_cnt += 1
            if early_stop_cnt > config['early_stop']:
                print(f"No improvment {config['early_stop']} consecutive epochs, early stopping")
                break

    print('Finished training after {} epochs'.format(epoch))
    return loss_record


def valid(valid_loader, model, device):
    # Make sure the model is in eval mode so that some modules like dropout are disabled and work normally.
    model.eval()
    # These are used to record information in validation.
    valid_loss = 0
    valid_acc = 0
    # Iterate the validation set by batches.
    for batch in valid_loader:
        # A batch consists of image data and corresponding labels.
        imgs, labels = batch
        # imgs = imgs.half()

        # We don't need gradient in validation.
        # Using torch.no_grad() accelerates the forward process.
        with torch.no_grad():
            logits = model(imgs.to(device))

        # We can still compute the loss (but not the gradient).
        loss = model.cal_loss(logits, labels.to(device))

        # Compute the accuracy for current batch.
        acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean().item()

        # Record the loss and accuracy.
        valid_loss += loss.item()
        valid_acc += acc

    # The average loss and accuracy for entire validation set is the average of the recorded values.
    valid_loss = valid_loss / len(valid_loader)
    valid_acc = valid_acc / len(valid_loader)
    return valid_loss, valid_acc
