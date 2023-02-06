# -*-coding:utf-8-*-
# Author: WSKH
# Blog: wskh0929.blog.csdn.net
# Time: 2022/12/12 11:05
# 定义训练循环
import torch
from tqdm import tqdm
from transformers import AdamW


def train(train_loader, valid_loader, val_questions, model, tokenizer, config, device):
    logging_step = config['logging_step']

    # Setup optimizer
    optimizer = AdamW(model.parameters(), lr=config['lr'])

    best_acc = -1  # 最佳模型的识别正确率
    # train_init_acc = valid(train_loader, model, tokenizer, val_questions, device)
    # valid_init_acc = valid(valid_loader, model, tokenizer, val_questions, device)

    # acc_record = {'train': [train_init_acc],
    #               'val': [valid_init_acc]}

    for epoch in range(config['num_epoch']):
        step = 1
        train_loss = train_acc = 0

        for data in tqdm(train_loader):
            # Load all data into GPU
            data = [i.to(device) for i in data]

            # Model inputs: input_ids, token_type_ids, attention_mask, start_positions, end_positions (Note: only "input_ids" is mandatory)
            # Model outputs: start_logits, end_logits, loss (return when start_positions/end_positions are provided)
            output = model(input_ids=data[0], token_type_ids=data[1], attention_mask=data[2], start_positions=data[3],
                           end_positions=data[4])

            # Choose the most probable start position / end position
            start_index = torch.argmax(output.start_logits, dim=1)
            end_index = torch.argmax(output.end_logits, dim=1)

            # Prediction is correct only if both start_index and end_index are correct
            train_acc += ((start_index == data[3]) & (end_index == data[4])).float().mean()
            train_loss += output.loss

            output.loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            step += 1

            ##### TODO: Apply linear learning rate decay #####

            # Print training loss and accuracy over past logging step
            if step % logging_step == 0:
                print(
                    f"Epoch {epoch + 1} | Step {step} | loss = {train_loss.item() / logging_step:.3f}, acc = {train_acc / logging_step:.3f}")
                train_loss = train_acc = 0

        val_acc = valid(valid_loader, model, tokenizer, val_questions, device)
        print(f"Validation | Epoch {epoch + 1} | acc = {val_acc / len(valid_loader):.3f}")


def valid(valid_loader, model, tokenizer, val_questions, device):
    model.eval()
    with torch.no_grad():
        val_acc = 0
        for i, data in enumerate(tqdm(valid_loader)):
            output = model(input_ids=data[0].squeeze(dim=0).to(device),
                           token_type_ids=data[1].squeeze(dim=0).to(device),
                           attention_mask=data[2].squeeze(dim=0).to(device))
            # prediction is correct only if answer text exactly matches
            val_acc += evaluate(data, output, tokenizer) == val_questions[i]["answer_text"]
    return val_acc


def evaluate(data, output, tokenizer):
    ##### TODO: Postprocessing #####
    # There is a bug and room for improvement in postprocessing
    # Hint: Open your prediction file to see what is wrong

    answer = ''
    max_prob = float('-inf')
    num_of_windows = data[0].shape[1]

    for k in range(num_of_windows):
        # Obtain answer by choosing the most probable start position / end position
        start_prob, start_index = torch.max(output.start_logits[k], dim=0)
        end_prob, end_index = torch.max(output.end_logits[k], dim=0)

        # Probability of answer is calculated as sum of start_prob and end_prob
        prob = start_prob + end_prob

        # Replace answer if calculated probability is larger than previous windows
        if prob > max_prob:
            max_prob = prob
            # Convert tokens to chars (e.g. [1920, 7032] --> "大 金")
            answer = tokenizer.decode(data[0][0][k][start_index: end_index + 1])

    # Remove spaces in answer (e.g. "大 金" --> "大金")
    return answer.replace(' ', '')
