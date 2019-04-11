import os
import scipy.io as scio
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np

from torch_baidu_ctc import ctc_loss, CTCLoss
from crnn import CRNN
from IIIT5K import IIIT5K
import crnn_helper


def run():
    train_dataset = IIIT5K()
    train_dataLoader = DataLoader(train_dataset, 64, shuffle=True, num_workers=4)
    test_dataset = IIIT5K(is_train=False)
    test_dataLoader = DataLoader(test_dataset, 100, shuffle=True, num_workers=4)
    
    crnn = CRNN(n_classes=train_dataset.num_classes, use_rnn=False).to("cuda")
    ctc_loss = CTCLoss(average_frames=True, reduction="mean", blank=0)
    optimizer = torch.optim.Adam(crnn.parameters(), lr=0.0001)

    for e in range(200):
        acc_count = 0
        total_loss = 0
        for i, batch_data in enumerate(train_dataLoader):
            image = batch_data["image"].type(torch.FloatTensor) / 255 - 0.5
            image = image.to("cuda")
            log_probs = crnn(image).log_softmax(2)
            targets = batch_data["label"].type(torch.IntTensor)
            targets_ = targets.reshape(-1)
            targets_ = targets_[targets_!=0]

            input_length = batch_data["input_length"]
            target_length = batch_data["target_length"]
            
            optimizer.zero_grad()
            loss = ctc_loss(log_probs, targets_, input_length, target_length)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
            # for i in range(log_probs.shape[0]):
            #     pre_string = crnn_helper.prob_string(log_probs[i])
            #     tag_string = crnn_helper.prob_string(targets[i])
            #     if pre_string == tag_string:
            #         acc_count += 1
        log_probs = log_probs.permute(1, 0, 2)
        for i in range(2):
            pre_string = crnn_helper.prob_string(log_probs[i])
            tag_string = crnn_helper.prob_string(targets[i])
            print(pre_string, tag_string)
        print("trian epoch: {0} loss: {1} acc {2}".format(e, total_loss, acc_count/len(train_dataset)))
        
        with torch.no_grad():
            acc_count = 0
            total_loss = 0
            for i, batch_data in enumerate(test_dataLoader):
                image = batch_data["image"].type(torch.FloatTensor) / 255 - 0.5
                image = image.to("cuda")
                log_probs = crnn(image).log_softmax(2)
                targets = batch_data["label"].type(torch.IntTensor)
                targets_ = targets.reshape(-1)
                targets_ = targets_[targets_!=0]

                input_length = batch_data["input_length"]
                target_length = batch_data["target_length"]
                test_loss = ctc_loss(log_probs, targets_, input_length, target_length)
                total_loss += test_loss.item()

                # for i in range(log_probs.shape[0]):
                #     pre_string = crnn_helper.prob_string(log_probs[i])
                #     tag_string = crnn_helper.prob_string(targets[i])
                #     if pre_string == tag_string:
                #         acc_count += 1
            log_probs = log_probs.permute(1, 0, 2)
            for i in range(2):
                pre_string = crnn_helper.prob_string(log_probs[i])
                tag_string = crnn_helper.prob_string(targets[i])
                print(pre_string, tag_string)
            print("test epoch: {0} loss: {1} acc {2}".format(e, total_loss, acc_count/len(test_dataset)))
if __name__ == "__main__":
    run()