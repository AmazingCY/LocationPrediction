# -*- coding: utf-8 -*-
# @Time : 2021/12/11 21:44
# @Author : Cao yu
# @File : train.py
# @Software: PyCharm

import torch
from torch.utils.data import DataLoader
import numpy as np
import os
from tensorboardX import SummaryWriter
from main import setting
from main import evaluation
from pre_dataset import PoiDataloader, AisDataloader, Split
from model import create_h0_strategy, LocationPredictionTrainer
import time

'''
Main train script to invoke from commandline.
'''

# select GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

if __name__ == "__main__":

    # parse settings and print which data is used
    dataset_setting = setting.Setting()
    dataset_setting.parse()
    dataset_sign = dataset_setting.parse_sign()
    print(dataset_setting)

    # loading data

    if dataset_sign == 'AIS':
        loc_loader = AisDataloader(dataset_setting.max_users, dataset_setting.min_checkins)
    else:
        loc_loader = PoiDataloader(dataset_setting.max_users, dataset_setting.min_checkins)

    # local test
    if torch.cuda.is_available():
        loc_loader.read(dataset_setting.dataset_file)
    else:
        loc_loader.read(dataset_setting.local_dataset_file)

    # batch_size warning
    assert dataset_setting.batch_size < loc_loader.user_count(), 'batch size must be lower than the amount of available users'
    # TRAIN DATASET
    dataset_train = loc_loader.create_dataset(dataset_setting.sequence_length, dataset_setting.batch_size, Split.TRAIN)
    dataloader_trian = DataLoader(dataset_train, batch_size=1, shuffle=False)

    # TEST DATASET
    dataset_test = loc_loader.create_dataset(dataset_setting.sequence_length, dataset_setting.batch_size, Split.TEST)
    dataloader_test = DataLoader(dataset_test, batch_size=1, shuffle=False)

    # create location prediction trainer
    trainer = LocationPredictionTrainer(dataset_setting.lambda_t, dataset_setting.lambda_s)  # initialization trainer
    h0_strategy = create_h0_strategy(dataset_setting.hidden_dim, dataset_setting.is_lstm)
    trainer.prepare(loc_loader.locations(), loc_loader.user_count(), dataset_setting.hidden_dim, \
                    dataset_setting.rnn_factory, dataset_setting.device, dataset_setting.is_transformer)
    evaluation_test = evaluation.Evaluation(dataset_test, dataloader_test, loc_loader.user_count(), \
                                            h0_strategy, trainer, dataset_setting,dataset_sign)
    print('{} {}'.format(trainer, dataset_setting.rnn_factory))

    # optimizer
    optimizer = torch.optim.Adam(trainer.parameters(), \
                                 lr=dataset_setting.learning_rate, weight_decay=dataset_setting.weight_decay)

    # training trick
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20, 30, 40], gamma=0.1)

    # record result
    writer = SummaryWriter('./log/exp1')

    # training loop
    for e in range(dataset_setting.epochs):
        start = time.time()
        h = h0_strategy.on_init(dataset_setting.batch_size, dataset_setting.device)
        dataset_train.shuffle_users()  # shuffle users before each epoch!
        # print(h.size())
        losses = []

        if dataset_sign == 'AIS':

            for i, (his_loc, his_t, his_SOG, his_COG, his_cor, \
                    tar_loc, tar_t, tar_SOG, tar_COG, tar_cor, \
                    reset_h, active_users) in enumerate(dataloader_trian):

                for j, reset in enumerate(reset_h):
                    if reset:
                        if dataset_setting.is_lstm:
                            hc = h0_strategy.on_reset(active_users[0][j])
                            h[0][0, j] = hc[0]
                            h[1][0, j] = hc[1]
                        else:
                            hc_ = h0_strategy.on_reset(active_users[0][j])
                            h[0, j] = hc_[0]
                            h[1, j] = hc_[1]

                h_loc = his_loc.squeeze().to(dataset_setting.device)
                h_t = his_t.squeeze().to(dataset_setting.device)
                h_SOG = his_SOG.squeeze().to(dataset_setting.device)
                h_COG = his_COG.squeeze().to(dataset_setting.device)
                h_cor = his_cor.squeeze().to(dataset_setting.device)
                t_loc = tar_loc.squeeze().to(dataset_setting.device)
                t_t = tar_t.squeeze().to(dataset_setting.device)
                t_SOG = tar_SOG.squeeze().to(dataset_setting.device)
                t_COG = tar_COG.squeeze().to(dataset_setting.device)
                t_cor = tar_cor.squeeze().to(dataset_setting.device)
                active_users = active_users.to(dataset_setting.device)

                optimizer.zero_grad()
                loss, h = trainer.loss_ais(h_loc, h_t, h_SOG, h_COG, h_cor, \
                                           t_loc, t_t, t_SOG, t_COG, t_cor, \
                                           h,active_users)
                loss.backward(retain_graph=True)
                losses.append(loss.item())
                optimizer.step()
        else:

            for i, (x, t, s, y, y_t, y_s, reset_h, active_users) in enumerate(dataloader_trian):
                # reset hidden states for newly added user

                for j, reset in enumerate(reset_h):
                    if reset:
                        if dataset_setting.is_lstm:
                            hc = h0_strategy.on_reset(active_users[0][j])
                            h[0][0, j] = hc[0]
                            h[1][0, j] = hc[1]
                        else:
                            hc_ = h0_strategy.on_reset(active_users[0][j])
                            h[0, j] = hc_[0]
                            h[1, j] = hc_[1]
                            # h[2, j] = hc_[2]
                            # h[3, j] = hc_[3]

                x = x.squeeze().to(dataset_setting.device)
                t = t.squeeze().to(dataset_setting.device)
                s = s.squeeze().to(dataset_setting.device)
                y = y.squeeze().to(dataset_setting.device)
                y_t = y_t.squeeze().to(dataset_setting.device)
                y_s = y_s.squeeze().to(dataset_setting.device)
                active_users = active_users.to(dataset_setting.device)

                optimizer.zero_grad()
                loss, h = trainer.loss_checkin(x, t, s, y, y_t, y_s, h, active_users)
                loss.backward(retain_graph=True)
                losses.append(loss.item())
                optimizer.step()

        # schedule learning rate:
        scheduler.step()

        # record loss
        writer.add_scalar("loss", np.mean(losses), e)
        
        # time record
        end = time.time()
        run_time = end - start
        
        # statistics result:
        if (e + 1) % 1 == 0:
            epoch_loss = np.mean(losses)
            print('***Epoch:{:d}/{:d}***'.format(e + 1, dataset_setting.epochs))
            print('Used learning rate:{:,}'.format(scheduler.get_lr()[0]), '-----',
                  'Avg Loss:{:.3f}'.format(epoch_loss))
            print('Train Time:{:.3f}'.format(run_time))
        if (e + 1) % dataset_setting.validate_epoch == 0:
            print('~~~ Test Set Evaluation (Epoch:{:d}) ~~~'.format(e + 1))
            evaluation_test.evaluate()
