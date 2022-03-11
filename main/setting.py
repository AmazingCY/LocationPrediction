# -*- coding: utf-8 -*-
# @Time : 2021/12/11 21:50
# @Author : Cao yu
# @File : setting.py
# @Software: PyCharm

import torch
import argparse
import sys
import os
from model import RnnFactory


class Setting:
    '''
    Defines all settings in a single place using a command line interface.
    '''

    def parse(self):
        self.guess_foursquare = any(
            ['Foursquare' in argv for argv in sys.argv])  # foursquare has different default args.
        self.guess_brightkite = any(['Brightkite' in argv for argv in sys.argv])
        self.guess_AIS = any(['AIS' in argv for argv in sys.argv])
        parser = argparse.ArgumentParser()
        if self.guess_foursquare:
            self.parse_foursquare(parser)
        elif self.guess_brightkite:
            self.parse_brightkite(parser)
        elif self.guess_AIS:
            self.parse_AIS(parser)
        else:
            self.parse_gowalla(parser)

        self.parse_arguments(parser)
        args = parser.parse_args()

        ###### settings ######
        # training
        self.gpu = args.gpu
        self.hidden_dim = args.hidden_dim
        self.weight_decay = args.weight_decay
        self.learning_rate = args.lr
        self.epochs = args.epochs
        self.rnn_factory = RnnFactory(args.rnn)
        self.is_lstm = self.rnn_factory.is_lstm()
        self.is_transformer = self.rnn_factory.is_transformer()
        self.lambda_t = args.lambda_t
        self.lambda_s = args.lambda_s

        # data management
        self.dataset_file = './data/{}'.format(args.dataset)
        self.local_dataset_file = "C:/Users/Caoyu/Desktop/MyProject/LocationPrediction/data/AIS/AIS_one_month_short.txt"
        self.max_users = 0  # 0 = use all available users
        self.sequence_length = 10
        self.batch_size = args.batch_size
        self.min_checkins = 50

        # evaluation
        self.validate_epoch = args.validate_epoch
        self.report_user = args.report_user

        ### CUDA Setup ###
        self.device = torch.device('cpu') if args.gpu == -1 else torch.device('cuda:0')

    def parse_arguments(self, parser):
        # training
        parser.add_argument('--gpu', default=-1, type=int, help='the gpu to use')
        parser.add_argument('--hidden-dim', default=32, type=int, help='hidden dimensions to use')
        parser.add_argument('--weight_decay', default=0.0, type=float, help='weight decay regularization')
        parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
        parser.add_argument('--epochs', default=50, type=int, help='amount of epochs')
        parser.add_argument('--rnn', default='transformer', type=str,
                            help='the GRU implementation to use: [rnn|gru|lstm]')

        # data management
        parser.add_argument('--dataset', default='Gowalla_filter.txt', type=str,
                            help='the data under ./data/<data.txt> to load')

        # evaluation
        parser.add_argument('--validate-epoch', default=5, type=int,
                            help='run each validation after this amount of epochs')
        parser.add_argument('--report-user', default=-1, type=int,
                            help='report every x user on evaluation (-1: ignore)')

    def parse_gowalla(self, parser):
        # defaults for gowalla data
        parser.add_argument('--batch-size', default=300, type=int,
                            help='amount of users to process in one pass (batching)')
        parser.add_argument('--lambda_t', default=0.1, type=float, help='decay factor for temporal data')
        parser.add_argument('--lambda_s', default=1000, type=float, help='decay factor for spatial data')

    def parse_foursquare(self, parser):
        # defaults for foursquare data
        parser.add_argument('--batch-size', default=500, type=int,
                            help='amount of users to process in one pass (batching)')
        parser.add_argument('--lambda_t', default=0.1, type=float, help='decay factor for temporal data')
        parser.add_argument('--lambda_s', default=100, type=float, help='decay factor for spatial data')

    def parse_brightkite(self, parser):
        # defaults for gowalla data
        parser.add_argument('--batch-size', default=600, type=int,
                            help='amount of users to process in one pass (batching)')
        parser.add_argument('--lambda_t', default=0.1, type=float, help='decay factor for temporal data')
        parser.add_argument('--lambda_s', default=1500, type=float, help='decay factor for spatial data')

    def parse_AIS(self, parser):
        # defaults for AIS data
        parser.add_argument('--batch-size', default=600, type=int,
                            help='amount of users to process in one pass (batching)')
        parser.add_argument('--lambda_t', default=0.1, type=float, help='decay factor for temporal data')
        parser.add_argument('--lambda_s', default=150, type=float, help='decay factor for spatial data')

    def parse_sign(self):
        if self.guess_AIS:
            return 'AIS'
        else:
            pass

    def __str__(self):
        if self.guess_foursquare:
            return ('parse with foursquare default settings') + '\n' + 'use device: {}'.format(self.device) + \
                   '\n' + ("Dataset:Foursquare")
        elif self.guess_brightkite:
            return ('parse with brightkite default settings') + '\n' + 'use device: {}'.format(self.device) + \
                   '\n' + ("Dataset:Brightkite")
        elif self.guess_AIS:
            return ('parse with AIS default settings') + '\n' + 'use device: {}'.format(self.device) + \
                   '\n' + ("Dataset:AIS")
        else:
            return ('parse with gowalla default settings') + '\n' + 'use device: {}'.format(self.device) + \
                   '\n' + ("Dataset:Gowalla")
