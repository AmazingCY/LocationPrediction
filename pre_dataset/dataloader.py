# -*- coding: utf-8 -*-
# @Time : 2021/12/11 22:26
# @Author : Cao yu
# @File : dataloader.py
# @Software: PyCharm

import os.path
import sys
import time as ti
from datetime import datetime
from .dataset import PoiDataset, AisDataset, Usage


class PoiDataloader():
    '''
    Creates datasets from our prepared Gowalla/Foursquare/Brightkite data files.
    The file consist of one check-in per line in the following format (tab separated):

    <user-id> <timestamp> <latitude> <longitude> <location-id>

    Check-ins for the same user have to be on continous lines.
    Ids for users and locations are recreated and continous from 0.
    '''

    def __init__(self, max_users=0, min_checkins=0):
        ''' max_users limits the amount of users to load.
        min_checkins discards users with less than this amount of checkins.
        '''

        self.max_users = max_users  # 0
        self.min_checkins = min_checkins  # 101

        self.user2id = {}
        self.poi2id = {}

        self.users = []
        self.times = []
        self.coords = []
        self.locs = []

    def create_dataset(self, sequence_length, batch_size, split, usage=Usage.MAX_SEQ_LENGTH, custom_seq_count=1):

        return PoiDataset(self.users.copy(), \
                          self.times.copy(), \
                          self.coords.copy(), \
                          self.locs.copy(), \
                          sequence_length, \
                          batch_size, \
                          split, \
                          usage, \
                          len(self.poi2id), \
                          custom_seq_count)

    def user_count(self):
        return len(self.users)

    def locations(self):
        return len(self.poi2id)

    def read(self, file):
        if not os.path.isfile(file):
            print('[Error]: Dataset not available: {}. Please follow instructions under ./data/README.md'.format(file))
            sys.exit(1)

        # collect all users with min check-ins:  default  为每个用户标号
        self.read_users(file)
        # collect check-ins for all collected users:    每个用户的位置信息
        self.read_pois(file)

    def read_users(self, file):
        f = open(file, 'r')
        lines = f.readlines()

        prev_user = int(float((lines[0].split('\t')[0])))
        visit_cnt = 0  # 计算总记录数
        for i, line in enumerate(lines):
            tokens = line.strip().split('\t')
            user = int(float(tokens[0]))
            if user == prev_user:
                visit_cnt += 1
            else:
                if visit_cnt >= self.min_checkins:
                    self.user2id[prev_user] = len(self.user2id)  # 给每个用户重新编号
                # else:
                #    print('discard user {}: to few checkins ({})'.format(prev_user, visit_cnt))
                prev_user = user
                visit_cnt = 1
                if self.max_users > 0 and len(self.user2id) >= self.max_users:
                    break  # restrict to max users

    def read_pois(self, file):
        f = open(file, 'r')
        lines = f.readlines()

        # store location ids
        user_time = []
        user_coord = []
        user_loc = []
        time_baseline = datetime(2009, 1, 1)  # base time
        prev_user = int(float((lines[0].split('\t')[0])))
        prev_user = self.user2id.get(prev_user)  # 新标号
        for i, line in enumerate(lines):
            tokens = line.strip().split('\t')
            user = int(float(tokens[0]))
            if self.user2id.get(user) is None:
                continue
            # user is not of interrest
            user = self.user2id.get(user)

            # time = (datetime.strptime(tokens[1], "%Y-%m-%dT%H:%M:%SZ") - datetime(1970, 1, 1)).total_seconds() # unix seconds
            time = float(tokens[1]) - ti.mktime(time_baseline.timetuple())
            lat = float(tokens[2])
            long = float(tokens[3])
            coord = (lat, long)
            location = tokens[4]  # location label

            if self.poi2id.get(location) is None:  # get-or-set locations
                self.poi2id[location] = len(self.poi2id)
            # 位置重新标号
            location = self.poi2id.get(location)
            # print(location)

            if user == prev_user:
                # insert in front!
                user_time.insert(0, time)
                user_coord.insert(0, coord)
                user_loc.insert(0, location)
            else:
                self.users.append(prev_user)
                self.times.append(user_time)
                self.coords.append(user_coord)
                self.locs.append(user_loc)

                # resart:
                prev_user = user
                user_time = [time]
                user_coord = [coord]
                user_loc = [location]

                # process also the latest user in the for loop
        self.users.append(prev_user)
        self.times.append(user_time)
        self.coords.append(user_coord)
        self.locs.append(user_loc)


class AisDataloader():
    '''
    Creates datasets from AIS data files.
    The file consist of one record per line in the following format (tab separated):

    <MMSI> <timestamp> <latitude> <longitude> <SOG> <COG> <Heading> <location-id>

    record for the same user have to be on continous lines.
    Ids for users and locations are recreated and continous from 0.
    '''

    def __init__(self, max_ships=0, min_records=0):
        '''
        max_ships limits the amount of ships to load.
        min_records discards users with less than this amount of records.
        '''

        self.max_ships = max_ships
        self.min_records = min_records

        self.MMSI2id = {}
        self.loc2id = {}

        self.ships = []
        self.times = []
        self.coords = []
        self.SOG = []
        self.COG = []
        self.locs = []

    def create_dataset(self, sequence_length, batch_size, split, usage=Usage.MAX_SEQ_LENGTH, custom_seq_count=1):

        return AisDataset(self.ships.copy(), \
                          self.times.copy(), \
                          self.coords.copy(), \
                          self.SOG.copy(), \
                          self.COG.copy(), \
                          self.locs.copy(), \
                          sequence_length, \
                          batch_size, \
                          split, \
                          usage, \
                          len(self.loc2id), \
                          custom_seq_count)

    def user_count(self):
        return len(self.ships)

    def locations(self):
        return len(self.loc2id)

    def read(self, file):
        if not os.path.isfile(file):
            print('[Error]: Dataset not available: {}. Please follow instructions in README.md'.format(file))
            sys.exit(1)

        # collect all users with min records:  default
        self.read_ships(file)
        # collect records for all collected ships:  default
        self.read_locations(file)

    def read_ships(self, file):
        f = open(file, 'r')
        lines = f.readlines()

        prev_ship = int(float((lines[0].split('\t')[0])))
        visit_cnt = 0  # 计算总记录数
        for i, line in enumerate(lines):
            tokens = line.strip().split('\t')
            ship = int(float(tokens[0]))
            if ship == prev_ship:
                visit_cnt += 1
            else:
                if visit_cnt >= self.min_records:
                    self.MMSI2id[prev_ship] = len(self.MMSI2id)  # # restrict to min users and recode ships
                # else:
                #    print('discard user {}: to few checkins ({})'.format(prev_user, visit_cnt))
                prev_ship = ship
                visit_cnt = 1
                if self.max_ships > 0 and len(self.MMSI2id) >= self.max_ships:
                    print("Have loaded max ships:{}".format(self.max_ships))
                    break  # restrict to max users

    def read_locations(self, file):
        f = open(file, 'r')
        lines = f.readlines()

        # store location ids
        record_time = []
        ship_coord = []
        ship_loc = []
        ship_SOG = []
        ship_COG = []
        time_baseline = datetime(2021, 1, 1)  # base time
        prev_ship = int(float((lines[0].split('\t')[0])))
        prev_ship = self.MMSI2id.get(prev_ship)  # 新标号
        for i, line in enumerate(lines):
            tokens = line.strip().split('\t')
            ship = int(float(tokens[0]))
            if self.MMSI2id.get(ship) is None:
                continue

            ship = self.MMSI2id.get(ship)

            # time = (datetime.strptime(tokens[1], "%Y-%m-%dT%H:%M:%SZ") - datetime(1970, 1, 1)).total_seconds() # unix seconds
            time = float(tokens[1]) - ti.mktime(time_baseline.timetuple())
            lat = float(tokens[2])
            long = float(tokens[3])
            coord = (lat, long)
            SOG = float(tokens[4])
            COG = float(tokens[5])
            location = tokens[6]  # location label

            if self.loc2id.get(location) is None:  # get-or-set locations
                self.loc2id[location] = len(self.loc2id)
            # recode loc label
            location = self.loc2id.get(location)
            # print(location)

            if ship == prev_ship:
                # insert in front!
                record_time.insert(0, time)
                ship_coord.insert(0, coord)
                ship_SOG.insert(0, SOG)
                ship_COG.insert(0, COG)
                ship_loc.insert(0, location)
            else:
                self.ships.append(prev_ship)
                self.times.append(record_time)
                self.coords.append(ship_coord)
                self.SOG.append(ship_SOG)
                self.COG.append(ship_COG)
                self.locs.append(ship_loc)

                # resart:
                prev_ship = ship
                record_time = [time]
                ship_coord = [coord]
                ship_SOG = [SOG]
                ship_COG = [COG]
                ship_loc = [location]

                # process also the latest user in the for loop
        self.ships.append(prev_ship)
        self.times.append(record_time)
        self.coords.append(ship_coord)
        self.SOG.append(ship_SOG)
        self.COG.append(ship_COG)
        self.locs.append(ship_loc)
