import tensorflow as tf
import numpy as np
import pickle
import copy
import random

class Memory:
    def __init__(self, max_size, enable_overlap=True):
        self.max_size = max_size
        self.total_number = 0
        self.index = 0
        self.enable_overlap = True
        self.data = []
        self.reset()

    def reset(self):
        self.total_number = 0
        del(self.data)
        self.data = [0 for _ in range(self.max_size)]

    def add(self, data_):
        if self.total_number >= self.max_size:
            self.index = 0
            self.data[self.index] = copy.deepcopy(data_)
        else:
            self.total_number += 1
            self.data[self.index] = copy.deepcopy(data_)
            self.index += 1
    
    def sample(self, sample_number, replace_option=False):
        return random.sample(self.data, sample_number)

    def load(self, path):
        with open(path, 'r') as f:
            data_dict = pickle.load(f)
            self.max_size = data_dict['size']
            self.total_number = data_dict['total_number']
            self.data = data_dict['data']

    def save(self, path):
        with open(path, 'w') as f:
            data_dict = {}
            data_dict['size'] = self.max_size
            data_dict['total_number'] = self.total_number
            data_dict['data'] = self.data
            pickle.dump(data_dict, f)