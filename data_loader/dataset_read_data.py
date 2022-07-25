#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" read datasets from existing files"""

import numpy as np
import logging
from sklearn.preprocessing import StandardScaler
from helpers.log_helper import LogHelper
from helpers.dir_utils import create_dir
from datetime import datetime
from pytz import timezone


class DataGenerator(object):

   

    def __init__(self, file_path, solution_path=None, normalize_flag=False, transpose_flag=False):

        self.inputdata = np.load(file_path)
        self.datasize, self.d = self.inputdata.shape

        data_dir = 'dataset/{}'.format(datetime.now(timezone('Asia/Hong_Kong')).strftime('%Y-%m-%d_%H-%M-%S-%f')[:-3])
        create_dir(data_dir)

        LogHelper.setup(log_path='{}/training.log'.format(data_dir),
                    level_str='INFO')

        _logger = logging.getLogger(__name__)
        _logger.info(print('Input data is \n', self.inputdata))

        if normalize_flag:
            self.inputdata = StandardScaler().fit_transform(self.inputdata)
            _logger.info(print('In the NORMALIZE block \n'))
            _logger.info(print('After normalizing \n', self.inputdata))

        if solution_path is None:
            gtrue = np.zeros(self.d)
        else:
            gtrue = np.load(solution_path)#DAG.npy
            if transpose_flag: 
                gtrue = np.transpose(gtrue)#Transposing the true DAG
                _logger.info(print('After transposing \n', gtrue))

        # (i,j)=1 => node i -> node j
        self.true_graph = np.int32(np.abs(gtrue) > 1e-3)
        _logger.info(print('True DAG absolutes values \n', self.true_graph))

    def gen_instance_graph(self, max_length, dimension, test_mode=False):
        seq = np.random.randint(self.datasize, size=(dimension))#dimension unclear
        input_ = self.inputdata[seq]
        return input_.T #Transpose of input_
    

    # Generate random batch for training procedure
    def train_batch(self, batch_size, max_length, dimension):
        input_batch = []

        for _ in range(batch_size):
            input_= self.gen_instance_graph(max_length, dimension)
            input_batch.append(input_)

        return input_batch
        _logger.info(print('Random input batch for training procedure\n', input_batch))
    
