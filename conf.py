# -*- coding: utf-8 -*-
# @Author: Jie Zhou
# @Time: 2019-08-04 18:13
# @Project: tensor2tensor-master
# @File: conf.py
# @Software: PyCharm


# server configuration
SERVER = "127.0.0.1"
LISTEN_PORT = 8501
SERVER_PORT = 8080
# tensor configuration order
# list of tensorflow-server names
SERVABLE_NAME = ['gect2t']
# list of usr_dirs
USR_DIR = ['/code_dir/src']
# list of problems
PROBLEM = ['english_grammar_error']
# list of data_dirs
DATA_DIR = ['/code_dir/t2t_finetune']
# list of models
MODEL_LIST = ['gect2t']
BPE_DICT = {
    'base_dir': '/bpe_dir/',
    'code': 'test.code.1',
    'vocab': 'test.vocab.1'
}
