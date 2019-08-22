# -*- coding: utf-8 -*-
# @Author: Jie Zhou
# @Time: 2019/7/31 上午11:22
# @Project: tensor2tensor-master
# @File: avg_all.py
# @Software: PyCharm

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensor2tensor.bin import t2t_avg_all

import tensorflow as tf

FLAGS = tf.flags.FLAGS
MODEL_DIR = 'backup_finetune_model'

def avg_checkpoints():
    FLAGS.model_dir = MODEL_DIR
    FLAGS.output_dir = MODEL_DIR + '/avg_model'
    FLAGS.n = 5
    t2t_avg_all.main(None)

if __name__ == "__main__":
    avg_checkpoints()
