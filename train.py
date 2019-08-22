# -*- coding: utf-8 -*-
# @Author: Jie Zhou
# @Time: 2019/7/27 下午4:30
# @Project: tensor2tensor-master
# @File: train.py.py
# @Software: PyCharm

from tensor2tensor.bin import t2t_trainer

import tensorflow as tf

FLAGS = tf.flags.FLAGS


def train(generate_data=True):
    FLAGS.problem = "english_grammar_error"
    FLAGS.model = "transformer"
    FLAGS.generate_data = True
    FLAGS.hparams_set = "transformer_big_single_gpu"
    FLAGS.t2t_usr_dir = "src"
    FLAGS.output_dir = "finetune"
    FLAGS.data_dir = "t2t_finetune"
    FLAGS.train_steps = 350000
    t2t_trainer.main(None)


if __name__ == "__main__":
    train()
