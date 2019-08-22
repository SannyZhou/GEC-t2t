# -*- coding: utf-8 -*-
# @Author: Jie Zhou
# @Time: 2019/7/27 下午4:30
# @Project: tensor2tensor-master
# @File: train.py.py
# @Software: PyCharm

from tensor2tensor.bin import t2t_decoder

import tensorflow as tf

FLAGS = tf.flags.FLAGS


def decode(generate_data=True):
    FLAGS.problem = "english_grammar_error"
    FLAGS.model = "transformer"
    FLAGS.hparams_set = "transformer_big_single_gpu"
    FLAGS.t2t_usr_dir = "src"
    FLAGS.output_dir = "finetune_dir"
    FLAGS.data_dir = "t2t_finetune"
    FLAGS.decode_hparams = "beam_size=4,alpha=0.6"
    FLAGS.decode_from_file = 'test_new.txt'
    FLAGS.decode_to_file = 'output_new.txt'
    t2t_decoder.main(None)


if __name__ == "__main__":
    decode()
