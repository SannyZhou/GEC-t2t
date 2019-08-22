# -*- coding: utf-8 -*-
# @Author: Jie Zhou
# @Time: 2019/7/31 下午3:18
# @Project: tensor2tensor-master
# @File: query-server.py.py
# @Software: PyCharm

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensor2tensor.serving import query

import tensorflow as tf

FLAGS = tf.flags.FLAGS


def main(argv):
    FLAGS.server = 'localhost:8501'
    FLAGS.servable_name = 't2t'
    FLAGS.problem = 'english_grammar_error'
    FLAGS.data_dir = "t2t_finetune"
    FLAGS.t2t_usr_dir = 'src'
    FLAGS.inputs_once = 'i try on docker'
    query.main(argv)


if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()

