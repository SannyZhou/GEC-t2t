# -*- coding: utf-8 -*-
# @Author: Jie Zhou
# @Time: 2019/7/31 上午11:22
# @Project: tensor2tensor-master
# @File: avg_all.py
# @Software: PyCharm

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensor2tensor.serving import export

import tensorflow as tf

FLAGS = tf.flags.FLAGS

def export_model():
    FLAGS.problem = "english_grammar_error"
    FLAGS.model = "transformer"
    FLAGS.hparams_set = "transformer_big_single_gpu"
    FLAGS.t2t_usr_dir = "src"
    FLAGS.output_dir = "backup_finetune_model/avg_model"
    FLAGS.data_dir = "t2t_finetune"
    export.main(None)


if __name__ == "__main__":
    export_model()
