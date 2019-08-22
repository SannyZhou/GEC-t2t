# -*- coding: utf-8 -*-
# @Author: Jie Zhou
# @Time: 2019/7/27 下午4:08
# @Project: tensor2tensor-master
# @File: gec_problem.py.py
# @Software: PyCharm

import os
import random

from tensor2tensor.data_generators import text_problems, text_encoder, problem
from tensor2tensor.utils import registry

import tensorflow as tf

BPE_DATASET = {'TRAIN': 'train',
'DEV': 'dev'}

def get_bpe_dataset(directory, filename):
    train_path = os.path.join(directory, filename)
    if not (tf.gfile.Exists(train_path + ".src") and
            tf.gfile.Exists(train_path + ".trg")):
        raise Exception("there should be some training/dev data in the tmp dir.")

    return train_path


@registry.register_problem
class EnglishGrammarError(text_problems.Text2TextProblem):

    @property
    def multiprocess_generate(self):
        """Whether to generate the data in multiple parallel processes."""
        return True

    @property
    def source_vocab_name(self):
        return "vocab.bpe.trg.30003"

    @property
    def target_vocab_name(self):
        return "vocab.bpe.trg.30003"

    def get_vocab(self, data_dir, is_target=False):
        """返回的是一个encoder，单词表对应的编码器"""
        vocab_filename = os.path.join(data_dir, self.target_vocab_name if is_target else self.source_vocab_name)
        if not tf.gfile.Exists(vocab_filename):
            raise ValueError("Vocab %s not found" % vocab_filename)
        return text_encoder.TokenTextEncoder(vocab_filename, replace_oov="UNK")

    def generate_samples(self, data_dir, tmp_dir, dataset_split):
        """Instance of token generator for the WMT en->zh task, training set."""
        train = dataset_split == problem.DatasetSplit.TRAIN
        dataset_path = (BPE_DATASET["TRAIN"] if train else BPE_DATASET["DEV"])
        train_path = get_bpe_dataset(tmp_dir, dataset_path)

        # Vocab
        src_token_path = (os.path.join(data_dir, self.source_vocab_name), self.source_vocab_name)
        tar_token_path = (os.path.join(data_dir, self.target_vocab_name), self.target_vocab_name)
        for token_path, vocab_name in [src_token_path, tar_token_path]:
            if not tf.gfile.Exists(token_path):
                token_tmp_path = os.path.join(tmp_dir, vocab_name)
                tf.gfile.Copy(token_tmp_path, token_path)
                with tf.gfile.GFile(token_path, mode="r") as f:
                    vocab_data = "<pad>\n<EOS>\n" + f.read() + "UNK\n"
                with tf.gfile.GFile(token_path, mode="w") as f:
                    f.write(vocab_data)

        return text_problems.text2text_txt_iterator(train_path + ".src",
                                                    train_path + ".trg")

    def generate_encoded_samples(self, data_dir, tmp_dir, dataset_split):
        """在生成数据的时候，主要是通过这个方法获取已编码样本的"""
        generator = self.generate_samples(data_dir, tmp_dir, dataset_split)
        encoder = self.get_vocab(data_dir)
        target_encoder = self.get_vocab(data_dir, is_target=True)
        return text_problems.text2text_generate_encoded(generator, encoder, target_encoder,
                                                        has_inputs=self.has_inputs)

    def feature_encoders(self, data_dir):
        source_token = self.get_vocab(data_dir)
        target_token = self.get_vocab(data_dir, is_target=True)
        return {
            "inputs": source_token,
            "targets": target_token,
        }

