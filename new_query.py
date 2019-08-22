# -*- coding: utf-8 -*-

# @Author: Jie Zhou
# @Time: 2019/7/31 下午3:18
# @Project: tensor2tensor-master
# @File: query-server.py.py
# @Software: PyCharm

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensor2tensor.serving import serving_utils
from tensor2tensor.utils import registry
from tensor2tensor.utils import usr_dir
from flask import Flask, request
from flask_cors import CORS
from flask import Response, json

import tensorflow as tf
import os
from subword_nmt import apply_bpe
import bpe_to_origin
import spacy

nlp = spacy.load("en")


app = Flask(__name__)
#CORS(app, resources = {r"/*": {"origins": "*"}})
app.config.from_object('conf')
servable_name_config = app.config['SERVABLE_NAME']
server_config = app.config['SERVER']
listen_port_config = app.config['LISTEN_PORT']
usr_dir_config = app.config['USR_DIR']
problem_config = app.config['PROBLEM']
data_dir_config = app.config['DATA_DIR']
port_config = app.config['SERVER_PORT']
FLAGS = tf.flags.FLAGS
model_config = app.config['MODEL_LIST']
model_list = {}
bpe_config = app.config['BPE_DICT']
#print(bpe_config)
codes_f = open(bpe_config['base_dir'] + bpe_config['code'], 'r')
bpe = apply_bpe.BPE(codes_f)
pattern = r',|\.|/|;|\'|`|\[|\]|<|>|\?|:|"|\{|\}|\~|!|@|#|\$|%|\^|&|\(|\)|-|=|\_|\+|，|。|、|；|‘|’|【|】|·|！| |…|（|）'

class T2T:

    def __init__(self, name, user_dir, data_dir, pro):
        self.problem = self.make_problem_fn(user_dir, data_dir, pro)
        self.request = self.make_request_fn(name)

    @staticmethod
    def make_request_fn(name):
        request_fn = serving_utils.make_grpc_request_fn(
            servable_name=name,
            server=server_config + ':' + str(listen_port_config),
            timeout_secs=100
        )
        return request_fn

    @staticmethod
    def make_problem_fn(user_dir, data_dir, pro):
        tf.logging.set_verbosity(tf.logging.INFO)
        usr_dir.import_usr_dir(user_dir)
        problem = registry.problem(pro)
        hparams = tf.contrib.training.HParams(
            data_dir=os.path.expanduser(data_dir)
        )
        problem.get_hparams(hparams)
        return problem


def init_fn():
    for i in range(len(model_config)):
        name = servable_name_config[i]
        user_dir = usr_dir_config[i]
        data_dir = data_dir_config[i]
        pro = problem_config[i]
        model_name = model_config[i]
        model = T2T(name, user_dir, data_dir, pro)
        model_list[model_name] = model


#@app.route('/grammar_check', methods=['POST'])
def grammar_check(inputs):
    print("IN!")
    '''
        data format: json
        example:
            {
                'model': 't2t',
                'content' 'This sentence would be checked by the model of grammar error correction.'
            }
        return: 
         {
                'corrected': output,
                'origin': origin
         }
    }

    '''
    #source = request.form['model']
    #inputs = request.form['content']
    source = 't2t'
    #inputs = 'People get certain disease because of genetic changes.'
    #origin = inputs
    inputs = ' '.join([token.orth_ for token in nlp(inputs)])
    print(inputs)
    inputs = bpe.process_line(inputs)
    origin = inputs
    print(inputs)
    outputs = serving_utils.predict([inputs], model_list[source].problem, model_list[source].request)
    outputs, = outputs
    output, score = outputs
    print(outputs)
    output = output[0: output.find('EOS') - 1]
    output = bpe_to_origin.bpe_to_origin_line(output)
    result = {
        'corrected': output,
        'origin': origin
        # 'origin': request.form['content']
    }
    print(result)
    return result


if __name__ == '__main__':
    init_fn()
    #print("testtest")
    grammar_check()
    #app.run()
