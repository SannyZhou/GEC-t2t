# -*- coding: utf-8 -*-
# @Author: Jie Zhou
# @Time: 2019/7/30 下午2:16
# @Project: tensor2tensor-master
# @File: bpe_to_origin.py
# @Software: PyCharm


def bpe_to_origin_text(bpe_text):
    result = []
    for line in bpe_text:
        bpe_text_list = line.split('@@ ')
        result.append(''.join(bpe_text_list))
    return result

def bpe_to_origin_line(bpe_line):
    bpe_line = bpe_line.split('@@ ')
    return ''.join(bpe_line)

def get_model_output(test_filename):
    test_file = open(test_filename, 'r')
    test_text = test_file.readlines()
    test_file.close()
    res =  bpe_to_origin_text(test_text)
    output_filename = '.'.join(test_filename.split('.')[:-1]) + '.out'
    output_file = open(output_filename, 'w')
    output_file.writelines(res)
    output_file.close()
    return (res, output_filename)

if __name__ == '__main__':
    get_model_output('output_new.txt')
