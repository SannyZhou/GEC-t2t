# GEC-t2t
Grammar Error Correction Based on Tensor2Tensor
<br>
A temp project of Deecamp.

## Train
The overall training procedure includes pretrain and finetune.
1. [Subword-nmt](https://github.com/rsennrich/subword-nmt)
<br> The input of this model should be a BPE format.


2. Pretrain
<br> In order to improve performance of this seq2seq task, the model needs to pretrain based on a large native corpus.
The source sentences are generated by denoising on native corpus. The denoising method refers to https://github.com/zhawe01/fairseq-gec.
The training step of pretrain depends on the size of native corpus and batchsize parameter, which should includes one epoch of native corpus.
<br> ***Tips:** The batchsize refers to the number of tokens.*

3. Finetune
<br> After pretrain, the model should be finetuned over gec corpus, such as CONLL-14. <br>
The training step depends on the loss and performance on your task. 

## Test
We use the tensorflow-serving on docker. 

## Reference
[Subword-nmt](https://github.com/rsennrich/subword-nmt)
[Tensor2Tensor](https://github.com/tensorflow/tensor2tensor)
