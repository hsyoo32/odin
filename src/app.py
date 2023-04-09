#coding=utf-8
#pylint: disable=no-member
#pylint: disable=no-name-in-module
#pylint: disable=import-error

from absl import app
from absl import flags

import sys
import os

import data_util
from trainer import Trainer
FLAGS = flags.FLAGS

flags.DEFINE_integer('pool', 10, 'Pool for negative sampling.')
flags.DEFINE_bool('use_gpu', True, 'Use GPU or not.')
flags.DEFINE_integer('gpu_id', 0, 'GPU ID.')
flags.DEFINE_integer('embedding_size', 64, 'Embedding size for embedding based models.')
flags.DEFINE_integer('epochs', 100, 'Max epochs for training.')
flags.DEFINE_float('lr', 0.001, 'Learning rate.')
flags.DEFINE_float('weight_decay', 5e-8, 'Weight decay.')
flags.DEFINE_integer('batch_size', 128, 'Batch Size.')
flags.DEFINE_float('disen_weight', 0.5, 'Weight for the disentanglement loss.')
flags.DEFINE_float('edge_weight', 1, 'Weight for edge score.')
flags.DEFINE_integer('neg_sample_rate', 4, 'Negative Sampling Ratio.')
flags.DEFINE_bool('shuffle', True, 'Shuffle the training set or not.')
flags.DEFINE_integer('num_workers', 6, 'Number of processes for training and testing.')
flags.DEFINE_string('option', '', 'Training options')
flags.DEFINE_string('input_file', '', 'Input filename')
flags.DEFINE_string('emb_file', '', 'Embeddings output filenmae')
# flags.DEFINE_string('emb_path', '', 'Embeddings output path')

def main(argv):
    flags_obj = FLAGS
    dm = data_util.DatasetManager(flags_obj)
    dm.get_dataset_info()
    trainer = Trainer(flags_obj, dm)
    trainer.train()

if __name__ == "__main__":
    app.run(main)

