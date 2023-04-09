# -*- coding: utf-8 -*-
import os
import sys

def odin_params():
	params = []
	disen_weights = [0.5]
	epochs = [200]
	num_neg_samples = [4]
	options = ['odin']

	for e in epochs:
		for n in num_neg_samples:
			for opt in options:
				for disen_weight in disen_weights:
					params.append(str(e)+'_'+str(n)+'_'+str(opt)+'_'+str(disen_weight))
	return params

gen_dataset = ["p2p-Gnutella08","wiki-vote",'jung','ciaodvd']
gen_emb_algo = ['odin']
gen_test_types = ['LP-uniform','LP-mixed','LP-biased']

# non-id (in), non-id (out)
# split_types = ['noniid-in-barrier','noniid-out-barrier']
split_types = ['noniid-in-barrier'] # 'noniid-in-barrier','noniid-out-barrier'

# K = -cap; when K = 0, it is iid
caps = [1.0,0.8,0.6,0.4,0.2,0.0]
caps = [1.0]
split_types_tmp = []
for split in split_types:
	for cap in caps:
		split_types_tmp.append(split+'_'+str(cap))
split_types = split_types_tmp
print(split_types)

# cross validation
fold_ids = ['u1','u2','u3','u4','u5']
fold_ids = ['u1'] 
num_embed_ = 120
# emb_mani = ["ss", "st", "tt", "stconcat"]
# overall node embedding is the concatenation of the source and target embeddings
emb_mani = ['stconcat']


dataset_idx = [1,2,3,4]
dataset_idx = [1]
algo_idx = [1] 
test_idx = [1,3]


#################
dataset = []
for i in dataset_idx:
	dataset.append(gen_dataset[i-1])
emb_algo = []
for i in algo_idx:
	emb_algo.append(gen_emb_algo[i-1])
test_types = []
for i in test_idx:
	test_types.append(gen_test_types[i-1])
##################

params_ = ['']

for mani in emb_mani:
	for data in dataset:
		for split_type in split_types:
			for algo in emb_algo:
				if 'odin' == algo:
					params = odin_params()

				for param in params:
					for fold_id in fold_ids:
						for test_type in test_types:

							if 'stconcat' == mani:
								num_embed = int(num_embed_/2)

							arg = 'python _execute_methods.py --dataset {} --emb_algo {} --split_type {} --fold_id {}\
								  --target {} --num_embed {} --mani {} --param {}'.format(data,algo,split_type,fold_id,test_type,num_embed,mani,param)
							os.system(arg)

