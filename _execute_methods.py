#-*- coding: utf-8 -*-
import os
import argparse
import _link_prediction as lp
import utility as util
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description="Run Test.")

    parser.add_argument('--dataset', nargs='?', default='arxiv_astroph',
                        help='Input dataset')   

    parser.add_argument('--emb_algo', nargs='?', default='node2vec',
                        help='Embedding algorithm')

    parser.add_argument('--split_type', nargs='?', default='2-Fold',
                        help='Number of fold')

    parser.add_argument('--fold_id', nargs='?', default='u1_SCC',
                        help='Fold ID')

    parser.add_argument('--target', nargs='?', default='prediction',
                        help='Target task')

    parser.add_argument('--num_embed', type=int, default=1,
                        help='Number of nodes.')

    parser.add_argument('--param', nargs='?', default='')

    parser.add_argument('--mani', nargs='?', default='')

    return parser.parse_args()

def test(args):
    # parameter settings
    data = args.dataset
    emb_algo = args.emb_algo
    split_type = args.split_type
    fold_id = args.fold_id
    target = args.target    
    num_embed = args.num_embed
    param = args.param
    mani = args.mani

    if param is None:
        param = ""
        emb_algo_param = emb_algo
    else:
        emb_algo_param = emb_algo + '_' + param
    #print("emb_algo_param: {}".format(emb_algo_param))

    # path settings and folder creation
    main_path = os.getcwd()
    file_path = os.getcwd() + "/_Data/" + data + "/" + split_type + "/" + fold_id + "/"
    embed_path = os.getcwd() + "/_Emb/" + data + "/" + split_type + "/" + emb_algo + "/"
    result_path = os.getcwd() + "/_Results/" + target + "/" + data + "/" + split_type + "/" + fold_id + "/" + emb_algo + "/"
    algo_path = os.getcwd() + "/src/"
    util.create_folder(file_path)
    util.create_folder(embed_path)
    util.create_folder(result_path)
    util.create_folder(algo_path)

    # input and output paths
    embed_filename = embed_path + "{}_{}_dim{}.emb".format(fold_id, emb_algo_param, num_embed)
    result_filename = result_path + "{}_dim{}.result".format(emb_algo_param, num_embed)
    ori_filename = file_path + "{}.edgelist".format(fold_id)
    train_filename = file_path + "{}.edgelist".format(fold_id)
    test_filename = file_path + "{}_test.edgelist".format(fold_id)
    # print(test_filename)
    # print(embed_filename)

    # if emb_algo in directed_algorithms:
    if mani == "ss": 
        result_filename = result_filename[:-7] + "_SS.result"
    elif mani == "tt":
        result_filename = result_filename[:-7] + "_TT.result"
    elif mani == "stconcat":
        result_filename = result_filename[:-7] + "_STconcat.result"

    # set the number of nodes
    train_edges_ = util.read_edges_from_file(ori_filename)
    nodes = set()
    for edge in train_edges_:
        nodes = nodes.union(set(edge[:2]))
    num_node = len(nodes)
    #print("num_node:{} max_node_id:{}".format(len(nodes), max(nodes)))

    # perform algorithms
    test_edges = util.read_edges_from_file(test_filename)
    train_embeddings = []
    # if not exist the embedding file, perform algorithm
    if not os.path.isfile(embed_filename): 
        perform_algorithm(algo_path, emb_algo_param, train_filename, embed_filename, num_embed, data, params=param)

    train_embeddings.append(np.zeros((num_node, num_embed)))
    if mani == "st" or mani == "stconcat":
        train_embeddings.append(np.zeros((num_node, num_embed))) # for target embeddings
        train_embeddings[0], train_embeddings[1] = util.read_directed_embeddings(embed_filename, n_node=num_node, n_embed=num_embed)
    else:
        if mani == "tt":
            train_embeddings[0] = util.read_embeddings(embed_filename+"2", n_node=num_node, n_embed=num_embed)
        else:
            train_embeddings[0] = util.read_embeddings(embed_filename, n_node=num_node, n_embed=num_embed)

    if mani == "stconcat":
        train_embeddings_ = []
        train_embeddings_.append(np.zeros((num_node, num_embed*2)))
        for i in range(num_node):
            train_embeddings_[0][i][:num_embed] = train_embeddings[0][i]
            train_embeddings_[0][i][num_embed:] = train_embeddings[1][i]
        train_embeddings = train_embeddings_
        num_embed *= 2

    # evaluation
    if 'LP-' in target:
        # Train edges should be same in all cases
        train_edges = util.read_edges_from_file(train_filename)            
        num_samples = ["1"]
        for num_sample in num_samples:
            if "LP-uniform" == target:
                train_uncon_filename = file_path + "{}_unconnected_ulp_train_{}times.edgelist".format(fold_id, num_sample)
                test_uncon_filename = file_path + "{}_unconnected_ulp_test_{}times.edgelist".format(fold_id, num_sample)
            elif "LP-mixed" == target:
                train_uncon_filename = file_path + "{}_unconnected_mlp_train_{}times.edgelist".format(fold_id, num_sample)
                test_uncon_filename = file_path + "{}_unconnected_mlp_test_{}times.edgelist".format(fold_id, num_sample)
            elif "LP-biased" == target:
                train_uncon_filename = file_path + "{}_unconnected_blp_train_{}times.edgelist".format(fold_id, num_sample)
                test_uncon_filename = file_path + "{}_unconnected_blp_test_{}times.edgelist".format(fold_id, num_sample)
            train_edges_uncon = util.read_edges_from_file(train_uncon_filename)
            test_edges_uncon = util.read_edges_from_file(test_uncon_filename)
            lp.compute_accuracy_using_classifier(train_edges, test_edges, train_edges_uncon, test_edges_uncon, train_embeddings, emb_algo, result_filename)
       
    print("Finish: perform evaluation - " + target)

def perform_algorithm(algo_path, algo, train_filename, embed_filename, num_embed, data, params=''):

    if "odin" in algo:
        if params != "":
            param = params.split('_')
            print(param)
            epochs = param[0]
            neg_sample_rate = param[1]
            option = param[2]
            disen_weight = param[3]

        os.chdir(algo_path)
        num_embed = int(num_embed/3)
        
        os.system("python app.py --embedding_size={} --epochs={} --option={} --disen_weight={} \
                  --input_file={} --emb_file={} --neg_sample_rate={}"
        .format(num_embed, epochs, option, disen_weight, train_filename, embed_filename, neg_sample_rate))
        
    print("Finish: perform algorithm - " + algo)

if __name__ == "__main__":
    args = parse_args()
    test(args)