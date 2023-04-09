import os
import pickle
import math
from tqdm import tqdm
import numpy as np
import scipy as sc
from scipy.sparse import csr_matrix
import scipy.io as sio

def read_edges_from_file(filename): # all columns
    with open(filename, "r") as f: 
        lines = f.readlines()
        edges = [str_list_to_int(line.split()) for line in lines]
    return edges

def read_edges_from_file_float(filename):
    with open(filename, "r") as f: 
        lines = f.readlines()
        edges = [str_list_to_float(line.split()) for line in lines]
    return edges

def read_edges_from_file_(filename): # two columns
    with open(filename, "r") as f:
        lines = f.readlines()
        edges = [str_list_to_int(line.split()[:2]) for line in lines]
    return edges

def read_edges_from_file_three_columns(filename): # three columns
    with open(filename, "r") as f:
        lines = f.readlines()
        edges = [str_list_to_int(line.split()[:3]) for line in lines]
    return edges

def read_edges_from_file_string(filename):
    with open(filename, "r") as f:
        lines = f.readlines()
        edges = [line.split()[:2] for line in lines]
        return edges

def read_edges_from_file_comma(filename):
    with open(filename, "r") as f:
        lines = f.readlines()
        edges = [str_list_to_int(line.split(",")) for line in lines]
    return edges

def convert_nodes_str_to_int(filename, output):
    edges = read_edges_from_file_string(filename)
    d = {}
    index = 0
    for edge in edges:
        if d.get(edge[0]) is None:
            d[edge[0]] = index
            index += 1
        if d.get(edge[1]) is None:
            d[edge[1]] = index
            index += 1
    with open(output, "w") as f:
        for edge in edges:
            f.writelines(str(d[edge[0]])+"\t"+str(d[edge[1]])+"\n")

def convert_signed_to_unsign(filename, output, output_neg):
    edges = read_edges_from_file_three_columns(filename)
    pos_edges = []
    neg_edges = []
    for edge in edges:
        if edge[2] > 0:
            pos_edges.append(edge)
        elif edge[2] < 0:
            neg_edges.append(edge)
    with open(output, "w") as f:
        for edge in pos_edges:
            f.writelines(str(edge[0])+"\t"+str(edge[1])+"\t1\n")
    with open(output_neg, "w") as f:
        for edge in neg_edges:
            f.writelines(str(edge[0])+"\t"+str(edge[1])+"\t-1\n")

def read_digraph_from_file(file_path):
    edges = read_edges_from_file_(file_path)
    d = {}
    for edge in edges:
        if d.get(edge[0]) is None:
            d[edge[0]] = []
        d[edge[0]].append(edge[1])
    return d

def read_sparse_matrix_from_file(filename):
    rows = []; cols = []; sign = []
    with open(filename, "r") as f:
        lines = f.readlines()
        for line in lines:
            edge = line.split()
            rows.append(int(edge[0]))
            cols.append(int(edge[1]))
            # rows.append(int(edge[1]))
            # cols.append(int(edge[0]))
            sign.append(float(edge[2]))
            # sign.append(float(edge[2]))
        csr = csr_matrix((sign,(rows,cols)))    
    return csr

def read_labels_from_file(filename, num_node):
    with open(filename, "r") as f:
        lines = f.readlines()
        num_label = int(lines[0])
        # print(str(num_label))
        labels = np.zeros((num_node, num_label))
        for line in lines[1:]:
            # print(line)
            label = line.split()
            label_list = str_list_to_int(label[1:])
            # print(label_list)
            for i in range(num_label):
                if i+1 in label_list:
                    labels[int(label[0]),i] = 1
            # print(labels[int(label[0])])
    return labels

def str_list_to_int(str_list):
    return [int(item) for item in str_list]

def str_list_to_float(str_list):
    return [float(item) for item in str_list]

def read_embeddings(filename, n_node, n_embed):
    embedding_matrix = np.zeros((n_node, n_embed))
    
    with open(filename, "r") as f:            
        lines = f.readlines()
        lines = lines[1:]  # skip the first line                
        for line in lines:
            emd = line.split()
            embedding_matrix[int(emd[0]), :] = str_list_to_float(emd[1:])
    return embedding_matrix

def read_directed_embeddings(filename, n_node, n_embed):
    embedding_matrix_source = np.zeros((n_node, n_embed)) #np.random.rand(n_node, n_embed)
    embedding_matrix_context = np.zeros((n_node, n_embed)) #np.random.rand(n_node, n_embed)

    with open(filename, "r") as f:
        lines = f.readlines()
        lines = lines[1:]  # skip the first line
        for line in lines:
            emd = line.split()
            embedding_matrix_source[int(emd[0]), :] = str_list_to_float(emd[1:])
    with open(filename+"2", "r") as f:
        lines = f.readlines()
        lines = lines[1:]  # skip the first line
        for line in lines:
            emd = line.split()
            embedding_matrix_context[int(emd[0]), :] = str_list_to_float(emd[1:])

    return embedding_matrix_source, embedding_matrix_context

def create_folder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print("Error: Creating directory - " + directory)



# def convert_to_ajacent_list(input_filename, output_filename):
#     edges = read_edges_from_file(input_filename)
#     d = {}
#     for edge in edges:
#         if d.get(edge[0]) is None:
#             d[edge[0]] = []
#         d[edge[0]].append(edge[1])
#     d = sorted(d.items())

#     with open(output_filename, "w+") as f:
#         for i in d:
#             f.writelines(str(i[0])+"\t")
#             for k in i[1]:
#                 f.writelines(str(k)+"\t")
#             f.writelines("\n")

# def convert_to_weighted_graph(input_filename, output_filename, weight):
# 	edges = read_edges_from_file(input_filename)
# 	with open(output_filename, "w+") as f:
# 		for i in range(len(edges)):
# 			f.writelines(str(edges[i][0]) + "\t" + str(edges[i][1]) + "\t" + str(weight) + "\n")

# def convert_to_zero_negative_dataset(input_filename, output_filename):
#     edges = read_edges_from_file(input_filename)
#     with open(output_filename, "w+") as f:
#         for i in range(len(edges)):
#             weight = edges[i][2] if edges[i][2] == 1 else 0  # 1->1,-1->0
#             f.writelines(str(edges[i][0]) + "\t" + str(edges[i][1]) + "\t" + str(weight) + "\n")

# def convert_to_positive_only_dataset(input_filename, output_filename):
#     edges = read_edges_from_file(input_filename)
#     with open(output_filename, "w+") as f:
#         for i in range(len(edges)):
#             if edges[i][2] == 1:
#                 f.writelines(str(edges[i][0]) + "\t" + str(edges[i][1]) + "\n")


# def sigmoid(x):
#     return 1 / (1 + np.exp(-x))

# def calculate_adamic_adar(node1, node2, train_adj_edge_dict):
#     rel = 0.0
#     if node1 not in train_adj_edge_dict or node2 not in train_adj_edge_dict:
#         return rel

#     inter_nhds = set(train_adj_edge_dict[node1]) & set(train_adj_edge_dict[node2])
#     for nhd in inter_nhds:
#         if len(train_adj_edge_dict[nhd]) <= 1:
#             continue
#         rel += 1.0 / math.log(len(train_adj_edge_dict[nhd]))
#     return rel

def aggregate_edge_emb(emb1, emb2):
    edge_emb = np.zeros(shape=len(emb1) * 2)
    edge_emb[:len(emb1)] = emb1
    edge_emb[len(emb1):] = emb2

    return edge_emb

# def calculate_edge_score(score_method, emb1, emb2):
#     if score_method == "dot_product": 
#         score = np.dot(emb1, emb2)
#     elif score_method == "cosine":
#         norm1 = np.linalg.norm(emb1)
#         norm2 = np.linalg.norm(emb2)
#         score = np.dot(emb1, emb2)/(norm1*norm2)
#         if math.isnan(score):
#             score = 0
#     elif score_method == "euclidean":
#         score = np.linalg.norm(emb1-emb2)

#     return score