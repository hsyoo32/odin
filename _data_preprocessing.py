from utility import *
import os
import snap
import random
from tqdm import tqdm
import numpy as np

def get_connected_graph_reindexed_v2(graph_type, train_edges, connected_train_filename, test_edges=None, connected_test_filename=None):
    if graph_type == "undirected":
        G = snap.TUNGraph.New()
    elif graph_type == "directed":
        G = snap.TNGraph.New()

    for edge in train_edges:
        if G.IsNode(edge[0]) == False:
            G.AddNode(edge[0])
        if G.IsNode(edge[1]) == False:
            G.AddNode(edge[1])
        G.AddEdge(edge[0], edge[1])

    # weakly connected component
    splited_graph = snap.GetMxWcc(G)

    j = 0
    for _ in splited_graph.Edges():
        j += 1
    print("{}/ ori_num_edges: {} wcc_num_edges: {}".format(connected_train_filename,len(train_edges),j))

    edges = []
    for edge in splited_graph.Edges():
        edges.append([edge.GetSrcNId(), edge.GetDstNId()])

    node_map = {}
    idx = 0
    for edge in edges:
        if node_map.get(edge[0]) is None:
            node_map[edge[0]] = idx
            idx += 1
        if node_map.get(edge[1]) is None:
            node_map[edge[1]] = idx
            idx += 1

    new_edges = []
    for edge in edges:
        new_edges.append([node_map[edge[0]], node_map[edge[1]]])

    with open(connected_train_filename, "w+") as f:
        for i in range(len(new_edges)):
            f.writelines(str(new_edges[i][0])+"\t"+str(new_edges[i][1])+"\n")

    if test_edges is not None:
        i = 0
        with open(connected_test_filename, "w+") as f:
            for edge in test_edges:
                if edge[0] in node_map and edge[1] in node_map:
                    i += 1
                    f.writelines(str(node_map[edge[0]]) + "\t" + str(node_map[edge[1]]) + "\n")
        print("{}/ ori_num_edges: {} wcc_num_edges: {}".format(connected_test_filename,len(test_edges),i))

def get_num_nodes_inmemory(edges):
    nodes = set()
    for edge in edges:
        nodes = nodes.union(set(edge[:2]))
    return len(nodes)

def get_num_nodes(filename):
    edges = read_edges_from_file_(filename)
    nodes = set()
    for edge in edges:
        nodes = nodes.union(set(edge[:2]))
    return len(nodes), max(nodes)

def get_num_nodes_(filename, filename_):
    edges = read_edges_from_file_(filename)
    edges_ = read_edges_from_file_(filename_)
    nodes = set()
    for edge in edges:
        nodes = nodes.union(set(edge[:2]))
    nodes_ = set()
    for edge in edges_:
        nodes_ = nodes_.union(set(edge[:2]))

    return "A:{} B:{} A|B:{} A&B:{}".format(len(nodes), len(nodes_), len(nodes|nodes_), len(nodes&nodes_))

def generate_unconnected_train_links(graph_type, train_filename, test_filename, train_unconnected_filename, 
    train_ratio_set, opt="", reverse_ratio=0):
    train_edges = read_edges_from_file(train_filename)
    test_edges = read_edges_from_file(test_filename)
    neighbors = {}  # dict, node_ID -> list_of_neighbors

    for edge in train_edges + test_edges:
        if neighbors.get(edge[0]) is None:
            neighbors[edge[0]] = []
        if neighbors.get(edge[1]) is None:
            neighbors[edge[1]] = []
        neighbors[edge[0]].append(edge[1])
        if graph_type == "undirected":
            neighbors[edge[1]].append(edge[0])
    
    nodes = set()
    for edge in train_edges:
        nodes = nodes.union(set(edge[:2]))
    nodes_list = list(nodes)
    
    # print("Generating unconnected train links has started.")  

    np.random.seed(100)
    neg_edges = []
    num_max_neg_edges = len(train_edges) * int(train_ratio_set[-1])

    if opt == "unidirectional":
        num_reverse = int(reverse_ratio * len(train_edges))
        #print("reverse_ratio = {}".format(reverse_ratio))
        pbar = tqdm(total = num_max_neg_edges)
        n = 0
        for edge in train_edges:
            pbar.update(1)
            if neighbors.get(edge[1]) and edge[0] in neighbors[edge[1]] or n >= num_reverse:
                flag = 0
                while flag == 0:
                    start_node = np.random.choice(nodes_list, size = 1)[0]
                    if neighbors.get(start_node) is not None:
                        neg_nodes = list(nodes.difference(set(neighbors[start_node] + [start_node])))
                        neg_node = np.random.choice(neg_nodes, size = 1)[0]
                        neg_edges.append([start_node, neg_node])
                        neighbors[start_node].append(neg_node)
                        flag = 1

            else:
                neg_edges.append([edge[1], edge[0]])
                n += 1
        pbar.close()
        #print("train uni: {} / {}; {}".format(n, len(train_edges), n/len(train_edges)*100))

    for n in range(len(train_ratio_set)):
        train_unconnected_ratio_filename = train_unconnected_filename[:-9] + "_" + str(train_ratio_set[n]) + "times.edgelist"
        num_neg_edges = len(train_edges) * int(train_ratio_set[n])
        #print('train_unconnected_ratio_filename: {}'.format(train_unconnected_ratio_filename))
        with open(train_unconnected_ratio_filename, "w+") as f:
            neg_edges_str = [str(x[0]) + "\t" + str(x[1]) + "\t" + str(0) + "\n" for x in neg_edges[:num_neg_edges]]
            f.writelines(neg_edges_str)

def generate_unconnected_test_links(graph_type, train_filename, test_filename, train_unconnected_filename, test_unconnected_filename, 
    test_ratio_set, opt="", reverse_ratio=0):
    train_edges = read_edges_from_file(train_filename)
    test_edges = read_edges_from_file(test_filename)
    train_unconnected_edges = read_edges_from_file(train_unconnected_filename[:-9] + "_" + str(test_ratio_set[-1]) + "times.edgelist") ###############
    neighbors = {}  # dict, node_ID -> list_of_neighbors

    for edge in train_edges + train_unconnected_edges + test_edges:
        if neighbors.get(edge[0]) is None:
            neighbors[edge[0]] = []
        if neighbors.get(edge[1]) is None:
            neighbors[edge[1]] = []
        neighbors[edge[0]].append(edge[1])
        if graph_type == "undirected":
            neighbors[edge[1]].append(edge[0])

    nodes = set()
    for edge in test_edges:
        nodes = nodes.union(set(edge[:2]))
    nodes_list = list(nodes)

    np.random.seed(100) 
    neg_edges = []
    num_max_neg_edges = len(test_edges) * int(test_ratio_set[-1])

    if opt == "unidirectional":
        num_reverse = int(reverse_ratio * len(test_edges))
        pbar = tqdm(total = num_max_neg_edges)
        n = 0
        for edge in test_edges:
            pbar.update(1)
            if neighbors.get(edge[1]) and edge[0] in neighbors[edge[1]] or n >= num_reverse:
                flag = 0
                while flag == 0:
                    start_node = np.random.choice(nodes_list, size = 1)[0]
                    if neighbors.get(start_node) is not None:
                        neg_nodes = list(nodes.difference(set(neighbors[start_node] + [start_node])))
                        neg_node = np.random.choice(neg_nodes, size = 1)[0]
                        neg_edges.append([start_node, neg_node])
                        neighbors[start_node].append(neg_node)
                        flag = 1
            else:
                neg_edges.append([edge[1], edge[0]])
                n += 1
        pbar.close()
        #print("test uni: {} / {}; {}".format(n, len(test_edges), n/len(test_edges)*100))
    
    for n in range(len(test_ratio_set)):
        test_unconnected_ratio_filename = test_unconnected_filename[:-9] + "_" + str(test_ratio_set[n]) + "times.edgelist"
        num_neg_edges = len(test_edges) * int(test_ratio_set[n])
        #print(test_unconnected_ratio_filename)
        with open(test_unconnected_ratio_filename, "w+") as f:
            neg_edges_str = [str(x[0]) + "\t" + str(x[1]) + "\t" + str(0) + "\n" for x in neg_edges[:num_neg_edges]]
            f.writelines(neg_edges_str)
    #print("Finish: write unconnected edges for test")

def noniid_split():
    # 'p2p-Gnutella08','wiki-vote','jung',ciaodvd'
    datasets = ['p2p-Gnutella08','wiki-vote','jung','ciaodvd'] 
    # 'noniid-in-barrier', 'noniid-out-barrier'
    split_types = ['noniid-in-barrier', 'noniid-out-barrier']
    fold_ids = ['u1','u2','u3','u4','u5']
    split_ratio = 0.2
    graph_type = 'directed'
    caps = [1.0,0.8,0.6,0.4,0.2,0.0]
    # caps = [0.0]

    for dataset in datasets:
        file_path = os.getcwd() + "/_Data/" + dataset + "/"
        input_file = file_path + dataset + '.txt'
        output_file = file_path + dataset
        input_edges = read_edges_from_file(input_file)
        get_connected_graph_reindexed_v2(graph_type, input_edges, output_file)

        edges = read_edges_from_file(output_file)
        indegree, outdegree = get_in_out_degrees(edges)
        num_nodes = len(indegree)
        num_test_edges = int(len(edges) * split_ratio)

        random.seed(0)
        random.shuffle(edges)

        for split_type in split_types:
            ks = caps

            # set seed 'u1','u2',...
            
            for cap, k in zip(caps, ks):
                random.seed(0)
                for it in fold_ids:
                    if cap == 0:
                        random.shuffle(edges)
                
                    fold_path = file_path + "{}/{}/{}".format(split_type+'_'+str(k), it, it)
                    create_folder(file_path + "{}/{}".format(split_type+'_'+str(k), it))
                    train_file = fold_path + ".edgelist"
                    test_file = fold_path + "_test.edgelist"
                    
                    train_edges = []
                    test_edges = []

                    # non-iid split from authority perspective 
                    if 'noniid-in' in split_type:
                        if 'barrier' in split_type:
                            split_flags = [0]*len(edges)
                            count = [0]*num_nodes
                            while True:
                                for i, edge in enumerate(edges):
                                    r = random.random()
                                    cap_ = 1/(indegree[edge[1]]**cap)

                                    if split_flags[i] == 0 and count[edge[1]] < indegree[edge[1]]-1 and r <= cap_:
                                        test_edges.append(edge)
                                        split_flags[i] = 1
                                        count[edge[1]] += 1
                                        if len(test_edges) == num_test_edges:
                                            break
                                if len(test_edges) == num_test_edges:
                                    break       

                    # non-iid split from hub perspective 
                    elif 'noniid-out' in split_type:
                        if 'barrier' in split_type:
                            split_flags = [0]*len(edges)
                            count = [0]*num_nodes
                            while True:
                                for i, edge in enumerate(edges):
                                    r = random.random()
                                    cap_ = 1/(outdegree[edge[0]]**cap)

                                    if split_flags[i] == 0 and count[edge[0]] < outdegree[edge[0]]-1 and r <= cap_:
                                        test_edges.append(edge)
                                        split_flags[i] = 1
                                        count[edge[0]] += 1
                                        if len(test_edges) == num_test_edges:
                                            break         
                                if len(test_edges) == num_test_edges:
                                    break    

                    for edge in edges:
                        if edge not in test_edges:
                            train_edges.append(edge)

                    #print('ori_num_nodes: {}'.format(get_num_nodes_inmemory(edges)))
                    get_connected_graph_reindexed_v2(graph_type, train_edges, train_file, test_edges, test_file)
                    #print('wcc_num_nodes: {}'.format(get_num_nodes_inmemory(read_edges_from_file(train_file))))

                    opt = ['ulp','blp','mlp']
                    r = ['1']
                    if "ulp" in opt:
                        #print("#####################opt: U-LP#######################")
                        uni_train = fold_path + "_unconnected_ulp_train.edgelist"
                        uni_test = fold_path + "_unconnected_ulp_test.edgelist"
                        generate_unconnected_train_links(graph_type, train_file, test_file, uni_train, r, opt="unidirectional", reverse_ratio=0.0)
                        generate_unconnected_test_links(graph_type, train_file, test_file, uni_train, uni_test, r, opt="unidirectional", reverse_ratio=0.0)   
                        
                    if "blp" in opt:
                        #print("#####################opt: B-LP#######################")
                        uni_train = fold_path + "_unconnected_blp_train.edgelist"
                        uni_test = fold_path + "_unconnected_blp_test.edgelist"
                        generate_unconnected_train_links(graph_type, train_file, test_file, uni_train, r, opt="unidirectional", reverse_ratio=1.0)
                        generate_unconnected_test_links(graph_type, train_file, test_file, uni_train, uni_test, r, opt="unidirectional", reverse_ratio=1.0)
                            
                    if "mlp" in opt:
                        #print("#####################opt: M-LP#######################")
                        uni_train = fold_path + "_unconnected_mlp_train.edgelist"
                        uni_test = fold_path + "_unconnected_mlp_test.edgelist"
                        generate_unconnected_train_links(graph_type, train_file, test_file, uni_train, r, opt="unidirectional", reverse_ratio=0.5)
                        generate_unconnected_test_links(graph_type, train_file, test_file, uni_train, uni_test, r, opt="unidirectional", reverse_ratio=0.5)

def get_node_degrees(edges):
    d = {}
    for edge in edges:
        if d.get(edge[0]) is None:
            d[edge[0]] = 0
        if d.get(edge[1]) is None:
            d[edge[1]] = 0
        d[edge[0]] += 1
        d[edge[1]] += 1
    return d

def get_in_out_degrees(edges):
    d_out = {}
    d_in = {}
    for edge in edges:
        if d_out.get(edge[0]) is None:
            d_out[edge[0]] = 0
        if d_out.get(edge[1]) is None:
            d_out[edge[1]] = 0
        
        if d_in.get(edge[1]) is None:
            d_in[edge[1]] = 0
        if d_in.get(edge[0]) is None:
            d_in[edge[0]] = 0

        d_out[edge[0]] += 1
        d_in[edge[1]] += 1

    d = get_node_degrees(edges)

    #print('0 out-degree nodes: {}/{} => {}%'.format(len(d)-len(d_out), len(d), (len(d)-len(d_out))/len(d)*100))
    #print('0 in-degree nodes: {}/{} => {}%'.format(len(d)-len(d_in), len(d), (len(d)-len(d_in))/len(d)*100))
    
    for node in d.keys():
        if d_in.get(node) is None:
            d_in[node] = 0
        if d_out.get(node) is None:
            d_out[node] = 0

    out = []
    ind = []
    for node in d_out.values():
        out.append(node)
    for node in d_in.values():
        ind.append(node)

    out = np.array(out)
    ind = np.array(ind)

    #print('out-degree max: {}, var: {}'.format(numpy.max(out), numpy.var(out)))
    #print('in-degree max: {}, var: {}'.format(numpy.max(ind), numpy.var(ind)))

    return d_in, d_out

if __name__ == "__main__":
    noniid_split()