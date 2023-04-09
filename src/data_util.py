#!/usr/local/anaconda3/envs/torch-1.0-py3/bin/python
#coding=utf-8
#pylint: disable=no-member
#pylint: disable=no-name-in-module
#pylint: disable=import-error

import scipy.sparse as sp
import numpy as np

class DatasetManager(object):

    def __init__(self, flags_obj):
        self.transformer = Transformer(flags_obj)
        self.edges_loader = EdgesLoader(flags_obj)
        self.input_file = flags_obj.input_file

    def get_dataset_info(self):
        self.edgelist = self.edges_loader.load(self.input_file)
        # targets per source
        self.tps = self.transformer.edges2tps(self.edgelist)
        # sources per target
        self.spt = self.transformer.edges2spt(self.edgelist)
        self.n_node = len(self.tps)
        self.indegree = np.array(self.transformer.edges2indegree(self.edgelist))
        self.outdegree = np.array(self.transformer.edges2outdegree(self.edgelist))

class EdgesLoader(object):

    def __init__(self, flags_obj):
        pass
    
    def load(self, filename):

        with open(filename, "r") as f: 
            lines = f.readlines()
            edges = [self.str_list_to_int(line.split()[:2]) for line in lines]
        return edges

    def str_list_to_int(self, str_list):
        return [int(item) for item in str_list]

class Transformer(object):

    def __init__(self, flags_obj):
        pass

    def edges2tps(self, edges):
        d = {}
        for edge in edges:
            if d.get(edge[0]) is None:
                d[edge[0]] = []
            if d.get(edge[1]) is None:
                d[edge[1]] = []
            d[edge[0]].append(edge[1])
        return d

    def edges2spt(self, edges):
        d = {}
        for edge in edges:
            if d.get(edge[0]) is None:
                d[edge[0]] = []
            if d.get(edge[1]) is None:
                d[edge[1]] = []
            d[edge[1]].append(edge[0])
        return d

    def edges2indegree(self, edges):
        d = {}
        for edge in edges:
            if d.get(edge[0]) is None:
                d[edge[0]] = 0
            if d.get(edge[1]) is None:
                d[edge[1]] = 0
            d[edge[1]] += 1
        l = sorted(d.items(), key=lambda x : x[0])
        indegree_list = []
        for _, indegree in l:
            indegree_list.append(indegree)

        return indegree_list

    def edges2outdegree(self, edges):
        d = {}
        for edge in edges:
            if d.get(edge[0]) is None:
                d[edge[0]] = 0
            if d.get(edge[1]) is None:
                d[edge[1]] = 0
            d[edge[0]] += 1
        l = sorted(d.items(), key=lambda x : x[0])
        outdegree_list = []
        for _, outdegree in l:
            outdegree_list.append(outdegree)

        return outdegree_list