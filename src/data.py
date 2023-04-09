#coding=utf-8
#pylint: disable=no-member
#pylint: disable=no-name-in-module
#pylint: disable=import-error

from torch.utils.data import Dataset, DataLoader
import numpy as np

class DataProcessor(object):

    def __init__(self, flags_obj):
        pass

    @staticmethod
    def get_ODIN_dataloader(flags_obj, dm):

        dataset = ODINDataset(flags_obj, dm)

        return DataLoader(dataset, batch_size=flags_obj.batch_size, shuffle=flags_obj.shuffle, 
                          num_workers=flags_obj.num_workers, drop_last=True)

class ODINDataset(Dataset):

    def __init__(self, flags_obj, dm):
        self.make_sampler(flags_obj, dm)

    def make_sampler(self, flags_obj, dm):
        train_edgelist = dm.edgelist
        train_tps = dm.tps
        train_spt = dm.spt
        indegree = dm.indegree
        outdegree = dm.outdegree

        self.sampler = ODINSampler(flags_obj, train_edgelist, train_tps, train_spt, flags_obj.neg_sample_rate, 
            indegree, outdegree, pool=flags_obj.pool)

    def __len__(self):

        return len(self.sampler.edgelist)

    def __getitem__(self, index):
        srcs, tars, tars_neg, srcs_neg, mask_int_tar, mask_int_src, mask_auth_up, mask_auth_down, \
        mask_hub_up, mask_hub_down, mask_hub_tar, mask_auth_src = self.sampler.sample(index)

        return srcs, tars, tars_neg, srcs_neg, mask_int_tar, mask_int_src, mask_auth_up, mask_auth_down, \
        mask_hub_up, mask_hub_down, mask_hub_tar, mask_auth_src

class ODINSampler(object):

    def __init__(self, flags_obj, edgelist, tps, spt, neg_sample_rate, indegree, outdegree, pool=10):
        self.tps = tps
        self.spt = spt
        self.edgelist = edgelist
        self.n_node = len(tps)
        self.indegree = indegree
        self.outdegree = outdegree
        # self.pool = pool
        self.neg_sample_rate = neg_sample_rate
        self.pool = self.neg_sample_rate

    def get_pos_src_tar(self, index):
        # source, target
        src = self.edgelist[index][0]
        tar = self.edgelist[index][1]

        return src, tar
    
    def generate_negative_samples(self, src, tar):

        negative_samples_tar = np.full(self.neg_sample_rate, -1, dtype=np.int64)
        negative_samples_src = np.full(self.neg_sample_rate, -1, dtype=np.int64)
        mask_type_int_tar = np.full(self.neg_sample_rate, False, dtype=np.bool)
        mask_type_int_src = np.full(self.neg_sample_rate, False, dtype=np.bool)
        
        mask_type_auth_up = np.full(self.neg_sample_rate, False, dtype=np.bool)
        mask_type_hub_up = np.full(self.neg_sample_rate, False, dtype=np.bool)
        mask_type_auth_down = np.full(self.neg_sample_rate, False, dtype=np.bool)
        mask_type_hub_down = np.full(self.neg_sample_rate, False, dtype=np.bool)

        mask_auth_src = np.full(self.neg_sample_rate, False, dtype=np.bool)
        mask_hub_tar = np.full(self.neg_sample_rate, False, dtype=np.bool)

        # source perspective negative sampling
        pos_tars = self.tps[src]
        tars_indegree = self.indegree[tar]
        neg_tars_greater = np.nonzero(self.indegree > tars_indegree)[0]
        neg_tars_greater = neg_tars_greater[np.logical_not(np.isin(neg_tars_greater, pos_tars))]
        num_neg_tars_greater = len(neg_tars_greater)

        neg_tars_smaller = np.nonzero(self.indegree < tars_indegree)[0]
        neg_tars_smaller = neg_tars_smaller[np.logical_not(np.isin(neg_tars_smaller, pos_tars))]
        num_neg_tars_smaller = len(neg_tars_smaller)

        # target perspective negative sampling
        pos_srcs = self.spt[tar]
        srcs_outdegree = self.outdegree[src]
        neg_srcs_greater = np.nonzero(self.outdegree > srcs_outdegree)[0]        
        neg_srcs_greater = neg_srcs_greater[np.logical_not(np.isin(neg_srcs_greater, pos_srcs))]
        num_neg_srcs_greater = len(neg_srcs_greater)

        neg_srcs_smaller = np.nonzero(self.outdegree < srcs_outdegree)[0]        
        neg_srcs_smaller = neg_srcs_smaller[np.logical_not(np.isin(neg_srcs_smaller, pos_srcs))]
        num_neg_srcs_smaller = len(neg_srcs_smaller)

        count = 0
        while True:
            if num_neg_srcs_greater >= self.pool:
                index = np.random.randint(num_neg_srcs_greater)
                src = neg_srcs_greater[index]
                while src in negative_samples_src:
                    index = np.random.randint(num_neg_srcs_greater)
                    src = neg_srcs_greater[index]

                mask_type_hub_up[count] = False
                mask_type_hub_down[count] = True
                mask_type_int_src[count] = True
                mask_auth_src[count] = True

                negative_samples_src[count] = src
                count += 1
                if count == self.neg_sample_rate:
                    break
            
            if num_neg_srcs_smaller >= self.pool:

                index = np.random.randint(num_neg_srcs_smaller)
                src = neg_srcs_smaller[index]
                while src in negative_samples_src:
                    index = np.random.randint(num_neg_srcs_smaller)
                    src = neg_srcs_smaller[index]  

                mask_type_hub_up[count] = True
                mask_type_hub_down[count] = False
                mask_type_int_src[count] = False
                mask_auth_src[count] = False

                negative_samples_src[count] = src
                count += 1
                if count == self.neg_sample_rate:
                    break

        count = 0
        while True:
            if num_neg_tars_greater >= self.pool:
                index = np.random.randint(num_neg_tars_greater)
                tar = neg_tars_greater[index]
                while tar in negative_samples_tar:
                    index = np.random.randint(num_neg_tars_greater)
                    tar = neg_tars_greater[index]

                mask_type_auth_up[count] = False
                mask_type_auth_down[count] = True
                mask_type_int_tar[count] = True
                mask_hub_tar[count] = True
                
                negative_samples_tar[count] = tar
                count += 1
                if count == self.neg_sample_rate:
                    break
            
            if num_neg_tars_smaller >= self.pool:

                index = np.random.randint(num_neg_tars_smaller)
                tar = neg_tars_smaller[index]
                while tar in negative_samples_tar:
                    index = np.random.randint(num_neg_tars_smaller)
                    tar = neg_tars_smaller[index]     

                mask_type_auth_up[count] = True
                mask_type_auth_down[count] = False
                mask_type_int_tar[count] = False
                mask_hub_tar[count] = False

                negative_samples_tar[count] = tar
                count += 1
                if count == self.neg_sample_rate:
                    break

        return negative_samples_tar, negative_samples_src, mask_type_int_tar, mask_type_int_src, \
        mask_type_auth_up, mask_type_auth_down, mask_type_hub_up, mask_type_hub_down, mask_hub_tar, mask_auth_src

    def sample(self, index):

        src, tar = self.get_pos_src_tar(index)
        srcs = np.full(self.neg_sample_rate, src, dtype=np.int64)
        tars = np.full(self.neg_sample_rate, tar, dtype=np.int64)
        tars_neg, srcs_neg, mask_int_tar, mask_int_src, mask_auth_up, mask_auth_down, mask_hub_up, \
            mask_hub_down, mask_hub_tar, mask_auth_src = self.generate_negative_samples(src, tar)

        return srcs, tars, tars_neg, srcs_neg, mask_int_tar, mask_int_src, mask_auth_up, mask_auth_down, \
            mask_hub_up, mask_hub_down, mask_hub_tar, mask_auth_src
