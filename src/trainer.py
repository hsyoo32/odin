#coding=utf-8
#pylint: disable=no-member
#pylint: disable=no-name-in-module
#pylint: disable=import-error

from tqdm import tqdm
import model
import data
import torch
import torch.optim as optim

class Trainer(object):

    def __init__(self, flags_obj, dm):
        self.dm = dm
        self.flags_obj = flags_obj
        self.lr = flags_obj.lr
        self.recommender = Recommender(flags_obj, dm)
        self.recommender.transfer_model()

    def train(self):
        self.dataloader = self.recommender.get_dataloader()
        self.optimizer = self.recommender.get_optimizer()

        for epoch in tqdm(range(self.flags_obj.epochs)):
            self.train_one_epoch(epoch, self.dataloader, self.optimizer)

        self.recommender.save_embeddings(self.flags_obj.emb_file)

    def train_one_epoch(self, epoch, dataloader, optimizer):
        total_loss = 0.0
        for batch_count, sample in enumerate(dataloader):
            optimizer.zero_grad()
            loss = self.recommender.get_loss(sample)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

class Recommender(object):

    def __init__(self, flags_obj, dm):
        self.dm = dm
        self.flags_obj = flags_obj
        self.set_device()
        self.set_model()

    def set_device(self):
        if not self.flags_obj.use_gpu:
            self.device = torch.device('cpu')
        else:
            self.device = torch.device('cuda:{}'.format(self.flags_obj.gpu_id))

    def transfer_model(self):
        self.model = self.model.to(self.device)

    def get_optimizer(self):
        return optim.Adam(self.model.parameters(), lr=self.flags_obj.lr, weight_decay=self.flags_obj.weight_decay, betas=(0.5, 0.99), amsgrad=True)

    def set_model(self):
        self.model = model.ODIN(self.dm.n_node, self.flags_obj)

    def get_dataloader(self):
        return data.DataProcessor.get_ODIN_dataloader(self.flags_obj, self.dm)

    def get_loss(self, sample):

        srcs, tars, tars_neg, srcs_neg, mask_int_tar, mask_int_src, mask_auth_up, mask_auth_down, \
        mask_hub_up, mask_hub_down, mask_hub_tar, mask_auth_src = sample
        srcs = srcs.to(self.device)
        tars = tars.to(self.device)
        tars_neg = tars_neg.to(self.device)
        srcs_neg = srcs_neg.to(self.device)
        mask_int_tar = mask_int_tar.to(self.device)
        mask_int_src = mask_int_src.to(self.device)
        mask_auth_up = mask_auth_up.to(self.device)
        mask_hub_up = mask_hub_up.to(self.device)
        mask_auth_down = mask_auth_down.to(self.device)
        mask_hub_down = mask_hub_down.to(self.device)
        mask_hub_tar = mask_hub_tar.to(self.device)
        mask_auth_src = mask_auth_src.to(self.device)

        loss = self.model(srcs, tars, tars_neg, srcs_neg, mask_int_tar, mask_int_src, mask_auth_up, 
                          mask_auth_down, mask_hub_up, mask_hub_down, mask_hub_tar, mask_auth_src)

        return loss

    def save_embeddings(self, emb_file):

        self.user_embeddings = self.model.get_user_embeddings()
        self.item_embeddings = self.model.get_item_embeddings()
        with open(emb_file, 'w+') as f:
            f.write(str(self.dm.n_node)+'\t'+str(self.flags_obj.embedding_size*3)+'\n')
            for cnt, emb in enumerate(self.user_embeddings):
                f.write(str(cnt))
                for v in emb:
                    f.write('\t'+str(v))
                f.write('\n')
        with open(emb_file+'2', 'w+') as f:
            f.write(str(self.dm.n_node)+'\t'+str(self.flags_obj.embedding_size*3)+'\n')
            for cnt, emb in enumerate(self.item_embeddings):
                f.write(str(cnt))
                for v in emb:
                    f.write('\t'+str(v))
                f.write('\n')
