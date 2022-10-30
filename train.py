import os
from tqdm import tqdm
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from Dataset import CustomDataset, ValidationDataset
from model import MF, Multi_Behavior_Graph_Base
from utils.loss import bpr_loss
from utils.metrics import Recall, NDCG
from utils.logger import Logger

BIGNUM = 1e8


class Training(object):
    def __init__(self, hyps):
        self.hyps = hyps

        self.train_ds = self.valid_ds = self.test_ds = None
        self.train_dl = self.valid_dl = self.test_dl = None
        self.data_set_init(hyps)

        self.device = None
        self.set_device(hyps)

        if hyps.model_name == 'MBGB':
            self.model = Multi_Behavior_Graph_Base(hyps, self.train_ds, self.device).to(self.device)
        else:
            self.model = MF(hyps, self.train_ds, self.device).to(self.device)

        if hyps.opt.lower() == 'adam':
            self.opt = optim.Adam(self.model.parameters(), lr=hyps.lr, amsgrad=True)
        else:
            self.opt = optim.SGD(self.model.parameters(), lr=hyps.lr, nesterov=True, momentum=0.9)

        self.scheduler = ReduceLROnPlateau(self.opt, mode='max', factor=0.75, patience=10,
                                           verbose=True, min_lr=5e-7, threshold_mode='abs', threshold=1e-3)

        self.metrics_dict = {'Recall@5': Recall(5), 'NDCG@5': NDCG(5),
                             'Recall@10': Recall(10), 'NDCG@10': NDCG(10),
                             'Recall@20': Recall(20), 'NDCG@20': NDCG(20),
                             'Recall@40': Recall(40), 'NDCG@40': NDCG(40),
                             'Recall@80': Recall(80), 'NDCG@80': NDCG(80)}
        self.max_metric = -1.0
        self.max_epoch = -1
        self.patience = 0

        self.logger_recall = Logger(self.hyps, name="Recall@10")
        self.logger_ndcg = Logger(self.hyps, name="NDCG@10")

    def set_device(self, hyps):
        if hyps.gpu:
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

    def data_set_init(self, hyps):
        self.train_ds = CustomDataset(hyps)
        self.valid_ds = ValidationDataset(hyps, self.train_ds, task='validation')
        self.test_ds = ValidationDataset(hyps, self.train_ds, task='test')

        self.train_dl = DataLoader(self.train_ds, hyps.batch_size, shuffle=True, num_workers=hyps.num_workers,
                                   pin_memory=True)
        self.valid_dl = DataLoader(self.valid_ds, hyps.batch_size, shuffle=False,
                                   num_workers=hyps.num_workers, pin_memory=True)
        self.test_dl = DataLoader(self.test_ds, hyps.batch_size, shuffle=False, num_workers=hyps.num_workers,
                                  pin_memory=True)

    def train(self):
        self.max_metric = -1.0
        self.max_epoch = -1
        self.patience = 0

        for epoch in range(self.hyps.epoch):
            self.train_one_epoch(epoch)
            self.validation(self.valid_dl)

            self.update_max(epoch)
            if (epoch != 0) and (epoch % self.hyps.next_sample == 0):
                self.train_dl.dataset.next_sampling()

            eval_metric = list(self.metrics_dict.values())[0]._metric
            if eval_metric < self.max_metric:
                self.patience += 1
                if self.patience == self.hyps.early_stop:
                    break
            else:
                self.patience = 0

            self.scheduler.step(eval_metric)

        print("Testing")
        self.validation(self.test_dl, task="Test evaluation")

        self.logger_recall.write_log()
        self.logger_ndcg.write_log()

    def train_one_epoch(self, epoch):
        self.model.train()
        # if self.hyps.model_name == 'MBGB':
        #     print(f"Weight of each behavior {self.hyps.behaviors}: {[i.item() for i in self.model.behavior_weight]}")
        total_loss = 0.0
        for i, data in enumerate(tqdm(self.train_dl, desc=f"Epoch {epoch}")):
            users, items = data
            self.opt.zero_grad()
            output = self.model(users.to(self.device), items.to(self.device))

            loss = bpr_loss(output, batch_size=self.train_dl.batch_size, loss_mode=self.hyps.loss_mode)
            total_loss += loss.item()
            loss.backward()
            self.opt.step()
        print(f"BPR Loss: {total_loss:.4f}")

    def validation(self, dl, task='Validation'):
        print(f"{task}")
        self.model.eval()
        for metric in self.metrics_dict:
            self.metrics_dict[metric].start()

        with torch.no_grad():
            for i, data in enumerate(dl):
                users, ground_truth, train_gt = data
                pred = self.model.evaluate(users.to(self.device))

                pred -= BIGNUM * train_gt.to(self.device)

                for metric in self.metrics_dict:
                    self.metrics_dict[metric](pred, ground_truth.to(self.device))

        for metric in self.metrics_dict:
            self.metrics_dict[metric].stop()

        self.logger_recall.read_log(self.metrics_dict[self.logger_recall.name]._metric)
        self.logger_ndcg.read_log(self.metrics_dict[self.logger_ndcg.name]._metric)
        print(("{:<20}" * len(self.metrics_dict)).format(*list(self.metrics_dict.keys())))
        print(("{:<20.8f}" * len(self.metrics_dict)).format(*[val._metric for val in list(self.metrics_dict.values())]))

    def update_max(self, epoch):

        metric_list = list(self.metrics_dict.values())
        metric = metric_list[0]._metric
        if metric > self.max_metric:
            self.max_metric = metric
            self.max_epoch = epoch

            if not os.path.exists(self.hyps.output_path):
                os.mkdir(self.hyps.output_path)
            torch.save(self.model.state_dict(),
                       os.path.join(self.hyps.output_path, f'best_{self.hyps.model_name}_tuning.pth'))


if __name__ == '__main__':
    pass
