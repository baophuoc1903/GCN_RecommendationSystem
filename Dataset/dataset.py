import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from .preprocess_data import negative_sampling


# from preprocess_data import negative_sampling
# from main import args_parser


def create_ground_truth(df, num_user, num_item):
    user, item = df["UserID"], df["ProductID"]
    index_tensor = torch.LongTensor([user, item])
    ground_truth = torch.sparse_coo_tensor(index_tensor,
                                           torch.ones(index_tensor.shape[1], dtype=torch.float),
                                           size=torch.Size([num_user, num_item]))
    checkins = index_tensor.t()

    return ground_truth, checkins


def aggregate_all_behavior(adj_dict):
    train_matrix = torch.zeros_like(list(adj_dict.values())[0].to_dense())
    for adj_matrix in adj_dict.values():
        train_matrix += adj_matrix.to_dense()

    train_matrix = torch.clamp(train_matrix, min=0, max=1)
    return train_matrix


class CustomDataset(Dataset):

    def __init__(self, hyps):
        self.hyps = hyps
        self.path = hyps.data_path
        self.name = hyps.data_name
        self.behaviors = hyps.behaviors
        self.__load_size()
        self.__create_adj_matrix()
        self.__calculate_user_item_degree()
        self.__generate_ground_truth()
        self.__load_item_graph()

        sampling_path = os.path.join(self.path, self.name, "sampling")
        if os.path.exists(sampling_path):
            num_files = len([f for f in os.listdir(sampling_path)
                             if os.path.isfile(os.path.join(sampling_path, f))])
            if num_files != self.hyps.epoch // self.hyps.next_sample:
                negative_sampling(os.path.join(self.path, self.name), self.hyps.epoch // self.hyps.next_sample)
        else:
            negative_sampling(os.path.join(self.path, self.name), self.hyps.epoch // self.hyps.next_sample)

        self.num_sampling = 0
        self.__read_training_sample(self.num_sampling)

    def __load_size(self):
        size_df = pd.read_csv(os.path.join(self.path, self.name, 'user_item_size.csv'))
        self.num_user, self.num_item = size_df.iloc[0]

    def __load_item_graph(self):
        self.item_graph = {}
        self.item_graph_degree = {}
        for behavior in self.behaviors:
            self.item_graph[behavior] = torch.load(
                os.path.join(self.path, self.name, 'item_graph', f'{behavior}_item_graph.pth')).to_dense()
            self.item_graph_degree[behavior] = self.item_graph[behavior].sum(dim=1).float().unsqueeze(-1)

    def __create_adj_matrix(self):
        self.adj_matrix_dict = {}

        for behavior in self.behaviors:
            data_df = pd.read_csv(os.path.join(self.path, self.name, behavior + '.csv'))
            user, item = data_df["UserID"], data_df["ProductID"]
            index_tensor = torch.LongTensor([user, item])
            num_interaction = index_tensor.shape[1]
            self.adj_matrix_dict[behavior] = torch.sparse_coo_tensor(index_tensor,
                                                                     torch.ones(num_interaction, dtype=torch.float),
                                                                     torch.Size([self.num_user, self.num_item]))

    def __calculate_user_item_degree(self):
        user_behavior = item_behavior = None
        for i in range(len(self.behaviors)):
            if i == 0:
                user_behavior = self.adj_matrix_dict[self.behaviors[i]].to_dense().sum(dim=1).unsqueeze(-1)
                item_behavior = self.adj_matrix_dict[self.behaviors[i]].to_dense().t().sum(dim=1).unsqueeze(-1)
            else:
                user_behavior = torch.cat(
                    (user_behavior, self.adj_matrix_dict[self.behaviors[i]].to_dense().sum(dim=1).unsqueeze(-1)), dim=1)
                item_behavior = torch.cat(
                    (item_behavior, self.adj_matrix_dict[self.behaviors[i]].to_dense().t().sum(dim=1).unsqueeze(-1)),
                    dim=1)
        self.user_behaviour_degree = user_behavior
        self.item_behaviour_degree = item_behavior

    def __generate_ground_truth(self):
        data_df = pd.read_csv(os.path.join(self.path, self.name, 'buy.csv'))

        self.ground_truth, self.checkins = create_ground_truth(data_df, self.num_user, self.num_item)
        self.train_matrix = aggregate_all_behavior(self.adj_matrix_dict).to_sparse_coo()

    def __read_training_sample(self, num_sampling):
        data_df = pd.read_csv(os.path.join(self.path, self.name, f'sampling/sampling_{num_sampling}.csv'))
        user, pos_item, neg_item = map(torch.LongTensor,
                                       [data_df["UserID"], data_df["ProductID"], data_df["NegProductID"]])

        self.train_sample = torch.concat([user.unsqueeze(-1), pos_item.unsqueeze(-1), neg_item.unsqueeze(-1)], dim=1)

    def next_sampling(self):
        self.num_sampling += 1
        self.num_sampling = min(self.num_sampling, self.hyps.epoch // self.hyps.next_sample - 1)
        self.__read_training_sample(self.num_sampling)

    def __getitem__(self, index):
        return self.train_sample[index, 0].unsqueeze(-1), self.train_sample[index, 1:]

    def __len__(self):
        return len(self.checkins)


class ValidationDataset(Dataset):
    def __init__(self, hypers, train_dataset, task='test'):
        self.path = hypers.data_path
        self.name = hypers.data_name
        self.train_gt = train_dataset.ground_truth
        self.num_user, self.num_item = train_dataset.num_user, train_dataset.num_item
        self.task = task
        self.__read_task()

    def __read_task(self):
        data_df = pd.read_csv(os.path.join(self.path, self.name, self.task + '.csv'))
        self.ground_truth, self.checkins = create_ground_truth(data_df, self.num_user, self.num_item)

    def __getitem__(self, index):
        return index, self.ground_truth[index].to_dense().squeeze(), self.train_gt[index].to_dense().squeeze()

    def __len__(self):
        return self.num_user


if __name__ == '__main__':
    pass
    # from torch.utils.data import DataLoader
    #
    # hyps = args_parser()
    # hyps.data_path = './'
    # hyps.epoch = 1
    # train_ds = CustomDataset(hyps)
    # train_dl = DataLoader(train_ds, 2, shuffle=False,
    #                       num_workers=0, pin_memory=True)
    #
    # iterator = iter(train_dl)
    # data = next(iterator)
    # print(data)
