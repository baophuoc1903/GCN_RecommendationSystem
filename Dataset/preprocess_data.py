import os
from pathlib import Path
from itertools import combinations
from collections import defaultdict
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
from shutil import rmtree


def read_data(df_path, df_name, verbose=True):
    df = pd.read_csv(Path(df_path).joinpath(df_name))

    if verbose:
        print(f"Number of users: {len(df['UserID'].unique())}")
        print(f"Number of items: {len(df['ProductID'].unique())}")

        with open(os.path.join(df_path, 'behavior_type.txt')) as fh:
            behaviors = fh.readline().strip('\n').split(',')
            for behavior in behaviors:
                min_interaction = df[df['BehaviorType'] == behavior].groupby('UserID')['ProductID'].nunique().min()
                max_interaction = df[df['BehaviorType'] == behavior].groupby('UserID')['ProductID'].nunique().max()
                print(f"\n{behavior.upper():^50}")
                print(f"Minimum number of unique item an user interacted with: {min_interaction}")
                print(f"Maximum number of unique item an user interacted with: {max_interaction}")

    return df


def df_split(df, data_path, num_val=2, save_name=("validation.csv", "test.csv")):
    print("\nBegin split dataset into training, validation and testing")
    with open(os.path.join(data_path, 'behavior_type.txt')) as fh:
        behaviors = fh.readline().strip('\n').split(',')

    for behavior in tqdm(behaviors):
        if behavior != 'buy':
            df_behavior = df[df["BehaviorType"] == behavior][["UserID", "ProductID"]]
            df_behavior.drop_duplicates(inplace=True)

            df_behavior.to_csv(os.path.join(data_path, f"{behavior}.csv"), index=False)
            continue

        # Group by user
        df_behavior = df[df["BehaviorType"] == behavior][["UserID", "ProductID"]]
        df_behavior.drop_duplicates(inplace=True)

        #  count number of interaction for a user
        interact_count_per_user = df_behavior.groupby("UserID").nunique().apply(list).to_dict()["ProductID"]

        # Initialize second dataframe
        valid_df = pd.DataFrame(columns=["UserID", "ProductID"])
        test_df = pd.DataFrame(columns=["UserID", "ProductID"])

        for user, cnt in interact_count_per_user.items():
            if cnt > num_val:
                list_item = df_behavior[df_behavior["UserID"] == user]
                idx = np.random.choice(len(list_item), size=num_val, replace=False)

                df_behavior = pd.concat([df_behavior, list_item.iloc[idx]]).drop_duplicates(keep=False)

                val_idx = idx[:num_val // 2]
                test_idx = idx[num_val // 2:]
                valid_df = pd.concat([valid_df, list_item.iloc[val_idx]]).drop_duplicates()
                test_df = pd.concat([test_df, list_item.iloc[test_idx]]).drop_duplicates()

        df_behavior.to_csv(os.path.join(data_path, f"{behavior}.csv"), index=False)
        valid_df.to_csv(os.path.join(data_path, save_name[0]), index=False)
        test_df.to_csv(os.path.join(data_path, save_name[1]), index=False)
    print("Done")


def reindex_user_item(data_path, df):
    new_df = df.copy()
    datasize_path = os.path.join(data_path, r"user_item_size.csv")

    with open(datasize_path, 'w') as fh:
        num_user = len(new_df['UserID'].unique())
        num_item = len(new_df['ProductID'].unique())
        fh.write("user,item\n")
        fh.write(f"{num_user},{num_item}")

    unique_item = {val: i for i, val in enumerate(sorted(new_df['ProductID'].unique()))}
    unique_user = {val: i for i, val in enumerate(sorted(new_df['UserID'].unique()))}
    new_df["UserID"] = new_df["UserID"].apply(lambda x: unique_user[x])
    new_df["ProductID"] = new_df["ProductID"].apply(lambda x: unique_item[x])

    new_df.to_csv(Path(data_path).joinpath(Path(data_path).name + '_reindex.csv'))
    return new_df, (num_user, num_item)


def create_one_item_graph(data_path, behavior, num_item):
    df = read_data(data_path, behavior + '.csv', verbose=False)

    user_item = defaultdict(lambda: [])
    for row in tqdm(df.iterrows()):
        user, item = row[1]
        user_item[user].append(item)

    indices = []
    for _, items in tqdm(user_item.items()):
        indices.extend(list(combinations(items, 2)))

    indices.extend([(i, i) for i in range(num_item)])

    indices = torch.LongTensor(indices).unique(dim=0)
    indices = torch.concat([indices, indices.flip(dims=(1,))], dim=0).unique(dim=0).t()

    item_graph = torch.sparse_coo_tensor(indices,
                                         torch.ones(indices.shape[1], dtype=torch.float),
                                         torch.Size([num_item, num_item]))
    return item_graph


def create_item_graphs(data_path):
    if not os.path.exists(os.path.join(data_path, "behavior_type.txt")):
        raise FileExistsError(f"Don't know behavior to build item graph. "
                              f"Create behavior_type.txt file store behavior to continue")
    if not os.path.exists(os.path.join(data_path, "user_item_size.csv")):
        raise FileExistsError(
            f"Don't know number of item to build graph. Please create a csv file to store number of item")

    _, num_item = read_num(data_path)

    with open(os.path.join(data_path, 'behavior_type.txt')) as fh:
        behaviors = fh.readline().strip('\n').split(',')
        if not behaviors:
            raise Exception(f"Don't have behavior to build graph")

        if not os.path.exists(os.path.join(data_path, "item_graph")):
            os.mkdir(os.path.join(data_path, "item_graph"))

        for idx, behavior in enumerate(behaviors):
            print(f"\nStart building item-item graph for behavior {idx + 1}: {behavior}")
            torch.save(create_one_item_graph(data_path, behavior, num_item),
                       os.path.join(data_path, "item_graph", f"{behavior}_item_graph.pth"))
            print("Done")


def read_num(path):
    size_df = pd.read_csv(os.path.join(path, 'user_item_size.csv'))
    num_user, num_item = size_df.iloc[0]
    return num_user, num_item


def user_interaction_generate(data_path):
    df = read_data(data_path, 'buy.csv', verbose=False)

    interaction = defaultdict(list)
    for user in df["UserID"].unique():
        interaction[user].extend(df[df["UserID"] == user]["ProductID"].tolist())

    return interaction


def negative_sampling(data_path, times=1):
    if os.path.exists(os.path.join(data_path, "sampling")):
        rmtree(os.path.join(data_path, "sampling"))

    os.mkdir(os.path.join(data_path, "sampling"))

    orig_df = read_data(data_path, 'buy.csv', verbose=False)
    user_num, item_num = read_num(data_path)
    user_interaction = user_interaction_generate(data_path)

    print("Start negative sampling for training")
    for i in tqdm(range(times)):
        train_df = orig_df.copy()
        neg_sample = []

        for _, row in train_df.iterrows():
            user = row["UserID"]
            try_cnt = 0
            while True:
                rand_item = torch.randint(0, item_num - 1, size=(1,))
                if rand_item not in user_interaction[user]:
                    break
                try_cnt += 1
                if try_cnt > 100000:
                    break
            neg_sample.append(rand_item.item())
        train_df["NegProductID"] = neg_sample

        train_df.to_csv(os.path.join(data_path, "sampling", f"sampling_{i}.csv"), index=False)
    print("Sampling done")


if __name__ == '__main__':
    df_name = 'taobao'
    path = (Path(__file__).parents[0]).joinpath(df_name)
    df_path = path.joinpath(path.name + '_reindex.csv')

    # Read data
    df = read_data(path, df_name+'.csv', verbose=True)

    # Reindex from 0 userID and productID
    new_df, (num_user, num_item) = reindex_user_item(path, df)

    # Split into train, valid, test purchase behavior
    df_split(new_df, path, num_val=2)

    # Create item-item graph for training item-to-item propagation
    create_item_graphs(path)


    # # =====Testing result=====
    # # Test split for each behavior
    # df = read_data(path, 'pv.csv', verbose=False)
    # new_df = df.groupby("ProductID")["UserID"].nunique()
    # print(new_df.sort_index())
    # print(len(new_df))

    # data_df = read_data(path, 'sampling.csv', verbose=False)
    #
    # user, pos_item, neg_item = map(torch.LongTensor,
    #                                [data_df["UserID"], data_df["ProductID"], data_df["NegProductID"]])
    # train_tmp = torch.concat([user.unsqueeze(-1), pos_item.unsqueeze(-1), neg_item.unsqueeze(-1)], dim=1)
    # print(train_tmp[0])
    # print(train_tmp.shape)
