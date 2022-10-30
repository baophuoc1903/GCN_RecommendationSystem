import argparse
from train import Training


def args_parser():
    parser = argparse.ArgumentParser(description="Hyperparameter for training process")

    # Model and data hyperparameter
    parser.add_argument('--model_name', type=str, default="MBGB", choices={"MBGB", "MF"},
                        help="Name of model to use: Graph-Base or Matrix factorization")
    parser.add_argument('--output_path', type=str, default='./run_larger', help="Path to save model when training")
    parser.add_argument('--pretrain_embed', action="store_true", help="Using a pretrain embedding or not")
    parser.add_argument('--pretrain_path', default='./run_larger/best_MF_1024.pth', help="Path to pretrain embedding")

    parser.add_argument('--embedding_size', type=int, default=512*2, help="Embedding size")
    parser.add_argument('--message_dropout', default=0.1, type=float, help="Percent of message to drop")
    parser.add_argument('--node_dropout', default=0.1, type=float, help="Percent of user-item node to drop")
    parser.add_argument('--sigmoid_weight', default=0.85, type=float, help="Weight for final score calculation")
    parser.add_argument('--regularize_weight', type=float, default=0.0001, help="Regularization weight")
    parser.add_argument('--loss_mode', default='sum', choices={"mean", "sum"}, help="Type of loss reducer")

    parser.add_argument('--behaviors', default=["pv", "buy", "cart", "fav"], type=list, nargs="+",
                        help="List of behavior")
    parser.add_argument('--behavior_weight', nargs="+", type=list, default=[1, 1, 1, 1],
                        help="Weight for each behavior, length equal to number of behaviors")
    parser.add_argument('--data_path', type=str, default='./Dataset', help="Path to dataset directory")
    parser.add_argument('--data_name', type=str, default='taobao_larger',
                        help="Name of dataset that is subdirectory in data_path")

    # Training hyperparameter
    parser.add_argument('--epoch', type=int, default=500, help="Number of training epochs")
    parser.add_argument('--next_sample', type=int, default=1,
                        help="Number of epoch after change to next data sample to continue training")
    parser.add_argument('--opt', type=str, default='adam', help="Optimization")
    parser.add_argument('--lr', type=float, default=1e-4, help="Learning rate")
    parser.add_argument('--gpu', action="store_false", help="Not using gpu?")

    parser.add_argument('--batch_size', type=int, default=256*4, help="Batch size")
    parser.add_argument('--num_workers', type=int, default=0, help="number of subprocess use to load data")
    parser.add_argument('--early_stop', type=int, default=500 ,
                        help="Stop the rest epochs when conditions are met aft"
                             "er given epoch")

    hyps = parser.parse_args()
    return hyps


if __name__ == '__main__':
    pass
    hyps = args_parser()
    print(hyps)

    model_train = Training(hyps)
    model_train.train()
