import argparse
from solver import Solver
from utils import *
from dataset import train_data_reduced, test_data_reduced


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_epochs", type=int, default=30)
    parser.add_argument("--print_every", type=int, default=1)

    parser.add_argument("--customNet", type=bool, default=False)

    # data visualization
    visualize_samples_image(train_data_reduced)
    visualize_label_distribution(train_data_reduced, "Distribution of samples in reduced training set")
    visualize_label_distribution(test_data_reduced, "Distribution of samples in reduced test set")

    args = parser.parse_args()
    solver = Solver(args)
    solver.fit()


if __name__ == "__main__":
    main()
