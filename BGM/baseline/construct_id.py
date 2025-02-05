import argparse
import pickle
import sys
import os
import numpy as np

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, parent_dir)

from models.base.DictTree import TreeNode

def get_args():
    parser = argparse.ArgumentParser()
    # Meta
    parser.add_argument('--dataroot', type=str, default='../data/')
    parser.add_argument("--meta_root", type=str, default="../data/")
    parser.add_argument("--meta_file", type=str, default="SCSShip_retrieval.pkl")
    parser.add_argument('--dataset', type=str,  default='SCSShip')

    args = parser.parse_args()

    args.meta_root = os.path.join(args.meta_root, args.dataset)

    return args

if __name__ == '__main__':
    # load meta file
    args = get_args()

    with open(os.path.join(args.meta_root, args.meta_file), 'rb') as fin:
        cfg = pickle.load(fin)

    dict_tree = TreeNode()

    imlist = cfg["imlist"]
    codes = []
    for i in range(len(imlist)):
        codes.append(i)
    codes = np.asarray(codes).reshape(-1, 1)
    dict_tree.insert_many(codes)

    cfg = {"mapping": codes, "dict_tree": dict_tree}
    output_file_name = f"autom_id_{args.dataset}_{len(imlist)}.pkl"
    with open(os.path.join(args.meta_root, output_file_name),'wb')as f:
        pickle.dump(cfg,f)
