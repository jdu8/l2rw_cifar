"""Seed the CIFAR-10 data directory from keras.

Use this on Colab where torchvision's download URL returns 403.
Run once before any training script:

    python seed_cifar10.py --data_root ./data
"""
import argparse
import os
import pickle

import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data_root', default='./data')
    args = p.parse_args()

    from tensorflow.keras.datasets import cifar10
    (x_tr, y_tr), (x_te, y_te) = cifar10.load_data()

    base = os.path.join(args.data_root, 'cifar-10-batches-py')
    os.makedirs(base, exist_ok=True)

    # torchvision expects rows in CHW-flattened order (3072 = 3*32*32)
    x_tr_flat = x_tr.transpose(0, 3, 1, 2).reshape(-1, 3072)
    x_te_flat = x_te.transpose(0, 3, 1, 2).reshape(-1, 3072)

    for i in range(5):
        s, e = i * 10000, (i + 1) * 10000
        with open(os.path.join(base, f'data_batch_{i+1}'), 'wb') as f:
            pickle.dump({'data': x_tr_flat[s:e], 'labels': y_tr[s:e, 0].tolist()}, f)

    with open(os.path.join(base, 'test_batch'), 'wb') as f:
        pickle.dump({'data': x_te_flat, 'labels': y_te[:, 0].tolist()}, f)

    with open(os.path.join(base, 'batches.meta'), 'wb') as f:
        pickle.dump({'label_names': ['airplane', 'automobile', 'bird', 'cat', 'deer',
                                     'dog', 'frog', 'horse', 'ship', 'truck']}, f)

    print(f'CIFAR-10 seeded at {base}  ({len(y_tr)} train / {len(y_te)} test)')


if __name__ == '__main__':
    main()
