import argparse
import random
import torch
import numpy as np
import sys

sys.path.append('../')

from dataloaders.WSI_Loader import wsi_tensor_loader

from model.HIPSS import HIPSS
from utils.train_eval_helper import train


def get_args_parser():
    parser = argparse.ArgumentParser('Training HIPSS Model',
                                     add_help=False)
    parser.add_argument('--opt', default='adam', type=str, help='Optimizer for training [Adam, SGD]')
    parser.add_argument('--lr', default=1e-2, type=float, help='Learning Rate')
    parser.add_argument('--reg', type=float, default=1e-5, help='weight decay (default: 1e-5)')
    parser.add_argument('--results_dir', default='', type=str, help='Results Directory')
    parser.add_argument('--n_classes', default=2, type=int, help='Number of classes')
    parser.add_argument('--epochs', default=100, type=int, help='Number of epochs')
    parser.add_argument('--folds', default=3, type=int, help='Number of folds')
    return parser


if __name__ == '__main__':
    print("HIPSS Model Training Started...")
    args = get_args_parser()
    args = args.parse_args()

    seed_list = []
    for seed in seed_list:
        all_test_auc = []
        all_val_auc = []
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        for i in range(args.folds):
            print("Fold {}".format(i))
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            # Replace with correct directory paths
            tensor_dir_train = f'/train_fold_{i}'
            tensor_dir_val = 'val'
            tensor_dir_test = 'test'

            wsi_loader_train = wsi_tensor_loader(tensor_dir_train)
            wsi_loader_val = wsi_tensor_loader(tensor_dir_val)
            wsi_loader_test = wsi_tensor_loader(tensor_dir_test)

            num_classes = args.n_classes
            hipss_model = HIPSS(num_classes=num_classes).to(device)
            for name, param in hipss_model.text_encoder.named_parameters():
                if "ssf_scale" not in name and "ssf_shift_" not in name:
                    param.requires_grad = False
            trainable_params = sum(p.numel() for p in hipss_model.parameters() if p.requires_grad)
            print('Number of Trainable Parameters: {}'.format(trainable_params))
            test_auc, val_auc = train(fold=i, args=args, train_loader=wsi_loader_train,
                                      val_loader=wsi_loader_val,
                                      test_loader=wsi_loader_test,
                                      model=hipss_model, device=device)

            all_test_auc.append(test_auc)
            all_val_auc.append(val_auc)
        print(f"{seed} Seed completed")
        print(np.mean(all_test_auc), np.std(all_test_auc))
