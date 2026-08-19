import os

import numpy as np
import torch
import torch.optim as optim
from tensorboardX import SummaryWriter
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import label_binarize
from sklearn.metrics import auc as calc_auc


def get_optim(model, args):
    if args.opt == "adam":
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.reg)
    elif args.opt == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, momentum=0.9,
                              weight_decay=args.reg)
    else:
        raise NotImplementedError
    return optimizer


def calculate_error(y_hat, y, device):
    y = y.to(device)
    error = 1. - y_hat.float().eq(y.float()).float().mean().item()
    return error


class Accuracy_Logger(object):
    """Accuracy logger"""

    def __init__(self, n_classes):
        super(Accuracy_Logger, self).__init__()
        self.n_classes = n_classes
        self.initialize()

    def initialize(self):
        self.data = [{"count": 0, "correct": 0} for i in range(self.n_classes)]

    def log(self, Y_hat, Y):
        Y_hat = int(Y_hat)
        Y = int(Y)
        self.data[Y]["count"] += 1
        self.data[Y]["correct"] += (Y_hat == Y)

    def log_batch(self, Y_hat, Y):
        Y_hat = np.array(Y_hat).astype(int)
        Y = np.array(Y).astype(int)
        for label_class in np.unique(Y):
            cls_mask = Y == label_class
            self.data[label_class]["count"] += cls_mask.sum()
            self.data[label_class]["correct"] += (Y_hat[cls_mask] == Y[cls_mask]).sum()

    def get_summary(self, c):
        count = self.data[c]["count"]
        correct = self.data[c]["correct"]

        if count == 0:
            acc = None
        else:
            acc = float(correct) / count

        return acc, correct, count


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=20, stop_epoch=60, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 20
            stop_epoch (int): Earliest epoch possible for stopping
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
        """
        self.patience = patience
        self.stop_epoch = stop_epoch
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf

    def __call__(self, epoch, val_loss, model, ckpt_name='checkpoint.pt'):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
        elif score <= self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience and epoch > self.stop_epoch:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, ckpt_name):
        """Saves model when validation loss decrease."""
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), ckpt_name)
        self.val_loss_min = val_loss


def train(fold, args, train_loader, val_loader, test_loader, model, device):
    print("Training Fold: {}".format(fold))
    n_classes = args.n_classes
    writer_dir = os.path.join(args.results_dir, str(fold))
    if not os.path.exists(writer_dir):
        os.makedirs(writer_dir)
    optimizer = get_optim(model, args)
    writer = SummaryWriter(log_dir=writer_dir, flush_secs=15)
    early_stopping = EarlyStopping(stop_epoch=40, verbose=True, patience=20)

    for epoch in range(args.epochs):
        train_loop(epoch, model, train_loader, optimizer, n_classes, writer, device)
        stop = validate(fold, epoch, model, val_loader, n_classes, early_stopping, writer, args.results_dir, device)
        if stop:
            break

    model.load_state_dict(torch.load(os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(fold))))

    val_auc, _ = summary(model, val_loader, n_classes, device)
    print('ROC AUC: {:.4f}'.format(val_auc))

    test_auc, acc_logger = summary(model, test_loader, n_classes, device)
    print('ROC AUC: {:.4f}'.format(test_auc))

    each_class_acc = []
    for i in range(args.n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        each_class_acc.append(acc)
        print('class {}: acc {:.4f}, correct {}/{}'.format(i, acc, correct, count))

        if writer:
            writer.add_scalar('final/test_class_{}_acc'.format(i), acc, 0)

    if writer:
        writer.add_scalar('final/test_auc', test_auc, 0)
        writer.close()

    return test_auc, val_auc


def train_loop(epoch, model, train_loader, optimizer, n_classes, writer, device):
    model.train()
    acc_logger = Accuracy_Logger(n_classes)
    train_loss = 0.
    train_error = 0.

    for batch_idx, (regions, label) in enumerate(train_loader):
        Y_prob, Y_hat, loss = model(regions, label)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        acc_logger.log(Y_hat, label)
        loss_value = loss.item()
        train_loss += loss_value
        error = calculate_error(Y_hat, label, device)
        train_error += error

    train_loss /= len(train_loader)
    train_error /= len(train_loader)

    print('Epoch: {}, train_loss: {:.4f}, train_error: {:.4f}'.format(epoch, train_loss, train_error))
    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {:.4f}, correct {}/{}'.format(i, acc, correct, count))
        if writer:
            writer.add_scalar('train/class_{}_acc'.format(i), acc, epoch)

    if writer:
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/error', train_error, epoch)


def validate(fold, epoch, model, val_loader, n_classes, early_stopping=None, writer=None, results_dir=None, device=None):
    print("Validation Fold: {}".format(fold))
    model.eval()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    val_error = 0.

    prob = np.zeros((len(val_loader), n_classes))
    labels = np.zeros(len(val_loader))

    all_pred = []
    all_label = []

    with torch.no_grad():
        for batch_idx, (regions, label) in enumerate(val_loader):
            Y_prob, Y_hat, loss = model(regions, label)
            acc_logger.log(Y_hat, label)
            prob[batch_idx] = Y_prob.cpu().numpy()
            labels[batch_idx] = label.item()
            error = calculate_error(Y_hat, label, device)
            val_error += error
            all_pred.append(Y_hat.item())
            all_label.append(label.item())

    val_error /= len(val_loader)

    if n_classes == 2:
        auc = roc_auc_score(labels, prob[:, 1])

    else:
        auc = roc_auc_score(labels, prob, multi_class='ovr')

    if writer:
        writer.add_scalar('val/auc', auc, epoch)

    print('\nVal Set, auc: {:.4f}'.format(auc))
    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {:.4f}, correct {}/{}'.format(i, acc, correct, count))

    if early_stopping:
        assert results_dir
        early_stopping(epoch, val_error, model, ckpt_name=os.path.join(results_dir, "s_{}_checkpoint.pt".format(fold)))

        if early_stopping.early_stop:
            print("Early stopping")
            return True

    return False


def summary(model, data_loader, n_classes, device):
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    model.eval()
    all_pred = []
    all_label = []
    all_probs = np.zeros((len(data_loader), n_classes))
    all_labels = np.zeros(len(data_loader))

    for batch_idx, (regions, label) in enumerate(data_loader):
        with torch.no_grad():
            Y_prob, Y_hat, loss = model(regions, label)

        acc_logger.log(Y_hat, label)
        probs = Y_prob.cpu().numpy()
        all_probs[batch_idx] = probs
        all_labels[batch_idx] = label.item()

        all_pred.append(Y_hat.item())
        all_label.append(label.item())

    if n_classes == 2:
        auc = roc_auc_score(all_labels, all_probs[:, 1])
    else:
        auc_values = []
        binary_labels = label_binarize(all_labels, classes=[i for i in range(n_classes)])
        for class_idx in range(n_classes):
            if class_idx in all_labels:
                fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], all_probs[:, class_idx])
                auc_values.append(calc_auc(fpr, tpr))
            else:
                auc_values.append(float('nan'))

        auc = np.nanmean(np.array(auc_values))

    return auc, acc_logger
