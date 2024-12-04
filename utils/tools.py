import torch
import os
import logging
import sys
import numpy as np
from sklearn.metrics import roc_curve
from sklearn.metrics import auc as cal_auc
import random

def _seed_everything(random_seed=41):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

def evaluate(model, data_loader):
    correct = 0
    total = 0
    with torch.no_grad():
        y_true, y_pred = [], []
        for i, data_dict in enumerate(data_loader):
            # get data
            img, label= data_dict['image'], data_dict['label']
            label = torch.where(data_dict['label'] != 0, 1, 0)
            # move data to GPU
            img = img.detach().cuda()
            label = label.detach().cuda()
            # model forward without considering gradient computation
            output = model.forward(img)
            probabilit_p = torch.softmax(output, dim=1)[:, 1]
            y_pred.append(probabilit_p.detach().cpu().numpy())
            y_true.append(label.cpu().numpy())

            # deal with acc
            _, prediction_class = torch.max(output, 1)
            correct += (prediction_class == label).sum().item()
            total += label.size(0)
    
    y_true, y_pred = np.concatenate(y_true), np.concatenate(y_pred)
    fpr, tpr, thresholds = roc_curve(y_true,y_pred,pos_label=1)
    AUC = cal_auc(fpr, tpr)
    acc = correct / total

    return AUC, acc

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.value = 0
        self.ave = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.value = val
        self.sum += val * n
        self.count += n
        self.ave = self.sum / self.count

def save_checkpoint(state, filepath='checkpoint.pth.tar'):
    torch.save(state, filepath)

# python 3.7
"""Utility functions for logging."""

__all__ = ['setup_logger']

DEFAULT_WORK_DIR = 'results'

def setup_logger(work_dir=None, logfile_name='log.txt', logger_name='logger'):
    """Sets up logger from target work directory.

    The function will sets up a logger with `DEBUG` log level. Two handlers will
    be added to the logger automatically. One is the `sys.stdout` stream, with
    `INFO` log level, which will print improtant messages on the screen. The other
    is used to save all messages to file `$WORK_DIR/$LOGFILE_NAME`. Messages will
    be added time stamp and log level before logged.

    NOTE: If `logfile_name` is empty, the file stream will be skipped. Also,
    `DEFAULT_WORK_DIR` will be used as default work directory.

    Args:
    work_dir: The work directory. All intermediate files will be saved here.
        (default: None)
    logfile_name: Name of the file to save log message. (default: `log.txt`)
    logger_name: Unique name for the logger. (default: `logger`)

    Returns:
    A `logging.Logger` object.

    Raises:
    SystemExit: If the work directory has already existed, of the logger with
        specified name `logger_name` has already existed.
    """

    logger = logging.getLogger(logger_name)
    if logger.hasHandlers():  # Already existed
        raise SystemExit(f'Logger name `{logger_name}` has already been set up!\n'
                            f'Please use another name, or otherwise the messages '
                            f'may be mixed between these two loggers.')

    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter("[%(asctime)s][%(levelname)s] %(message)s")
    # Print log message with `INFO` level or above onto the screen.
    sh = logging.StreamHandler(stream=sys.stdout)
    sh.setLevel(logging.DEBUG)
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    if not logfile_name:
        return logger

    work_dir = work_dir or DEFAULT_WORK_DIR
    logfile_name = os.path.join(work_dir, logfile_name)

    os.makedirs(work_dir, exist_ok=True)

    # Save log message with all levels in log file.
    fh = logging.FileHandler(logfile_name)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger