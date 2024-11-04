import numpy as np
import logging
import os
from sklearn.metrics import accuracy_score, confusion_matrix, cohen_kappa_score, classification_report, f1_score
import json
import random

def count_params(model):
    param_num = sum(p.numel() for p in model.parameters())
    return param_num / 1e6


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, length=0):
        self.length = length
        self.reset()

    def reset(self):
        if self.length > 0:
            self.history = []
        else:
            self.count = 0
            self.sum = 0.0
        self.val = 0.0
        self.avg = 0.0

    def update(self, val, num=1):
        if self.length > 0:
            # currently assert num==1 to avoid bad usage, refine when there are some explict requirements
            assert num == 1
            self.history.append(val)
            if len(self.history) > self.length:
                del self.history[0]

            self.val = self.history[-1]
            self.avg = np.mean(self.history)
        else:
            self.val = val
            self.sum += val * num
            self.count += num
            self.avg = self.sum / self.count


logs = set()


def init_log(name, level=logging.INFO):
    if (name, level) in logs:
        return
    logs.add((name, level))
    logger = logging.getLogger(name)
    logger.setLevel(level)
    ch = logging.StreamHandler()
    ch.setLevel(level)
    if "SLURM_PROCID" in os.environ:
        rank = int(os.environ["SLURM_PROCID"])
        logger.addFilter(lambda record: rank == 0)
    else:
        rank = 0
    format_str = "[%(asctime)s %(filename)s %(funcName)s][line:%(lineno)d] %(levelname)s %(message)s"
    formatter = logging.Formatter(format_str)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger


def acc(true, pred):
    kappa = cohen_kappa_score(true, pred)
    OA = accuracy_score(true, pred)
    #f1 = f1_score(true, pred)
    cm = confusion_matrix(true, pred)
    producer_accuracy = []
    for i in range(cm.shape[1]):
        tp = cm[i][i]
        fp = np.sum(cm[:, i]) - tp
        producer_acc_i = tp / (tp + fp)
        producer_accuracy.append(producer_acc_i)

    return kappa, OA, producer_accuracy

def read_json_file(json_file_path):
     with open(json_file_path, 'r') as file:
         data = json.load(file)
     return data

def data_split(root, num_class, train_sample_number, val_sample_number, seed):
        random.seed(seed)

        data = read_json_file(os.path.join(root, "data.json"))

        train_lis = []
        val_lis = []
        test_lis = []

        for i in range(num_class):
            sample_data = list(filter(lambda x: x['label'] == str(i), data))
            sample_i_lis_len  = len(sample_data)
            sample_i_train = random.sample(sample_data, int(sample_i_lis_len * train_sample_number))
            sample_i_res = list(filter(lambda x: x not in sample_i_train, sample_data))
            sample_i_val = random.sample(sample_i_res, int(sample_i_lis_len * val_sample_number))
            sample_i_test = list(filter(lambda x: x not in sample_i_val, sample_i_res))
            
            train_lis += sample_i_train
            val_lis += sample_i_val
            test_lis += sample_i_test

        return train_lis, val_lis, test_lis