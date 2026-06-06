
import os
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score, recall_score, f1_score


def compute_metrics(y_true, y_pred, y_prob=None):
    metrics = {}

    # Accuracy
    metrics['accuracy'] = accuracy_score(y_true, y_pred)

    # AUROC
    if y_prob is not None and len(np.unique(y_true)) > 1:
        try:
            metrics['auroc'] = roc_auc_score(y_true, y_prob)
        except ValueError:
            metrics['auroc'] = 0.0
    else:
        metrics['auroc'] = 0.0

    # AUPRC
    if y_prob is not None and len(np.unique(y_true)) > 1:
        try:
            metrics['auprc'] = average_precision_score(y_true, y_prob)
        except ValueError:
            metrics['auprc'] = 0.0
    else:
        metrics['auprc'] = 0.0

    # Recall / Sensitivity
    metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)

    if len(np.unique(y_true)) == 2:
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    else:
        metrics['specificity'] = 0.0

    # F1
    metrics['f1'] = f1_score(y_true, y_pred, zero_division=0)

    return metrics


def save_results_fold(save_path, slide_ids, y_true, y_pred):
    results = np.array([slide_ids, y_true, y_pred]).T
    np.save(save_path, results, allow_pickle=True)


def save_metrics_csv(save_path, all_metrics):

    ordered_metrics = []
    for m in all_metrics:
        ordered = {'fold': m['fold']}
        for k, v in m.items():
            if k != 'fold':
                ordered[k] = v
        ordered_metrics.append(ordered)

    df = pd.DataFrame(ordered_metrics)

    metric_cols = [col for col in df.columns if col != 'fold']
    mean_row = {'fold': 'mean'}
    std_row = {'fold': 'std'}

    for col in metric_cols:
        mean_row[col] = df[col].mean()
        std_row[col] = df[col].std()

    df = pd.concat([df, pd.DataFrame([mean_row, std_row])], ignore_index=True)

    df.to_csv(save_path, index=False)


def load_split_csv(split_path):
    df = pd.read_csv(split_path)
    train_slides = df['train'].dropna().tolist()
    test_slides = df['test'].dropna().tolist()
    return train_slides, test_slides


def create_exp_name(args, model_name):

    hyperparams = []

    if hasattr(args, 'lr'):
        hyperparams.append('lr{}'.format(args.lr))
    if hasattr(args, 'epochs'):
        hyperparams.append('ep{}'.format(args.epochs))
    if hasattr(args, 'projection_dim'):
        hyperparams.append('pd{}'.format(args.projection_dim))
    if hasattr(args, 'num_heads'):
        hyperparams.append('nh{}'.format(args.num_heads))

    if hyperparams:
        hyperparam_str = '_'.join(hyperparams)
    else:
        hyperparam_str = 'default'

    exp_name = '{}_{}_{}'.format(model_name, hyperparam_str, args.seed)

    return exp_name


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)