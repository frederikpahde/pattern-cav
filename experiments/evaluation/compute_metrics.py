import torch
import tqdm
import numpy as np
from utils.metrics import get_accuracy, get_f1, get_auc_label


def compute_model_scores(
        model: torch.nn.Module,
        dl: torch.utils.data.DataLoader,
        device: str,
        limit_batches: int=None):
    model.to(device).eval()
    model_outs = []
    ys = []
    for i, (x_batch, y_batch) in enumerate(tqdm.tqdm(dl)):
        if limit_batches and limit_batches == i:
            break
        model_out = model(x_batch.to(device)).detach().cpu()
        model_outs.append(model_out)
        ys.append(y_batch)

    model_outs = torch.cat(model_outs)
    y_true = torch.cat(ys)

    return model_outs, y_true


def compute_metrics(model_outs, y_true, class_names=None, prefix="", suffix=""):
    accuracy, standard_err = get_accuracy(model_outs, y_true, se=True)
    results = {
        f"{prefix}accuracy{suffix}": accuracy,
        f"{prefix}accuracy_standard_err{suffix}": standard_err,
        f"{prefix}f1{suffix}": get_f1(model_outs, y_true)
    }

    if class_names:
        results_auc = {f"AUC_{class_names[class_id]}": get_auc_label(y_true, model_outs, class_id)
                       for class_id in range(model_outs.shape[1])}
        results = {**results, **results_auc}

    return results

def compute_tcav_metrics_batch(grad, cav):
    grad = grad if grad.dim() > 2 else grad[..., None, None]
    metrics = {
        'TCAV_pos': ((grad * cav[..., None, None]).sum(1).flatten() > 0).sum().item(),
        'TCAV_neg': ((grad * cav[..., None, None]).sum(1).flatten() < 0).sum().item(),
        'TCAV_pos_mean': ((grad * cav[..., None, None]).sum(1).mean((1, 2)).flatten() > 0).sum().item(),
        'TCAV_neg_mean': ((grad * cav[..., None, None]).sum(1).mean((1, 2)).flatten() < 0).sum().item(),
        'TCAV_sensitivity': (grad * cav[..., None, None]).sum(1).abs().flatten().numpy()

    }
    return metrics

def aggregate_tcav_metrics(TCAV_pos, TCAV_neg, TCAV_pos_mean, TCAV_neg_mean, TCAV_sens_list):

    eps = 1e-8
    TCAV_sens_list = np.concatenate(TCAV_sens_list)
    tcav_quotient = TCAV_pos / (TCAV_neg + TCAV_pos + eps)
    mean_tcav_quotient = TCAV_pos_mean / (TCAV_neg_mean + TCAV_pos_mean + eps)
    mean_tcav_sensitivity = TCAV_sens_list.mean()
    mean_tcav_sensitivity_sem = np.std(TCAV_sens_list) / np.sqrt(len(TCAV_sens_list))
    quotient_sderr = np.sqrt(tcav_quotient * (1 - tcav_quotient) / (TCAV_neg + TCAV_pos + eps))
    mean_quotient_sderr = np.sqrt(mean_tcav_quotient * (1 - mean_tcav_quotient) / (TCAV_neg_mean + TCAV_pos_mean + eps))

    metrics = {
        "tcav_quotient": tcav_quotient,
        "quotient_sderr": quotient_sderr,
        "mean_tcav_quotient": mean_tcav_quotient,
        "mean_quotient_sderr": mean_quotient_sderr,
        "mean_tcav_sensitivity": mean_tcav_sensitivity,
        "mean_tcav_sensitivity_sem": mean_tcav_sensitivity_sem
    }

    return metrics