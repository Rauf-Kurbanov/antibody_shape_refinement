import numpy as np
import torch


def sin_cos_to_angle(x):
    sin_phi, cos_phi, sin_psi, cos_psi = x
    return np.math.atan2(sin_phi, cos_phi), np.math.atan2(sin_psi, cos_psi)


def angle_metrics(pred, test, lengths_test):
    mean_var_phi = 0
    mean_var_psi = 0
    cnt = 0

    for i in range(len(lengths_test)):
        for j in range(lengths_test[i]):
            cnt += 1
            pred_phi, pred_psi = sin_cos_to_angle(pred[i][j])
            test_phi, test_psi = sin_cos_to_angle(test[i][j])
            mean_var_phi += np.abs(pred_phi - test_phi)
            mean_var_psi += np.abs(pred_psi - test_psi)

    mean_var_phi = mean_var_phi / cnt
    mean_var_psi = mean_var_psi / cnt

    return mean_var_phi, mean_var_psi


def scalar_prod(v1, v2):
    return torch.sum(v1 * v2, dim=-1)


def distance_between_atoms(loop):
    v1 = loop[:, :-1]
    v2 = loop[:, 1:]
    return (v1 - v2).norm(dim=-1)


def angles_between_atoms(loop, lengths, on_cpu=False):
    loop = loop.reshape(loop.shape[0], -1, 3)
    a = loop[:, :-2]
    b = loop[:, 1:-1]
    c = loop[:, 2:]
    ba = a - b
    bc = c - b
    norm_ba = ba.norm(dim=-1)
    norm_bc = bc.norm(dim=-1)
    # remove possible nans
    mask = torch.zeros_like(norm_ba)
    for i, l in enumerate(lengths):
        l = l * 3 - 2
        mask[i, l + 1:] = 1
    norm_ba = norm_ba + mask
    norm_bc = norm_bc + mask
    res = scalar_prod(ba, bc) / norm_ba / norm_bc
    return res


def rmsd(pred, test, lengths):
    msd = torch.sum((pred - test) ** 2, dim=-1).sum(dim=-1) / lengths.float()
    return torch.sqrt(msd)


def coordinate_metrics(pred, test, lengths, on_cpu=False):
    metrics = {}
    pred = pred.reshape(pred.shape[0], -1, 3)
    test = test.reshape(pred.shape[0], -1, 3)

    mae_batch = (pred - test).norm(dim=-1).mean(-1)
    rmsd_batch = rmsd(pred, test, lengths)
    metrics['mae'] = mae_batch.mean()
    metrics['mae_min'] = mae_batch.min()
    metrics['mae_max'] = mae_batch.max()
    metrics['mae_median'] = mae_batch.median()
    metrics['rmsd'] = rmsd_batch.mean()
    metrics['rmsd_max'] = rmsd_batch.max()
    metrics['rmsd_min'] = rmsd_batch.min()
    metrics['rmsd_median'] = rmsd_batch.median()
    metrics['diff_neighbours_dist'] = torch.mean(abs(distance_between_atoms(pred) -
                                                     distance_between_atoms(test)))
    metrics['diff_angles'] = torch.mean(abs(angles_between_atoms(pred, lengths, on_cpu) -
                                            angles_between_atoms(test, lengths, on_cpu)))
    # todo distance between ends metric

    return metrics
