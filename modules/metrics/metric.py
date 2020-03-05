import numpy as np
import torch
import itertools


def sin_cos_to_angle(x):
    return np.math.atan2(x[0], x[1]), np.math.atan2(x[2], x[3])


def angle_metrics(pred, test, lengths_test, threshold=0.01):
    angle_var_phi = []
    angle_var_psi = []

    for i in range(len(lengths_test)):
        for j in range(lengths_test[i]):
            pred_phi, pred_psi = sin_cos_to_angle(pred[i][j])
            test_phi, test_psi = sin_cos_to_angle(test[i][j])
            angle_var_phi.append(np.abs(pred_phi - test_phi))
            angle_var_psi.append(np.abs(pred_psi - test_psi))

    mean_var_phi = np.mean(angle_var_phi)
    mean_var_psi = np.mean(angle_var_psi)
    accuracy_phi = len(list(filter(lambda x: x < threshold, angle_var_phi))) / len(angle_var_phi)
    accuracy_psi = len(list(filter(lambda x: x < threshold, angle_var_psi))) / len(angle_var_psi)

    return mean_var_phi, accuracy_phi, mean_var_psi, accuracy_psi


def scalar_prod(v1, v2):
    return torch.sum(v1 * v2, dim=-1)


def norm(v1):
    return torch.sqrt(scalar_prod(v1, v1))


def distance_between_atoms(loop, on_cpu=False):
    loop = loop.reshape(loop.shape[0], -1, 3) if on_cpu else loop.view(loop.shape[0], -1, 3)
    v1 = loop[:, :-1]
    v2 = loop[:, 1:]
    return norm(v1 - v2)


def angles_between_atoms(loop, lengths, on_cpu=False):
    loop = loop.reshape(loop.shape[0], -1, 3) if on_cpu else loop.view(loop.shape[0], -1, 3)
    a = loop[:, :-2]
    b = loop[:, 1:-1]
    c = loop[:, 2:]
    ba = a - b
    bc = c - b
    norm_ba = norm(ba)
    norm_bc = norm(bc)
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
    return torch.sqrt(msd).mean()


def coordinate_metrics(pred, test, lengths, on_cpu=False):
    metrics = {}
    pred = pred.reshape(pred.shape[0], -1, 3) if on_cpu else pred.view(pred.shape[0], -1, 3)
    test = test.reshape(pred.shape[0], -1, 3) if on_cpu else test.view(pred.shape[0], -1, 3)

    metrics['mae'] = torch.mean(norm(pred - test))
    metrics['rmsd'] = rmsd(pred, test, lengths)
    metrics['diff_neighbours_dist'] = torch.mean(abs(distance_between_atoms(pred, on_cpu) -
                                                     distance_between_atoms(test, on_cpu)))
    metrics['diff_angles'] = torch.mean(abs(angles_between_atoms(pred, lengths, on_cpu) -
                                            angles_between_atoms(test, lengths, on_cpu)))
    # todo distance between ends metric

    return metrics
