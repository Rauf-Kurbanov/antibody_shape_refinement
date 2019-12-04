import numpy as np
import torch


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


def distance_diff(a1, a2, b1, b2):
    return abs(torch.dist(a1, a2) - torch.dist(b1, b2))


def distance_diff_percent(a1, a2, b1, b2):
    return distance_diff(a1, a2, b1, b2) / torch.dist(b1, b2)


def coordinate_metrics(pred, test, lengths):
    metrics = {}
    ae = []
    dist_diff = []
    dist_diff_p = []
    neighbour_dist_diff = []
    neighbour_dist_diff_p = []
    for i in range(len(lengths)):
        for j in range(lengths[i]):
            ae.append(torch.dist(pred[i][j][:3], test[i][j][:3]))
            ae.append(torch.dist(pred[i][j][3:6], test[i][j][3:6]))
            ae.append(torch.dist(pred[i][j][6:], test[i][j][6:]))

            neighbour_dist_diff.append(distance_diff(pred[i][j][:3], pred[i][j][3:6],
                                                     test[i][j][:3], test[i][j][3:6]))
            neighbour_dist_diff.append(distance_diff(pred[i][j][3:6], pred[i][j][6:],
                                                     test[i][j][3:6], test[i][j][6:]))

            neighbour_dist_diff_p.append(distance_diff_percent(pred[i][j][:3], pred[i][j][3:6],
                                                               test[i][j][:3], test[i][j][3:6]))
            neighbour_dist_diff_p.append(distance_diff_percent(pred[i][j][3:6], pred[i][j][6:],
                                                               test[i][j][3:6], test[i][j][6:]))

            if j != lengths[i] - 1:
                neighbour_dist_diff.append(distance_diff(pred[i][j][6:], pred[i][j + 1][:3],
                                                         test[i][j][6:], test[i][j + 1][:3]))
                neighbour_dist_diff_p.append(distance_diff_percent(pred[i][j][6:], pred[i][j + 1][:3],
                                                                   test[i][j][6:], test[i][j + 1][:3]))

        dist_diff.append(distance_diff(pred[i][0][:3], pred[i][lengths[i] - 1][6:],
                                       test[i][0][:3], test[i][lengths[i] - 1][6:]))
        dist_diff_p.append(distance_diff_percent(pred[i][0][:3], pred[i][lengths[i] - 1][6:],
                                                 test[i][0][:3], test[i][lengths[i] - 1][6:]))

    metrics['mae'] = torch.mean(torch.stack(ae))
    metrics['diff_ends_dist'] = torch.mean(torch.stack(dist_diff))
    metrics['diff_neighbours_dist'] = torch.mean(torch.stack(neighbour_dist_diff))
    metrics['diff_ends_dist_p'] = torch.mean(torch.stack(dist_diff_p))
    metrics['diff_neighbours_dist_p'] = torch.mean(torch.stack(neighbour_dist_diff_p))

    # TODO angles metric

    return metrics
