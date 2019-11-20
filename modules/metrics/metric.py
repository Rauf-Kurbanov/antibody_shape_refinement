import numpy as np

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
