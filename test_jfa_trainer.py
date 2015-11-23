import numpy
import numpy.linalg
import scipy.io

from train import GMM_Machine as GMMMachine
from train import GMM_Stats as GMMStats
from train import JFA_Base as JFABase
from train import JFA_Machine as JFAMachine
from train import JFA_Trainer as JFATrainer

F1 = numpy.array(
    [0.3833, 0.4516, 0.6173, 0.2277, 0.5755, 0.8044, 0.5301, 0.9861, 0.2751, 0.0300, 0.2486, 0.5357]).reshape((6, 2))
F2 = numpy.array(
    [0.0871, 0.6838, 0.8021, 0.7837, 0.9891, 0.5341, 0.0669, 0.8854, 0.9394, 0.8990, 0.0182, 0.6259]).reshape((6, 2))
F = [F1, F2]

N1 = numpy.array([0.1379, 0.1821, 0.2178, 0.0418]).reshape((2, 2))
N2 = numpy.array([0.1069, 0.9397, 0.6164, 0.3545]).reshape((2, 2))
N = [N1, N2]

gs11 = GMMStats(2, 3)
gs11.n = N1[:, 0]
gs11.sum_Px = F1[:, 0].reshape(2, 3)
gs12 = GMMStats(2, 3)
gs12.n = N1[:, 1]
gs12.sum_Px = F1[:, 1].reshape(2, 3)

gs21 = GMMStats(2, 3)
gs21.n = N2[:, 0]
gs21.sum_Px = F2[:, 0].reshape(2, 3)
gs22 = GMMStats(2, 3)
gs22.n = N2[:, 1]
gs22.sum_Px = F2[:, 1].reshape(2, 3)

TRAINING_STATS = [
    [gs11, gs12],  # person 1
    [gs21, gs22]  # person 2
]
# m
UBM_MEAN = numpy.array([0.1806, 0.0451, 0.7232, 0.3474, 0.6606, 0.3839])
# UBM_MEAN = scipy.io.loadmat("models/ubm_means")
# E
UBM_VAR = numpy.array([0.6273, 0.0216, 0.9106, 0.8006, 0.7458, 0.8131])
# UBM_VAR = scipy.io.loadmat("models/ubm_variances")

# d
M_d = numpy.array([0.4106, 0.9843, 0.9456, 0.6766, 0.9883, 0.7668])
# v
M_v = numpy.array([0.3367, 0.4116, 0.6624, 0.6026, 0.2442, 0.7505, 0.2955,
                   0.5835, 0.6802, 0.5518, 0.5278, 0.5836]).reshape((6, 2))
# u
M_u = numpy.array([0.5118, 0.3464, 0.0826, 0.8865, 0.7196, 0.4547, 0.9962,
                   0.4134, 0.3545, 0.2177, 0.9713, 0.1257]).reshape((6, 2))

_z1 = numpy.array([0.3089, 0.7261, 0.7829, 0.6938, 0.0098, 0.8432])
_z2 = numpy.array([0.9223, 0.7710, 0.0427, 0.3782, 0.7043, 0.7295])
_y1 = numpy.array([0.2243, 0.2691])
_y2 = numpy.array([0.6730, 0.4775])
_x1 = numpy.array([0.9976, 0.8116, 0.1375, 0.3900]).reshape((2, 2))
_x2 = numpy.array([0.4857, 0.8944, 0.9274, 0.9175]).reshape((2, 2))

M_z = [_z1, _z2]
M_y = [_y1, _y2]
M_x = [_x1, _x2]


def test_JFATrainInitialize():
    # Check that the initialization is consistent and using the rng (cf. issue #118)

    eps = 1e-10

    # UBM GMM
    ubm = GMMMachine(2, 3)
    ubm.mean_supervector = UBM_MEAN
    ubm.variance_supervector = UBM_VAR

    ## JFA
    jfa_base = JFABase(ubm, 512, 2)
    # first round
    jfa_machine = JFAMachine(jfa_base)
    jfa_trainer = JFATrainer(jfa_machine)
    training_data = scipy.io.loadmat("./data/stats/fa_train_eigenvoices_stats.mat")
    jfa_trainer.train(training_data)


def check_dimensions(t1, t2):
    assert len(t1) == len(t2)
    for i in range(len(t1)):
        assert t1.shape[i] == t2.shape[i]


def checkBlitzEqual(t1, t2):
    check_dimensions(t1, t2)
    for i in t1.shape[0]:
        for j in t1.shape[1]:
            if len(t1.shape) > 2:
                for k in t1.shape[2]:
                    assert t1[i, j, k] == t2[i, j, k]
            else:
                assert t1[i, j] == t2[i, j]


def checkBlitzClose(t1, t2, eps):
    numpy.allclose(t1, t2, eps)


test_JFATrainInitialize()
