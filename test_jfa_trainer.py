import numpy
import numpy.linalg
import numpy as np
# import nose.tools
import random
from train import GMM_Machine as GMMMachine
from train import GMM_Stats as GMMStats
from train import JFA_Machine as JFAMachine
from train import JFA_Base as JFABase
from train import JFA_Trainer as JFATrainer


def equals(x, y, epsilon):
    return (abs(x - y) < epsilon).all()


spk_ids = [0, 0, 1, 1]
# Define Training set and initial values for tests
F1 = numpy.array([0.3833, 0.4516, 0.6173, 0.2277, 0.5755, 0.8044, 0.5301,
                  0.9861, 0.2751, 0.0300, 0.2486, 0.5357]).reshape((6, 2))
F2 = numpy.array([0.0871, 0.6838, 0.8021, 0.7837, 0.9891, 0.5341, 0.0669,
                  0.8854, 0.9394, 0.8990, 0.0182, 0.6259]).reshape((6, 2))
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

TRAINING_STATS = [[gs11, gs12], [gs21, gs22]]
# m
UBM_MEAN = numpy.array([0.1806, 0.0451, 0.7232, 0.3474, 0.6606, 0.3839])
# E
UBM_VAR = numpy.array([0.6273, 0.0216, 0.9106, 0.8006, 0.7458, 0.8131])
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


def test_JFATrainer_updateYandV():
    # test the JFATrainer for updating Y and V

    v_ref = numpy.array([0.7228, 0.7892, 0.6475, 0.6080, 0.8631, 0.8416,
                         1.6512, 1.6068, 0.0500, 0.0101, 0.4325, 0.6719]).reshape((6, 2))

    y1 = numpy.array([0., 0.])
    y2 = numpy.array([0., 0.])
    y3 = numpy.array([0.9630, 1.3868])
    y4 = numpy.array([0.0426, -0.3721])
    y = [y1, y2]

    # call the updateY function
    ubm = GMMMachine(2, 3)
    ubm.mean_supervector = UBM_MEAN
    ubm.variance_supervector = UBM_VAR
    m = JFABase(ubm, 2, 2)
    t = JFATrainer(m, TRAINING_STATS)
    m.u = M_u
    m.v = M_v
    m.d = M_d
    t.__X__ = M_x
    t.__Y__ = y
    t.__Z__ = M_z
    t.e_step_v(m, TRAINING_STATS)
    t.m_step_v(m, TRAINING_STATS)

    # Expected results(JFA cookbook, matlab)
    assert equals(t.__Y__[0], y3, 2e-4)
    assert equals(t.__Y__[1], y4, 2e-4)
    assert equals(m.v, v_ref, 2e-4)


def test_JFATrainer_updateXandU():
    # test the JFATrainer for updating X and U

    u_ref = numpy.array([0.6729, 0.3408, 0.0544, 1.0653, 0.5399, 1.3035,
                         2.4995, 0.4385, 0.1292, -0.0576, 1.1962, 0.0117]).reshape((6, 2))

    x1 = numpy.array([0., 0., 0., 0.]).reshape((2, 2))
    x2 = numpy.array([0., 0., 0., 0.]).reshape((2, 2))
    x3 = numpy.array([0.2143, 1.8275, 3.1979, 0.1227]).reshape((2, 2))
    x4 = numpy.array([-1.3861, 0.2359, 5.3326, -0.7914]).reshape((2, 2))
    x = [x1, x2]

    # call the updateX function
    ubm = GMMMachine(2, 3)
    ubm.mean_supervector = UBM_MEAN
    ubm.variance_supervector = UBM_VAR
    m = JFABase(ubm, 2, 2)
    t = JFATrainer()
    t.initialize(m, TRAINING_STATS)
    m.u = M_u
    m.v = M_v
    m.d = M_d
    t.__X__ = x
    t.__Y__ = M_y
    t.__Z__ = M_z
    t.e_step_u(m, TRAINING_STATS)
    t.m_step_u(m, TRAINING_STATS)

    # Expected results(JFA cookbook, matlab)
    assert equals(t.__X__[0], x3, 2e-4)
    assert equals(t.__X__[1], x4, 2e-4)
    assert equals(m.u, u_ref, 2e-4)


def test_JFATrainer_updateZandD():
    # test the JFATrainer for updating Z and D

    d_ref = numpy.array([0.3110, 1.0138, 0.8297, 1.0382, 0.0095, 0.6320])

    z1 = numpy.array([0., 0., 0., 0., 0., 0.])
    z2 = numpy.array([0., 0., 0., 0., 0., 0.])
    z3_ref = numpy.array([0.3256, 1.8633, 0.6480, 0.8085, -0.0432, 0.2885])
    z4_ref = numpy.array([-0.3324, -0.1474, -0.4404, -0.4529, 0.0484, -0.5848])
    z = [z1, z2]

    # call the updateZ function
    ubm = GMMMachine(2, 3)
    ubm.mean_supervector = UBM_MEAN
    ubm.variance_supervector = UBM_VAR
    m = JFABase(ubm, 2, 2)
    t = JFATrainer()
    t.initialize(m, TRAINING_STATS)
    m.u = M_u
    m.v = M_v
    m.d = M_d
    t.__X__ = M_x
    t.__Y__ = M_y
    t.__Z__ = z
    t.e_step_d(m, TRAINING_STATS)
    t.m_step_d(m, TRAINING_STATS)

    # Expected results(JFA cookbook, matlab)
    assert equals(t.__Z__[0], z3_ref, 2e-4)
    assert equals(t.__Z__[1], z4_ref, 2e-4)
    assert equals(m.d, d_ref, 2e-4)


def test_JFATrainAndEnrol():
    # Train and enroll a JFAMachine

    # Calls the train function
    ubm = GMMMachine(2, 3)
    ubm.mean_supervector = UBM_MEAN
    ubm.variance_supervector = UBM_VAR
    mb = JFABase(ubm, 2, 2)
    t = JFATrainer()
    t.initialize(mb, TRAINING_STATS)
    mb.u = M_u
    mb.v = M_v
    mb.d = M_d
    bob.learn.em.train_jfa(t, mb, TRAINING_STATS, initialize=False)

    v_ref = numpy.array([[0.245364911936476, 0.978133261775424], [0.769646805052223, 0.940070736856596],
                         [0.310779202800089, 1.456332053893072],
                         [0.184760934399551, 2.265139705602147], [0.701987784039800, 0.081632150899400],
                         [0.074344030229297, 1.090248340917255]], 'float64')
    u_ref = numpy.array([[0.049424652628448, 0.060480486336896], [0.178104127464007, 1.884873813495153],
                         [1.204011484266777, 2.281351307871720],
                         [7.278512126426286, -0.390966087173334], [-0.084424326581145, -0.081725474934414],
                         [4.042143689831097, -0.262576386580701]], 'float64')
    d_ref = numpy.array(
        [9.648467e-18, 2.63720683155e-12, 2.11822157653706e-10, 9.1047243e-17, 1.41163442535567e-10, 3.30581e-19],
        'float64')

    eps = 1e-10
    assert numpy.allclose(mb.v, v_ref, eps)
    assert numpy.allclose(mb.u, u_ref, eps)
    assert numpy.allclose(mb.d, d_ref, eps)

    # Calls the enroll function
    m = JFAMachine(mb)

    Ne = numpy.array([0.1579, 0.9245, 0.1323, 0.2458]).reshape((2, 2))
    Fe = numpy.array(
        [0.1579, 0.1925, 0.3242, 0.1234, 0.2354, 0.2734, 0.2514, 0.5874, 0.3345, 0.2463, 0.4789, 0.5236]).reshape(
        (6, 2))
    gse1 = GMMStats(2, 3)
    gse1.n = Ne[:, 0]
    gse1.sum_Px = Fe[:, 0].reshape(2, 3)
    gse2 = GMMStats(2, 3)
    gse2.n = Ne[:, 1]
    gse2.sum_Px = Fe[:, 1].reshape(2, 3)

    gse = [gse1, gse2]
    t.enroll(m, gse, 5)

    y_ref = numpy.array([0.555991469319657, 0.002773650670010], 'float64')
    z_ref = numpy.array(
        [8.2228e-20, 3.15216909492e-13, -1.48616735364395e-10, 1.0625905e-17, 3.7150503117895e-11, 1.71104e-19],
        'float64')
    assert numpy.allclose(m.y, y_ref, eps)
    assert numpy.allclose(m.z, z_ref, eps)

    # Testing exceptions
    nose.tools.assert_raises(RuntimeError, t.initialize, mb, [1, 2, 2])
    nose.tools.assert_raises(RuntimeError, t.initialize, mb, [[1, 2, 2]])
    nose.tools.assert_raises(RuntimeError, t.e_step_u, mb, [1, 2, 2])
    nose.tools.assert_raises(RuntimeError, t.e_step_u, mb, [[1, 2, 2]])
    nose.tools.assert_raises(RuntimeError, t.m_step_u, mb, [1, 2, 2])
    nose.tools.assert_raises(RuntimeError, t.m_step_u, mb, [[1, 2, 2]])

    nose.tools.assert_raises(RuntimeError, t.e_step_v, mb, [1, 2, 2])
    nose.tools.assert_raises(RuntimeError, t.e_step_v, mb, [[1, 2, 2]])
    nose.tools.assert_raises(RuntimeError, t.m_step_v, mb, [1, 2, 2])
    nose.tools.assert_raises(RuntimeError, t.m_step_v, mb, [[1, 2, 2]])

    nose.tools.assert_raises(RuntimeError, t.e_step_d, mb, [1, 2, 2])
    nose.tools.assert_raises(RuntimeError, t.e_step_d, mb, [[1, 2, 2]])
    nose.tools.assert_raises(RuntimeError, t.m_step_d, mb, [1, 2, 2])
    nose.tools.assert_raises(RuntimeError, t.m_step_d, mb, [[1, 2, 2]])

    nose.tools.assert_raises(RuntimeError, t.enroll, m, [[1, 2, 2]], 5)


def test_JFATrainInitialize():
    # Check that the initialization is consistent and using the rng (cf. issue #118)

    eps = 1e-10

    # UBM GMM
    ubm = GMMMachine(2, 3)
    ubm.mean_supervector = UBM_MEAN
    ubm.variance_supervector = UBM_VAR

    ## JFA
    jfa_base = JFABase(ubm, 2, 2)
    # first round
    rng = random.randint
    jfa_machine = JFAMachine(jfa_base)
    jfa_trainer = JFATrainer(jfa_machine)
    jfa_trainer.train(TRAINING_STATS)

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


def test_estimateXandU():
    # estimateXandU
    JFATrainer.estimateXandU(F, N, UBM_MEAN, UBM_VAR, M_d, M_v, M_u, M_z, M_y, M_x, spk_ids)

    # JFA cookbook reference
    x_ref = np.array([
        [0.2143, 3.1979],
        [1.8275, 0.1227],
        [-1.3861, 5.3326],
        [0.2359, -0.7914],
    ])
    checkBlitzClose(M_x, x_ref, eps)


def test_estimateYandV():
    # estimateXandU
    JFATrainer.estimateYandV(F, N, UBM_MEAN, UBM_VAR, M_d, M_v, M_u, M_z, M_y, M_x, spk_ids)

    # JFA cookbook reference
    y_ref = np.array([
        [0.9630, 1.3868],
        [0.04255, -0.3721],
    ])
    checkBlitzClose(M_y, y_ref, eps)


def test_estimateZandD():
    # estimateXandU
    JFATrainer.estimateZandD(F, N, UBM_MEAN, UBM_VAR, M_d, M_v, M_u, M_z, M_y, M_x, spk_ids)
    # JFA cookbook reference
    z_ref = np.array([
        [0.3256, 1.8633, 0.6480, 0.8085, -0.0432, 0.2885],
        [-0.3324, -0.1474, -0.4404, -0.4529, 0.0484, -0.5848],
    ])
    checkBlitzClose(M_z, z_ref, eps)


def test_JFATrainer_updateYandV():

    F1 = F[0:1,:].transpose(1,0)
    F2 = F[2:3,:].transpose(1,0)
    Ft = [F1, F2]

    N1 = N[0:1, :].transpose(1, 0)
    N2 = N[2:3, :].transpose(1, 0)
    Nt = [N1, N2]

    vt = M_v.transpose(1, 0)
    ut = M_u.transpose(1, 0)

    # std::vector<blitz::Array<double,1> > zt;
    z1 = M_z[0, :]
    z2 = M_z[1, :]
    zt = [z1, z2]

    # std::vector<blitz::Array<double,1> > yt;
    y1 = [0, 0]
    y2 = [0, 0]
    yt = [y1, y2]

    x1 = M_x[0:1, :].transpose(1, 0)
    x2 = M_x[2:3, :].transpose(1, 0)
    xt = [x1, x2]

    # updateYandV
    ubm = GMMMachine(2,3)
    ubm.setMeanSupervector(UBM_MEAN)
    ubm.setVarianceSupervector(UBM_VAR)
    jfa_base_m = JFABase(ubm, 2, 2)
    jfa_base_m.setU(ut)
    jfa_base_m.setV(vt)
    jfa_base_m.setD(M_d)

    jfa_base_t = JFABaseTrainer(jfa_base_m)
    jfa_base_t.setStatistics(Nt, Ft)
    jfa_base_t.setSpeakerFactors(xt, yt, zt)
    jfa_base_t.precomputeSumStatisticsN()
    jfa_base_t.precomputeSumStatisticsF()

    jfa_base_t.updateY()
    jfa_base_t.updateV()

    # JFA cookbook reference
    v_ref = np.array([
        [0.7228, 0.7892],
        [0.6475, 0.6080],
        [0.8631, 0.8416],
        [1.6512, 1.6068],
        [0.0500, 0.0101],
        [0.4325, 0.6719],
    ])
    # y_ref
    y1_ref = np.array([0.9630, 1.3868])
    y2_ref = np.array([0.0426, -0.3721])

    checkBlitzClose(jfa_base_m.getV(), v_ref, eps)
    checkBlitzClose(jfa_base_t.getY()[0], y1_ref, eps)
    checkBlitzClose(jfa_base_t.getY()[1], y2_ref, eps)

def jfa_setup():
    ubm = GMMMachine(2,3)
    ubm.setMeanSupervector(m);
    ubm.setVarianceSupervector(E);

    # updateXandU
    jfa_base_m = JFABaseMachine(ubm, 2, 2)
    jfa_base_m.setU(ut)
    jfa_base_m.setV(vt)
    jfa_base_m.setD(M_d)

    jfa_base_t = JFABaseTrainer(jfa_base_m)
    jfa_base_t.setStatistics(Nt,Ft)
    jfa_base_t.setSpeakerFactors(xt,yt,zt)
    jfa_base_t.precomputeSumStatisticsN()
    jfa_base_t.precomputeSumStatisticsF()
    return jfa_base_m, jfa_base_t


def test_JFATrainer_updateXandU():

    jfa_base_m, jfa_base_t = jfa_setup()
    jfa_base_t.updateX()
    jfa_base_t.updateU()

    # JFA cookbook reference
    # u_ref
    u_ref = np.array([ [0.6729, 0.3408], [0.0544, 1.0653], [0.5399, 1.3035], [2.4995, 0.4385], [0.1292, -0.0576], [1.1962, 0.0117], ])
    # x_ref
    x1_ref = np.array([ [0.2143, 1.8275], [3.1979, 0.1227] ])
    x2_ref = np.array([ [-1.3861, 0.2359], [5.3326, -0.7914] ])

    checkBlitzClose(jfa_base_m.getU(), u_ref, eps)
    checkBlitzClose(jfa_base_t.getX()[0], x1_ref, eps)
    checkBlitzClose(jfa_base_t.getX()[1], x2_ref, eps)


def test_JFATrainer_updateZandD():

    jfa_base_m, jfa_base_t = jfa_setup()
    jfa_base_t.updateZ()
    jfa_base_t.updateD()

    # JFA cookbook reference
    # d_ref
    d_ref = np.array([0.3110, 1.0138, 0.8297, 1.0382, 0.0095, 0.6320 ])
    # z_ref
    z1_ref = np.array([0.3256, 1.8633, 0.6480, 0.8085, -0.0432, 0.2885])
    z2_ref = np.array([-0.3324, -0.1474, -0.4404, -0.4529, 0.0484, -0.5848])

    checkBlitzClose(jfa_base_m.getD(), d_ref, eps)
    checkBlitzClose(jfa_base_t.getZ()[0], z1_ref, eps)
    checkBlitzClose(jfa_base_t.getZ()[1], z2_ref, eps)


def test_JFATrainer_train():
    jfa_base_m, jfa_base_t = jfa_setup()
    jfa_base_t.train(Nt,Ft,1)


def test_JFATrainer_enrol():

    jfa_base_m, jfa_base_t = jfa_setup()
    # enrol
    jfa_base_m.setU(ut)
    jfa_base_m.setV(vt)
    jfa_base_m.setD(d)
    jfa_base_t = JFABaseTrainer(jfa_base_m)

    jfa_m = JFAMachine(jfa_base_m)

    jfa_t = JFATrainer(jfa_m, jfa_base_t)
    jfa_t.enrol(N1, F1, 5)

    sample = GMMStats(2,3)
    sample.T = 50
    sample.log_likelihood = -233
    sample.n = N1[:, 0]
    for g in [0,1]:
        f = sample.sumPx[g, :]
        slice0 = g * 3
        slice1 = (g+1)*3 - 1
        f = F1[slice0, slice1, 0]
        sample_ = GMMStats(sample)
        print(sample.n, sample.sumPx)
    jfa_m.forward(sample_, score)

print("fails:")
try:
    test_JFATrainer_updateYandV()
except:
    print("test_JFATrainer_updateYandV()")
try:
    test_JFATrainer_updateXandU()
except:
    print("test_JFATrainer_updateXandU()")
try:
    test_JFATrainer_updateZandD()
except:
    print("test_JFATrainer_updateZandD()")
try:
    test_JFATrainAndEnrol()
except:
    print("test_JFATrainAndEnrol()")
try:
    test_JFATrainInitialize()
except:
    print("test_JFATrainInitialize()")

test_JFATrainInitialize()
test_JFATrainAndEnrol()
test_JFATrainer_updateYandV()
test_JFATrainer_updateXandU()
test_JFATrainer_updateZandD()
