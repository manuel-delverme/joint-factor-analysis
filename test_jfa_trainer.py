#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
# Tue Jul 19 12:16:17 2011 +0200
#
# Copyright (C) 2011-2014 Idiap Research Institute, Martigny, Switzerland

"""Test JFA trainer package
"""

import numpy
import numpy.linalg

import bob.core.random
import nose.tools

from bob.learn.em import GMMStats, GMMMachine, JFABase, JFAMachine, ISVBase, ISVMachine, JFATrainer, ISVTrainer


def equals(x, y, epsilon):
  return (abs(x - y) < epsilon).all()

# Define Training set and initial values for tests
F1 = numpy.array( [0.3833, 0.4516, 0.6173, 0.2277, 0.5755, 0.8044, 0.5301,
  0.9861, 0.2751, 0.0300, 0.2486, 0.5357]).reshape((6,2))
F2 = numpy.array( [0.0871, 0.6838, 0.8021, 0.7837, 0.9891, 0.5341, 0.0669,
  0.8854, 0.9394, 0.8990, 0.0182, 0.6259]).reshape((6,2))
F=[F1, F2]

N1 = numpy.array([0.1379, 0.1821, 0.2178, 0.0418]).reshape((2,2))
N2 = numpy.array([0.1069, 0.9397, 0.6164, 0.3545]).reshape((2,2))
N=[N1, N2]

gs11 = GMMStats(2,3)
gs11.n = N1[:,0]
gs11.sum_px = F1[:,0].reshape(2,3)
gs12 = GMMStats(2,3)
gs12.n = N1[:,1]
gs12.sum_px = F1[:,1].reshape(2,3)

gs21 = GMMStats(2,3)
gs21.n = N2[:,0]
gs21.sum_px = F2[:,0].reshape(2,3)
gs22 = GMMStats(2,3)
gs22.n = N2[:,1]
gs22.sum_px = F2[:,1].reshape(2,3)

TRAINING_STATS = [[gs11, gs12], [gs21, gs22]]
UBM_MEAN = numpy.array([0.1806, 0.0451, 0.7232, 0.3474, 0.6606, 0.3839])
UBM_VAR = numpy.array([0.6273, 0.0216, 0.9106, 0.8006, 0.7458, 0.8131])
M_d = numpy.array([0.4106, 0.9843, 0.9456, 0.6766, 0.9883, 0.7668])
M_v = numpy.array( [0.3367, 0.4116, 0.6624, 0.6026, 0.2442, 0.7505, 0.2955,
  0.5835, 0.6802, 0.5518, 0.5278,0.5836]).reshape((6,2))
M_u = numpy.array( [0.5118, 0.3464, 0.0826, 0.8865, 0.7196, 0.4547, 0.9962,
  0.4134, 0.3545, 0.2177, 0.9713, 0.1257]).reshape((6,2))

z1 = numpy.array([0.3089, 0.7261, 0.7829, 0.6938, 0.0098, 0.8432])
z2 = numpy.array([0.9223, 0.7710, 0.0427, 0.3782, 0.7043, 0.7295])
y1 = numpy.array([0.2243, 0.2691])
y2 = numpy.array([0.6730, 0.4775])
x1 = numpy.array([0.9976, 0.8116, 0.1375, 0.3900]).reshape((2,2))
x2 = numpy.array([0.4857, 0.8944, 0.9274, 0.9175]).reshape((2,2))
M_z=[z1, z2]
M_y=[y1, y2]
M_x=[x1, x2]


def test_JFATrainer_updateYandV():
  # test the JFATrainer for updating Y and V

  v_ref = numpy.array( [0.7228, 0.7892, 0.6475, 0.6080, 0.8631, 0.8416,
    1.6512, 1.6068, 0.0500, 0.0101, 0.4325, 0.6719]).reshape((6,2))

  y1 = numpy.array([0., 0.])
  y2 = numpy.array([0., 0.])
  y3 = numpy.array([0.9630, 1.3868])
  y4 = numpy.array([0.0426, -0.3721])
  y=[y1, y2]

  # call the updateY function
  ubm = GMMMachine(2,3)
  ubm.mean_supervector = UBM_MEAN
  ubm.variance_supervector = UBM_VAR
  m = JFABase(ubm,2,2)
  t = JFATrainer()
  t.initialize(m, TRAINING_STATS)
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

  u_ref = numpy.array( [0.6729, 0.3408, 0.0544, 1.0653, 0.5399, 1.3035,
    2.4995, 0.4385, 0.1292, -0.0576, 1.1962, 0.0117]).reshape((6,2))

  x1 = numpy.array([0., 0., 0., 0.]).reshape((2,2))
  x2 = numpy.array([0., 0., 0., 0.]).reshape((2,2))
  x3 = numpy.array([0.2143, 1.8275, 3.1979, 0.1227]).reshape((2,2))
  x4 = numpy.array([-1.3861, 0.2359, 5.3326, -0.7914]).reshape((2,2))
  x  = [x1, x2]

  # call the updateX function
  ubm = GMMMachine(2,3)
  ubm.mean_supervector = UBM_MEAN
  ubm.variance_supervector = UBM_VAR
  m = JFABase(ubm,2,2)
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
  z=[z1, z2]

  # call the updateZ function
  ubm = GMMMachine(2,3)
  ubm.mean_supervector = UBM_MEAN
  ubm.variance_supervector = UBM_VAR
  m = JFABase(ubm,2,2)
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
  ubm = GMMMachine(2,3)
  ubm.mean_supervector = UBM_MEAN
  ubm.variance_supervector = UBM_VAR
  mb = JFABase(ubm, 2, 2)
  t = JFATrainer()
  t.initialize(mb, TRAINING_STATS)
  mb.u = M_u
  mb.v = M_v
  mb.d = M_d
  bob.learn.em.train_jfa(t,mb, TRAINING_STATS, initialize=False)

  v_ref = numpy.array([[0.245364911936476, 0.978133261775424], [0.769646805052223, 0.940070736856596], [0.310779202800089, 1.456332053893072],
        [0.184760934399551, 2.265139705602147], [0.701987784039800, 0.081632150899400], [0.074344030229297, 1.090248340917255]], 'float64')
  u_ref = numpy.array([[0.049424652628448, 0.060480486336896], [0.178104127464007, 1.884873813495153], [1.204011484266777, 2.281351307871720],
        [7.278512126426286, -0.390966087173334], [-0.084424326581145, -0.081725474934414], [4.042143689831097, -0.262576386580701]], 'float64')
  d_ref = numpy.array([9.648467e-18, 2.63720683155e-12, 2.11822157653706e-10, 9.1047243e-17, 1.41163442535567e-10, 3.30581e-19], 'float64')

  eps = 1e-10
  assert numpy.allclose(mb.v, v_ref, eps)
  assert numpy.allclose(mb.u, u_ref, eps)
  assert numpy.allclose(mb.d, d_ref, eps)

  # Calls the enroll function
  m = JFAMachine(mb)

  Ne = numpy.array([0.1579, 0.9245, 0.1323, 0.2458]).reshape((2,2))
  Fe = numpy.array([0.1579, 0.1925, 0.3242, 0.1234, 0.2354, 0.2734, 0.2514, 0.5874, 0.3345, 0.2463, 0.4789, 0.5236]).reshape((6,2))
  gse1 = GMMStats(2,3)
  gse1.n = Ne[:,0]
  gse1.sum_px = Fe[:,0].reshape(2,3)
  gse2 = GMMStats(2,3)
  gse2.n = Ne[:,1]
  gse2.sum_px = Fe[:,1].reshape(2,3)

  gse = [gse1, gse2]
  t.enroll(m, gse, 5)

  y_ref = numpy.array([0.555991469319657, 0.002773650670010], 'float64')
  z_ref = numpy.array([8.2228e-20, 3.15216909492e-13, -1.48616735364395e-10, 1.0625905e-17, 3.7150503117895e-11, 1.71104e-19], 'float64')
  assert numpy.allclose(m.y, y_ref, eps)
  assert numpy.allclose(m.z, z_ref, eps)
  
  #Testing exceptions
  nose.tools.assert_raises(RuntimeError, t.initialize, mb, [1,2,2])  
  nose.tools.assert_raises(RuntimeError, t.initialize, mb, [[1,2,2]])
  nose.tools.assert_raises(RuntimeError, t.e_step_u, mb, [1,2,2])  
  nose.tools.assert_raises(RuntimeError, t.e_step_u, mb, [[1,2,2]])
  nose.tools.assert_raises(RuntimeError, t.m_step_u, mb, [1,2,2])  
  nose.tools.assert_raises(RuntimeError, t.m_step_u, mb, [[1,2,2]])
  
  nose.tools.assert_raises(RuntimeError, t.e_step_v, mb, [1,2,2])  
  nose.tools.assert_raises(RuntimeError, t.e_step_v, mb, [[1,2,2]])  
  nose.tools.assert_raises(RuntimeError, t.m_step_v, mb, [1,2,2])  
  nose.tools.assert_raises(RuntimeError, t.m_step_v, mb, [[1,2,2]])  
    
  nose.tools.assert_raises(RuntimeError, t.e_step_d, mb, [1,2,2])  
  nose.tools.assert_raises(RuntimeError, t.e_step_d, mb, [[1,2,2]])
  nose.tools.assert_raises(RuntimeError, t.m_step_d, mb, [1,2,2])  
  nose.tools.assert_raises(RuntimeError, t.m_step_d, mb, [[1,2,2]])
  
  nose.tools.assert_raises(RuntimeError, t.enroll, m, [[1,2,2]],5)
  


def test_ISVTrainAndEnrol():
  # Train and enroll an 'ISVMachine'

  eps = 1e-10
  d_ref = numpy.array([0.39601136, 0.07348469, 0.47712682, 0.44738127, 0.43179856, 0.45086029], 'float64')
  u_ref = numpy.array([[0.855125642430777, 0.563104284748032], [-0.325497865404680, 1.923598985291687], [0.511575659503837, 1.964288663083095], [9.330165761678115, 1.073623827995043], [0.511099245664012, 0.278551249248978], [5.065578541930268, 0.509565618051587]], 'float64')
  z_ref = numpy.array([-0.079315777443826, 0.092702428248543, -0.342488761656616, -0.059922635809136 , 0.133539981073604, 0.213118695516570], 'float64')

  # Calls the train function
  ubm = GMMMachine(2,3)
  ubm.mean_supervector = UBM_MEAN
  ubm.variance_supervector = UBM_VAR
  mb = ISVBase(ubm,2)
  t = ISVTrainer(4.)
  t.initialize(mb, TRAINING_STATS)
  mb.u = M_u
  for i in range(10):
    t.e_step(mb, TRAINING_STATS)
    t.m_step(mb)

  assert numpy.allclose(mb.d, d_ref, eps)
  assert numpy.allclose(mb.u, u_ref, eps)

  # Calls the enroll function
  m = ISVMachine(mb)

  Ne = numpy.array([0.1579, 0.9245, 0.1323, 0.2458]).reshape((2,2))
  Fe = numpy.array([0.1579, 0.1925, 0.3242, 0.1234, 0.2354, 0.2734, 0.2514, 0.5874, 0.3345, 0.2463, 0.4789, 0.5236]).reshape((6,2))
  gse1 = GMMStats(2,3)
  gse1.n = Ne[:,0]
  gse1.sum_px = Fe[:,0].reshape(2,3)
  gse2 = GMMStats(2,3)
  gse2.n = Ne[:,1]
  gse2.sum_px = Fe[:,1].reshape(2,3)

  gse = [gse1, gse2]
  t.enroll(m, gse, 5)
  assert numpy.allclose(m.z, z_ref, eps)
  
  #Testing exceptions
  nose.tools.assert_raises(RuntimeError, t.initialize, mb, [1,2,2])  
  nose.tools.assert_raises(RuntimeError, t.initialize, mb, [[1,2,2]])
  nose.tools.assert_raises(RuntimeError, t.e_step, mb, [1,2,2])  
  nose.tools.assert_raises(RuntimeError, t.e_step, mb, [[1,2,2]])
  nose.tools.assert_raises(RuntimeError, t.enroll, m, [[1,2,2]],5)
  


def test_JFATrainInitialize():
  # Check that the initialization is consistent and using the rng (cf. issue #118)

  eps = 1e-10

  # UBM GMM
  ubm = GMMMachine(2,3)
  ubm.mean_supervector = UBM_MEAN
  ubm.variance_supervector = UBM_VAR

  ## JFA
  jb = JFABase(ubm, 2, 2)
  # first round
  rng = bob.core.random.mt19937(0)
  jt = JFATrainer()
  #jt.rng = rng
  jt.initialize(jb, TRAINING_STATS, rng)
  u1 = jb.u
  v1 = jb.v
  d1 = jb.d

  # second round
  rng = bob.core.random.mt19937(0)
  jt.initialize(jb, TRAINING_STATS, rng)
  u2 = jb.u
  v2 = jb.v
  d2 = jb.d

  assert numpy.allclose(u1, u2, eps)
  assert numpy.allclose(v1, v2, eps)
  assert numpy.allclose(d1, d2, eps)
    

def test_ISVTrainInitialize():

  # Check that the initialization is consistent and using the rng (cf. issue #118)
  eps = 1e-10

  # UBM GMM
  ubm = GMMMachine(2,3)
  ubm.mean_supervector = UBM_MEAN
  ubm.variance_supervector = UBM_VAR

  ## ISV
  ib = ISVBase(ubm, 2)
  # first round
  rng = bob.core.random.mt19937(0)
  it = ISVTrainer(10)
  #it.rng = rng
  it.initialize(ib, TRAINING_STATS, rng)
  u1 = ib.u
  d1 = ib.d

  # second round
  rng = bob.core.random.mt19937(0)
  #it.rng = rng
  it.initialize(ib, TRAINING_STATS, rng)
  u2 = ib.u
  d2 = ib.d

  assert numpy.allclose(u1, u2, eps)
  assert numpy.allclose(d1, d2, eps)
