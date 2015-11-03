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

import nose.tools
import random

from train import GMM_Machine as GMMMachine
from train import GMM_Stats as  GMMStats
from train import JFA_Machine as JFAMachine
from train import JFA_Base as JFABase
from train import JFA_Trainer as JFATrainer


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
#m
UBM_MEAN = numpy.array([0.1806, 0.0451, 0.7232, 0.3474, 0.6606, 0.3839])
#E
UBM_VAR = numpy.array([0.6273, 0.0216, 0.9106, 0.8006, 0.7458, 0.8131])
#d
M_d = numpy.array([0.4106, 0.9843, 0.9456, 0.6766, 0.9883, 0.7668])
#v
M_v = numpy.array( [0.3367, 0.4116, 0.6624, 0.6026, 0.2442, 0.7505, 0.2955,
  0.5835, 0.6802, 0.5518, 0.5278,0.5836]).reshape((6,2))
#u
M_u = numpy.array( [0.5118, 0.3464, 0.0826, 0.8865, 0.7196, 0.4547, 0.9962,
  0.4134, 0.3545, 0.2177, 0.9713, 0.1257]).reshape((6,2))

#z
z1 = numpy.array([0.3089, 0.7261, 0.7829, 0.6938, 0.0098, 0.8432])
#z
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
  


def test_JFATrainInitialize():
  # Check that the initialization is consistent and using the rng (cf. issue #118)

  eps = 1e-10

  # UBM GMM
  ubm = GMMMachine(2,3)
  ubm.mean_supervector = UBM_MEAN
  ubm.variance_supervector = UBM_VAR

  ## JFA
  jfa_base = JFABase(ubm, 2, 2)
  # first round
  rng = random.randint
  jfa_machine = JFAMachine(jfa_base)
  jt = JFATrainer(jfa_machine, TRAINING_STATS, rng)
  u1 = jb.u
  v1 = jb.v
  d1 = jb.d

  # second round
  rng = random.randint
  jt.initialize(jb, TRAINING_STATS, rng)
  u2 = jb.u
  v2 = jb.v
  d2 = jb.d

  assert numpy.allclose(u1, u2, eps)
  assert numpy.allclose(v1, v2, eps)
  assert numpy.allclose(d1, d2, eps)

    def check_dimensions(t1, t2):
        assert len(t1) == (t2)
        for i in len(t1):
            assert t1.shape[i] == t2.shape[i]
      
    def checkBlitzEqual(t1, t2):
        check_dimensions( t1, t2);
        for i in t1.shape[0]:
            for j in t1.shape[1]:
                if len(t1.shape) > 2:
                    for k in t1.shape[2]:
                        assert t1[i, j, k] == t2[i, j, k]
                else:
                    assert t1[i, j] == t2[i, j]
      
      
    def checkBlitzClose(t1, t2, eps):
        np.allclose(t1, t2, eps)
      
      
      """
      
      BOOST_FIXTURE_TEST_SUITE( test_setup, T )
      
      BOOST_AUTO_TEST_CASE( test_estimateXandU )
      {
        // estimateXandU
        bob::trainer::jfa::estimateXandU(F,N,m,E,d,v,u,z,y,x,spk_ids);
      
        // JFA cookbook reference
        blitz::Array<double,2> x_ref(4,2);
        x_ref = 0.2143, 3.1979,
            1.8275, 0.1227,
            -1.3861, 5.3326,
            0.2359,  -0.7914;
      
        checkBlitzClose(x, x_ref, eps);
      }
      
      BOOST_AUTO_TEST_CASE( test_estimateYandV )
      {
        // estimateXandU
        bob::trainer::jfa::estimateYandV(F,N,m,E,d,v,u,z,y,x,spk_ids);
      
        // JFA cookbook reference
        blitz::Array<double,2> y_ref(2,2);
        y_ref = 0.9630, 1.3868,
            0.04255, -0.3721;
      
        checkBlitzClose(y, y_ref, eps);
      }
      
      BOOST_AUTO_TEST_CASE( test_estimateZandD )
      {
        // estimateXandU
        bob::trainer::jfa::estimateZandD(F,N,m,E,d,v,u,z,y,x,spk_ids);
      
        // JFA cookbook reference
        blitz::Array<double,2> z_ref(2,6);
        z_ref = 0.3256, 1.8633, 0.6480, 0.8085, -0.0432, 0.2885,
            -0.3324, -0.1474, -0.4404, -0.4529, 0.0484, -0.5848;
      
        checkBlitzClose(z, z_ref, eps);
      }
      """
    def test_JFATrainer_updateYandV():
        #Ft;
        #F1 = F(blitz::Range(0,1),blitz::Range::all()).transpose(1,0);
        #F2 = F(blitz::Range(2,3),blitz::Range::all()).transpose(1,0);
        #Ft.append(F1);
        #Ft.append(F2);
      
        std::vector<blitz::Array<double,2> > Nt;
        blitz::Array<double,2> N1 = N(blitz::Range(0,1),blitz::Range::all()).transpose(1,0);
        blitz::Array<double,2> N2 = N(blitz::Range(2,3),blitz::Range::all()).transpose(1,0);
        Nt.push_back(N1);
        Nt.push_back(N2);
      
        blitz::Array<double,2> vt = v.transpose(1,0);
        blitz::Array<double,2> ut = u.transpose(1,0);
      
        std::vector<blitz::Array<double,1> > zt;
        blitz::Array<double,1> z1 = z(0,blitz::Range::all());
        blitz::Array<double,1> z2 = z(1,blitz::Range::all());
        zt.push_back(z1);
        zt.push_back(z2);
      
        std::vector<blitz::Array<double,1> > yt;
        blitz::Array<double,1> y1(2);
        blitz::Array<double,1> y2(2);
        y1 = 0;
        y2 = 0;
        yt.push_back(y1);
        yt.push_back(y2);
      
        std::vector<blitz::Array<double,2> > xt;
        blitz::Array<double,2> x1 = x(blitz::Range(0,1),blitz::Range::all()).transpose(1,0);
        blitz::Array<double,2> x2 = x(blitz::Range(2,3),blitz::Range::all()).transpose(1,0);
        xt.push_back(x1);
        xt.push_back(x2);
      
        // updateYandV
        boost::shared_ptr<bob::machine::GMMMachine> ubm(new bob::machine::GMMMachine(2,3));
        ubm->setMeanSupervector(m);
        ubm->setVarianceSupervector(E);
        bob::machine::JFABaseMachine jfa_base_m(ubm, 2, 2);
        jfa_base_m.setU(ut);
        jfa_base_m.setV(vt);
        jfa_base_m.setD(d);
        bob::trainer::JFABaseTrainer jfa_base_t(jfa_base_m);
        jfa_base_t.setStatistics(Nt,Ft);
        jfa_base_t.setSpeakerFactors(xt,yt,zt);
        jfa_base_t.precomputeSumStatisticsN();
        jfa_base_t.precomputeSumStatisticsF();
      
        jfa_base_t.updateY();
        jfa_base_t.updateV();
      
        // JFA cookbook reference
        // v_ref
        blitz::Array<double,2> v_ref(6,2);
        v_ref = 0.7228, 0.7892,
                0.6475, 0.6080,
                0.8631, 0.8416,
                1.6512, 1.6068,
                0.0500, 0.0101,
                0.4325, 0.6719;
        // y_ref
        blitz::Array<double,1> y1_ref(2);
        y1_ref = 0.9630, 1.3868;
        blitz::Array<double,1> y2_ref(2);
        y2_ref = 0.0426, -0.3721;
      
        checkBlitzClose(jfa_base_m.getV(), v_ref, eps);
        checkBlitzClose(jfa_base_t.getY()[0], y1_ref, eps);
        checkBlitzClose(jfa_base_t.getY()[1], y2_ref, eps);
      }
      
      BOOST_AUTO_TEST_CASE( test_JFATrainer_updateXandU )
      {
        std::vector<blitz::Array<double,2> > Ft;
        blitz::Array<double,2> F1 = F(blitz::Range(0,1),blitz::Range::all()).transpose(1,0);
        blitz::Array<double,2> F2 = F(blitz::Range(2,3),blitz::Range::all()).transpose(1,0);
        Ft.push_back(F1);
        Ft.push_back(F2);
      
        std::vector<blitz::Array<double,2> > Nt;
        blitz::Array<double,2> N1 = N(blitz::Range(0,1),blitz::Range::all()).transpose(1,0);
        blitz::Array<double,2> N2 = N(blitz::Range(2,3),blitz::Range::all()).transpose(1,0);
        Nt.push_back(N1);
        Nt.push_back(N2);
      
        blitz::Array<double,2> vt = v.transpose(1,0);
        blitz::Array<double,2> ut = u.transpose(1,0);
      
        std::vector<blitz::Array<double,1> > zt;
        blitz::Array<double,1> z1 = z(0,blitz::Range::all());
        blitz::Array<double,1> z2 = z(1,blitz::Range::all());
        zt.push_back(z1);
        zt.push_back(z2);
      
        std::vector<blitz::Array<double,1> > yt;
        blitz::Array<double,1> y1 = y(0,blitz::Range::all());
        blitz::Array<double,1> y2 = y(1,blitz::Range::all());
        yt.push_back(y1);
        yt.push_back(y2);
      
        std::vector<blitz::Array<double,2> > xt;
        blitz::Array<double,2> x1(2,2);
        x1 = 0;
        blitz::Array<double,2> x2(2,2);
        x2 = 0;
        xt.push_back(x1);
        xt.push_back(x2);
      
        // updateXandU
        boost::shared_ptr<bob::machine::GMMMachine> ubm(new bob::machine::GMMMachine(2,3));
        ubm->setMeanSupervector(m);
        ubm->setVarianceSupervector(E);
        bob::machine::JFABaseMachine jfa_base_m(ubm, 2, 2);
        jfa_base_m.setU(ut);
        jfa_base_m.setV(vt);
        jfa_base_m.setD(d);
        bob::trainer::JFABaseTrainer jfa_base_t(jfa_base_m);
        jfa_base_t.setStatistics(Nt,Ft);
        jfa_base_t.setSpeakerFactors(xt,yt,zt);
        jfa_base_t.precomputeSumStatisticsN();
        jfa_base_t.precomputeSumStatisticsF();
      
        jfa_base_t.updateX();
        jfa_base_t.updateU();
      
        // JFA cookbook reference
        // u_ref
        blitz::Array<double,2> u_ref(6,2);
        u_ref = 0.6729, 0.3408,
                0.0544, 1.0653,
                0.5399, 1.3035,
                2.4995, 0.4385,
                0.1292, -0.0576,
                1.1962, 0.0117;
        // x_ref
        blitz::Array<double,2> x1_ref(2,2);
        x1_ref = 0.2143, 1.8275,
                 3.1979, 0.1227;
        blitz::Array<double,2> x2_ref(2,2);
        x2_ref = -1.3861, 0.2359,
                  5.3326, -0.7914;
      
        checkBlitzClose(jfa_base_m.getU(), u_ref, eps);
        checkBlitzClose(jfa_base_t.getX()[0], x1_ref, eps);
        checkBlitzClose(jfa_base_t.getX()[1], x2_ref, eps);
      }
      
      BOOST_AUTO_TEST_CASE( test_JFATrainer_updateZandD )
      {
        std::vector<blitz::Array<double,2> > Ft;
        blitz::Array<double,2> F1 = F(blitz::Range(0,1),blitz::Range::all()).transpose(1,0);
        blitz::Array<double,2> F2 = F(blitz::Range(2,3),blitz::Range::all()).transpose(1,0);
        Ft.push_back(F1);
        Ft.push_back(F2);
      
        std::vector<blitz::Array<double,2> > Nt;
        blitz::Array<double,2> N1 = N(blitz::Range(0,1),blitz::Range::all()).transpose(1,0);
        blitz::Array<double,2> N2 = N(blitz::Range(2,3),blitz::Range::all()).transpose(1,0);
        Nt.push_back(N1);
        Nt.push_back(N2);
      
        blitz::Array<double,2> vt = v.transpose(1,0);
        blitz::Array<double,2> ut = u.transpose(1,0);
      
        std::vector<blitz::Array<double,1> > zt;
        blitz::Array<double,1> z1(6);
        z1 = 0;
        blitz::Array<double,1> z2(6);
        z2 = 0;
        zt.push_back(z1);
        zt.push_back(z2);
      
        std::vector<blitz::Array<double,1> > yt;
        blitz::Array<double,1> y1 = y(0,blitz::Range::all());
        blitz::Array<double,1> y2 = y(1,blitz::Range::all());
        yt.push_back(y1);
        yt.push_back(y2);
      
        std::vector<blitz::Array<double,2> > xt;
        blitz::Array<double,2> x1 = x(blitz::Range(0,1),blitz::Range::all()).transpose(1,0);
        blitz::Array<double,2> x2 = x(blitz::Range(2,3),blitz::Range::all()).transpose(1,0);
        xt.push_back(x1);
        xt.push_back(x2);
      
        // updateZandD
        boost::shared_ptr<bob::machine::GMMMachine> ubm(new bob::machine::GMMMachine(2,3));
        ubm->setMeanSupervector(m);
        ubm->setVarianceSupervector(E);
        bob::machine::JFABaseMachine jfa_base_m(ubm, 2, 2);
        jfa_base_m.setU(ut);
        jfa_base_m.setV(vt);
        jfa_base_m.setD(d);
        bob::trainer::JFABaseTrainer jfa_base_t(jfa_base_m);
        jfa_base_t.setStatistics(Nt,Ft);
        jfa_base_t.setSpeakerFactors(xt,yt,zt);
        jfa_base_t.precomputeSumStatisticsN();
        jfa_base_t.precomputeSumStatisticsF();
      
        jfa_base_t.updateZ();
        jfa_base_t.updateD();
      
        // JFA cookbook reference
        // d_ref
        blitz::Array<double,1> d_ref(6);
        d_ref = 0.3110, 1.0138, 0.8297, 1.0382, 0.0095, 0.6320;
        // z_ref
        blitz::Array<double,1> z1_ref(6);
        z1_ref = 0.3256, 1.8633, 0.6480, 0.8085, -0.0432, 0.2885;
        blitz::Array<double,1> z2_ref(6);
        z2_ref = -0.3324, -0.1474, -0.4404, -0.4529, 0.0484, -0.5848;
      
        checkBlitzClose(jfa_base_m.getD(), d_ref, eps);
        checkBlitzClose(jfa_base_t.getZ()[0], z1_ref, eps);
        checkBlitzClose(jfa_base_t.getZ()[1], z2_ref, eps);
      }
      
      BOOST_AUTO_TEST_CASE( test_JFATrainer_train )
      {
        std::vector<blitz::Array<double,2> > Ft;
        blitz::Array<double,2> F1 = F(blitz::Range(0,1),blitz::Range::all()).transpose(1,0);
        blitz::Array<double,2> F2 = F(blitz::Range(2,3),blitz::Range::all()).transpose(1,0);
        Ft.push_back(F1);
        Ft.push_back(F2);
      
        std::vector<blitz::Array<double,2> > Nt;
        blitz::Array<double,2> N1 = N(blitz::Range(0,1),blitz::Range::all()).transpose(1,0);
        blitz::Array<double,2> N2 = N(blitz::Range(2,3),blitz::Range::all()).transpose(1,0);
        Nt.push_back(N1);
        Nt.push_back(N2);
      
        blitz::Array<double,2> vt = v.transpose(1,0);
        blitz::Array<double,2> ut = u.transpose(1,0);
      
        std::vector<blitz::Array<double,1> > zt;
        blitz::Array<double,1> z1(6);
        z1 = 0;
        blitz::Array<double,1> z2(6);
        z2 = 0;
        zt.push_back(z1);
        zt.push_back(z2);
      
        std::vector<blitz::Array<double,1> > yt;
        blitz::Array<double,1> y1 = y(0,blitz::Range::all());
        blitz::Array<double,1> y2 = y(1,blitz::Range::all());
        yt.push_back(y1);
        yt.push_back(y2);
      
        std::vector<blitz::Array<double,2> > xt;
        blitz::Array<double,2> x1 = x(blitz::Range(0,1),blitz::Range::all()).transpose(1,0);
        blitz::Array<double,2> x2 = x(blitz::Range(2,3),blitz::Range::all()).transpose(1,0);
        xt.push_back(x1);
        xt.push_back(x2);
      
        // train
        boost::shared_ptr<bob::machine::GMMMachine> ubm(new bob::machine::GMMMachine(2,3));
        ubm->setMeanSupervector(m);
        ubm->setVarianceSupervector(E);
        bob::machine::JFABaseMachine jfa_base_m(ubm, 2, 2);
        jfa_base_m.setU(ut);
        jfa_base_m.setV(vt);
        jfa_base_m.setD(d);
        bob::trainer::JFABaseTrainer jfa_base_t(jfa_base_m);
        jfa_base_t.train(Nt,Ft,1);
      }
      
      BOOST_AUTO_TEST_CASE( test_JFATrainer_enrol )
      {
        std::vector<blitz::Array<double,2> > Ft;
        blitz::Array<double,2> F1 = F(blitz::Range(0,1),blitz::Range::all()).transpose(1,0);
        blitz::Array<double,2> F2 = F(blitz::Range(2,3),blitz::Range::all()).transpose(1,0);
        Ft.push_back(F1);
        Ft.push_back(F2);
      
        std::vector<blitz::Array<double,2> > Nt;
        blitz::Array<double,2> N1 = N(blitz::Range(0,1),blitz::Range::all()).transpose(1,0);
        blitz::Array<double,2> N2 = N(blitz::Range(2,3),blitz::Range::all()).transpose(1,0);
        Nt.push_back(N1);
        Nt.push_back(N2);
      
        blitz::Array<double,2> vt = v.transpose(1,0);
        blitz::Array<double,2> ut = u.transpose(1,0);
      
        std::vector<blitz::Array<double,1> > zt;
        blitz::Array<double,1> z1 = z(0,blitz::Range::all());
        blitz::Array<double,1> z2 = z(1,blitz::Range::all());
        zt.push_back(z1);
        zt.push_back(z2);
      
        std::vector<blitz::Array<double,1> > yt;
        blitz::Array<double,1> y1 = y(0,blitz::Range::all());
        blitz::Array<double,1> y2 = y(1,blitz::Range::all());
        yt.push_back(y1);
        yt.push_back(y2);
      
        std::vector<blitz::Array<double,2> > xt;
        blitz::Array<double,2> x1 = x(blitz::Range(0,1),blitz::Range::all()).transpose(1,0);
        blitz::Array<double,2> x2 = x(blitz::Range(2,3),blitz::Range::all()).transpose(1,0);
        xt.push_back(x1);
        xt.push_back(x2);
      
        // enrol
        boost::shared_ptr<bob::machine::GMMMachine> ubm(new bob::machine::GMMMachine(2,3));
        ubm->setMeanSupervector(m);
        ubm->setVarianceSupervector(E);
        boost::shared_ptr<bob::machine::JFABaseMachine> jfa_base_m(new bob::machine::JFABaseMachine(ubm, 2, 2));
        jfa_base_m->setU(ut);
        jfa_base_m->setV(vt);
        jfa_base_m->setD(d);
        bob::trainer::JFABaseTrainer jfa_base_t(*jfa_base_m);
      
        bob::machine::JFAMachine jfa_m(jfa_base_m);
      
        bob::trainer::JFATrainer jfa_t(jfa_m, jfa_base_t);
        jfa_t.enrol(N1,F1,5);
      
        double score;
        bob::machine::GMMStats sample(2,3);
        sample.T = 50;
        sample.log_likelihood = -233;
        sample.n = N1(blitz::Range::all(),0);
        for(int g=0; g<2; ++g) {
          blitz::Array<double,1> f = sample.sumPx(g,blitz::Range::all());
          f = F1(blitz::Range(g*3,(g+1)*3-1),0);
        }
        boost::shared_ptr<const bob::machine::GMMStats> sample_(new bob::machine::GMMStats(sample));
      //  std::cout << sample.n << sample.sumPx;
        jfa_m.forward(sample_, score);
      }
      */

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
#test_JFATrainer_updateYandV()
#test_JFATrainer_updateXandU()
#test_JFATrainer_updateZandD()