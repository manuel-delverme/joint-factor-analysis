# -*- coding: UTF-8 -*-
#!/usr/bin/python3
import copy
import os
import itertools
import scipy.sparse
import scipy.io.wavfile as wavfile
import glob

import warnings
import sklearn.mixture
import numpy as np

import cheatcodes

try:
    from bob.ap import Ceps
except ImportError:
    def Ceps(*args):
        print("bob not installed, returning random")
        return lambda signal: np.random.random((12, len(signal)))

JFA_PATH = "../jfa_cookbook"
MODELS_PATH = JFA_PATH + "/models"
LISTS_PATH = JFA_PATH + "/lists"


class GMM_Stats(object):
    def __init__(self, n_gaussians, n_inputs):
        self.n_gaussians = n_gaussians
        self.n_inputs = n_inputs
        self.n = None
        self.sum_Px = None
        self.sum_Pxx = None
        self.t = None


class GMM_Machine(object):
    def __init__(self, n_components, n_inputs):
        self.n_inputs = n_inputs
        self.n_components = n_components

    def getNInputs(self):
        return self.n_inputs

    def getNGaussians(self):
        return self.n_components


class JFA_Base(object):
    def __init__(self, ubm, dimEigenChannels, dimEigenVoices):
        self.n_components = None
        self.n_inputs = None
        assert dimEigenChannels > 0  # full JFA only now
        assert dimEigenVoices > 0  # full JFA only now
        self.ubm = ubm
        self.dimEigenChannels = dimEigenChannels
        self.dimEigenVoices = dimEigenVoices
        self.U = np.zeros((self.getDimSupervector(), dimEigenChannels))
        self.V = np.zeros((self.getDimSupervector(), dimEigenVoices))
        self.d = np.zeros(self.getDimSupervector())


        # next part is inti'd later
        self.cache_mean = None
        self.cache_sigma = None

    def getDimC(self):
        raise Exception("use getNrUBMGaussians")
        return self.ubm.getNGaussians()
  
    def getDimD(self):
        warnings.warn("use getNrUBMInputs")
        return self.ubm.getNInputs()
  
    def getDimSupervector(self):
        return self.ubm.getNGaussians() * self.ubm.getNInputs()

    def getDimCD(self):
        warnings.warn("use DimSupervector")
        return self.getDimC() * self.getDimD()

    def getDimEigenChannels(self):
        return self.dimEigenChannels

    def getDimEigenVoices(self):
        return self.dimEigenVoices

    def getDimRu(self):
        raise Exception("use DimEigenChannels")
        return self.dimEigenChannels

    def getDimRv(self):
        raise Exception("use DimEigenVoices")
        return self.dimEigenVoices

    def getNInputs(self):
        return self.n_inputs

    def getNGaussians(self):
        return self.n_components

    def getNrUbmGaussians(self):
        return self.ubm.getNGaussians()

    def getNrUbmInputs(self):
        return self.ubm.getNInputs()

    def updateU(self, U):
        warnings.warn("Should only be used by the trainer for efficiency reason, or for testing purpose.")
        self.U = U

    def updateV(self, V):
        warnings.warn("Should only be used by the trainer for efficiency reason, or for testing purpose.")
        self.V = V

    def updateD(self, d):
        warnings.warn("Should only be used by the trainer for efficiency reason, or for testing purpose.")
        self.d = d

    def getU(self):
        # rows:eigenchannels cols:supervectorDimensionality
        return self.U

    def getV(self):
        return self.V

    def getD(self):
        return self.d

    @property
    def shape(self):
        return (self.ubm.getNGaussians(), self.ubm.getNInputs(),
                self.dimEigenChannels, self.dimEigenVoices)

    @property
    def supervector_length(self):
      return self.ubm.getNInputs() * self.ubm.getNGaussians()

    def getUbm(self):
        return self.ubm

    def resize(self, ru, rv):
        self.dimEigenChannels = ru
        self.dimEigenVoices = rv
        self.U.resize(self.U.shape[0], ru)
        self.V.resize(self.V.shape[0], rv)


class JFA_Machine(object):
    def __init__(self, jfa_base_machine):
        self.jfa_base = jfa_base_machine

        self.y = jfa_base_machine.getDimEigenVoices()
        self.z = jfa_base_machine.getDimSupervector()
        self.y_for_x = jfa_base_machine.getDimEigenVoices()
        self.z_for_x = jfa_base_machine.getDimSupervector()
        self.x = jfa_base_machine.getDimEigenChannels()

    def getDimSupervector(self):
        return self.jfa_base.getDimSupervector()

    def getBase(self):
        return self.jfa_base

    def getV(self):
        return self.jfa_base.V

    def cacheSupervectors(self):
        # Put supervectors in cache
        self.cache_mean = self.jfa_base.getUbm().getMeanSupervector()
        self.cache_sigma = self.jfa_base.getUbm().getVarianceSupervector()


class JFA_Trainer(object):
    def __init__(self, jfa_machine, n_iter_train=10):
        self.jfa_base_machine = jfa_machine.getBase()
        self.cache_ubm_var = self.jfa_base_machine.ubm.variance_supervector
        self.cache_ubm_mean = self.jfa_base_machine.ubm.mean_supervector
        self.jfa_machine = jfa_machine
        self.n_iter_train = n_iter_train

        dimEigenChannels = self.jfa_base_machine.getDimEigenChannels()
        dimSupervector = self.jfa_base_machine.getDimSupervector()
        nrUbmGaussians = self.jfa_base_machine.getNrUbmGaussians()
        dimEigenVoices = self.jfa_base_machine.getDimEigenVoices()
        nrUbmInputs = self.jfa_base_machine.getNrUbmInputs()

        # Cache/Precomputation
        # U
        self.cache_UtSigmaInv = np.zeros((dimEigenChannels,  dimSupervector)) # U'Σ⁻¹
        self.cache_UProd = np.zeros((nrUbmGaussians, dimEigenChannels, dimEigenChannels))
        self.cache_IdPlusUProd_ih = np.zeros((dimEigenChannels, dimEigenChannels))
        self.cache_Fn_x_ih = np.zeros((dimSupervector, 1))
        self.cache_A1_x = np.zeros((nrUbmGaussians, dimEigenChannels, dimEigenChannels))
        self.cache_A2_x = np.zeros((dimSupervector, dimEigenChannels))
        # V
        self.cache_VtSigmaInv = np.zeros((dimEigenVoices,  dimSupervector))
        self.cache_VProd = np.zeros((nrUbmGaussians, dimEigenVoices, dimEigenVoices))
        self.cache_IdPlusVProd_i = np.zeros((dimEigenVoices, dimEigenVoices))
        self.cache_Fn_y_i = np.zeros(dimSupervector)
        self.cache_A1_y = np.zeros((nrUbmGaussians, dimEigenVoices, dimEigenVoices))
        self.cache_A2_y = np.zeros((dimSupervector, dimEigenVoices))
        # D
        self.cache_DtSigmaInv = np.zeros(dimSupervector)
        self.cache_DProd = np.zeros(dimSupervector)
        self.cache_IdPlusDProd_i = np.zeros(dimSupervector)
        self.cache_Fn_z_i = np.zeros(dimSupervector)
        self.cache_A1_z = np.zeros(dimSupervector)
        self.cache_A2_z = np.zeros(dimSupervector)

        # will be used later
        self.Nacc = []
        self.Facc = []
        self.x = None
        self.z = None
        self.y = None
        self.U = None
        self.V = None
        self.D = None
        self.n_speakers = None  # number of gmm_stats

    def getX(self):
        return self.jfa_base_machine.getX()

    def getZ(self):
        return self.jfa_base_machine.getZ()

    def getY(self):
        return self.jfa_base_machine.getY()

    def getU(self):
        return self.jfa_base_machine.getU()

    def getV(self):
        return self.jfa_base_machine.getV()

    def getD(self):
        return self.jfa_base_machine.getD()

    def precomputeSumStatisticsN(self, training_data):
        warnings.warn("CHEATING HERE")
        return

        self.Nacc = []
        for session in training_data:
            Nsum = sum([mixture.n for mixture in session])
            self.Nacc.append(Nsum)

    def precomputeSumStatisticsF(self, training_data):
        warnings.warn("CHEATING HERE")
        return

        self.Facc = []
        ubm = self.jfa_base_machine.getUbm()
        Fsum = np.empty(self.jfa_machine.getDimSupervector())

        for session in training_data:
            Fsum.fill(0)
            for mixture_component in session:
                #  dim = size(m, 1) / n_mixtures;
                gaussiansPerMixture = len(Fsum) // ubm.getNGaussians()
                for index, slice_start in enumerate(range(0, len(Fsum), gaussiansPerMixture)):
                    slice_end = slice_start + gaussiansPerMixture
                    Fsum_gaussian = Fsum[slice_start: slice_end]
                    # Warning! pointers here!
                    Fsum_gaussian += mixture_component.sum_Px[index]
            self.Facc.append(Fsum)

    def initializeUVD(self):
        nr_eigenvoices = self.jfa_base_machine.getDimEigenVoices()
        shape = (nr_eigenvoices, self.Facc.shape[1])
        v = np.random.random_sample(shape)  # * sum(E,2) * 0.001;
        self.jfa_base_machine.updateV(v)
        self.jfa_base_machine.updateU(np.random.random_sample(self.jfa_base_machine.getU().shape))
        self.jfa_base_machine.updateD(np.random.random_sample(self.jfa_base_machine.getD().shape))

    def train(self, training_data):
        # assert type(training_data) == list
        # assert type(training_data[0]) == list
        # assert type(training_data[0][0]) == GMM_Stats

        # TODO find a better way to do this
        spk_logical = training_data['spk_logical']
        spk_ids = [str(spk_id[0][0]) for spk_id in spk_logical]
        unique_ids = [str(spk_id) for spk_id in np.unique(spk_ids)]
        self.Nacc = training_data['N']
        self.Facc = training_data['F']
        lookup = {}
        for (num, name) in enumerate(unique_ids):
            lookup[name] = num
        spk_ids = [lookup[spk_id] for spk_id in spk_ids]

        n_speakers = len(unique_ids)
        n_sessions = len(spk_ids)
        self.n_speakers = n_speakers


        self.initializeUVD()
        self.initializeXYZ(n_speakers, n_sessions)

        self.precomputeSumStatisticsN(training_data)
        self.precomputeSumStatisticsF(training_data)


        for _ in range(self.n_iter_train):
            self.updateY(training_data, unique_ids, spk_ids)
            self.updateV(training_data)
        self.updateY(training_data)

        for _ in range(self.n_iter_train):
            self.updateX(training_data)
            self.updateU(training_data)
        self.updateX(training_data)

        for _ in range(self.n_iter_train):
            self.updateZ(training_data)
            self.updateD(training_data)

    def updateY(self, gmmStats, unique_ids, spk_ids):
        nr_samples, N1 = self.Nacc.shape
        nr_samples, long_num = self.Facc.shape
        v = self.getV()
        nr_eigen_voices, v1 = v.shape
        assert v1 == long_num
        A = np.zeros((nr_eigen_voices, nr_eigen_voices)) * N1
        C = np.zeros((nr_eigen_voices, long_num))

        dim = self.Facc.shape[1]//self.Nacc.shape[1]
        cols_N = self.Nacc.shape[1]
        index_map = []
        for j in range(cols_N):
            index_map += [j] * dim
        # self.y = self.n_speakers
        self.computeVtΣInv()
        self.computeVProd()
        #for person_id in range(len(self.Nacc)):
        for person_id in unique_ids:
            self.computeIdPlusVProd_i(person_id)
            self.computeFn_y_i(gmmStats, person_id)
            self.updateY_i(person_id)

    def updateV(self, training_data):
        # Initializes the cache accumulator
        self.cache_A1_y.fill(0.)
        nrGaussians = self.jfa_base_machine.getNrUbmGaussians()
        # Loops over all people
        self.cache_A2_y.fill(0.)
        for person_id in range(len(self.Nacc)):
            self.computeIdPlusVProd_i(person_id)
            self.computeFn_y_i(training_data, person_id)

            #Needs to return values to be accumulated for estimating V
            y = self.y[person_id]
            self.PLACEHOLDER_EigenVoicesSquared = self.cache_IdPlusVProd_i + sum([yi * yj for yi, yj in itertools.combinations(y, 2)])
            for gaussian_nr in range(nrGaussians):
                A1_y_c = self.cache_A1_y[gaussian_nr]
                A1_y_c += self.PLACEHOLDER_EigenVoicesSquared * self.Nacc[person_id][gaussian_nr]
            for i in range(y.shape[0]):
                for j in range(y.shape[1]):
                    self.cache_A2_y += self.cache_Fn_y_i[i] * y[j]
        dim = self.jfa_base_machine.getNrUbmInputs()
        V = self.jfa_base_machine.getV()
        for gaussian_nr in range(nrGaussians):
            A1 = self.cache_A1_y[gaussian_nr]
            self.PLACEHOLDER_EigenVoicesSquared = np.linalg.inv(A1)
            slice0, slice1 = gaussian_nr * dim, (gaussian_nr+1) * dim - 1
            A2 = self.cache_A2_y[slice0:slice1]
            V[slice0:slice1] = np.dot(A2, self.PLACEHOLDER_EigenVoicesSquared)
        self.jfa_base_machine.updateV(V)

    def computeVtΣInv(self):
        V = self.jfa_machine.getV()
        Vt = V.transpose(1, 0)
        Σ = self.cache_ubm_var
        self.cache_VtΣInv = []
        for i in range(Vt.shape[0]):
            row = []
            for j in range(Vt.shape[1]):
                row.append(Vt[i, j] / Σ[j])  # Vt * diag(Σ)^-1
            self.cache_VtΣInv.append(row)

        # TODO: proper matrix multiplication
        warnings.warn("this multiplication was not tested")
        self.cache_VtΣInv = np.array(self.cache_VtΣInv)

    def computeVProd(self):
        V = self.jfa_machine.getV()
        Σ = self.cache_ubm_var
        for gaussian_nr in range(self.jfa_base_machine.getNrUbmGaussians()):
            nr_ubm_inputs = self.jfa_base_machine.getNrUbmInputs()
            slice0 = gaussian_nr * nr_ubm_inputs
            slice1 = (gaussian_nr + 1) * (nr_ubm_inputs - 1)
            Vv_c = V[slice0:slice1]
            Vt_c = Vv_c.transpose(1, 0)
            Σ_c = Σ[slice0:slice1]

            for i in range(Vt_c.shape[0]):
                for j in range(Vt_c.shape[1]):
                    self.tmp_rvD = Vt_c[i, j] / Σ_c[j]  # Vt_c * diag(Σ)^-1
            self.cache_VProd[gaussian_nr] = np.dot(Vv_c, self.tmp_rvD)

    def updateY_i(self, person_id):
        self.tmp_rv = np.dot(self.cache_VtΣInv, self.cache_Fn_y_i)
        self.y[person_id] = np.dot(self.cache_IdPlusVProd_i, self.tmp_rv)

    def computeIdPlusVProd_i(self, person_id):
        Ni = self.Nacc[person_id]
        self.PLACEHOLDER_EigenVoicesSquared = np.eye(self.jfa_base_machine.getDimEigenVoices())

        nr_gaussians = self.jfa_base_machine.getNrUbmGaussians()
        for gaussian_nr in range(nr_gaussians):
            VProd_c = self.cache_VProd[gaussian_nr]
            self.PLACEHOLDER_EigenVoicesSquared += VProd_c * Ni[gaussian_nr]
        " l(s) = I + v∗Σ^-1 N(s)v "  # posterior distribution of hidden variables
        " posterior distribution of y(s) conditioned on thheacusting observation of speaker = l^-1(s)v*Σ^-1 F˜(s) and covariance matrix l^-1(s) "
        self.cache_IdPlusVProd_i = np.linalg.inv(self.PLACEHOLDER_EigenVoicesSquared)  # self.cache_IdPlusVProd_i = ( I+Vt*diag(Σ)^-1*Ni*V)^-1")

    def computeFn_y_i(self, training_data, person_id):
        if hasattr(self, "tmp_CD_b"):
            del self.tmp_CD_b
        # Compute Fn_yi = sum_{sessions h}(N_{i,h}*(o_{i,h} - m - D*z_{i} - U*x_{i,h}) (Normalised first order statistics)
        Fi = self.Facc[person_id]
        m = self.cache_ubm_mean
        d = self.jfa_base_machine.getD()
        z = self.z[person_id]

        dimSupervector = self.jfa_base_machine.getDimSupervector()
        self.tmp_CD = repelem(Fi, np.empty(dimSupervector))

        self.cache_Fn_y_i = Fi - self.tmp_CD * (m + d * z)  # Fn_yi = sum_{sessions h}(N_{i,h}*(o_{i,h} - m - D*z_{i})
        self.cache_Fn_y_i = fixShape(self.cache_Fn_y_i)
        X = self.x[person_id]
        U = self.jfa_base_machine.getU()
        for h in range(X.shape[1]):  # Loops over the sessions
            Xh = X[:, h]
            Xh = fixShape(Xh)  # Xh = x_{i,h} (length: dimEigenChannels)
            assert Xh.shape[0] == self.jfa_base_machine.getDimEigenChannels()
            # self.tmp_CD_b = U * Xh  # self.tmp_CD_b = U*x_{i,h}")
            Nih = training_data[person_id][h].n
            self.tmp_CD = fixShape(repelem(Nih, self.tmp_CD))
            b = self.tmp_CD * np.dot(U, Xh)  # N_{i,h} * U * x_{i,h}
            self.cache_Fn_y_i -= b
        # Fn_yi = sum_{sessions h}(N_{i,h}*(o_{i,h} - m - D*z_{i} - U*x_{i,h})

    def initializeXYZ(self, nr_speakers, nr_sessions):

        x = []  # z = [np.empty(1)]
        y = []  # y = [np.empty(1)] 
        z = []  # x = [np.empty((1,1))] 

        dimEigenChannels = self.jfa_base_machine.getDimEigenChannels()
        dimEigenVoices = self.jfa_base_machine.getDimEigenVoices()
        dimSupervector = self.jfa_base_machine.getDimSupervector()

        for session in range(nr_speakers):
            z.append(np.zeros(dimSupervector))
            y.append(np.zeros(dimEigenVoices))
            x.append(np.zeros((dimEigenChannels, nr_sessions)))
        self.setSpeakerFactors(x, y, z)

    def updateX(self, gmm_stats):
        # Precomputation
        self.computeUtΣInv()
        self.computeUProd()
        for person_id in range(len(gmm_stats)):
            n_session_i = self.x[person_id].shape[1]
            for session_id in range(n_session_i):
                self.computeIdPlusUProd_ih(gmm_stats, person_id, session_id)
                self.computeFn_x_ih(gmm_stats, person_id, session_id)
                self.updateX_ih(person_id, session_id)

    def updateU(self, gmm_stats):
        #// Initializes the cache accumulator
        self.cache_A1_x.fill(0.)
        self.cache_A2_x.fill(0.)
        # Loops over all people
        for person_id in range(len(gmm_stats)):
            n_session_i = self.x[person_id].shape[1]
            for h in range(n_session_i):
                self.computeIdPlusUProd_ih(gmm_stats, person_id, h)
                self.computeFn_x_ih(gmm_stats,person_id, h)

                # Needs to return values to be accumulated for estimating U
                x = fixShape(self.x[person_id][:, h])
                self.tmp_ruru = self.cache_IdPlusUProd_ih
                for i in range(x.shape[0]):
                    for j in range(x.shape[1]):
                        self.tmp_ruru += x[i] * x[j]
                for c in range(self.jfa_base_machine.getNrUbmGaussians()):
                    A1_x_c = self.cache_A1_x[c]
                    A1_x_c += self.tmp_ruru * gmm_stats[person_id][h].n[c]
                for i in range(self.cache_Fn_x_ih.shape[0]):
                    for j in range(x.shape[1]):
                        self.cache_A2_x += self.cache_Fn_x_ih[i] * x[j]
        dim = self.jfa_base_machine.getNrUbmInputs()
        U = self.jfa_base_machine.getU()
        for c in range(self.jfa_base_machine.getNrUbmGaussians()):
            A1 = self.cache_A1_x[c]
            self.tmp_ruru = np.linalg.inv(A1)
            slice0 = c * dim
            slice1 = (c + 1) * dim - 1
            A2 = self.cache_A2_x[slice0:slice1]
            slice0 = c * dim
            slice1 = (c + 1) * dim - 1
            U[slice0:slice1] = A2 * self.tmp_ruru
        self.jfa_base_machine.updateU(U)

    def computeIdPlusUProd_ih(self, gmm_stats, person_id, h):
        Nih = gmm_stats[person_id][h].n
        self.tmp_ruru = np.eye(self.jfa_base_machine.getDimEigenChannels())
        for gaussian_nr in range(self.jfa_base_machine.getNrUbmGaussians()):
            UProd_c = self.cache_UProd[gaussian_nr]
            self.tmp_ruru += UProd_c * Nih[gaussian_nr]
        self.cache_IdPlusUProd_ih = np.linalg.inv(self.tmp_ruru)

    def computeFn_x_ih(self, gmm_stats, person_id, session_id):
        if hasattr(self, "tmp_CD_b"):
            del self.tmp_CD_b
        Fih = gmm_stats[person_id][session_id].sum_Px
        m = self.cache_ubm_mean
        d = self.jfa_base_machine.getD()
        z = self.z[person_id]
        Nih = gmm_stats[person_id][session_id].n
        self.tmp_CD = fixShape(repelem(Nih, self.tmp_CD))
        for c in range(self.jfa_base_machine.getNrUbmGaussians()):
            slice0 = c * self.jfa_base_machine.getNrUbmInputs()
            slice1 = (c + 1) * self.jfa_base_machine.getNrUbmInputs()
            self.cache_Fn_x_ih[slice0:slice1] = fixShape(Fih[c])
        print("CHECK NEXT")
        self.tmp_CD_b = (m + d * z)
        self.tmp_CD_b = fixShape(self.tmp_CD_b)
        self.cache_Fn_x_ih = fixShape(self.cache_Fn_x_ih)
        self.cache_Fn_x_ih -= self.tmp_CD * self.tmp_CD_b   # Fn_x_ih = N_{i,h}*(o_{i,h} - m - D*z_{i})

        y = self.y[person_id]
        V = self.jfa_base_machine.getV()
        self.tmp_CD_b = np.dot(V, y)
        self.cache_Fn_x_ih -= self.tmp_CD * self.tmp_CD_b

    def updateZ(self, gmm_stats):
        # Precomputation
        self.computeDtΣInv()
        self.computeDProd()
        # Loops over all people
        for person_id in range(len(self.Nacc)):
            self.computeIdPlusDProd_i(person_id)
            self.computeFn_z_i(gmm_stats, person_id)
            self.updateZ_i(person_id)

    def updateD(self, gmm_stats):
        # Initializes the cache accumulator
        self.cache_A1_z = 0.
        self.cache_A2_z = 0.
        for person_id in range(len(self.Nacc)):
            self.computeIdPlusDProd_i(person_id)
            self.computeFn_z_i(gmm_stats, person_id)

            # Needs to return values to be accumulated for estimating D
            z = self.z[person_id]
            self.tmp_CD = repelem(self.Nacc[person_id], self.tmp_CD)
            self.cache_A1_z += (self.cache_IdPlusDProd_i + z * z) * self.tmp_CD
            self.cache_A2_z += self.cache_Fn_z_i * z

        d = self.cache_A2_z / self.cache_A1_z
        self.jfa_base_machine.updateD(d)

    def setSpeakerFactors(self, x, y, z):
        # len x and y is nr of people
        assert len(y) == self.n_speakers or len(z) == self.n_speakers

        NrEigenChannels = self.jfa_base_machine.getDimEigenChannels()
        NrEigenVoices = self.jfa_base_machine.getDimEigenVoices()
        supervectorDim = self.jfa_base_machine.getDimSupervector()

        for xi in x:
            assert len(xi.shape) == 2
            assert xi.shape[0] == NrEigenChannels
        for (yi, zi) in zip(y, z):
            assert len(yi.shape) == 1
            assert len(zi.shape) == 1
            assert yi.shape[0] == NrEigenVoices
            assert zi.shape[0] == supervectorDim

        self.x = x
        self.y = y
        self.z = z

    def computeUtΣInv(self):
        U = self.jfa_base_machine.getU()
        Ut = U.transpose(1,0)
        σ = self.cache_ubm_var
        for i in range(Ut.shape[0]):
            for j in range(Ut.shape[1]):
                self.cache_UtΣInv = Ut[i,j] / σ[j]; # Ut * diag(sigma)^-1

    def computeUProd(self):
        U = self.jfa_base_machine.getU()
        σ = self.cache_ubm_var
        nr_ubm_inputs = self.jfa_base_machine.getNrUbmInputs()
        for gaussian_nr in range(self.jfa_base_machine.getNrUbmGaussians()):
            # UProd_c = self.cache_UProd[gaussian_nr]
            slice0 = gaussian_nr * nr_ubm_inputs
            slice1 = (gaussian_nr + 1) * nr_ubm_inputs - 1
            Uu_c = U[slice0:slice1]
            Ut_c = Uu_c.transpose(1, 0)
            σ_c = σ[slice0:slice1]
            for i in range(Ut_c.shape[0]):
                for j in range(Ut_c.shape[1]):
                    self.tmp_ruD = Ut_c[i, j] / σ_c[j]  # Ut_c * diag(sigma)^-1
            self.cache_UProd[gaussian_nr] = np.dot(self.tmp_ruD, Uu_c)

    def updateX_ih(self, person_id, session_id):
        # Computes xih = Axih * Cus * Fn_x_ih
        old_x = self.x[person_id][:, session_id]
        # self.tmp_ru = self.cache_UtSigmaInv * self.cache_Fn_x_ih =
        # = Ut*diag(sigma)^-1 * N_{i,h}*(o_{i,h} - m - D*z_{i} - V*y_{i})
        self.tmp_ru = np.dot(self.cache_UtSigmaInv, self.cache_Fn_x_ih)
        x = np.dot(self.cache_IdPlusUProd_ih, self.tmp_ru)
        assert fixShape(old_x).shape == x.shape
        x = unfixShape(x)
        warnings.warn("remove ugly hack")
        self.x[person_id][:, session_id] = x

    def computeIdPlusDProd_i(self, person_id):
        Ni = self.Nacc[person_id]
        self.tmp_CD = repelem(Ni, self.tmp_CD) # m_tmp_CD = Ni 'repmat'
        self.cache_IdPlusDProd_i = 1.; # self.cache_IdPlusDProd_i = Id
        self.cache_IdPlusDProd_i += self.cache_DProd * self.tmp_CD; # self.cache_IdPlusDProd_i = I+Dt*diag(sigma)^-1*Ni*D
        self.cache_IdPlusDProd_i = 1 / self.cache_IdPlusDProd_i; # self.cache_IdPlusVProd_i = (I+Dt*diag(sigma)^-1*Ni*D)^-1

    def computeFn_z_i(self, gmm_stats, person_id):
        Fi = self.Facc[person_id]
        m = self.cache_ubm_mean
        V = self.jfa_machine.getV()
        y = self.y[person_id]
        self.tmp_CD = repelem(self.Nacc[person_id], self.tmp_CD)
        self.tmp_CD_b = np.dot(V, y)
        self.cache_Fn_z_i = Fi - self.tmp_CD * (m + self.tmp_CD_b)  # Fn_yi = suself.{sessions h}(N_{i,h}*(o_{i,h} - m - V*y_{i})

        X = self.x[person_id]
        U = self.jfa_machine.getU()
        for session_id in range(X.shape[0]):
            Nh = gmm_stats[person_id][session_id].n # Nh = N_{i,h} (length: C)
            self.tmp_CD = repelem(Nh, self.tmp_CD)
            Xh = X[:, session_id] # Xh = x_{i,h} (length: dimEigenChannels)
            self.tmp_CD_b = np.dot(U, Xh)
            self.cache_Fn_z_i -= self.tmp_CD * self.tmp_CD_b
            # Fn_z_i = suself.{sessions h}(N_{i,h}*(o_{i,h} - m - V*y_{i} - U*x_{i,h})

    def computeDtΣInv(self):
        d = self.jfa_base_machine.getD()
        σ = self.cache_ubm_var
        self.cache_DtSigmaInv = d / σ  # Dt * diag(sigma)^-1

    def computeDProd(self):
        d = self.jfa_base_machine.getD()
        σ = self.cache_ubm_var
        self.cache_DProd = d / σ * d  # Dt * diag(sigma)^-1 * D

    def updateZ_i(self, person_id):
        # m_tmp_CD = m_cache_DtSigmaInv * m_cache_Fn_z_i =
        # = Dt*diag(sigma)^-1 * sum_{sessions h}(N_{i,h}*(o_{i,h} - m - V*y_{i} - U*x_{i,h})
        self.z[person_id] = self.cache_IdPlusDProd_i * self.cache_DtSigmaInv * self.cache_Fn_z_i


def collect_suf_stats(data, m, v, w):
    nr_mixtures = len(w)
    dim = len(m)

    gammas = gaussian_posteriors(data, m, v, w)
    N = np.sum(gammas, 2)
    F = data * gammas
    F = np.reshape(F, nr_mixtures * dim, 1)
    return N, F


def gaussian_posteriors(data, m, v, w):
    nr_mixtures = len(w)
    nr_frame = len(data[0])
    dim = len(data)

    g = sklearn.mixture.GMM(nr_mixtures)
    g.fit(data)

    gammas = np.empty((nr_mixtures,1))
    for ii in range(nr_mixtures):
        gammas[ii] = 1 # gaussian_function(data, a[ii], m[:, ii], v[:, ii])

    # normalize
    gammas /= sum(gammas)
    return gammas


def file_to_list(file_name):
    with open(file_name) as data_file:
        lst = [row.split() for row in data_file.read().split("\n")]
    # return lst[:-2]
    return np.loadtxt(file_name)


def parse_list(file_name):
    with open(file_name) as recording:
        lst = recording.read().split()
    logical, physical = zip(*[str.split(row, "=") for row in lst])
    return logical, physical


def collect_suf_stats(data, m, v, w):
    nr_mixtures = len(w)
    dim = len(m)

    gammas = gaussian_posteriors(data, m, v, w)

    # zero order stats for each gaussian are just
    # the sum of the posteriors (soft counts)
    N = sum(gammas, 2)  # TODO: along the second dimension

    # first order stats is jsut a posterior weighted sum
    F = data * gammas
    np.reshape(F, (nr_mixtures * dim, 1))
    return N, F


def extract_features(recording_files, nr_ceps=12):
    print("skipping features")
    return  Ceps()(range(100))
    nr_utt_in_ubm = 300

    win_length_ms = 25  # The window length of the cepstral analysis in milliseconds
    win_shift_ms = 10  # The window shift of the cepstral analysis in milliseconds
    nr_filters = 24  # NOTSURE The number of filter bands
    nr_ceps = nr_ceps  # The number of cepstral coefficients
    f_min = 0.  # NOTSURE The minimal frequency of the filter bank
    f_max = 4000.  # NOTSURE The maximal frequency of the filter bank
    delta_win = 2  # NOTSURE The integer delta value used for computing the first and second order derivatives
    pre_emphasis_coef = 0.97  # NOTSURE The coefficient used for the pre-emphasis
    dct_norm = True  # NOTSURE A factor by which the cepstral coefficients are multiplied
    mel_scale = True  # Tell whether cepstral features are extracted on a linear (LFCC) or Mel (MFCC) scale
    # TODO add feature wrapping

    if glob.has_magic(recording_files):
        recording_files = glob.glob(recording_files)

    rate, ubm_wav = wavfile.read(recording_files.pop())
    for recording_file in recording_files:
        rate, signal = wavfile.read(recording_file)
        ubm_wav = np.append(ubm_wav, signal)
    c = Ceps(rate, win_length_ms, win_shift_ms, nr_filters, nr_ceps, f_min, f_max, delta_win, pre_emphasis_coef,
                    mel_scale, dct_norm)
    ubm_wav = np.cast['float'](ubm_wav)  # vector should be in **float**
    mfcc = c(ubm_wav)
    return mfcc


def train_ubm(nr_mixtures, features):
    gmm = sklearn.mixture.GMM(nr_mixtures)
    # TODO should use Baum-Welch algorithm?
    gmm.fit(features)

    gmm_machine = GMM_Machine(gmm.n_components, len(features))
    return gmm_machine


class JFA:
    def __init__(self, ubm, ru, rv, n_iter_train, n_iter_enrol):
        """
            UBM The Universal Backgroud Model
            dimEigenChannels  size of U (CD x dimEigenChannels)
            dimEigenVoices  size of V (CD x dimEigenVoices)
        """
        self.training_iterations = n_iter_train
        self.enroll_iterations = n_iter_enrol
        jfa_base = JFA_Base()
        jfa_machine = JFA_Machine(jfa_base)
        self.jfa_trainer = JFA_Trainer(jfa_machine, jfa_base_trainer)

        if ubm is None:
            self.ubm = self.train_ubm()
        else:
            self.ubm = ubm

        self.ru = ru
        self.rv = rv
        dimCD = self.ubm.getNInputs() * self.ubm.getNGaussians()
        self.U = np.zeros((dimCD, ru))
        self.V = np.zeros((dimCD, rv))
        self.d = np.zeros((1, dimCD))
        self.cache_Fn_y_i = None

    def train_ubm(self, nr_mixtures, features):
        gmm = sklearn.mixture.GMM(nr_mixtures)
        # TODO should use Baum-Welch algorithm?
        gmm.fit(features)
        self.gmm = gmm
        return gmm

    def train_enroler(self, train_files, enroler_file):
        self.jfa_base = jfa_statsHolder(self.ubm, self.U, self.V)

        gmm_stats = []
        print("skipping GMMS")
        #for k in l_files:
        #  stats = GMMStats()
        #  gmm_stats.append(stats)
        trainer = jfa_trainer(self.training_iterations)
        trainer.train(gmm_stats)

    def enroll(self, student_gmm):
        self.trainer.enroll(self.jfa_machine, student_gmm, self.enroll_iterations)

    def score(self, model, data):
        return model.forward(data)


def estimate_x_and_u(F, N, m, E, d, v, u, z, y, x, spk_ids):
    """
     F - matrix of first order statistics (not centered). The rows correspond
         to training segments. Number of columns is given by the supervector
         dimensionality. The first n collums correspond to the n dimensions
         of the first Gaussian component, the second n collums to second·
         component, and so on.
     N - matrix of zero order statistics (occupation counts of Gaussian
         components). The rows correspond to training segments. The collums
         correspond to Gaussian components.
     S - NOT USED by this function; reserved for second order statistics
     m - speaker and channel independent mean supervector (e.g. concatenated
         UBM mean vectors)
     E - speaker and channel independent variance supervector (e.g. concatenated
         UBM variance vectors)
     d - Row vector that is the diagonal from the diagonal matrix describing the
         remaining speaker variability (not described by eigenvoices). Number of
         columns is given by the supervector dimensionality.
     v - The rows of matrix v are 'eigenvoices'. (The number of rows must be the
         same as the number of columns of matrix y). Number of columns is given
         by the supervector dimensionality.
     u - The rows of matrix u are 'eigenchannels'. (The number of rows must be
         the same as the number of columns of matrix x) Number of columns is
         given by the supervector dimensionality.
     y - matrix of speaker factors corresponding to eigenvoices. The rows
         correspond to speakers (values in vector spk_ids are the indices of the
         rows, therfore the number of the rows must be (at least) the highest
         value in spk_ids). The columns correspond to eigenvoices (The number
         of columns must the same as the number of rows of matrix v).
     z - matrix of speaker factors corresponding to matrix d. The rows
         correspond to speakers (values in vector spk_ids are the indices of the
         rows, therfore the number of the rows must be (at least) the highest
         value in spk_ids). Number of columns is given by the supervector·
         dimensionality.
     x - NOT USED by this function; used by other JFA function as
         matrix of channel factors. The rows correspond to training
         segments. The columns correspond to eigenchannels (The number of columns
         must be the same as the number of rows of matrix u)
     spk_ids - column vector with rows corresponding to training segments and
         integer values identifying a speaker. Rows having same values identifies
         segments spoken by same speakers. The values are indices of rows in
         y and z matrices containing corresponding speaker factors.
    """
    T, C = N.shape
    T, CD = F.shape

    assert CD % C == 0
    assert N.shape[0] == F.shape[0] == spk_ids.shape[0] == x.shape[0]
    assert m.shape[0] == F.shape[1] == E.shape[0] == d.shape[0] == v.shape[1] == u.shape[1] == z.shape[1]
    D = CD / C

    ru, _ = v.shape
    rv, _ = u.shape
    assert y.shape[1] == rv and x.shape[1] == ru
    assert y.shape[0] == z.shape[0]
    Nspk, _ = z.shape

    uEut = np.empty((C, ru, rv))
    # blitz::firstIndex i;
    # blitz::secondIndex j;
    tmp1 = np.empty((ru, D))

    # for(int c=0; c<C; ++c)
    for c in range(C):

        #   blitz::Array<double,2> uEuT_c = uEuT(c, blitz::Range::all(), blitz::Range::all());
        uEut_c = uEut[c]
    #   blitz::Array<double,2> u_elements = u(blitz::Range::all(), blitz::Range(c*D, (c+1)*D-1));
        u_elements = u[:, c*D: (c+1)*D-1 ]
    #   blitz::Array<double,1> e_elements = E(blitz::Range(c*D, (c+1)*D-1));
        e_elements = E[c*D: (c+1)*D-1]

    #   tmp1 = u_elements(i,j) / e_elements(j);
        for i in range(C):
            for j in range(ru):
                tmp1 = u_elements[i, j] / e_elements[j]
        # blitz::Array<double,2> u_transposed = u_elements.transpose(1,0);
        u_transposed = u_elements.transpose(1, 0)
        #  math::prod(tmp1, u_transposed, uEuT_c);
        np.prod(tmp1, u_transposed, uEut_c)
    #}

    # // 3/ Determine the vector of speaker ids
    # //    Assume that samples from the same speaker are consecutive

    # std::vector<uint32_t> ids;
    ids = np.vector(type=np.uint32_t)
    last_elem = 0
    for ind in range(T):
        if ids.empty():
            ids.push_back(spk_ids(ind))
            last_elem = spk_ids(ind)
        elif last_elem != spk_ids(ind):
            ids.push_back(spk_ids(ind))
            last_elem = spk_ids(ind)

    # // 4/ Main computation
    # // Allocate working arrays
    spk_shift = np.empty(CD, dtype='double')
    tmp2 = np.empty(CD, dtype='double')
    tmp3 = np.empty(ru, dtype='double')
    Fh = np.empty(CD, dtype='double')
    L = np.empty((ru, ru), dtype='double')
    Linv = np.empty((ru, ru), dtype='double')

    # std::vector<uint32_t>::iterator it;
    it = iter
    cur_start_ind = 0
    # int cur_end_ind;
    pass
  # for(it=ids.begin(); it<ids.end(); ++it)
    for id in enumerate(ids):
        # // a/ Determine speaker sessions
  #   uint32_t cur_elem=*it;
        cur_elem = id(iter)
        cur_start_ind = 0
        while spk_ids(cur_start_ind) != cur_elem:
            cur_start_ind += 1
            cur_end_ind = cur_start_ind
            for ind in range(cur_start_ind+1, T):
                cur_end_ind = ind
                if spk_ids(ind) != cur_elem:
                    cur_end_ind = ind - 1
                    break
        #   // b/ Compute speaker shift
        spk_shift = m
        y_ii = y[cur_elem]
        math.prod(y_ii, v, tmp2)
        spk_shift += tmp2
        z_ii = z[cur_elem]
        spk_shift += z_ii * d

        # // c/ Loop over speaker session
        #   for(int jj=cur_start_ind; jj<=cur_end_ind; ++jj)
        for jj in range(cur_start_ind, cur_end_ind):
            #     blitz::Array<double,1> Nhint = N(jj, blitz::Range::all());
            Nhint = N[jj]
            core.repelem(Nhint, tmp2)
            Fh = F[jj] - tmp2 * spk_shift
            # // L=Identity
            L.set_everything_to(0.)
            for k in range(ru):
                L[k, k] = 1.

            for c in range(C):
                uEuT_c = uEut[c]
                L += uEuT_c * N[jj, c]

            # // inverse L
            np.invert(L, Linv)

            # // update x
            x_jj = x[jj]
            Fh /= E
            print ("WTF IS the next line")
            uu = u[:, :]
            for i in range(10): print ("WTF happened")
            raise Exception("WTFFF")
            u_t = uu.transpose(1, 0)
            tmp3 = Fh * u_t
            x_jj = tmp3 * Linv
    return x, u


def main():
    nr_ceps = 12
    features = extract_features("data/ubm/*", nr_ceps=nr_ceps)
    dataset_size, speaker_factors = features.shape
    nr_mixtures = speaker_factors

    # C = nr_mixtures
    # F = dataset_size
    # Rc = channel_factors
    # Rc = 10
    # Rs = speaker_factors
    # CF = C * F

    # m = np.empty((CF, 1))
    # u = np.empty((CF, Rc))
    # v = np.empty((CF, Rs))
    # d = scipy.sparse.dia_matrix((CF, CF))
    # Σ = scipy.sparse.dia_matrix((CF, CF))
    # Lambda = (m.u, v, d, Σ)

    ubm_machine = train_ubm(nr_mixtures=2, features=features)
    # cheatcodes.plot_gmms([ubm], [features])

    # x and Ux
    classifier = JFA(ubm_machine, 2, 2, 1, 1)
    classifier.train_enroler("du", "ud")

    eps = 1e-4
    F1 = np.array( [0.3833, 0.4516, 0.6173, 0.2277, 0.5755, 0.8044, 0.5301, 0.9861, 0.2751, 0.0300, 0.2486, 0.5357]).reshape((6,2))
    F2 = np.array( [0.0871, 0.6838, 0.8021, 0.7837, 0.9891, 0.5341, 0.0669, 0.8854, 0.9394, 0.8990, 0.0182, 0.6259]).reshape((6,2))
    F = [F1, F2]


    N1 = np.array([0.1379, 0.1821, 0.2178, 0.0418]).reshape((2,2))
    N2 = np.array([0.1069, 0.9397, 0.6164, 0.3545]).reshape((2,2))
    N = [N1, N2]

    m = np.array([0.1806, 0.0451, 0.7232, 0.3474, 0.6606, 0.3839])
    E = np.array([0.6273, 0.0216, 0.9106, 0.8006, 0.7458, 0.8131])
    d = np.array([0.4106, 0.9843, 0.9456, 0.6766, 0.9883, 0.7668])
    v = np.array([
        [0.3367, 0.6624, 0.2442, 0.2955, 0.6802, 0.5278],
        [0.4116, 0.6026, 0.7505, 0.5835, 0.5518, 0.5836]
    ])
    u = np.array([
        [0.5118, 0.0826, 0.7196, 0.9962, 0.3545, 0.9713],
        [0.3464, 0.8865, 0.4547, 0.4134, 0.2177, 0.1257]
    ])
    z = np.array([
        [0.3089, 0.7261, 0.7829, 0.6938, 0.0098, 0.8432],
        [0.9223, 0.7710, 0.0427, 0.3782, 0.7043, 0.7295]
    ])
    y = np.array([0.2243, 0.2691, 0.6730, 0.4775])
    x = np.array([
        [0.9976, 0.1375],
        [0.8116, 0.3900],
        [0.4857, 0.9274],
        [0.8944, 0.9175]
    ])
    spk_ids = np.array([0, 0, 1, 1])

    F1 = np.array( [0.3833, 0.4516, 0.6173, 0.2277, 0.5755, 0.8044, 0.5301, 0.9861, 0.2751, 0.0300, 0.2486, 0.5357]).reshape((6,2))
    F2 = np.array( [0.0871, 0.6838, 0.8021, 0.7837, 0.9891, 0.5341, 0.0669, 0.8854, 0.9394, 0.8990, 0.0182, 0.6259]).reshape((6,2))
    F=[F1, F2]

    N1 = np.array([0.1379, 0.1821, 0.2178, 0.0418]).reshape((2,2))
    N2 = np.array([0.1069, 0.9397, 0.6164, 0.3545]).reshape((2,2))
    N=[N1, N2]

    gs11 = statsHolder(2,3) # 2 gausians, 3 input
    gs11.n = N1[:,0]
    gs11.sum_Px = F1[:,0].reshape(2,3)
    gs12 = statsHolder(2,3) # 2 gausians, 3 input
    gs12.n = N1[:,1]
    gs12.sum_Px = F1[:,1].reshape(2,3)

    gs21 = statsHolder(2,3) # 2 gausians, 3 input
    gs21.n = N2[:,0]
    gs21.sum_Px = F2[:,0].reshape(2,3)
    gs22 = statsHolder(2,3) # 2 gausians, 3 input
    gs22.n = N2[:,1]
    gs22.sum_Px = F2[:,1].reshape(2,3)

    TRAINING_STATS = [[gs11, gs12], [gs21, gs22]]

    # u = estimate_x_and_u(F[0], N[0], m, E, d, v, u, z, y, x, spk_ids)
    # print(u)

    # more_data = glob.glob("data/more_data/*")
    # pca_components = 2
    # pca = PCA(n_components=pca_components)
    # database = [extract_features(recording) for recording in more_data]
    # pca.fit(database)

    models = {}
    speaker_names = os.listdir("data/")
    for speaker_name in speaker_names:
        session_files = glob.glob("data/{}/*".format(speaker_name))
        models[speaker_name] = {}
        for i, session_file in enumerate(session_files):
            speaker_session_features = extract_features(session_file)
            speaker_session_gmm = train_ubm(2, features)
            models[speaker_name][i] = speaker_gmm
    print("done")

    dim = len(m) / nr_mixtures

    # dim is the real width but the data is stored as (n,1)
    m = np.reshape(m, (dim, nr_mixtures))
    v = np.reshape(m, (dim, nr_mixtures))

    datasets = ['enroll_stats', 'fa_train_eigenchannels_stats', 'fa_train_eigenvoices_stats', 'test_stats']

    for dataset in datasets:
        list_file = "{}/{}.lst".format(LISTS_PATH, dataset)
        spk_logical, spk_physical = parse_list(list_file)
        nr_sessions = len(spk_logical)

        N = np.empty((nr_sessions, nr_mixtures))
        F = np.empty((nr_sessions, nr_mixtures))

        for i in range(len(spk_physical)):
            session_file = "{}/{}.ascii".format(JFA_PATH, spk_physical[i])
            data = file_to_list(session_file)
            with cheatcodes.Timer('collect_suf_stats'):
                Ni, Fi = collect_suf_stats(data, m, v, w);
            N[i] = Ni
            F[i] = Fi
        out_stats_file = "data/stats/{}.mat".format(dataset)
        print("saving to", out_stats_file)
        pickle.dump({
            'N': N,
            'F': F,
            'spk_logical': spk_logical,
        },
            open(out_stats_file, "w")
        )


"""
# follow http://www1.icsi.berkeley.edu/Speech/presentations/AFRL_ICSI_visit2_JFA_tutorial_icsitalk.pdf

# 2/ GMM Training
    nr_gaussians = 256
    iterk = 25
    iterg_train = 25
    end_acc = 0.0001
    var_thd = 0.0001
    update_weights = True
    update_means = True
    update_variances = True
    norm_KMeans = True

    for gaussianr_n in nr_gaussians:
        print "training", gaussianr_n
        print "merging", gaussianr_n
        C = np.array(nr_gaussians)
        F = np.array(acusitc_features_vector_size)

        m = GMM(M, D, features)


# 3/ JFA Training

#use:
# Patrick Kenny. “Joint factor analysis of speaker and session variability: Theory and algorithms”. In: CRIM,
# Montreal,(Report) CRIM-06/08-13 (2005).

#or better:

# Patrick Kenny et al. “A study of interspeaker variability in speaker verification”. In: Audio, Speech, and
# Language Processing, IEEE Transactions on 16.5 (2008), pp. 980–988.

#or read ../jfa_cookbook/*


    dimEigenChannels = 50 # The dimensionality of the subspace
    relevance_factor = 4
    nr_iter_train = 10
    nr_iter_enrol = 1

    for M in Ms:
        print "M", M
        for speaker in speakers
            print "speaker", speaker
            features = read_data(nr_samples, speaker, session)
            D = nr_samples * nr_something_else
            gmm = GMM(M, D, features)
            gmm.train(features)



# 4/ JFA Enrolment and scoring
    iterg_enrol = 1
    convergence_threshold = 0.0001
    variance_threshold = 0.0001
    relevance_factor = 4
    responsibilities_threshold = 0
"""


def repmat(src, dst):
    m = dst.shape[0] / src.shape[0]
    n = dst.shape[1] / src.shape[1]
    for i in range(m):
        for j in range(n):
            slice00 = src.shape[0] * i
            slice01 = src.shape[0] * (i + 1) - 1
            slice10 = src.shape[1] * j
            slice11 = src.shape[1] * (j + 1) - 1
            dst_mn = dst[slice00:slice01, slice10:slice11]
        dst_mn = src


def repelem(src, dst):
    # Function which replicates the elements of an input 1D array,
    # generating a new (larger) 1D array. 
    if (dst.shape[0] % src.shape[0]) != 0:
        raise Exception("dst.shape[0] ({}) is not a multiple of src.shape[0] ({})".format(dst.shape[0], src.shape[0]))
    warnings.warn("move to cheatcodes")
    warnings.warn("check")
    result = np.repeat(src, dst.shape[0]//src.shape[0])
    return result

def fixShape(arr):
    if len(arr.shape) > 1:
        return arr
    r, = arr.shape
    newShape = (r, 1)
    return np.reshape(arr, newShape)

def unfixShape(arr):
    warnings.warn("remove ugly hack")
    if len(arr.shape) == 1:
        return arr
    r, c = arr.shape
    assert c == 1
    newShape = (r,)
    return np.reshape(arr, newShape)

if "__main__" == __name__:
    main()
else:
    print("imported, quitting")
