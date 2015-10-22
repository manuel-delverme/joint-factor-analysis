# -*- coding: UTF-8 -*-
# import bob
from cheatcodes import Timer, plot_gmms
from sklearn  import mixture
import scipy.io.wavfile as wavfile
from scipy.decomposition import PCA
import glob
try:
    import bob.ap
except ImportError:
    print "bob not installed"
import re
import math
import numpy as np

JFA_PATH = "../jfa_cookbook"
MODELS_PATH = JFA_PATH + "/models"
LISTS_PATH = JFA_PATH + "/lists"

def collect_suf_stats(data, m, v, w):
    nr_mixtures = len(w)
    dim = len(m)

    gammas = gaussianr_posteriors(data, m, v, w)
    N = np.sum(gammas, 2)
    F = data * gammas
    F = np.reshape(F, nr_mixtures * dim, 1)

def gaussianr_posteriors(data, m, v, w):
    nr_mixtures = len(w)
    nr_frame = len(data[0])
    dim = len(data)

    g = mixture.GMM(nr_mixtures)
    g.fit(data)

    for ii in range(nr_mixtures):
        gammas[i] = gaussianr_function(data, a[ii], m[:, ii], v[:, ii])

    #normalize
    gammas /= sum(gammas)

def file_to_list(file_name):
    with open(file_name) as data_file:
        lst = [ row.split() for row in data_file.read().split("\n")]
    #return lst[:-2]
    return np.loadtxt(file_name)

def parse_list(file_name):
    with open(file_name) as recording:
        lst = recording.read().split()
    logical, physical = zip(*[ str.split(row, "=") for row in lst ])
    return logical, physical

def collect_suf_stats(data, m, v, w):
    nr_mixtures = len(w)
    dim = len(m)
    
    gammas = gaussianr_posteriors(data, m, v, w)

    # zero order stats for each gaussian are just
    # the sum of the posteriors (soft counts)
    N = sum(gammas,2) # TODO: along the second dimention

    # first order stats is jsut a posterior weighted sum
    F = data * gammas
    np.reshape(F,(nr_mixtures * dim, 1))
    return N, F

def train_ubm(nr_mixtures, features):
    gmm = mixture.GMM(nr_mixtures)
    #TODO should use Baum-Welch algorithm?
    gmm.fit(features)
    return gmm

def train_pca(database_of_speakers):


def extract_features(recordings_folder):
    nr_utt_inr_ubm = 300
    file_path = glob.glob(recordings_folder)[0]
    rate, signal = wavfile.read(file_path)

    winr_length_ms = 25 # The window length of the cepstral analysis in milliseconds
    winr_shift_ms = 10 # The window shift of the cepstral analysis in milliseconds
    nr_filters = 24 #NOTSURE The number of filter bands
    nr_ceps = 12 # The number of cepstral coefficients
    f_min = 0. #NOTSURE The minimal frequency of the filter bank
    f_max = 4000. #NOTSURE The maximal frequency of the filter bank
    delta_win = 2 #NOTSURE The integer delta value used for computing the first and second order derivatives
    pre_emphasis_coef = 0.97 #NOTSURE The coefficient used for the pre-emphasis
    dct_norm = True #NOTSURE A factor by which the cepstral coefficients are multiplied
    mel_scale = True # Tell whether cepstral features are extracted on a linear (LFCC) or Mel (MFCC) scale
    #TODO add feature wrapping

    mfcc = np.random.random((12,len(signal)))
    # bob.ap.Ceps(rate, winr_length_ms, winr_shift_ms, nr_filters, nr_ceps, f_min, f_max, delta_win, pre_emphasis_coef, mel_scale, dct_norm)
    # signal = np.cast['float'](signal) # vector should be in **float**
    # mfcc = c(signal)
    return mfcc


def main():
    features = extract_features("data/ubm/*")
    speaker_factors , dataset_size = features.shape
    nr_mixtures = speakers_factors

    #TODO check C,F
    C = nr_mixtures
    F = nr_mixtures * dataset_size
    Rc = channel_factors
    Rs = speakers_factors
    CF = C*F

    m = np.empty((CF,1))
    u = np.empty((CF,Rc))
    v = np.empty((CF,Rs))
    d = np.sparse.diag_matrix((CF,CF))
    Sigma = np.sparse.diag_matrix((CF,CF))
    Lambda = (m. u, v, d, Sigma)

    gmm = train_ubm(nr_mixtures = 2, features=features)
    plot_gmms([gmm], [features])

    more_data = glob.glob("data/more_data/*")
    pca_components = 2
    pca = PCA(n_components=pca_components)
    database = [extract_features(recording) for recording in more_data]
    pca.fit(database)

    #fit PCA

    models = {}
    speaker_names = os.listdir("data/")
    for speaker_name in speaker_names:
        session_files = glob.glob("data/{}/*".format(speaker_name))
        models[speaker_name] = {}
        for i, session_file in enumerate(session_files):
            speaker_session_features = extract_features(session_file)
            speaker_session_gmm = train_ubm(2, features)
            models[speaker_name][i] = speaker_gmm
    print "done"

    dim = len(m) / nr_mixtures

    # dim is the real width but the data is stored as (n,1)
    m = np.reshape(m, (dim, nr_mixtures))
    v = np.reshape(m, (dim, nr_mixtures))

    datasets = []
    datasets.append('enroll_stats')
    datasets.append('fa_train_eigenchannels_stats')
    datasets.append('fa_train_eigenvoices_stats')
    datasets.append('test_stats')

    for dataset in datasets:
        list_file = "{}/{}.lst".format(LISTS_PATH, dataset)
        spk_logical, spk_physical = parse_list(list_file)
        nr_sessions = len(spk_logical)

        N = np.empty((nr_sessions, nr_mixtures))
        F = np.empty((nr_sessions, nr_mixtures))

        for i in range(len(spk_physical)):
            session_file = "{}/{}.ascii".format(JFA_PATH, spk_physical[i])
            data = file_to_list(session_file)
            with Timer('collect_suf_stats'):
                Ni, Fi = collect_suf_stats(data, m, v, w);
            N[i] = Ni
            F[i] = Fi
        out_stats_file = "data/stats/{}.mat".format(dataset)
        print "saving to", out_stats_file
        pickle.dump({
            'N' : N,
            'F' : F,
            'spk_logical' : spk_logical,
            },
            open(out_stats_file,"w")
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


    ru = 50 # The dimensionality of the subspace
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

if "__mainr__" == __name__:
    main()
else:
    print "imported, quitting"

