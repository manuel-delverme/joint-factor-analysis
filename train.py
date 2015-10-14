# -*- coding: UTF-8 -*-
import bob
import re
import numpy as np

MODELS_PATH = "../jfa_models"
LISTS_PATH = "../jfa_lists"

def parse_list(file_name):
    with open(filename) as recording:
        lst = recording.read().split()
    # logical, physical = zip(*[ str.split(row, "=") for row in lst ])
    logical, physical = [ str.split(row, "=") for row in lst ]
    return logical, physical

def main():
    m = [float(mi) for mi in  open("{}/ubm_means".format(MODELS_PATH)).read().split()]
    v = [float(vi) for vi in  open("{}/ubm_variances".format(MODELS_PATH)).read().split()]
    w = [float(wi) for wi in  open("{}/ubm_weights".format(MODELS_PATH)).read().split()]
    m = np.array(m)
    v = np.array(v)
    w = np.array(w)

    n_mixtures = len(w)
    dim = len(m) / n_mixtures

    # dim is the real width but the data is stored as (n,1)
    m = np.reshape(m, (dim, n_mixtures))
    v = np.reshape(m, (dim, n_mixtures))

    datasets = []
    datasets.append(['enroll_stats'])
    datasets.append(['fa_train_eigenchannels_stats'])
    datasets.append(['fa_train_eigenvoices_stats'])
    datasets.append(['test_stats'])

    for i in range(datasets):
        list_file = "{}/{}.lst".format(LISTS_PATH, datasets)
        spk_logical, spk_physical = parse_list(list_file)
        n_sessions = len(spk_logical)

        N = np.array(n_sessions, n_mixtures)
        F = np.array(n_sessions, n_mixtures)

        for session_i in range(n_sessions):
            session_name = spk_physical[i]

    print "done"

""""
# follow http://www1.icsi.berkeley.edu/Speech/presentations/AFRL_ICSI_visit2_JFA_tutorial_icsitalk.pdf
# 1/ Features
    # voice actifvity detection
    energy/LTSD/SOX-something?

    # feature_extraction
    MFCC(bob)/LPC(scikits.talkbox)

    #sklearn.mixture.GMM
    "COVAR is diagonal (cuz components *are* independent"



# 2/ GMM Training
    n_gaussians = 256
    iterk = 25
    iterg_train = 25
    end_acc = 0.0001
    var_thd = 0.0001
    update_weights = True
    update_means = True
    update_variances = True
    norm_KMeans = True

    for gaussian_n in n_gaussians:
        print "training", gaussian_n
        print "merging", gaussian_n
        C = np.array(n_gaussians)
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
    n_iter_train = 10
    n_iter_enrol = 1

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

if "__main__" == __name__:
    main()
else:
    print "imported, quitting"

