import time
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

class Timer(object):
    def __init__(self, name=None):
        self.name = name

    def __enter__(self):
        self.tstart = time.time()

    def __exit__(self, type, value, traceback):
        if self.name:
            print('[%s]' % self.name,)
        print('Elapsed: %s' % (time.time() - self.tstart))


def plot_gmms(gmms, datasets):
    # n_classes = len(datasets)
    n_classifiers = len(gmms)

    plt.figure()

    for index, (gmm, X) in enumerate(zip(gmms, datasets)):
        ax = plt.subplot(111, aspect='equal')
        Y_ = gmm.predict(X)

        for n, color in enumerate('rg'):
            plt.scatter(X[Y_ == n, 0], X[Y_ == n, 1], s=20, marker='x', color=color)
            v, w = np.linalg.eigh(gmm._get_covars()[n][:2, :2])
            u = w[0] / np.linalg.norm(w[0])
            angle = np.arctan2(u[1], u[0])
            angle = 180 + (180 * angle / np.pi)  # convert to degrees
            v *= 9
            center = gmm.means_[n, :2]
            width = v[0]
            height = v[1]
            print(center, width, height, angle)
            ell = mpl.patches.Ellipse(xy=center, width=width, height=height, angle=angle, color=color)
            ell.set_clip_box(ax.bbox)
            ell.set_alpha(0.5)
            ax.add_artist(ell)

        plt.xticks(())
        plt.yticks(())
        plt.title("title here")
        plt.legend(loc='lower right', prop=dict(size=12))
        plt.show()
        return

def random_like(vector):
     return np.array([np.random.normal(0., 1.) for _ in vector])
