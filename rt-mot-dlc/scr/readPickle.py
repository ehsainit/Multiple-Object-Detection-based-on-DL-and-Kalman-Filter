import pickle
import sys

import numpy as np
import scipy.misc
from matplotlib import pyplot as plt


def readPickle(f):
    pickle_in = open(f, 'rb')
    dict1 = pickle.load(pickle_in)
    img = dict1["scoremapofimage: "]
    scmap = img[0]
    offmat = img[1]
    scmap = np.squeeze(scmap)
    scmap_part = scmap[:, :, 2]
    maxloc = np.unravel_index(np.argmax(scmap_part),
                              scmap_part.shape)
    test = np.extract(scmap_part >= 0.99999, scmap_part)
    test = np.where(scmap_part >= 0.99999845)
    # print(test[0][0],test[1][1])
    # maxloc1 = scmap_part[test[0][0],test[1][1]]
    # print(maxloc1)
    offset = np.array(offmat[maxloc][0])[::-1]
    pos_f8 = (np.array(maxloc).astype('float') * 8.0 + 0.5 * 8.0 +
              offset)
    pose = np.hstack((pos_f8[::-1],
                      [scmap[maxloc][2]]))

    print(pos_f8)
    rgb = scipy.misc.toimage(scmap_part)
    plt.imshow(rgb)
    plt.show()


if __name__ == '__main__':
    if len(sys.argv) == 2:
        f = sys.argv[1]

        readPickle(f)
    else:
        print('Usage: python3 readPickle.py [filename]')
