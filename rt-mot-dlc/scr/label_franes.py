import sys

import deeplabcut


def label_frames(config):
    deeplabcut.label_frames(config)


if __name__ == '__main__':
    if len(sys.argv) == 2:
        conf = sys.argv[1]
        label_frames(conf)
