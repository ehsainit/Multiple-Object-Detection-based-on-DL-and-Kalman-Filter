'''
Adopted: DeeperCut by Eldar Insafutdinov
https://github.com/eldar/pose-tensorflow
'''
from scr.pose_estimation.PoseNet import PoseNet


def pose_net(cfg):
    cls = PoseNet
    return cls(cfg)
