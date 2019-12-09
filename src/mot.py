import random

import numpy as np
import scipy.ndimage as ndimage
import scipy.ndimage.filters as filters
from scipy.optimize import linear_sum_assignment

from kalman_filter import KalmanFilter


def get_random_color(n):
    ''' generate rgb using a list comprehension '''
    ret = []
    r = int(random.random() * 256)
    g = int(random.random() * 256)
    b = int(random.random() * 256)
    step = 256 / n
    for i in range(n):
        r += step
        g += step
        b += step
        r = int(r) % 256
        g = int(g) % 256
        b = int(b) % 256
    return (r, g, b)


class Track:
    def __init__(self, detection, trackId):
        self.track_id = trackId
        self.KF = KalmanFilter(detection)
        self.prediction = detection
        self.trace = []  # trace path
        self.skipped_frames = 0
        self.color = get_random_color(self.track_id)

    def predict(self, detection=None):
        self.prediction = self.KF.predict()
        if detection is not None:
            self.KF.update(detection)


class Tracker:

    def __init__(self, max_frames_to_skip):
        self.max_frames_to_skip = max_frames_to_skip
        self.id_count = 1
        self.tracks = []
        self.iter = 0
        self.det = []

    def extract_locmaxima(self, scmap, locref):
        neighborhood_size = 5
        threshold = 0.99999
        data = scmap[:, :, 2]
        locref = locref[:, :, 2]
        data_max = filters.maximum_filter(data, neighborhood_size)
        maxima = (data == data_max)
        data_min = filters.minimum_filter(data, neighborhood_size)
        diff = ((data_max - data_min) > threshold)
        maxima[diff == 0] = 0
        labeled, num_objects = ndimage.label(maxima)
        slices = ndimage.find_objects(labeled)
        detections = []
        for dy, dx in slices:
            maxloc = (dy.start, dx.start)
            offset = np.array(locref[maxloc])[::-1]
            pos_f8 = (np.array(maxloc).astype('float') * 8.0 + 0.5 * 8.0 + offset)
            pose = np.array([pos_f8[1], pos_f8[0], 0.0, 0.0])
            detections.append(pose)
        return detections

    def cost_metric(self, detections, N, M):
        cost_matrix = np.zeros(shape=(N, M))  # Cost matrix
        for i in range(len(self.tracks)):
            for j in range(len(detections)):
                gating_dist = self.tracks[i].KF.mahalanobis_dist(detections[j])
                cost_matrix[i][j] = gating_dist
        return cost_matrix

    def assignment_problem(self, costrix):
        row_ind, col_ind = linear_sum_assignment(costrix)
        return row_ind, col_ind

    def track(self, scmap, locref, bs):
        print("ITERATION NUMBRE : " + str(self.iter))
        self.iter += 1
        if bs == 1:
            scmap = np.squeeze(scmap)
            # pose extractions
            print("begin extracting detections")
            detections = self.extract_locmaxima(scmap, locref)
            self.det = detections
            print("end extracting detctions")
        else:
            pass
            # batchsize, ny, nx, num_joints = scmap.shape
            # locref = locref.reshape(batchsize, nx * ny, num_joints, 2)
            # # at the moment, should be kinda threshold for multiple pose extraction + data association
            # maxloc = np.argmax(scmap.reshape(batchsize, nx * ny, num_joints), axis=1)
            # x, y = np.unravel(maxloc, dims(ny, nx))
            # dz = np.zeros((batchsize, num_joints, 3))
            # for i in range(batchsize):
            #     for k in range(num_joints):
            #         dz[1, k, 2] = locref[1, maxloc[1, k], k, :]
            #         dz[1, k, 2] = scmap[1, y[1, k], x[1, k], k]
            # x = x.astype('float32') * 8.0 + .5 * 8.0 + dz[:, :, 0]
            # y = y.astype('float32') * 8.0 + .5 * 8.0 + dz[:, :, 1]
            # likelihood = dz[:, :, 2]

        if len(self.tracks) == 0:
            print("# No Tracks Found, Craeting New Tracks")
            for i in range(len(detections)):
                track = Track(detections[i], self.id_count)
                self.id_count += 1

                self.tracks.append(track)
            print("# number of created tracks: " + str(len(self.tracks)))
        N = len(self.tracks)
        M = len(detections)
        print(detections, N, M)
        cost_matrix = self.cost_metric(detections, N, M)
        rows, cols = self.assignment_problem(cost_matrix)
        matches = self._matching(rows, cols, cost_matrix, detections, N, M)
        for i in range(len(matches)):
            if matches[i] is not None:
                self.tracks[i].skipped_frames = 0
                self.tracks[i].predict(detections[matches[i]])
            else:
                self.tracks[i].predict()

    def _matching(self, rows, cols, costmatrix, detections, N, M):
        c = 0
        matches = [None] * N
        for i in range(len(rows)):
            matches[rows[i]] = cols[i]
            c += 1
        print("matches number: " + str(c))
        unmatching = []
        for i in range(len(matches)):
            if matches[i] is not None:
                if costmatrix[i][matches[i]] > 150:
                    print("unmatches found")
                    matches[i] = None
                    unmatching.append(i)
            else:
                self.tracks[i].skipped_frames += 1
        del_tracks = []
        for i in range(len(self.tracks)):
            if self.tracks[i].skipped_frames > self.max_frames_to_skip:
                del_tracks.append(self.tracks[i])

        if len(del_tracks) > 0:
            for i in range(len(del_tracks)):
                del self.tracks[i]
                del matches[i]

        for i in range(len(detections)):
            if i not in matches:
                track = Track(np.array(detections[i]), self.id_count)
                self.id_count += 1
                self.tracks.append(track)
        return matches
