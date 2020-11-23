import torch
import numpy as np
import torchvision
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2


class Tracklet(object):
    last_ID = 0

    def __init__(self, bounding_box, feature):
        """
        inputs
        - bounding_box: Detected or associated bounding box; It is composed of [x1, y1, x2, y2], so it is converted to
        [x, y, w, h] for states.
        - feature: Embedding feature; Feature is extracted by roi_align, so it is same dimension through all instances.

        class variables
        - particle_numbers: The number of particles
        - ID: Unique number to the instance
        - states: List of state; State is composed of [x, y, w, h]. The maximum length of states is 60
        - cur_feature: The latest feature of the instance
        - vel_x, vel_y: The velocity of instance
        - no_tracking_count: The fail number of associating
        - particles: The list of particles; The particle is composed of [x, y, w, h, weight].
        """

        self.particle_numbers = 100
        self.ID = Tracklet.last_ID
        Tracklet.last_ID += 1

        # state: [x, y, w, h]
        # bounding_box: [x1, y1, x2, y2]
        self.states = np.zeros((0, 4))
        state = np.asarray(bounding_box).reshape((1, 4))
        state[0, 2] = state[0, 2] - state[0, 0]
        state[0, 3] = state[0, 3] - state[0, 1]
        self.states = np.concatenate((self.states, state), axis=0)

        self.cur_feature = feature
        self.vel_x = 0.0
        self.vel_y = 0.0
        self.no_tracking_count = 0

        # particle initialization
        self.particles = np.zeros((self.particle_numbers, 5))
        self.particles[:, 0] = self.states[0, 0] + self.states[0, 2] * np.random.randn(self.particle_numbers)
        self.particles[:, 1] = self.states[0, 1] + self.states[0, 3] * np.random.randn(self.particle_numbers)
        self.particles[:, 2] = self.states[0, 2] + np.random.randn(self.particle_numbers)
        self.particles[:, 3] = self.states[0, 3] + np.random.randn(self.particle_numbers)
        self.particles[:, 4] = 1/self.particle_numbers

    def update(self, bounding_box, feature):

        if bounding_box is not None:
            state = np.asarray(bounding_box).reshape((1, 4))
            state[0, 2] = state[0, 2] - state[0, 0]
            state[0, 3] = state[0, 3] - state[0, 1]
            self.states = np.concatenate((self.states, state), axis=0)
            self.cur_feature = feature

            if len(self.states) > 60:
                self.states = np.delete(self.states, obj=0, axis=0)
            self.no_tracking_count = 0

        elif bounding_box is None:
            self.no_tracking_count += 1

        self.calculate_vel()
        self.particle_update()
        self.particle_predict()

    def calculate_vel(self):
        if len(self.states) >= 2:
            self.vel_x = (self.states[-1, 0] + self.states[-1, 2]/2.0) - (self.states[-2, 0] + self.states[-2, 2]/2.0)
            self.vel_y = (self.states[-1, 1] + self.states[-1, 3]/2.0) - (self.states[-2, 1] + self.states[-2, 3]/2.0)

    def particle_update(self):

        weights = np.zeros((self.particle_numbers))
        # 1. weights of particles are updateed by calculating distance between detected bounding box
        distance = (self.particles[:, 0] - self.states[-1, 0]) ** 2 + (self.particles[:, 1] - self.states[-1, 1]) ** 2
        distance = np.sqrt(distance)
        distance /= np.sum(distance)    # sum of distances is 1
        weights += distance

        self.particles[:, 4] = weights

    def particle_predict(self):
        generating_numbers = 0
        predicted = np.zeros((self.particle_numbers, 5), dtype=np.float)
        predicted[:, 4] = 1/self.particle_numbers

        # particles are sorted by descending order on weight
        for i in range(self.particle_numbers):
            num = np.int(np.ceil(self.particles[i, 4] * self.particle_numbers))
            if generating_numbers + num > self.particle_numbers:
                num = self.particle_numbers - generating_numbers

            # particles are sampled by weights and states
            predicted[generating_numbers:generating_numbers + num, 0] = self.particles[i, 0] + self.vel_x + np.random.randn(num)
            predicted[generating_numbers:generating_numbers + num, 1] = self.particles[i, 1] + self.vel_y + np.random.randn(num)
            predicted[generating_numbers:generating_numbers + num, 2] = np.mean(self.states[-4:, 2]) + np.random.randn(num)
            predicted[generating_numbers:generating_numbers + num, 3] = np.mean(self.states[-4:, 3]) + np.random.randn(num)

            generating_numbers += num
            if generating_numbers is num:
                break


class Tracker(object):
    def __init__(self, max_tracklets=100):
        self.active = False
        self.tracklets = []
        self.max_tracklets = 50

    def forward(self, detector_bb, feature_map):

        if not self.active:
            features = self.extract_features(bounding_box=detector_bb, feature_map=feature_map)
            for i in range(len(detector_bb)):
                tracklet = Tracklet(detector_bb[i, :4], features[i, :])
                self.tracklets.append(tracklet)
            self.active = True

        else:
            candidates_bb = self.generate_candidates()
            detector_features = self.extract_features(bounding_box=detector_bb, feature_map=feature_map)
            candidates_features = self.extract_features(bounding_box=candidates_bb, feature_map=feature_map)

            detector_score_matrix = self.construct_score_matrix(detector_features, bounding_box=detector_bb)
            candidates_score_matrix = self.construct_score_matrix(candidates_features, bounding_box=candidates_bb)
            self.matching(score_matrix=[detector_score_matrix, candidates_score_matrix], bounding_bb=[detector_bb, candidates_bb],
                          features=[detector_features, candidates_features])

    def generate_candidates(self, num=10):
        candidates = np.zeros((0, 4))

        for tracklet in self.tracklets:
            candidates = np.concatenate((candidates, tracklet.particles[:, :4]), axis=0)

        return np.float32(candidates)

    def construct_score_matrix(self, features, bounding_box):

        m = len(self.tracklets)
        n = bounding_box.shape[0]

        # appearance sim, motion sim, iou sim
        cost_a = np.zeros((m, n))
        cost_i = np.zeros((m, n))
        cost_m = np.zeros((m, n))

        for i in range(m):
            dist = (self.tracklets[i].cur_feature[None, :] - features.numpy()) ** 2
            cost_a[i, :] = np.exp(-np.sum(dist.numpy(), axis=1))

            temp = np.zeros((n, 2))
            temp[:, 0] = self.tracklets[i].states[-1, 0]
            temp[:, 1] = bounding_box[:, 0]
            x1 = np.max(temp, axis=1)

            temp[:, 0] = self.tracklets[i].states[-1, 1]
            temp[:, 1] = bounding_box[:, 1]
            y1 = np.max(temp, axis=1)

            temp[:, 0] = self.tracklets[i].states[-1, 0] + self.tracklets[i].states[-1, 2]
            temp[:, 1] = bounding_box[:, 2]
            x2 = np.min(temp, axis=1)

            temp[:, 0] = self.tracklets[i].states[-1, 1] + self.tracklets[i].states[-1, 3]
            temp[:, 1] = bounding_box[:, 3]
            y2 = np.min(temp, axis=1)

            temp = np.zeros((n, 2))
            temp[:, 1] = x2 - x1
            temp_ = np.zeros((n, 2 ))
            temp_[:, 1] = y2 - y1
            intersection = np.max(temp, axis=1) * np.max(temp_, axis=1)
            area1 = self.tracklets[i].states[-1, 2] * self.tracklets[i].states[-1, 3]
            area2 = (bounding_box[:, 2] - bounding_box[:, 0]) * (bounding_box[:, 3] - bounding_box[:, 1])
            iou = intersection / (area1 + area2 - intersection)
            cost_i[i, :] = iou

            bb_ctr_x = (bounding_box[:, 0] + bounding_box[:, 2]) * 0.5
            bb_ctr_y = (bounding_box[:, 1] + bounding_box[:, 3]) * 0.5
            tracklet_ctr_x = self.tracklets[i].states[-1, 0] + self.tracklets[i].states[-1, 2] * 0.5 + self.tracklets[i].vel_x
            tracklet_ctr_y = self.tracklets[i].states[-1, 1] + self.tracklets[i].states[-1, 3] * 0.5 + self.tracklets[i].vel_y

            dist = (bb_ctr_x - bb_ctr_y) ** 2 + (tracklet_ctr_x - tracklet_ctr_y) ** 2
            cost_m[i, :] = np.exp(-dist)

        return 0.2 * cost_a + 0.2 * cost_i + 0.6 * cost_m

    def matching(self, score_matrix, bounding_bb, features):
        detector_score_matrix, candidates_score_matrix = score_matrix[0], score_matrix[1]
        detector_bb, candidates_bb = bounding_bb[0], bounding_bb[1]
        detector_features, candidates_features = features[0], features[1]

        # 1. Associate detector bounding boxes with tracklets
        row_ind, col_ind = linear_sum_assignment(-detector_score_matrix)
        row_ind_, col_ind_ = linear_sum_assignment(-candidates_score_matrix)
        mask = detector_score_matrix[row_ind, col_ind] >= 0.1
        mask_ = candidates_score_matrix[row_ind_, col_ind_] >= 0.2

        matched_row_ind = []
        for i in range(len(row_ind)):
            if mask[i]:
                self.tracklets[row_ind[i]].update(detector_bb[col_ind[i], :4], detector_features[col_ind[i]])
                matched_row_ind.append(row_ind[i])

        # 2. Associate candidates bounding boxes with tracklets
        for i in range(len(row_ind_)):
            if row_ind_[i] in matched_row_ind:
                continue
            elif mask_[i]:
                self.tracklets[row_ind_[i]].update(candidates_bb[col_ind_[i], :4], candidates_features[col_ind_[i]])
                matched_row_ind.append(row_ind_[i])

        # 3. updates tracklets missed tracking count
        for i in range(len(self.tracklets)):
            if i not in matched_row_ind:
                self.tracklets[i].update(bounding_box=None, feature=None)

        # 4. check weather detector bounding boxes not to be matches would be initialized or not by IOU.
        no_detector_matching = set(range(len(detector_bb)))
        no_detector_matching = no_detector_matching - set(col_ind[mask])

        for i in no_detector_matching:
            if (len(self.tracklets)) >= self.max_tracklets:
                break
            flag = True
            for tracklet in self.tracklets:
                if tracklet.no_tracking_count > 5:
                    continue
                x1 = np.max((tracklet.states[-1, 0], detector_bb[i, 0]))
                y1 = np.max((tracklet.states[-1, 1], detector_bb[i, 1]))
                x2 = np.min((tracklet.states[-1, 0] + tracklet.states[-1, 2], detector_bb[i, 2]))
                y2 = np.min((tracklet.states[-1, 1] + tracklet.states[-1, 3], detector_bb[i, 3]))

                intersection = np.max((0, x2 - x1)) * np.max((0, y2 - y1))
                area1 = tracklet.states[-1, 2] * tracklet.states[-1, 3]
                area2 = (detector_bb[i, 2] - detector_bb[i, 0]) * (detector_bb[i, 3] - detector_bb[i, 1])
                iou = intersection / (area1 + area2 - intersection)

                if iou > 0.4:
                    flag = False
                    break
            if flag:
                tracklet = Tracklet(detector_bb[i, :4], detector_features[i, :])
                self.tracklets.append(tracklet)

        # 5. terminalization
        for t in self.tracklets:
            if t.no_tracking_count == 5:
                del t

    def extract_features(self, bounding_box, feature_map, img_size=(1080, 1920)):
        """
        :param bounding_box: numpy, [x1, y1, x2, y2]
        :param feature_map: pytorch tensor
        """
        c, h, w = feature_map.shape
        spatial_ratio = img_size[0] / h

        features = torchvision.ops.roi_align(input=feature_map[None, :], boxes=[torch.from_numpy(bounding_box)], output_size=(8, 16), spatial_scale=spatial_ratio,
                                             sampling_ratio=2)
        K = features.shape[0]
        features = features.reshape((K, -1))

        return features

    def drawing_boxes(self, frame):
        for i in range(len(self.tracklets)):
            tracklet = self.tracklets[i]
            if tracklet.no_tracking_count > 0:
                continue
            p1 = (int(tracklet.states[-1, 0]), int(tracklet.states[-1, 1]))
            p2 = (int(tracklet.states[-1, 0] + tracklet.states[-1, 2]), int(tracklet.states[-1, 1] + tracklet.states[-1, 3]))
            cv2.rectangle(img=frame, pt1=p1, pt2=p2, color=(0, 0, 255), thickness=3, lineType=cv2.LINE_AA)
            cv2.putText(img=frame, text=str(tracklet.ID), org=p1, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.0,
                        color=(255, 255, 255), thickness=2, lineType=cv2.LINE_AA)
        return frame


