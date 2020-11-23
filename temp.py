from __future__ import division
import torch
import torchvision
import numpy as np
from scipy.optimize import linear_sum_assignment
import cv2

class Instance:
    last_ID = 0

    def __init__(self, box=None, feature=None, particle_numbers=100):
        """
        inputs
        - box: bounding box; It is composed of [x1, y1, x2, y2]
        - feature: Embedding feature; Feature is extracted by roi align.
        - size_std: standard deviates of width and height to generate particles
        """
        self.ID = Instance.last_ID
        Instance.last_ID += 1
        self.no_tracking_count = 0

        """
        tracklet: [x1, y1, x2, y2, ctr_x, ctr_y, w, h, v_x, v_y]
        (x1, y1): left top location
        (x2, y2): right bottom locatoin
        (ctr_x, ctr_y): the center of location
        (w, h): width and height
        (v_x, v_y): velocity of x-axis and y-axis
        """
        self.tracklets = np.zeros((0, 10))
        tracklet = self.make_tracklet(box=box)
        self.tracklets = np.concatenate((self.tracklets, tracklet), axis=0)
        self.feature = feature

        self.location_std = (tracklet[0, 6], tracklet[0, 7])
        """
        particles: [x1, y1, x2, y2, ctr_x, ctr_y, w, h, weight]
        """
        self.particle_numbers = particle_numbers
        self.particles = np.zeros((self.particle_numbers, 9))
        self.particles[:, 4] = self.tracklets[0, 4] + self.tracklets[0, 6] * np.random.randn(self.particle_numbers) * (
                    self.no_tracking_count + 1)
        self.particles[:, 5] = self.tracklets[0, 5] + self.tracklets[0, 7] * np.random.randn(self.particle_numbers) * (
                    self.no_tracking_count + 1)
        self.particles[:, 6] = self.tracklets[0, 6] + np.random.randn(self.particle_numbers)
        self.particles[:, 7] = self.tracklets[0, 7] + np.random.randn(self.particle_numbers)
        self.particles[:, 8] = 1 / self.particle_numbers
        self.particles[:, 0] = self.particles[:, 4] - self.particles[:, 6] / 2
        self.particles[:, 1] = self.particles[:, 5] - self.particles[:, 7] / 2
        self.particles[:, 2] = self.particles[:, 4] + self.particles[:, 6] / 2
        self.particles[:, 3] = self.particles[:, 5] + self.particles[:, 6] / 2

    def update(self, box=None, feature=None, detector=True):
        if box is None:
            self.no_tracking_count += 1
        else:
            flag = False

            if detector:
                if self.no_tracking_count != 0:
                    flag = True
                    self.no_tracking_count = 0
                self.feature = feature
            else:
                self.no_tracking_count += 1

            tracklet = self.make_tracklet(box=box, flag=flag)
            self.tracklets = np.concatenate((self.tracklets, tracklet), axis=0)

            if len(self.tracklets) > 60:
                self.tracklets = np.delete(self.tracklets, obj=0, axis=0)

    def make_tracklet(self, box, flag=False):
        tracklet = np.zeros((1, 10))
        tracklet[0, :4] = np.asarray(box).reshape((1, 4))
        tracklet[0, 4] = (tracklet[0, 2] + tracklet[0, 0]) / 2
        tracklet[0, 5] = (tracklet[0, 3] + tracklet[0, 1]) / 2
        tracklet[0, 6] = (tracklet[0, 2] - tracklet[0, 0])
        tracklet[0, 7] = (tracklet[0, 3] - tracklet[0, 1])

        if len(self.tracklets) != 0:
            if self.no_tracking_count == 0:
                if flag:
                    tracklet[0, 8] = self.tracklets[-1, 8]
                    tracklet[0, 9] = self.tracklets[-1, 9]
                else:
                    tracklet[0, 8] = 0.7 * (tracklet[0, 4] - self.tracklets[-1, 4]) + 0.3 * self.tracklets[-1, 8]
                    tracklet[0, 9] = 0.7 * (tracklet[0, 5] - self.tracklets[-1, 5]) + 0.3 * self.tracklets[-1, 9]

            else:
                tracklet[0, 8] = self.tracklets[-1, 8]
                tracklet[0, 9] = self.tracklets[-1, 9]

        return tracklet


class Tracker:
    def __init__(self, max_instances=50, particle_number=100, score_weights=(0.2, 0.4, 0.4), img_size=(1080, 1920)):
        self.active = False
        self.instances = []
        self.max_instances = max_instances
        self.particle_number = particle_number
        self.img_size = img_size
        self.score_weights = score_weights

        self.colors = np.random.randint(low=0, high=255, size=(1000, 3))

        self.prev_detected_instances = None
        self.cur_detected_instances = None

    def tracking(self, boxes, feature_map):
        if not self.active:
            features = self.extract_features(boxes=boxes, feature_map=feature_map)
            for i in range(len(boxes)):
                instance = Instance(box=boxes[i, :4], feature=features[i, :], particle_numbers=self.particle_number)
                self.instances.append(instance)
            self.active = True

            self.prev_detected_instances = np.array(self.instances)

        else:
            detector_features = self.extract_features(boxes=boxes, feature_map=feature_map)

            m = len(self.instances)
            n = boxes.shape[0]
            #detector_score_matrix = self.construct_score_matrix(features=detector_features, boxes=boxes,
            #                                                    weights=self.score_weights, size=(m, n))
            detector_score_matrix = self.construct_score_matrix_gating(features=detector_features, boxes=boxes, vel_thres=1.5,
                                                                       std=(2, 0.01, 0.01))
            self.matching(score_matrix=detector_score_matrix, boxes=boxes, features=detector_features,
                          feature_map=feature_map)

    def construct_score_matrix(self, features, boxes, weights=(1 / 3, 1 / 3, 1 / 3), size=None):
        # score matrix size : (m x n)
        instance_features = []
        instance_tracklets = []

        for instance in self.instances:
            instance_features.append(instance.feature)
            instance_tracklets.append(instance.tracklets[-1, :])

        instance_features = np.asarray(instance_features)
        instance_tracklets = np.asarray(instance_tracklets)
        instance_boxes = instance_tracklets[:, :4]
        instance_boxes[:, 0] += instance_tracklets[:, -2]
        instance_boxes[:, 2] += instance_tracklets[:, -2]
        instance_boxes[:, 1] += instance_tracklets[:, -1]
        instance_boxes[:, 3] += instance_tracklets[:, -1]
        instance_locatoin = instance_tracklets[:, 4:6]
        instance_locatoin[:, 0] += instance_tracklets[:, -2]
        instance_locatoin[:, 1] += instance_tracklets[:, -1]
        location2 = np.zeros((size[1], 2))
        location2[:, 0] = (boxes[:, 0] + boxes[:, 2]) / 2
        location2[:, 1] = (boxes[:, 1] + boxes[:, 3]) / 2

        score_a = self.cal_appearance_score(features1=instance_features, features2=features)
        score_i = self.cal_iou_score(boxes1=instance_boxes, boxes2=boxes)
        score_m = self.cal_motion_score(location1=instance_locatoin, location2=location2)

        return weights[0] * score_a + weights[1] * score_i + weights[2] * score_m

    def cal_appearance_score(self, features1, features2, std_a=0.01):
        # features : (n x K)
        m, K = features1.shape
        n, _ = features2.shape

        score_a = 1 - np.matmul(features1, np.transpose(features2))
        score_a = np.exp(-0.5 * (score_a/std_a) ** 2)
        # score_a = (score_a / 2) + 0.5
        # score_a = np.sum((features1.reshape(m, 1, K) - features2.reshape(1, n, K)) ** 2, axis=2)
        # score_a = np.sqrt(score_a)
        # score_a = np.exp(-score_a)

        return score_a

    def cal_iou_score(self, boxes1, boxes2):
        m, n = len(boxes1), len(boxes2)

        temp1 = np.tile(boxes1[:, 0].reshape(m, 1), (1, n))
        temp2 = np.tile(boxes2[:, 0].reshape(1, n), (m, 1))
        x1 = np.max((temp1, temp2), axis=0)

        temp1 = np.tile(boxes1[:, 1].reshape(m, 1), (1, n))
        temp2 = np.tile(boxes2[:, 1].reshape(1, n), (m, 1))
        y1 = np.max((temp1, temp2), axis=0)

        temp1 = np.tile(boxes1[:, 2].reshape(m, 1), (1, n))
        temp2 = np.tile(boxes2[:, 2].reshape(1, n), (m, 1))
        x2 = np.min((temp1, temp2), axis=0)

        temp1 = np.tile(boxes1[:, 3].reshape(m, 1), (1, n))
        temp2 = np.tile(boxes2[:, 3].reshape(1, n), (m, 1))
        y2 = np.min((temp1, temp2), axis=0)

        width = np.clip(x2 - x1, a_min=0, a_max=None)
        height = np.clip(y2 - y1, a_min=0, a_max=None)
        intersection = width * height
        area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

        denominator = (area1.reshape(m, 1) + area2.reshape(1, n)) - intersection
        if np.allclose(denominator, 0.0):
            print(denominator)
        iou = intersection / denominator

        return iou

    def cal_motion_score(self, location1, location2, motion_std=40.0):
        dist = (location1[:, 0].reshape(-1, 1) / self.img_size[1] - location2[:, 0].reshape(1, -1) / self.img_size[
            1]) ** 2 + \
               (location1[:, 1].reshape(-1, 1) / self.img_size[0] - location2[:, 1].reshape(1, -1) / self.img_size[
                   0]) ** 2
        dist = np.sqrt(dist)
        score_m = np.exp(-dist * 10)
        return score_m

    def construct_score_matrix_gating(self, features, boxes, vel_thres=1.5, std=(1, 1, 0.01)):
        std_h, std_p, std_v = std
        instance_features = []
        instance_tracklets = []

        for instance in self.instances:
            instance_features.append(instance.feature)
            instance_tracklets.append(instance.tracklets[-1, :])
        instance_features = np.array(instance_features)
        instance_tracklets = np.array(instance_tracklets)

        ctr_boxes = np.zeros(shape=(len(boxes), 6))
        ctr_boxes[:, :4] = boxes
        ctr_boxes[:, 4] = (boxes[:, 2] + boxes[:, 0]) / 2
        ctr_boxes[:, 5] = (boxes[:, 3] + boxes[:, 1]) / 2

        g = self.gating_function(boxes=ctr_boxes, tracklets=instance_tracklets, vel_thres=vel_thres, std_h=std_h, std_p=std_p, std_v=std_v)
        a = self.cal_appearance_score(features1=instance_features, features2=features)

        score_matrix = a * g

        return score_matrix

    def gating_function(self, boxes, tracklets, vel_thres=1.5, std_h=1, std_p=1, std_v=1):
        m, n = len(tracklets), len(boxes)
        g = np.zeros(shape=(m, n))

        # the agreement between the bounding box height of target and detection
        h = (tracklets[:, 6].reshape(m, 1) - (boxes[:, 3] - boxes[:, 1]).reshape(1, n)) / tracklets[:, 6].reshape(m, 1)
        h = np.exp((h ** 2) / (-2 * std_h))

        # distance between boxes and target, using velocity for fast-moving object
        vel = np.sqrt(np.sum(tracklets[:, -2:] ** 2, axis=1))
        mask = (vel < vel_thres)
        g[mask, :] = ((tracklets[mask, 4].reshape(-1, 1) - boxes[:, 4].reshape(1, -1)) / self.img_size[1]) ** 2 + \
                     ((tracklets[mask, 5].reshape(-1, 1) - boxes[:, 5].reshape(1, -1)) / self.img_size[0]) ** 2
        g[mask, :] = np.sqrt(g[mask, :])
        g[mask, :] = np.exp(-g[mask, :] / (2 * std_p))

        p = boxes[:, 4:6].reshape(1, -1, 2) - tracklets[~mask, 4:6].reshape(-1, 1, 2)  # m x n x 2
        v = tracklets[~mask, -1:-3:-1].reshape(-1, 1, 2)  # m x 1 x 2

        g[~mask, :] = np.abs(np.sum(v * p, axis=2)) / np.sqrt(np.sum(v ** 2, axis=2)).reshape(-1, 1)
        g[~mask, :] = np.exp(g[~mask, :] / (-2 * std_v))
        g = g * h

        return g

    def matching(self, score_matrix, boxes, features, feature_map):

        tracked_state = np.zeros((0, 4))
        # 1. Consider detector bounding boxes
        row_ind, col_ind = linear_sum_assignment(-score_matrix)
        mask = (score_matrix[row_ind, col_ind] >= 0.3)

        matched_row_ind = []
        matched_col_ind = []
        for i in range(len(row_ind)):
            if mask[i]:
                self.instances[row_ind[i]].update(box=boxes[col_ind[i], :4], feature=features[col_ind[i], :])
                matched_row_ind.append(row_ind[i])
                matched_col_ind.append(col_ind[i])
                tracked_state = np.concatenate((tracked_state, boxes[col_ind[i], :4].reshape(1, 4)), axis=0)
        self.cur_detected_instances = np.array(self.instances)[matched_row_ind]

        # 2. Associate particles boxes with instances not to be detected
        no_match_idx = []
        for i in range(len(self.instances)):
            if i in matched_row_ind:
                continue
            else:
                instance = self.instances[i]
                particles = instance.particles
                particles_features = self.extract_features(boxes=particles[:, :4], feature_map=feature_map)

                """
                score_a = self.cal_appearance_score(features1=instance.feature.reshape(1, -1),
                                                    features2=particles_features)

                box1 = instance.tracklets[-1, :4].reshape(1, 4)
                box1[0, 0] += instance.tracklets[-1, -2]
                box1[0, 1] += instance.tracklets[-1, -1]
                box1[0, 2] += instance.tracklets[-1, -2]
                box1[0, 3] += instance.tracklets[-1, -1]
                score_i = self.cal_iou_score(boxes1=box1, boxes2=particles[:, :4])

                location1 = instance.tracklets[-1, 4:6] + instance.tracklets[-1, -2:]
                location1 = location1.reshape(1, -1)
                location2 = particles[:, 4:6]
                location2 = location2.reshape(-1, 2)
                score_m = self.cal_motion_score(location1=location1, location2=location2, motion_std=20.0)
                score_matrix = score_a / 3 + \
                               score_i / 3 + \
                               score_m / 3
                """
                #g = self.gating_function(boxes=particles, tracklets=instance.tracklets[-1, :].reshape(1, -1), vel_thres=1.5,
                #                         std_h=2, std_p=0.01, std_v=0.01)
                a = self.cal_appearance_score(features1=instance.feature.reshape(1, -1), features2=particles_features)
                score_matrix = a

                idx = np.argmax(score_matrix)
                if score_matrix[0, idx] < 0.8:
                    box = np.mean(particles[:, :4], axis=0)
                    self.instances[i].update(box=box, feature=None, detector=False)
                    no_match_idx.append(i)
                    continue

                self.instances[i].update(box=particles[idx, :4], feature=particles_features[idx, :], detector=False)
                tracked_state = np.concatenate((tracked_state, particles[idx, :4].reshape(1, 4)), axis=0)

        # 3. particle update and predict
        mask = np.sum(np.equal(self.prev_detected_instances.reshape(1, -1), self.cur_detected_instances.reshape(-1, 1)),
                      axis=1)
        matched_instances = self.prev_detected_instances[mask]
        prev_loc = []
        cur_loc = []
        for inst in matched_instances:
            prev_loc.append(inst.tracklets[-2, 4:6])
            cur_loc.append(inst.tracklets[-1, 4:6])
        prev_loc = np.array(prev_loc)
        cur_loc = np.array(cur_loc)

        for i in range(len(self.instances)):
            self.particle_update(instance=self.instances[i], prev_loc=prev_loc, cur_loc=cur_loc)
            self.particle_predict(instance=self.instances[i])

        self.prev_detected_instances = self.cur_detected_instances
        # 4. check weather detector bounding boxes not to be matched would be initialized or not by IOU

        no_matched_col_ind = np.asarray(range(len(boxes)))
        no_matched_col_ind = np.setdiff1d(no_matched_col_ind, np.asarray(matched_col_ind), assume_unique=True)
        no_matched_detector_boxes = boxes[no_matched_col_ind, :]
        no_matched_detector_features = features[no_matched_col_ind, :]
        self.initialization(boxes=no_matched_detector_boxes, tracked_state=tracked_state,
                            features=no_matched_detector_features)

        # 5. terminalization
        idx = 0
        while idx < len(self.instances):
            if self.instances[idx].no_tracking_count > 30:
                del self.instances[idx]
            else:
                idx += 1

    def particle_update(self, instance, prev_loc, cur_loc):
        if instance.no_tracking_count == 0:
            dist = ((instance.particles[:, 4] - instance.tracklets[-1, 4]) / self.img_size[1]) ** 2 + (
                        (instance.particles[:, 5] - instance.tracklets[-1, 5]) / self.img_size[0]) ** 2
            dist = np.sqrt(dist)
            dist = np.exp(-dist / 0.005)
            dist /= np.sum(dist)

            instance.particles[:, -1] = dist
        elif instance.no_tracking_count != 0:
            normalization = np.array([self.img_size[1], self.img_size[0]]).reshape(1, 1, 2)
            prev_loc_dist = (instance.tracklets[-1, 4:6].reshape(1, 1, 2) - prev_loc.reshape(1, -1, 2)) / normalization
            cur_loc_dist = (instance.particles[:, 4:6].reshape(-1, 1, 2) - cur_loc.reshape(1, -1, 2)) / normalization

            dist = np.sum(np.sqrt(np.sum((cur_loc_dist - prev_loc_dist) ** 2, axis=2)), axis=1)
            dist = np.exp(-dist)
            dist /= np.sum(dist)

            instance.particles[:, -1] = dist

    def particle_predict(self, instance):
        generating_numbers = 0
        pred_particles = np.zeros((instance.particle_numbers, 9))
        pred_particles[:, -1] = 1 / instance.particle_numbers

        for i in range(instance.particle_numbers):
            num = np.int(np.ceil(instance.particles[i, -1] * instance.particle_numbers))
            if generating_numbers + num > instance.particle_numbers:
                num = instance.particle_numbers - generating_numbers

            pred_particles[generating_numbers:generating_numbers + num, 4] = instance.particles[i, 4] \
                                                                             + instance.tracklets[-1, -2] \
                                                                             + np.random.randn(num) * np.min(
                ((instance.no_tracking_count + 1), 5))
            pred_particles[generating_numbers:generating_numbers + num, 5] = instance.particles[i, 5] \
                                                                             + instance.tracklets[-1, -1] \
                                                                             + np.random.randn(num) * np.min(
                ((instance.no_tracking_count + 1), 5))
            pred_particles[generating_numbers:generating_numbers + num, 6] = np.mean(instance.tracklets[-4:, 6]) \
                                                                             + np.random.randn(num)
            pred_particles[generating_numbers:generating_numbers + num, 7] = np.mean(instance.tracklets[-4:, 7]) \
                                                                             + np.random.randn(num)

            generating_numbers += num
            if generating_numbers == instance.particle_numbers:
                break

        instance.particles = pred_particles
        instance.particles[:, 4] = np.clip(instance.particles[:, 4], a_min=0.0, a_max=self.img_size[1])
        instance.particles[:, 5] = np.clip(instance.particles[:, 5], a_min=0.0, a_max=self.img_size[0])
        instance.particles[:, 0] = np.clip(instance.particles[:, 4] - instance.particles[:, 6] / 2, a_min=0.0,
                                           a_max=None)
        instance.particles[:, 1] = np.clip(instance.particles[:, 5] - instance.particles[:, 7] / 2, a_min=0.0,
                                           a_max=None)
        instance.particles[:, 2] = np.clip(instance.particles[:, 4] + instance.particles[:, 6] / 2, a_min=None,
                                           a_max=self.img_size[1])
        instance.particles[:, 3] = np.clip(instance.particles[:, 5] + instance.particles[:, 7] / 2, a_min=None,
                                           a_max=self.img_size[0])

    def initialization(self, boxes, tracked_state, features):
        if len(boxes) == 0:
            return None
        if len(self.instances) > self.max_instances:
            return None
        """
        If IOU is less than threshold comparing with tracked bounding boxes, the tracker makes the new instance.
        """
        m, n = len(boxes), len(tracked_state)
        temp1 = np.tile(boxes[:, 0].reshape(m, 1), (1, n))
        temp2 = np.tile(tracked_state[:, 0].reshape(1, n), (m, 1))
        x1 = np.max((temp1, temp2), axis=0)

        m, n = len(boxes), len(tracked_state)
        temp1 = np.tile(boxes[:, 0].reshape(m, 1), (1, n))
        temp2 = np.tile(tracked_state[:, 0].reshape(1, n), (m, 1))
        x1 = np.max((temp1, temp2), axis=0)

        temp1 = np.tile(boxes[:, 1].reshape(m, 1), (1, n))
        temp2 = np.tile(tracked_state[:, 1].reshape(1, n), (m, 1))
        y1 = np.max((temp1, temp2), axis=0)

        temp1 = np.tile(boxes[:, 2].reshape(m, 1), (1, n))
        temp2 = np.tile(tracked_state[:, 2].reshape(1, n), (m, 1))
        x2 = np.min((temp1, temp2), axis=0)

        temp1 = np.tile(boxes[:, 3].reshape(m, 1), (1, n))
        temp2 = np.tile(tracked_state[:, 3].reshape(1, n), (m, 1))
        y2 = np.min((temp1, temp2), axis=0)

        width = np.clip(x2 - x1, a_min=0, a_max=None)
        height = np.clip(y2 - y1, a_min=0, a_max=None)
        intersection = width * height
        area1 = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        area2 = (tracked_state[:, 2] - tracked_state[:, 0]) * (tracked_state[:, 3] - tracked_state[:, 1])

        """
        intersection: (m x n)
        area1: (m, )
        area2: (n, )
        """
        denominator = (area1.reshape(m, 1) + area2.reshape(1, n)) - intersection
        iou = intersection / denominator

        flag = (iou > 0.2)
        flag = np.sum(flag, axis=1)

        new_boxes = boxes[flag == 0]
        new_features = features[flag == 0]
        for box, feature in zip(new_boxes, new_features):
            instance = Instance(box=box[:4], feature=feature, particle_numbers=self.particle_number)
            self.instances.append(instance)

    def extract_features(self, boxes, feature_map):
        c, h, w = feature_map.shape
        spatial_scale = self.img_size[0] / h

        features = torchvision.ops.roi_align(input=feature_map[None, :], boxes=[torch.from_numpy(boxes).float()],
                                             output_size=(16, 8), spatial_scale=spatial_scale, sampling_ratio=-1)
        K = features.shape[0]
        features = features.reshape((K, 3, -1))
        hist = torch.zeros(size=(K, 30))
        for i in range(K):
            hist_r = torch.histc(features[i, 0, :], bins=10, min=0, max=1)
            hist_g = torch.histc(features[i, 1, :], bins=10, min=0, max=1)
            hist_b = torch.histc(features[i, 2, :], bins=10, min=0, max=1)

            hist[i, :] = torch.cat((hist_r, hist_g, hist_b), dim=0)

        # features = features.reshape((K, -1)).numpy()
        features = hist.numpy()
        features = features / np.linalg.norm(features, axis=1).reshape(K, 1)
        return features

    def drawing_boxes(self, frame):
        for i in range(len(self.instances)):
            instance = self.instances[i]
            p1 = (int(instance.tracklets[-1, 0]), int(instance.tracklets[-1, 1]))
            p2 = (int(instance.tracklets[-1, 2]), int(instance.tracklets[-1, 3]))
            color = (self.colors[instance.ID % 1000, :])
            color = (int(color[0]), int(color[1]), int(color[2]))

            if instance.no_tracking_count == 0:
                color = (0, 0, 255)

            cv2.rectangle(img=frame, pt1=p1, pt2=p2, color=color, thickness=3, lineType=8, shift=0)

            for particle in instance.particles:
                center = (int(particle[4]), int(particle[5]))
                cv2.circle(img=frame, center=center, radius=1, color=color, thickness=-1)
            #text = "ID_{0}, vel={1:.2f}, {2:.2f}".format(instance.ID, instance.tracklets[-1, -2],
            #                                             instance.tracklets[-1, -1])

            cv2.putText(img=frame, text=str(instance.ID), org=p1, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.0,
                        color=(255, 255, 255), thickness=2, lineType=cv2.LINE_AA)
        return frame
