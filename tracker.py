from __future__ import division
import torch
import torchvision
import numpy as np
from scipy.optimize import linear_sum_assignment
import cv2
from collections import deque


class Player:
    next_ID = 0

    def __init__(self, box, num_particles, img_size=(1080, 1920)):
        """
        :param box: (8, ), [x1, y1, x2, y2, ctr_x, ctr_y, w, h, v_x, v_y]
        :param feature: (1, K)
        :param num_particles: scalar
        :param img_size: (h, w)
        """

        self.ID = Player.next_ID
        Player.next_ID += 1
        self.img_size = img_size
        self.no_tracking_cnt = 0
        self.detector_tracking_cnt = 0

        """
        tracklet: [x1, y1, x2, y2, ctr_x, ctr_y, w, h, v_x, v_y]
        (x1, y1): left top location
        (x2, y2): right bottom locatoin
        (ctr_x, ctr_y): the center of location
        (w, h): width and height
        (v_x, v_y): velocity of x-axis and y-axis
        """
        self.ind = 1
        self.tracklets = np.zeros(shape=(60, 10))
        self.tracklets[0, :8] = np.array(box).reshape((1, 8))
        self.average_shape = np.zeros(2)
        self.average_shape[:] = self.tracklets[0, 6:8]
        self.flag = True

        self.counter = 0
        self.detector_tracked_ctr = deque(maxlen=5)
        self.detector_tracked_ctr.append(np.array([self.tracklets[0, 4], self.tracklets[0, 5], self.counter]))

        """
        particles: [x1, y1, x2, y2, ctr_x, ctr_y, w, h, v_x, v_y, weight]
        """
        self.num_particles = num_particles
        self.particles = np.zeros(shape=(self.num_particles, 11))
        self.particles[:, 4:6] = self.tracklets[0, 4:6] + (self.tracklets[0, 6:8].reshape(1, 2) * 0.5) * np.random.randn(
            self.num_particles, 2)
        self.particles[:, 6:8] = self.tracklets[0, 6:8]
        self.particles[:, 8:10] = 0.0
        self.particles[:, 10] = 1 / self.num_particles
        self.particles[:, 0:2] = self.particles[:, 4:6] - self.particles[:, 6:8] * 0.5
        self.particles[:, 2:4] = self.particles[:, 4:6] + self.particles[:, 6:8] * 0.5

    def _update(self, particle, detector):
        self.counter += 1

        if detector:
            self.detector_tracking_cnt += 1
            self.no_tracking_cnt = 0
            item = np.array([particle[4], particle[5], self.counter])
            self.detector_tracked_ctr.append(item)
        else:
            self.detector_tracking_cnt = 0
            self.no_tracking_cnt += 1

        if self.ind == 60:
            self.tracklets[:59, :] = self.tracklets[1:60, :]
        else:
            self.ind += 1

        self.tracklets[self.ind - 1, :] = particle[0:10]

    def _particle_update(self, box, prev_loc, cur_loc, std_p, std_v, mode):
        norm = np.array((self.img_size[1], self.img_size[0])).reshape(1, 2)
        score = 0
        if mode == 0:
            loc_score = np.sqrt(np.sum(((self.particles[:, 4:6] - box[4:6].reshape(1, 2)) / norm) ** 2, axis=1))
            loc_score = np.exp(-0.5 * ((loc_score / std_p) ** 2))

            vel = (box[4:6] - self.tracklets[self.ind - 1, 4:6]).reshape(1, 2)
            vel_score = np.sqrt(np.sum(((self.particles[:, 8:10] - vel) / norm) ** 2, axis=1))
            vel_score = np.exp(-0.5 * ((vel_score / std_v) ** 2))

            score = loc_score * vel_score + 1e-8

            self.average_shape = 0.25 * box[6:8] + 0.75 * self.average_shape
        elif mode == 1:
            norm = norm.reshape(1, 1, 2)
            prev_loc_dist = (self.tracklets[self.ind - 1, 4:6].reshape(1, 2) - prev_loc.reshape(-1, 2))

            temp = np.sum(prev_loc ** 2, axis=1)
            ind = np.argsort(temp)
            idx = np.min([len(ind), 20])
            ind = ind[:idx]

            prev_loc_dist = prev_loc_dist[ind, :].reshape(1, idx, 2) / norm
            cur_loc_dist = (self.particles[:, 4:6].reshape(-1, 1, 2) - cur_loc[ind, :].reshape(1, -1, 2)) / norm

            score = np.mean(np.sqrt(np.sum((prev_loc_dist - cur_loc_dist) ** 2, axis=2)), axis=1)
            score = np.exp(-score) + 1e-8

        idx = np.argmax(score)
        tracklet = self.particles[idx, :]

        if mode == 0:
            if len(self.detector_tracked_ctr) >= 5:
                tracklet[8:10] = (tracklet[4:6] - self.detector_tracked_ctr[0][0:2]) / (self.counter - self.detector_tracked_ctr[0][2] + 1)
                self.detector_tracked_ctr.popleft()
            elif self.flag:
                tracklet[8:10] = 0.5 * (tracklet[4:6] - self.tracklets[self.ind-1, 4:6]) + 0.5 * (self.tracklets[self.ind-1, 8:10])
            else:
                tracklet[8:10] = (tracklet[4:6] - self.tracklets[self.ind - self.no_tracking_cnt - 1, 4:6]) / (self.no_tracking_cnt + 1)
            self.flag = True
        elif mode == 1:
            #tracklet[8:10] = 0.5 * self.tracklets[self.ind-1, 8:10]
            tracklet[8:10] = 0.3 * (tracklet[4:6] - self.tracklets[self.ind-1, 4:6]) + 0.7 * self.tracklets[self.ind - self.no_tracking_cnt-1, 8:10]
            self.flag = False

        self._update(tracklet, detector=True if mode == 0 else False)
        self.particles[:, -1] = (score / np.sum(score))

        return tracklet

    def _particle_predict(self):
        g_num = 0
        pred_particles = np.zeros(shape=(self.num_particles, 11))
        pred_particles[:, -1] = 1 / self.num_particles

        ind = np.argsort(-self.particles[:, -1])

        scale_p = self.average_shape.reshape(1, 2) * np.power(0.5, np.min((len(self.tracklets), 10))) / np.power(0.5, np.min((self.no_tracking_cnt, 7)))
        scale_v = self.tracklets[self.ind-1, 8:10].reshape(1, 2) * np.power(0.1, np.min((len(self.tracklets), 10))) / np.power(0.1, np.min((self.no_tracking_cnt, 7)))

        for i in ind:
            num = np.int(np.around(self.particles[i, -1] * self.num_particles))
            if g_num + num > self.num_particles:
                num = self.num_particles - g_num

            pred_particles[g_num:g_num + num, 4:6] = self.particles[i, 4:6] + self.tracklets[self.ind - 1, 8:10] + scale_p * np.random.randn(num, 2)
            pred_particles[g_num:g_num + num, 6:8] = self.average_shape
            g_num += num
            if g_num >= self.num_particles:
                break

        self.particles = pred_particles
        self.particles[:, 4] = np.clip(self.particles[:, 4], a_min=0.0, a_max=self.img_size[1])
        self.particles[:, 5] = np.clip(self.particles[:, 5], a_min=0.0, a_max=self.img_size[0])
        self.particles[:, 0:2] = self.particles[:, 4:6] - self.particles[:, 6:8] * 0.5
        self.particles[:, 2:4] = self.particles[:, 4:6] + self.particles[:, 6:8] * 0.5
        self.particles[:, 8:10] = self.tracklets[self.ind - 1, 8:10] + np.abs(np.random.randn(self.num_particles, 2)) * scale_v


class PlayerTracker:
    def __init__(self, max_players=50, num_particles=250, init_thres=0.2, detect_thres=0.4, vel_thres=2.0, std_s=2,
                 std_p=0.01, std_m=0.01, std_v=1.0, img_size=(1080, 1920)):
        self.max_players = max_players
        self.num_particles = num_particles
        self.img_size = img_size
        self.init_thres = init_thres
        self.detect_thres = detect_thres
        self.vel_thres = vel_thres
        self.std_s = std_s
        self.std_p = std_p
        self.std_m = std_m
        self.std_v = std_v

        self.active = False
        self.players = []
        self.prev_players = None
        self.cur_players = None

        self.colors = np.random.randint(low=0, high=255, size=(1000, 3))

    def _tracking(self, boxes):
        boxes = self._extend_boxes(boxes=boxes)
        if not self.active:
            for i in range(len(boxes)):
                player = Player(box=boxes[i, :], num_particles=self.num_particles)
                self.players.append(player)
            self.active = True
            self.prev_players = np.array(self.players)

        else:
            # 1. Construct Score matrix using detector boxes, features, tracklets of target
            score_mat = self._calc_score_mat(boxes=boxes, vel_thres=self.vel_thres, std_s=self.std_s, std_p=self.std_p,
                                             std_m=self.std_m)

            # 2. Association targets with detector boxes.
            matched_row_ind, matched_col_ind = self._matching(score_mat=score_mat, detect_thres=self.detect_thres)

            # 3. Particle update and predict
            # 3.1 consider player associated with bounding box
            selected_particles = []
            for i in range(len(matched_row_ind)):
                player = self.players[matched_row_ind[i]]
                box = boxes[matched_col_ind[i], :]
                particle = player._particle_update(box=box, mode=0, prev_loc=None, cur_loc=None, std_p=self.std_p,
                                                   std_v=self.std_v)
                selected_particles.append(particle)
                player._particle_predict()

            # 3.2 consider player not to be associated with bounding box
            no_matched_row_ind = np.setdiff1d(np.array(range(len(self.players))), matched_row_ind, assume_unique=True)
            prev_loc, cur_loc = self._intersection_prev_cur(matched_row_ind=matched_row_ind)

            for i in range(len(no_matched_row_ind)):
                player = self.players[no_matched_row_ind[i]]
                particle = player._particle_update(box=None, mode=1, prev_loc=prev_loc, cur_loc=cur_loc,
                                                   std_p=self.std_p, std_v=self.std_v)
                selected_particles.append(particle)
                player._particle_predict()
            self.prev_players = self.cur_players

            # 4. Initialize detector bounding boxes below IoU threshold with tracted boxes
            selected_particles = np.array(selected_particles)
            self._initialization(matched_col_ind=matched_col_ind, boxes=boxes, selected_particles=selected_particles)

            # 5. terminalize when player is missed during some frames.
            self._terminalization()

    def _calc_score_mat(self, boxes, vel_thres, std_s, std_p, std_m):
        player_tracklets = []
        player_particles_ctr = []
        for player in self.players:
            player_tracklets.append(player.tracklets[player.ind - 1, :])
            player_particles_ctr.append(player.particles[:, 4:6])
        player_tracklets = np.array(player_tracklets)
        player_particles_ctr = np.array(player_particles_ctr)

        g_mat = self._calc_gating_mat(tracklets=player_tracklets, boxes=boxes, vel_thres=vel_thres, std_s=std_s,
                                      std_p=std_p, std_m=std_m)
        p_mat = self._calc_particles_mat(particles_ctr=player_particles_ctr, boxes=boxes, std_p=std_p)

        score_mat = g_mat * p_mat
        return score_mat

    def _calc_gating_mat(self, tracklets, boxes, vel_thres, std_s, std_p, std_m):
        m, n = len(tracklets), len(boxes)
        g = np.zeros(shape=(m, n))

        # 1. Bounding box height score
        h = (tracklets[:, 7].reshape(m, 1) - boxes[:, 7].reshape(1, n)) / tracklets[:, 7].reshape(m, 1)
        h = np.exp(-0.5 * ((h / std_s) ** 2))

        # 2. Motion score depending on velocity threshhold
        vel = np.sqrt(np.sum(tracklets[:, -2:] ** 2, axis=1))
        mask = (vel < vel_thres)

        # 2.1 low velocity --> score by distance between bounding boxes and targets
        g[mask, :] = ((tracklets[mask, 4].reshape(-1, 1) + tracklets[mask, -2].reshape(-1, 1) - boxes[:, 4].reshape(1,
                                                                                                                    -1)) /
                      self.img_size[1]) ** 2 + \
                     ((tracklets[mask, 5].reshape(-1, 1) + tracklets[mask, -1].reshape(-1, 1) - boxes[:, 5].reshape(1,
                                                                                                                    -1)) /
                      self.img_size[0]) ** 2
        g[mask, :] = np.exp(-0.5 * g[mask, :] / (std_p ** 2))

        # 2.2 high velocity --> score by distance between bounding boxes and lines of targets
        norm = np.array([self.img_size[1], self.img_size[0]]).reshape(1, 1, 2)
        p = (boxes[:, 4:6].reshape(1, -1, 2) - tracklets[~mask, 4:6].reshape(-1, 1, 2)) / norm
        n = (tracklets[~mask, -1:-3:-1].reshape(-1, 1, 2)) / norm[:, :, -1::-1]
        n[:, 0, 0] *= -1
        g[~mask, :] = np.abs(np.sum(n * p, axis=2)) / np.sqrt(np.sum(n ** 2, axis=2)).reshape(-1, 1)
        g[~mask, :] = np.exp(-0.5 * ((g[~mask, :] / std_m) ** 2))
        return g * h

    def _calc_particles_mat(self, particles_ctr, boxes, std_p):
        # particles_ctr : (n x100 x 2)
        # boxes : (m x 8)
        norm = np.array([self.img_size[1], self.img_size[0]]).reshape((1, 1, 1, 2))
        particles_mat = np.sum(
            ((particles_ctr.reshape(-1, 1, self.num_particles, 2) - boxes[:, 4:6].reshape(1, -1, 1, 2)) / norm) ** 2,
            axis=3)
        particles_mat = np.exp(-0.5 * particles_mat / ((std_p * 4) ** 2))
        particles_mat = np.sum(particles_mat, axis=2) / self.num_particles
        return particles_mat

    def _matching(self, score_mat, detect_thres):
        row_ind, col_ind = linear_sum_assignment(-score_mat)
        mask = (score_mat[row_ind, col_ind] > detect_thres)
        matched_row_ind = row_ind[mask]
        matched_col_ind = col_ind[mask]

        return np.array(matched_row_ind), np.array(matched_col_ind)

    def _initialization(self, matched_col_ind, boxes, selected_particles):
        no_matched_col_ind = np.setdiff1d(np.array(range(len(boxes))), matched_col_ind, assume_unique=True)
        no_matched_boxes = boxes[no_matched_col_ind, :]

        if len(no_matched_boxes) == 0:
            return None
        if len(self.players) > self.max_players:
            return None

        m, n = len(no_matched_boxes), len(selected_particles)
        temp1 = np.tile(no_matched_boxes[:, 0].reshape(m, 1), (1, n))
        temp2 = np.tile(selected_particles[:, 0].reshape(1, n), (m, 1))
        x1 = np.max((temp1, temp2), axis=0)

        temp1 = np.tile(no_matched_boxes[:, 1].reshape(m, 1), (1, n))
        temp2 = np.tile(selected_particles[:, 1].reshape(1, n), (m, 1))
        y1 = np.max((temp1, temp2), axis=0)

        temp1 = np.tile(no_matched_boxes[:, 2].reshape(m, 1), (1, n))
        temp2 = np.tile(selected_particles[:, 2].reshape(1, n), (m, 1))
        x2 = np.min((temp1, temp2), axis=0)

        temp1 = np.tile(no_matched_boxes[:, 3].reshape(m, 1), (1, n))
        temp2 = np.tile(selected_particles[:, 3].reshape(1, n), (m, 1))
        y2 = np.min((temp1, temp2), axis=0)

        width = np.clip(x2 - x1, a_min=0, a_max=None)
        height = np.clip(y2 - y1, a_min=0, a_max=None)
        intersection = width * height
        area1 = no_matched_boxes[:, 6] * no_matched_boxes[:, 7]
        area2 = selected_particles[:, 6] * selected_particles[:, 7]

        denominator = (area1.reshape(m, 1) + area2.reshape(1, n)) - intersection
        iou = intersection / denominator
        mask = np.sum((iou > self.init_thres), axis=1)

        new_boxes = no_matched_boxes[mask == 0]

        for box in new_boxes:
            player = Player(box=box, num_particles=self.num_particles, img_size=self.img_size)
            self.players.append(player)

    def _terminalization(self):
        idx = 0
        while idx < len(self.players):
            if self.players[idx].no_tracking_cnt > 30:
                del self.players[idx]
            else:
                idx += 1
        return True

    def _drawing_boxes(self, frame):
        for player in self.players:
            p1 = (int(player.tracklets[player.ind - 1, 0]), int(player.tracklets[player.ind - 1, 1]))
            p2 = (int(player.tracklets[player.ind - 1, 2]), int(player.tracklets[player.ind - 1, 3]))
            color = (self.colors[player.ID % 1000, :])
            color = (int(color[0]), int(color[1]), int(color[2]))

            if player.no_tracking_cnt == 0:
                color = (0, 0, 255)

            cv2.rectangle(img=frame, pt1=p1, pt2=p2, color=color, thickness=3, lineType=8, shift=0)

            for particle in player.particles:
                center = (int(particle[4]), int(particle[5]))
                cv2.circle(img=frame, center=center, radius=1, color=color, thickness=-1)
            text = "ID_{0}".format(player.ID)
            # text = "ID_{0}, vel={1:.2f}, {2:.2f}".format(player.ID, player.tracklets[player.ind-1, -2],
            #                                             player.tracklets[player.ind-1, -1])

            cv2.putText(img=frame, text=text, org=p1, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5,
                        color=(255, 255, 255), thickness=2, lineType=cv2.LINE_AA)
        return frame

    def _extend_boxes(self, boxes):
        m = len(boxes)
        extended_boxes = np.zeros(shape=(m, 8))
        extended_boxes[:, :4] = boxes[:, :4]
        extended_boxes[:, 4:6] = (boxes[:, 0:2] + boxes[:, 2:4]) / 2
        extended_boxes[:, 6:8] = (boxes[:, 2:4] - boxes[:, 0:2])

        return extended_boxes

    def _intersection_prev_cur(self, matched_row_ind):
        self.cur_players = np.array(self.players)[matched_row_ind]
        x = np.sum(np.equal(self.prev_players.reshape(-1, 1), self.cur_players.reshape(1, -1)),
                                  axis=1)
        repeat_detection = self.prev_players[x==1]
        prev_loc = []
        cur_loc = []
        for player in repeat_detection:
            prev_loc.append(player.tracklets[player.ind - 2, 4:6])
            cur_loc.append(player.tracklets[player.ind - 1, 4:6])
        prev_loc = np.array(prev_loc)
        cur_loc = np.array(cur_loc)

        return prev_loc, cur_loc

