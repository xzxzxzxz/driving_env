#!/usr/bin/env python
# -*- coding: utf-8 -*-
# define the track class
import scipy.io 
import scipy.optimize
import numpy as np
from math import sqrt, cos, sin, tan, pi


class Track:

    def __init__(self, dt, name, horizon=50, deviation=0, threshold=5):
        """
        track class
        input: track.mat filename
               dt: timestep, must match the dt in vehicle class
               horizon: search area, must match the horizon in vehicle class
               deviation: for a parallel lane, indicates lateral deviation with base lane
               numPrePts: num of predicted waypoints at a certain timestep
               threshold: if farther away from lane than threshold, then terminate the episode
        state: x, y, psi for the waypoints
               size: length of the reference trajectory
               threshold: if deviation larger than threshold then report failure
               currentIndex: index of waypoint corresponding to current vehicle position
        """
        waypoints_mat = scipy.io.loadmat("sine_curve.mat")[name]
        self.x     = waypoints_mat[0][:]
        self.y     = waypoints_mat[1][:] + deviation
        self.psi   = (waypoints_mat[2][:]) % (2 * pi)
        self.size  = waypoints_mat.shape[1]
        self.threshold = threshold
        self.longi_safe = 4
        self.later_safe = 2
        self.currentIndex = 0
        self.obstacleIndex = 0
        self.obstacleDistance = 0
        self.obstacleSpeed = 0
        self.dt = dt
        self.horizon = horizon

    def getStartPosYaw(self):
        """
        randomly choose a waypoint in the first 1/6 of the traj as starting point
        output: x, y, psi of the starting waypoint
        """
        self.currentIndex = np.random.random_integers(0, int(self.size/6))
        return self.x[self.currentIndex], self.y[self.currentIndex], self.psi[self.currentIndex]

    def setStartPosObstacle(self, delta_index = 5000, speed=10, two_obstacles=False):
        """
        choose a waypoint "delta_index" length after the currentIndex as the staring pos of obstacle
        Note: delta_index != 0, furthermore -> self,obstacleIndex==0 indicates no obstacle initiated
        """
        self.obstacleIndex = self.currentIndex + delta_index
        self.obstacleSpeed = speed
        if two_obstacles:
            self.obstacleDistance = np.random.random_integers(300, 500)
        return self.obstacleIndex

    def obstacleMove(self):
        """
        given the speed of the obstacle, get the new pos, and assign to self
        """
        pX = self.x[self.obstacleIndex] + self.obstacleSpeed * cos(self.psi[self.obstacleIndex]) * self.dt
        pY = self.y[self.obstacleIndex] + self.obstacleSpeed * sin(self.psi[self.obstacleIndex]) * self.dt
        self.obstacleIndex, _ = self.searchClosestPt(pX, pY, standard_index=self.obstacleIndex)

    def getRef(self, X, Y, phi, v_x):
        """
        get the reference position and angle of the vehicle with respect to the traj
        input: position: X, Y current position of the vehicle
               phi: current yaw angle of the vehicle
               v_x: current longitudinal speed
               self.dt: timestep length of the vehicle simulator
               self.horizon: max num of timesteps to predict
        output: array(ref): first half: distance error | second half: normalized angle error
                            size = self.getRefSize() = 2 * self.numPrePts
                status：1 if completed, 0 if normal driving, -1 if deviation large enough
        """
        tList = self.dt * np.array([0, self.horizon])
        errorList = []
        relaAngList = []
        debug_view = []
        
        for t in tList:
            pX = X + v_x * cos(phi) * t
            pY = Y + v_x * sin(phi) * t
            index, distanceMin = self.searchClosestPt(pX, pY, standard_index=self.currentIndex)
            debug_view.append([pX, pY])
            debug_view.append([self.x[index], self.y[index]])
            
            if t == 0:
                self.currentIndex = index

            delta_x = self.x[index] - pX
            delta_y = self.y[index] - pY
            relYaw = (self.psi[index] - phi) % (2 * pi)
            if relYaw > pi:
                relYaw -= 2 * pi

            relY = - delta_x * sin(phi) + delta_y * cos(phi)  # + if traj left of vehicle, - else
            errorList.append(distanceMin / self.threshold * np.sign(relY))
            relaAngList.append(relYaw / (pi / 3))

        PreviewIndex = index 

        if abs(errorList[0]) > 1:
            status = -1
        elif PreviewIndex == self.size:    # size - currentIndex < horizon?
            status = 1
        else:
            status = 0

        ref = errorList + relaAngList
        return np.array(ref), status, debug_view

    def getRefObstacle(self, vh_state):
        """
        get the reference position and angle of the vehicle with respect to the obstacle
        input: vh_state: ego vehicle state [X, Y, phi, v_x, v_y, r, d_f]
        output: array(ref): [vX_over_errX, err_x, err_y]
                status：0 if normal driving, -1 if hit the safety boundary
        """
        delta_x = self.x[self.obstacleIndex] - vh_state[0]
        delta_y = self.y[self.obstacleIndex] - vh_state[1]
        distSqr = delta_x*delta_x + delta_y*delta_y
        err_y = (delta_y - delta_x * tan(vh_state[2])) * cos(vh_state[2])
        err_x = sqrt(distSqr - err_y**2)
        err_phi = self.psi[self.obstacleIndex] - vh_state[2]
        err_vx = self.obstacleSpeed * cos(err_phi) - vh_state[3]
        err_vy = self.obstacleSpeed * sin(err_phi) - vh_state[4]
        err_r = - vh_state[5]

        refObstacle = [np.array([err_x, err_y, err_phi, err_vx, err_vy, err_r])]

        if abs(err_x) < 4 and abs(err_y) < 2:
            collision = True
        else:
            collision = False

        # if there are two obetacles
        if self.obstacleDistance:
            delta_x = self.x[self.obstacleIndex+self.obstacleDistance] - vh_state[0]
            delta_y = self.y[self.obstacleIndex+self.obstacleDistance] - vh_state[1]
            distSqr = delta_x * delta_x + delta_y * delta_y
            err_y = (delta_y - delta_x * tan(vh_state[2])) * cos(vh_state[2])
            err_x = sqrt(distSqr - err_y ** 2)
            err_phi = self.psi[self.obstacleIndex+self.obstacleDistance] - vh_state[2]
            err_vx = self.obstacleSpeed * cos(err_phi) - vh_state[3]
            err_vy = self.obstacleSpeed * sin(err_phi) - vh_state[4]
            err_r = - vh_state[5]

            refObstacle.append(np.array([err_x, err_y, err_phi, err_vx, err_vy, err_r]))

            if abs(err_x) < 4 and abs(err_y) < 2:
                collision = True
            else:
                pass # collision = collision

            # if the ego is before both obstacles, delete the last obstacle and create a new one
            if err_x < 0:
                self.obstacleIndex = self.obstacleIndex + self.obstacleDistance
                self.obstacleDistance = np.random.random_integers(300, 500)

        return refObstacle, collision

    def searchClosestPt(self, pX, pY, standard_index):
        """
        search for the closest point with the designated point on the traj
        input:  pX, pY: accurate point position
                self.horizon: search in +/- 10* horizon
                standard_index: search around the standard index
        output: indexMin: closest traj point's index
                distMin:  distance from the closest traj point to the accurate point

        (one trick: only compare dist**2, save the time used to compute sqrt)
        (another trick: only consider waypoints with index in currentIndex +/- 10*horizon range)
        """
        indexMin = standard_index
        distSqrMin = (pX - self.x[standard_index])**2 + (pY - self.y[standard_index])**2
        for index in range(max(standard_index - self.horizon * 10, 0), min(standard_index + self.horizon * 10, self.size)):
            distSqr = (pX - self.x[index])**2 + (pY - self.y[index])**2
            if distSqr < distSqrMin:
                indexMin = index
                distSqrMin = distSqr
        return indexMin, sqrt(distSqrMin)

    @staticmethod
    def getRefSize():
        """
        returns the size of the reference trajectory
        """
        return 4

    @staticmethod
    def getRefObSize():
        """
        returns the size of the obstacle observation
        :return:
        """
        return 3
