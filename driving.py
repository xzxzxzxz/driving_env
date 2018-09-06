from vehicle import vehicle
from tracks import Tra
from math import sin, cos, pi, atan, degrees, sqrt
import numpy as np
import csv
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches
from scipy.io import savemat


class LaneKeepingChanging:
    """
    Driving scenario with three lanes, starting at the center lanes.

    """

    def __init__(self,
                 dt=0.02,
                 vh_linear_model=False,
                 vh_fixed_speed=False,
                 vh_random_model=False,
                 vh_random_seed=0,
                 vh_err_bound=0.3,
                 vh_side_force=0,
                 track_data='Tra_curve2',
                 track_horizon=50,
                 story_index=0
                 ):
        self.vehicle = vehicle(dt,
                               vh_linear_model,
                               vh_fixed_speed,
                               vh_random_model,
                               vh_random_seed,
                               vh_err_bound,
                               vh_side_force
                               )
        self.track = []
        self.track.append(Tra(dt, track_data, track_horizon))
        self.track.append(Tra(dt, track_data, track_horizon, deviation=3))
        self.track.append(Tra(dt, track_data, track_horizon, deviation=-3))
        self.trajectory = []
        self.index_list = []
        self.track_select = []
        self.story_index = story_index

    def reset(self):
        track_index = self.get_track_index(0)
        self.track_select = [track_index]
        # randomly generated initial state
        X0, Y0, phi0 = self.track[track_index].getStartPosYaw()
        self.vehicle.reset_state(X0, Y0, phi0, initial_speed=2)
        vh_state = self.vehicle.get_state()
        self.trajectory = [vh_state]
        # synchronize currentIndex
        for i in range(3):
            self.track[i].currentIndex = self.track[track_index].currentIndex
        # get track measurement
        ref_state = []
        ref_status = []
        for i in range(3):
            ref, status = self.track[i].getRef(vh_state[0], vh_state[1], vh_state[2], vh_state[3])
            ref_state.append(ref)
            ref_status.append(status)

        self.index_list = [self.track[track_index].currentIndex]

        obs = np.append(vh_state[3:], ref_state[track_index])
        return obs

    def step(self, action, steps):
        track_index = self.get_track_index(steps)
        self.track_select.append(track_index)
        # simulate the vehicle forward
        self.vehicle.simulate(action)
        vh_state = self.vehicle.get_state()
        self.trajectory.append(vh_state)
        # get track measurement
        ref_state = []
        ref_status = []
        for i in range(3):
            temp_ref_state, temp_ref_status = self.track[i].getRef(vh_state[0], vh_state[1], vh_state[2], vh_state[3])
            ref_state.append(temp_ref_state)
            ref_status.append(temp_ref_status)
        self.index_list.append(self.track[track_index].currentIndex)
        obs = np.append(vh_state[3:], ref_state[track_index])
        status = ref_status[track_index]
        # calculate reward
        r = self.reward_func(obs, status)
        return obs, r, status != 0

    def reward_func(self, obs, status):
        """
        input: obs: vehicle states [v_x, v_y, r, d_f] + [ref] (see getRef in tracks.py)
               status: 1 if completed, 0 if normal driving, -1 if deviation large enough
        """
        q_error = 20

        ref = obs[4:]
        error = ref[:int(ref.size/2)]
        relyaw = ref[int(ref.size/2):]

        cost = q_error * np.sum(error[0]**2 - 0.8**2)
        v_along = obs[0] * cos(relyaw[0] * pi / 3) + obs[1] * sin(relyaw[0] * pi / 3)
        v_vertical = obs[1] * cos(relyaw[0] * pi / 3) - obs[0] *sin(relyaw[0] * pi / 3)
        reward = v_along - abs(v_vertical)

        if status == -1:
            terminalReward = -1000
        elif status == 1:
            terminalReward = 0
        else:
            terminalReward = 0

        return - cost + 2 * reward + terminalReward

    def get_track_index(self, steps):
        track_index = 0

        if self.story_index == 0:
            pass

        elif self.story_index == 1:
            if steps < 200:
                track_index = 0
            elif steps < 400:
                track_index = 1
            elif steps < 600:
                track_index = 0
            elif steps < 800:
                track_index = 2
            else:
                track_index = 0

        elif self.story_index == 2:
            if steps < 333:
                track_index = 0
            elif steps < 666:
                track_index = 1
            else:
                track_index = 0

        elif self.story_index == 3:
            if steps < 333:
                track_index = 0
            elif steps < 666:
                track_index = 2
            else:
                track_index = 0

        return track_index

    def getObservSpaceDim(self):
        return self.vehicle.getObservSpaceDim() + Tra.getRefSize()

    def getActionSpaceDim(self):
        return self.vehicle.getActionSpaceDim()

    def getActionUpperBound(self):
        return self.vehicle.getActionUpperBound()

    def getActionLowerBound(self):
        return self.vehicle.getActionLowerBound()

    def writeTraj(self, dir):
        """
        write the current trajectory to a csv file
        input: dir : directory of the csv file (same with the video dir)
        return: null (save the csv file and plot speed profile)
        """
        f = open(dir + 'traj.csv', 'w')
        log_writer = csv.writer(f, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        log_writer.writerow(['X', 'Y', 'phi', 'v_x', 'v_y', 'r', 'steering'])
        for i in range(len(self.trajectory)):
            log_writer.writerow(self.trajectory[i])
        f.close()

    def render_traj_history(self):
        # plot the speed profile
        phi_array = np.array(self.trajectory)[:, 2]
        for i in range(len(self.trajectory)):
            phi_array[i] = phi_array[i]% (2 * pi)
            if phi_array[i] > pi:
                phi_array[i] -= 2 * pi
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(phi_array, 'g--', label='yaw angle phi')
        # ax.plot(np.array(self.trajectory)[:, 3], '--', label='longitudinal speed v_x')
        ax.plot(np.array(self.trajectory)[:, 4], 'b--', label='lateral speed v_y')
        ax.plot(np.array(self.trajectory)[:, 5], 'r--', label='yaw rate r')
        ax.plot(np.array(self.trajectory)[:, 6], 'y--', label='steering angle d_f')
        ax.legend()
        plt.show()

    def render(self, interval, itr, dir, show=0):
        NumtimeStep = len(self.trajectory)
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        plt.axis('scaled')
        track0Plot, = plt.plot([], [], color='green')
        track0Plot.set_zorder = 0
        track1Plot, = plt.plot([], [], color='green')
        track1Plot.set_zorder = 0
        track2Plot, = plt.plot([], [], color='green')
        track2Plot.set_zorder = 0
        tracksPlot = [track0Plot, track1Plot, track2Plot]
        tracks = [self.track[0], self.track[1], self.track[2]]
        self.carPlot = patches.Rectangle((0,0), 3.8, 1.8, 0, color='green')
        self.carPlot.set_zorder = 1
        ax.add_patch(self.carPlot)
        plt.xlabel('X(m)')
        plt.ylabel('Y(m)')
        plt.ylim(-5, 5)
        animate = animation.FuncAnimation(fig, self.updatePlot, NumtimeStep,
            fargs = (tracks, ax, tracksPlot), interval = interval)
        if show:
            plt.show()
        else:
            Writer = animation.writers['ffmpeg']
            writer = Writer(fps=50, metadata=dict(artist='Me'), bitrate=1800)
            name = dir + 'video' + str(itr) +".mp4"
            animate.save(name,writer=writer)
            data = {'trajectory': np.array(self.trajectory), 'indexList': np.array(self.index_list), 'trackX': self.track[0].x, 'trackY': self.track[0].y}
            savemat(dir + 'data' + str(itr) + '.mat', data)

    def updatePlot(self, num, tracks, ax, tracksPlot):
        self.carPlot = updatePlot(num, self.trajectory, self.index_list, tracks, ax, self.carPlot, tracksPlot, self.track_select)

def updatePlot(num,
               traj,
               indexList,
               tracks,
               ax,
               carPlot,
               tracksPlot,
               trackselect=None,
               obstacle_traj=None,
               obstacle_plot=None):
    """
    the function used to update one frame of the plot
    :param num: steps index in a trajectory
    :param traj: the traj list recording the host vehicle's position and yaw
    :param indexList: the list storing the vehicle's position in the track
    :param tracks: the list of tracks to be ploted
    :param carPlot: the plot variable of the vehicle (rectangular)
    :param tracksPlot: the list of the plots pg the tracks (lines)
    :param trackselect: the list storing hte selected track
    :param obstacle_traj: the list storing the obstacle traj, including the position and yaw
    :param obstacle_plot:  the list storing the obstacle plot variable (rectangular)
    :return: void
    """
    stateK = traj[num]
    x = stateK[0] - sqrt(1.8**2+3.8**2)/2 * cos(stateK[2]+atan(1.8/3.8))
    y = stateK[1] - sqrt(1.8**2+3.8**2)/2 * sin(stateK[2]+atan(1.8/3.8))
    index = indexList[num]
    # plot lanes
    for i, track, trackPlot in zip(range(len(tracks)), tracks, tracksPlot):
        if trackselect:
            if trackselect[num] == i:
                trackPlot.set_color('green')
            else:
                trackPlot.set_color('red')
        TrackX = track.x[max(0, index - 500):min(track.size, index + 1001)]
        TrackY = track.y[max(0, index - 500):min(track.size, index + 1001)]
        trackData = np.array([TrackX, TrackY])
        trackPlot.set_data(trackData)
        trackPlot.axes.set_xlim(TrackX[0], TrackX[-1])
        trackPlot.set_zorder = 0
    # plot vehicle
    stateK = traj[num]
    x = stateK[0] - sqrt(1.8**2+3.8**2)/2 * cos(stateK[2]+atan(1.8/3.8))
    y = stateK[1] - sqrt(1.8**2+3.8**2)/2 * sin(stateK[2]+atan(1.8/3.8))
    carPlot.remove()
    carPlot = patches.Rectangle((x,y), 3.8, 1.8, degrees(stateK[2]), color='green')
    carPlot.set_zorder = 1
    ax.add_patch(carPlot)
    # plot obstacle
    if obstacle_plot and obstacle_traj:
        stateOb = obstacle_traj[num]
        xOb = stateOb[0] - sqrt(1.8 ** 2 + 3.8 ** 2) / 2 * cos(stateOb[2] + atan(1.8 / 3.8))
        yOb = stateOb[1] - sqrt(1.8 ** 2 + 3.8 ** 2) / 2 * sin(stateOb[2] + atan(1.8 / 3.8))
        obstacle_plot.set_xy((xOb, yOb))
        obstacle_plot._angle = degrees(stateOb[2])
        obstacle_plot.visible = True
    return carPlot
