from vehicle import vehicle
from tracks import Track
from math import sin, cos, pi, atan, degrees, sqrt
import numpy as np
import csv
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches


class Driving:
    """
    Driving scenario with three lanes, starting at the center lane.
    Consisting many stories, including
        lane keeping (0), lane changing (1,2,3), and obstacle avoidance (4)

    """

    def __init__(self,
                 dt=0.02,
                 vh_linear_model=False,
                 vh_fixed_speed=False,
                 vh_random_model=False,
                 vh_random_seed=0,
                 vh_err_bound=0.3,
                 vh_side_force=0,
                 track_data='sine_curve',
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
        self.tracks = []
        self.tracks.append(Track(dt, track_data, track_horizon))
        self.tracks.append(Track(dt, track_data, track_horizon, deviation=3))
        self.tracks.append(Track(dt, track_data, track_horizon, deviation=-3))
        self.story_index = story_index
        self.trajectory = []
        self.index_list = []
        self.track_select = []
        self.debug_view = []  # stores the closest point, the foresee point, and foresee closest point
        self.obstacle_info = self.get_obstacle_info()
        self.obstacles_traj = []
        if len(self.obstacle_info):
            for i in range(len(self.obstacle_info)):
                self.obstacles_traj.append([])
        # reset in init.
        self.reset()

    def get_observation(self):
        # return needed observation in an array:
        vh_state = self.vehicle.get_state()
        ref_state = []
        ref_status = []
        ref_debugview = []
        for i in range(3):
            temp_ref_state, temp_ref_status, debugview = self.tracks[i].getRef(vh_state[0], vh_state[1], vh_state[2], vh_state[3])
            ref_state.append(temp_ref_state)
            ref_status.append(temp_ref_status)
            ref_debugview.append(debugview)

        obs = np.append(vh_state[3:], ref_state[self.track_index])
        status = ref_status[self.track_index]
        self.debug_view.append(ref_debugview[self.track_index])
        return obs, status

    def reset(self):
        self.step_num = 0
        self.track_index = self.get_track_index(0)
        self.track_select = [self.track_index]

        # randomly generated initial state
        X0, Y0, phi0 = self.tracks[self.track_index].getStartPosYaw()
        self.vehicle.reset_state(X0, Y0, phi0, initial_speed=15)
        vh_state = self.vehicle.get_state()
        self.trajectory = [vh_state]

        # synchronize currentIndex
        for i in range(3):
            self.tracks[i].currentIndex = self.tracks[self.track_index].currentIndex
        
        # initiate obstacle
        if len(self.obstacle_info):
            ref_obst = []
            obstacle_fail = False
            for i in range(len(self.obstacle_info)):
                self.tracks[self.obstacle_info[i][0]].setStartPosObstacle(self.obstacle_info[i][1],
                                                                          self.obstacle_info[i][2],
                                                                          self.obstacle_info[i][3])
                obstacle_pos = [[self.tracks[self.obstacle_info[i][0]].x[
                                     self.tracks[self.obstacle_info[i][0]].obstacleIndex],
                                 self.tracks[self.obstacle_info[i][0]].y[
                                     self.tracks[self.obstacle_info[i][0]].obstacleIndex],
                                 self.tracks[self.obstacle_info[i][0]].psi[
                                     self.tracks[self.obstacle_info[i][0]].obstacleIndex]]]
                if self.obstacle_info[i][3]:
                    obstacle_pos.append([self.tracks[self.obstacle_info[i][0]].x[
                                             self.tracks[self.obstacle_info[i][0]].obstacleIndex +
                                             self.tracks[self.obstacle_info[i][0]].obstacleDistance],
                                         self.tracks[self.obstacle_info[i][0]].y[
                                             self.tracks[self.obstacle_info[i][0]].obstacleIndex +
                                             self.tracks[self.obstacle_info[i][0]].obstacleDistance],
                                         self.tracks[self.obstacle_info[i][0]].psi[
                                             self.tracks[self.obstacle_info[i][0]].obstacleIndex +
                                             self.tracks[self.obstacle_info[i][0]].obstacleDistance]])
                self.obstacles_traj[i] = [obstacle_pos]
                refObstacle, collision = \
                    self.tracks[self.obstacle_info[i][0]].getRefObstacle(vh_state=vh_state)
                ref_obst.append(refObstacle)
                obstacle_fail = obstacle_fail or collision


        self.index_list = [self.tracks[self.track_index].currentIndex]

        obs, _ = self.get_observation()
        return obs

    def step(self, action):
        self.step_num += 1
        self.track_index = self.get_track_index(self.step_num)
        self.track_select.append(self.track_index)

        # simulate the vehicle forward
        self.vehicle.simulate(action)
        vh_state = self.vehicle.get_state()
        self.trajectory.append(vh_state)

        # simulate the obstacles forward
        obstacle_fail = False
        if len(self.obstacle_info):
            ref_obst = []
            for i in range(len(self.obstacle_info)):
                self.tracks[self.obstacle_info[i][0]].obstacleMove()
                obstacle_pos = [[self.tracks[self.obstacle_info[i][0]].x[
                                     self.tracks[self.obstacle_info[i][0]].obstacleIndex],
                                 self.tracks[self.obstacle_info[i][0]].y[
                                     self.tracks[self.obstacle_info[i][0]].obstacleIndex],
                                 self.tracks[self.obstacle_info[i][0]].psi[
                                     self.tracks[self.obstacle_info[i][0]].obstacleIndex]]]
                if self.obstacle_info[i][3]:
                    obstacle_pos.append([self.tracks[self.obstacle_info[i][0]].x[
                                             self.tracks[self.obstacle_info[i][0]].obstacleIndex +
                                             self.tracks[self.obstacle_info[i][0]].obstacleDistance],
                                         self.tracks[self.obstacle_info[i][0]].y[
                                             self.tracks[self.obstacle_info[i][0]].obstacleIndex +
                                             self.tracks[self.obstacle_info[i][0]].obstacleDistance],
                                         self.tracks[self.obstacle_info[i][0]].psi[
                                             self.tracks[self.obstacle_info[i][0]].obstacleIndex +
                                             self.tracks[self.obstacle_info[i][0]].obstacleDistance]])
                self.obstacles_traj[i].append(obstacle_pos)
                refObstacle, collision = \
                    self.tracks[self.obstacle_info[i][0]].getRefObstacle(vh_state=vh_state)
                ref_obst.append(refObstacle)
                obstacle_fail = obstacle_fail or collision

        self.index_list.append(self.tracks[self.track_index].currentIndex)
        obs, status = self.get_observation()
        # calculate reward
        if len(self.obstacle_info) and obstacle_fail:
            status = -1
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
        # v_vertical = obs[1] * cos(relyaw[0] * pi / 3) - obs[0] *sin(relyaw[0] * pi / 3)
        reward = v_along # - abs(v_vertical)

        if status == -1:
            terminalReward = -1000
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

        else:
            pass

        return track_index

    def get_obstacle_info(self):
        """
        get the info of the obstacles (if applicable)
        :return: list of obstacle info for each obstacle, which is
                 [track index, how far, how fast, whether there are multiple obstacles on one track]
        """
        obstacle_info = []

        if self.story_index <= 3:
            pass

        elif self.story_index == 4:
            obstacle_info.append([0, 500, 10, False])

        elif self.story_index == 5:
            obstacle_info.append([0, 500, 10, False])
            obstacle_info.append([1, 300, 10, False])

        elif self.story_index == 6:
            obstacle_info.append([0, 300, 10, False])
            obstacle_info.append([1, 500, 10, False])

        elif self.story_index == 7:
            obstacle_info.append([1, 300, 10, True])
            obstacle_info.append([2, 500, 5, True])
            obstacle_info.append([0, 700, 8, False])

        return obstacle_info

    def getObservSpaceDim(self):
        return self.vehicle.getObservSpaceDim() + Track.getRefSize()

    def getActionSpaceDim(self):
        return self.vehicle.getActionSpaceDim()

    def getActionUpperBound(self):
        return self.vehicle.getActionUpperBound()

    def getActionLowerBound(self):
        return self.vehicle.getActionLowerBound()

    def writeTraj(self, filename):
        """
        write the current trajectory to a csv file
        input: dir : directory of the csv file (same with the video dir)
        return: null (save the csv file)
        """
        f = open(filename + '.csv', 'w')
        log_writer = csv.writer(f, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        log_writer.writerow(['X', 'Y', 'phi', 'v_x', 'v_y', 'r', 'steering'])
        for i in range(len(self.trajectory)):
            log_writer.writerow(self.trajectory[i])
        f.close()

    def render_traj_history(self):
        """
        visualize the history of the key vehicle states
        return: null (plot speed profile)
        """
        phi_array = np.array(self.trajectory)[:, 2]
        for i in range(len(self.trajectory)):
            phi_array[i] = phi_array[i] % (2 * pi)
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

    def render(self, filename, show=False, debugview=False):
        NumtimeStep = len(self.trajectory)
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        plt.axis('scaled')

        # plot the lanes (tracks)
        self.tracksPlot = []
        for i in range(len(self.tracks)):
            trackPlot,  = plt.plot([], [], color='green')
            self.tracksPlot.append(trackPlot)
            self.tracksPlot[i].set_zorder = 0

        # plot the vehicle
        self.carPlot = patches.Rectangle((0, 0), 3.8, 1.8, 0, color='green')
        self.carPlot.set_zorder = 1
        ax.add_patch(self.carPlot)

        # plot the three debug view lines which connect:
        # (1) current pos and future pos
        # (2) current pos and current closest point
        # (3) future pos and future closest point
        self.debugviewPlot = []
        for i in range(3):
            self.debugviewPlot.append(patches.Arrow(0, 0, 0, 0, width=0.1))
            self.debugviewPlot[i].set_zorder = 2
            ax.add_patch(self.debugviewPlot[i])

        # plot the obstacles if any
        self.obstaclePlot = []
        if len(self.obstacle_info):
            for i in range(len(self.obstacle_info)):
                one_track_obstacle_plot = [patches.Rectangle((0, 0), 3.8, 1.8, 0, color='red')]
                if self.obstacle_info[i][3]:
                    one_track_obstacle_plot.append(patches.Rectangle((0, 0), 3.8, 1.8, 0, color='red'))
                self.obstaclePlot.append(one_track_obstacle_plot)

                self.obstaclePlot[i][0].set_zorder = 1
                ax.add_patch(self.obstaclePlot[i][0])
                if self.obstacle_info[i][3]:
                    self.obstaclePlot[i][1].set_zorder = 1
                    ax.add_patch(self.obstaclePlot[i][1])

        plt.xlabel('X(m)')
        plt.ylabel('Y(m)')
        plt.ylim(-5, 5)
        animate = animation.FuncAnimation(fig,
                                          self.updatePlot,
                                          frames=NumtimeStep,
                                          fargs=(ax, debugview),
                                          interval=100)
        if show:
            plt.show()
        else:
            Writer = animation.writers['ffmpeg']
            writer = Writer(fps=50, metadata=dict(artist='Me'), bitrate=1800)
            name = filename + ".mp4"
            animate.save(name, writer=writer)

    def updatePlot(self, frames, ax, debugview):
        updatePlot(frames,
                   self.trajectory,
                   self.index_list,
                   self.tracks,
                   self.debug_view,
                   ax,
                   debugview,
                   self.carPlot,
                   self.tracksPlot,
                   self.debugviewPlot,
                   self.track_select,
                   self.obstacles_traj,
                   self.obstaclePlot)


def updatePlot(frames,
               traj,
               indexList,
               tracks,
               debug_view_data,
               ax,
               debugviewBool,
               carPlot,
               tracksPlot,
               debugviewPlot,
               trackselect=None,
               obstacles_traj=None,
               obstacles_plot=None):
    """
    the function used to update one frame of the plot
    :param frames: steps index in a trajectory
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
    stateK = traj[frames]
    x = stateK[0] - sqrt(1.8**2+3.8**2)/2 * cos(stateK[2]+atan(1.8/3.8))
    y = stateK[1] - sqrt(1.8**2+3.8**2)/2 * sin(stateK[2]+atan(1.8/3.8))
    index = indexList[frames]
    # plot lanes
    for i, track, trackPlot in zip(range(len(tracks)), tracks, tracksPlot):
        if trackselect:
            if trackselect[frames] == i:
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
    stateK = traj[frames]
    x = stateK[0] - sqrt(1.8**2+3.8**2)/2 * cos(stateK[2]+atan(1.8/3.8))
    y = stateK[1] - sqrt(1.8**2+3.8**2)/2 * sin(stateK[2]+atan(1.8/3.8))
    carPlot.set_xy((x, y))
    carPlot._angle = degrees(stateK[2])
    ax.add_patch(carPlot)

    # plot debugview
    if debugviewBool:
        for i in range(3):
            debugviewPlot[i].remove()
        data = debug_view_data[frames]
        debugviewPlot[0] = patches.Arrow(data[0][0],
                                         data[0][1],
                                         data[1][0] - data[0][0],
                                         data[1][1] - data[0][1],
                                         width=0.1)
        debugviewPlot[1] = patches.Arrow(data[2][0],
                                         data[2][1],
                                         data[3][0] - data[2][0],
                                         data[3][1] - data[2][1],
                                         width=0.1)
        debugviewPlot[2] = patches.Arrow(data[0][0],
                                         data[0][1],
                                         data[2][0] - data[0][0],
                                         data[2][1] - data[0][1],
                                         width=0.1)
        for i in range(3):
            ax.add_patch(debugviewPlot[i])

    # plot obstacle
    for one_track_obstacle_plot, obstacle_traj in zip(obstacles_plot, obstacles_traj):
        statesOb = obstacle_traj[frames]
        for stateOb, obstacle_plot in zip(statesOb, one_track_obstacle_plot):
            xOb = stateOb[0] - sqrt(1.8 ** 2 + 3.8 ** 2) / 2 * cos(stateOb[2] + atan(1.8 / 3.8))
            yOb = stateOb[1] - sqrt(1.8 ** 2 + 3.8 ** 2) / 2 * sin(stateOb[2] + atan(1.8 / 3.8))
            obstacle_plot.set_xy((xOb, yOb))
            obstacle_plot._angle = degrees(stateOb[2])
            obstacle_plot.visible = True
            ax.add_patch(obstacle_plot)
