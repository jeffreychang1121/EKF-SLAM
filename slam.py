from __future__ import division
import numpy as np
import slam_utils
import tree_extraction
import math
from scipy.stats import chi2

##### Motion Model #####
def motion_model(u, dt, ekf_state, vehicle_params):
    """
    Computes the discretized motion model for the given vehicle as well as its Jacobian
    :param u:
    :param dt:
    :param ekf_state:
    :param vehicle_params:
    :return:
    f(x,u), a 3x1 vector corresponding to motion x_{t+1} - x_t given the odometry u.
    df/dX, the 3x3 Jacobian of f with respect to the vehicle state (x, y, phi)
    """

    # vehicle parameters
    a = vehicle_params['a']
    b = vehicle_params['b']
    L = vehicle_params['L']
    H = vehicle_params['H']

    # transform between ve and vc
    ve = u[0]
    alpha = u[1]
    vc = ve / (1 - math.tan(alpha) * H / L)

    # parameters for motion model
    x, y, phi = ekf_state['x'][0], ekf_state['x'][1], ekf_state['x'][2]
    # make sure phi is between -pi and +pi
    phi = slam_utils.clamp_angle(phi)

    # create motion model
    motion = np.zeros((3,1))

    motion[0,0] = dt * (vc * math.cos(phi) - (vc/L) * math.tan(alpha) * (a * math.sin(phi) + b * math.cos(phi)))
    motion[1,0] = dt * (vc * math.sin(phi) + (vc/L) * math.tan(alpha) * (a * math.cos(phi) - b * math.sin(phi)))
    motion[2,0] = dt * (vc * math.tan(alpha)) / L

    # create prediction model
    G = np.identity(3)
    G[0, 2] = dt * (-vc * math.sin(phi) - vc / L * math.tan(alpha) * (a * math.cos(phi) - b * math.sin(phi)))
    G[1, 2] = dt * (vc * math.cos(phi) + vc / L * math.tan(alpha) * (-a * math.sin(phi) - b * math.cos(phi)))

    return motion, G


##### Prediction Model #####
def odom_predict(u, dt, ekf_state, vehicle_params, sigmas):
    """
    Perform the propagation step of the EKF filter given an odometry measurement u
    and time step dt where u = (ve, alpha) as shown in the vehicle/motion model.
    :param u:
    :param dt:
    :param ekf_state:
    :param vehicle_params:
    :param sigmas:
    :return:
    returns the new ekf_state
    """

    ### Motion Model, Prediction Model ###
    N = len(ekf_state['x'])
    motion, G = motion_model(u, dt, ekf_state, vehicle_params)

    # motion model noise
    xy_noise = sigmas['xy']
    phi_noise = sigmas['phi']

    ### Prediction Model ###
    # update mu and sigma based on prediction

    # mu
    motion = motion.reshape(-1)
    ekf_state['x'][0] += motion[0]
    ekf_state['x'][1] += motion[1]
    ekf_state['x'][2] += motion[2]
    ekf_state['x'][2] = slam_utils.clamp_angle(ekf_state['x'][2])

    # sigma
    G_t = np.block([[G, np.zeros((3,N-3))], [np.zeros((N-3,3)), np.identity(N-3)]])
    sigma_t = np.dot(np.dot(G_t, ekf_state['P'].copy()), G_t.T)
    sigma_t[0:3, 0:3] += np.diag((xy_noise**2, xy_noise**2, phi_noise**2))

    ekf_state['P'] = slam_utils.make_symmetric(sigma_t)

    return ekf_state

##### Measurement Model #####
def gps_update(gps, ekf_state, sigmas):
    """
    Perform a measurement update of the EKF state given a GPS measurement (x,y)
    Returns the updated ekf_state.
    :param gps:
    :param ekf_state:
    :param sigmas:
    :return:
    """

    # measurement noise
    gps_noise = sigmas['gps']

    # check validity
    r = gps.reshape((2,1)) - ekf_state['x'][:2].copy().reshape((2,1))
    S = slam_utils.make_symmetric(ekf_state['P'][0:2,0:2].copy() + 10e-6)
    d = np.dot(np.dot(r.T, slam_utils.invert_2x2_matrix(S)), r)
    if d > 13.8:
        return ekf_state

    ### Kalman Gain ###
    x = ekf_state['x'][0:3].copy().reshape((3,1))
    P = ekf_state['P'][0:3,0:3].copy()

    H = np.asarray([[1,0,0],[0,1,0]])
    Q = np.diag((gps_noise**2, gps_noise**2))

    # Kalman Gain update
    K_t = np.dot(P, H.T)
    K_t = np.dot(K_t, slam_utils.invert_2x2_matrix(np.dot(np.dot(H, P), H.T) + Q))

    ### Measurement Model ###
    # update mu and sigma based on measurement
    x_t = x + np.dot(K_t, r)
    P_t = np.dot(np.identity(3) - np.dot(K_t, H), P)
    ekf_state['x'][0:3] = x_t.reshape(-1)
    ekf_state['P'][0:3,0:3] = slam_utils.make_symmetric(P_t)

    return ekf_state

##### Observation Model #####
def laser_measurement_model(ekf_state, landmark_id):
    """
    Returns the measurement model for a (range,bearing) sensor observing the
    mapped landmark with id 'landmark_id' along with its jacobian.
    :param ekf_state:
    :param landmark_id:
    :return:
    h(x, l_id): the 2x1 predicted measurement vector [r_hat, theta_hat].

    dh/dX: For a measurement state with m mapped landmarks, i.e. a state vector of
           dimension 3 + 2*m, this should return the full 2 by 3+2m Jacobian
           matrix corresponding to a measurement of the landmark_id'th feature.
    """

    # landmark ekf_state
    x_l = ekf_state['x'][3 + landmark_id * 2]
    y_l = ekf_state['x'][4 + landmark_id * 2]
    x = ekf_state['x'][0]
    y = ekf_state['x'][1]
    phi = ekf_state['x'][2]

    ### Observation Model ###
    z_t = np.zeros((2,1))
    q = np.sqrt((x_l - x)**2 + (y_l - y)**2)

    z_t[0,:] = q
    z_t[1,:] = np.arctan2(y_l - y, x_l - x) - phi
    z_t[1,:] = slam_utils.clamp_angle(z_t[1,:])

    # Compute Jacobian for observation
    H = np.zeros((2, len(ekf_state['x'])))

    H[0,0] = -(x_l - x) / q
    H[0,1] = -(y_l - y) / q
    H[0,2] = 0
    H[0,3 + landmark_id*2] = (x_l - x) / q
    H[0,4 + landmark_id*2] = (y_l - y) / q

    H[1, 0] = (y_l - y) / q**2
    H[1, 1] = -(x_l - x) / q**2
    H[1, 2] = -1
    H[1, 3 + landmark_id * 2] = -(y_l - y) / q**2
    H[1, 4 + landmark_id * 2] = (x_l - x) / q**2

    return z_t, H

def initialize_landmark(ekf_state, tree):
    """
    Initialize a newly observed landmark in the filter state, increasing its
    dimension by 2
    :param ekf_state:
    :param tree:
    :return:
    Returns the new ekf_state
    """
    n_landmark = ekf_state['num_landmarks']
    n_state = n_landmark * 2 + 3
    tree_xy = slam_utils.tree_to_global_xy([tree], ekf_state)

    # add new landmarks into state
    x_new = np.zeros(n_state+2)
    x_new[:n_state] = ekf_state['x']
    x_new[n_state] = tree_xy[0]
    x_new[n_state+1] = tree_xy[1]

    # add new landmarks covariance
    P_new = np.zeros((n_state+2, n_state+2))
    P_new[0:n_state, 0:n_state] = ekf_state['P']
    P_new[n_state:n_state+2, n_state:n_state+2] = np.diag((104,104)) # 105

    # update state and covariance
    ekf_state['x'] = x_new
    ekf_state['P'] = P_new
    ekf_state['num_landmarks'] += 1

    return ekf_state

##### Measurement Model #####
def laser_update(trees, assoc, ekf_state, sigmas, params):
    """
    The diameter component of the measurement can be discarded
    :param trees: a list of measurements, where each measurement is a tuple (range, bearing, diameter)
    :param assoc: the data association for the given set of trees, i.e. trees[i] is an observation of the
    ith landmark. If assoc[i] == -1, initialize a new landmark with the function initialize_landmark
    in the state for measurement i. If assoc[i] == -2, discard the measurement as it is too ambiguous to use
    :param ekf_state:
    :param sigmas:
    :param params:
    :return:
    Returns the ekf_state
    """

    # measurement noise
    range_noise = sigmas['range']
    bearing_noise = sigmas['bearing']

    Q = np.diag((range_noise**2, bearing_noise**2))

    for i in range(len(trees)):
        # add state with new landmark
        if assoc[i] == -1:
            ekf_state = initialize_landmark(ekf_state, trees[i])
        # discard ambiguous measurement
        elif assoc[i] == -2:
            continue
        # update state with existing landmark
        else:
            landmark_id = assoc[i]
            z_t, H = laser_measurement_model(ekf_state, landmark_id)
            x = ekf_state['x']
            P = ekf_state['P']
            N = len(ekf_state['x'])

            # Kalman Gain update
            K_t = np.dot(P, H.T)
            K_t = np.dot(K_t, slam_utils.invert_2x2_matrix(np.dot(np.dot(H, P), H.T) + Q))

            z = np.asarray([trees[i][0], trees[i][1]]).reshape((2,1))
            ekf_state['x'] = x + np.dot(K_t, (z - z_t)).reshape(-1)
            ekf_state['P'] = np.dot(np.identity(N) - np.dot(K_t, H), P)

        ekf_state['x'][2] = slam_utils.clamp_angle(ekf_state['x'][2])
        ekf_state['P'] = slam_utils.make_symmetric(ekf_state['P'] + 10e-6)

    return ekf_state

def compute_data_association(ekf_state, measurements, sigmas, params):
    """
    Computes measurement data association.
    Given a robot and map state and a set of (range,bearing) measurements,
    this function should compute a good data association, or a mapping from
    measurements to landmarks.
    :param ekf_state:
    :param measurements:
    :param sigmas:
    :param params:
    :return:
    Returns an array 'assoc' such that:
        assoc[i] == j if measurement i is determined to be an observation of landmark j,
        assoc[i] == -1 if measurement i is determined to be a new, previously unseen landmark, or,
        assoc[i] == -2 if measurement i is too ambiguous to use and should be discarded.
    """


    if ekf_state["num_landmarks"] == 0:
        # set association to init new landmarks for all measurements
        return [-1 for _ in measurements]

    range_noise = sigmas['range']
    bearing_noise = sigmas['bearing']
    Q = np.diag((range_noise**2, bearing_noise**2))

    n_measurements = len(measurements)
    n_landmarks = ekf_state["num_landmarks"]
    P = ekf_state['P'].copy()
    assoc = [-2 for _ in range(n_measurements)]
    M = np.zeros((n_measurements, n_landmarks))


    for i in range(n_landmarks):
        z_t, H = laser_measurement_model(ekf_state, i)

        for j in range(n_measurements):

            z = np.asarray(measurements[j][0:2]).reshape((2,1))
            r = z - z_t
            S = np.dot(np.dot(H, P), H.T) + Q
            d = np.dot(np.dot(r.T, slam_utils.invert_2x2_matrix(S)), r)
            M[j, i] = d

    # threshold for ambiguity
    alpha = chi2.ppf(0.95, df=2)
    A = alpha * np.ones((n_measurements, n_measurements))
    M_new = np.concatenate((M, A), axis=1)
    result = slam_utils.solve_cost_matrix_heuristic(M_new)
    result = np.asarray(result)

    update_index = result[:, 1] < n_landmarks
    result_update = result[update_index]

    for i in range(len(result_update)):
        assoc[result_update[i, 0]] = result_update[i, 1]

    remain_index = ~update_index

    for i in range(len(M)):
        if (remain_index[i] and np.min(M[result[i, 0], :]) >= chi2.ppf(0.9999999, df=2)):
            assoc[result[i, 0]] = -1
        elif (remain_index[i] and np.min(M[result[i, 0], :]) < chi2.ppf(0.9999999, df=2)):
            assoc[result[i, 0]] = -2

    return assoc

def run_ekf_slam(events, ekf_state_0, vehicle_params, filter_params, sigmas):
    last_odom_t = -1
    ekf_state = {
        'x': ekf_state_0['x'].copy(),
        'P': ekf_state_0['P'].copy(),
        'num_landmarks': ekf_state_0['num_landmarks']
    }
    
    state_history = {
        't': [0],
        'x': ekf_state['x'],
        'P': np.diag(ekf_state['P'])
    }

    if filter_params["do_plot"]:
        plot = slam_utils.init_plot()

    for i, event in enumerate(events):
        t = event[1][0]
        if i % 1000 == 0:
            print("t = {}".format(t))

        if event[0] == 'gps':
            gps_msmt = event[1][1:]
            ekf_state = gps_update(gps_msmt, ekf_state, sigmas)

        elif event[0] == 'odo':
            if last_odom_t < 0:
                last_odom_t = t
                continue
            u = event[1][1:]
            dt = t - last_odom_t
            ekf_state = odom_predict(u, dt, ekf_state, vehicle_params, sigmas)
            last_odom_t = t

        else:
            # Laser
            scan = event[1][1:]
            trees = tree_extraction.extract_trees(scan, filter_params)
            assoc = compute_data_association(ekf_state, trees, sigmas, filter_params)
            ekf_state = laser_update(trees, assoc, ekf_state, sigmas, filter_params)
            if filter_params["do_plot"]:
                slam_utils.do_plot(state_history['x'], ekf_state, trees, scan, assoc, plot, filter_params)

        
        state_history['x'] = np.vstack((state_history['x'], ekf_state['x'][0:3]))
        state_history['P'] = np.vstack((state_history['P'], np.diag(ekf_state['P'][:3,:3])))
        state_history['t'].append(t)

    return state_history


def main():
    odo = slam_utils.read_data_file("data/DRS.txt")
    gps = slam_utils.read_data_file("data/GPS.txt")
    laser = slam_utils.read_data_file("data/LASER.txt")

    # collect all events and sort by time
    events = [('gps', x) for x in gps]
    events.extend([('laser', x) for x in laser])
    events.extend([('odo', x) for x in odo])

    events = sorted(events, key = lambda event: event[1][0])

    vehicle_params = {
        "a": 3.78,
        "b": 0.50, 
        "L": 2.83,
        "H": 0.76
    }

    filter_params = {
        # measurement params
        "max_laser_range": 75, # meters

        # general...
        "do_plot": True,
        "plot_raw_laser": True,
        "plot_map_covariances": False

        # Add other parameters here if you need to...
    }

    # Noise values
    sigmas = {
        # Motion model noise
        "xy": 0.05,
        "phi": 0.5 * np.pi/180,

        # Measurement noise
        "gps": 3,
        "range": 0.5,
        "bearing": 5 * np.pi/180
    }

    # Initial filter state
    ekf_state = {
        "x": np.array([gps[0,1], gps[0,2], 36 * np.pi / 180]),
        "P": np.diag([.1, .1, 1]),
        "num_landmarks": 0
    }

    run_ekf_slam(events, ekf_state, vehicle_params, filter_params, sigmas)

if __name__ == '__main__':
    main()
