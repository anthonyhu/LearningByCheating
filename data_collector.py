"""
Stores tuples of (birdview, measurements, rgb).

Run from top level directory.
Sample usage -

python3 bird_view/data_collector.py \
        --dataset_path $PWD/data \
        --frame_skip 10 \
        --frames_per_episode 1000 \
        --n_episodes 100 \
        --port 3000 \
        --n_vehicles 0 \
        --n_pedestrians 0
"""
import argparse
import math
from enum import Enum

from pathlib import Path

import numpy as np
import tqdm
import lmdb

from bird_view.utils import bz_utils as bu
from bird_view.utils import carla_utils as cu

from benchmark import make_suite
from bird_view.models.common import crop_birdview
from bird_view.models.controller import PIDController
from bird_view.models.roaming import RoamingAgentMine

import torchvision
import carla

HACK_MAX_DISTANCE_TO_TRAFFIC_LIGHT = 40 # meters, sometimes we got an issue on the distance to traffic_light, let's just dump those states...


def _debug(observations, agent_debug):
    import cv2

    processed = cu.process(observations)

    control = observations['control']
    control = [control.steer, control.throttle, control.brake]
    control = ' '.join(str('%.2f' % x).rjust(5, ' ') for x in control)
    real_control = observations['real_control']
    real_control = [real_control.steer, real_control.throttle, real_control.brake]
    real_control = ' '.join(str('%.2f' % x).rjust(5, ' ') for x in real_control)

    canvas = np.uint8(observations['rgb']).copy()
    rows = [x * (canvas.shape[0] // 10) for x in range(10+1)]
    cols = [x * (canvas.shape[1] // 10) for x in range(10+1)]

    WHITE = (255, 255, 255)
    CROP_SIZE = 192
    X = 176
    Y = 192 // 2
    R = 2

    def _write(text, i, j):
        cv2.putText(
                canvas, text, (cols[j], rows[i]),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, WHITE, 1)

    _command = {
            1: 'LEFT',
            2: 'RIGHT',
            3: 'STRAIGHT',
            4: 'FOLLOW',
            }.get(int(observations['command']), '???')

    _write('Command: ' + _command, 1, 0)
    _write('Velocity: %.1f' % np.linalg.norm(observations['velocity']), 2, 0)
    _write('Real: %s' % control, -5, 0)
    _write('Control: %s' % control, -4, 0)

    r = 2
    birdview = cu.visualize_birdview(crop_birdview(processed['birdview']))

    def _dot(x, y, color):
        x = int(x)
        y = int(y)
        birdview[176-r-x:176+r+1-x,96-r+y:96+r+1+y] = color

    _dot(0, 0, [255, 255, 255])

    ox, oy = observations['orientation']
    R = np.array([
        [ox,  oy],
        [-oy, ox]])

    u = np.array(agent_debug['waypoint']) - np.array(agent_debug['vehicle'])
    u = R.dot(u[:2])
    u = u * 4

    _dot(u[0], u[1], [255, 255, 255])

    def _stick_together(a, b):
        h = min(a.shape[0], b.shape[0])

        r1 = h / a.shape[0]
        r2 = h / b.shape[0]

        a = cv2.resize(a, (int(r1 * a.shape[1]), int(r1 * a.shape[0])))
        b = cv2.resize(b, (int(r2 * b.shape[1]), int(r2 * b.shape[0])))

        return np.concatenate([a, b], 1)

    full = _stick_together(canvas, birdview)

    bu.show_image('full', full)



class NoisyAgent(RoamingAgentMine):
    """
    Each parameter is in units of frames.
    State can be "drive" or "noise".
    """
    # CONSTANT for traffic light state
    LightState = Enum('State', 'Red Yellow Green')   # corresponds to 1, 2, 3 in value

    # Magic numbers which seems to work, at least on Town05/Town02
    distance_to_detect_US = 60
    distance_to_detect_EU = 30
    UGLY_HARDCODED_MIN_DIST_TO_TRAFFIC_LIGHT_EU = 5
    max_angle_between_traffic_light_and_car_US = 35
    max_angle_between_traffic_light_and_car_EU = 35

    def __init__(self, env, noise=None):
        super().__init__(env._player, resolution=1, threshold_before=7.5, threshold_after=5.)

        # self.params = {'drive': (100, 'noise'), 'noise': (10, 'drive')}
        self.params = {'drive': (100, 'drive')}

        self.steps = 0
        self.state = 'drive'
        self.noise_steer = 0
        self.last_throttle = 0
        self.noise_func = noise if noise else lambda: np.random.uniform(-0.25, 0.25)

        self.speed_control = PIDController(K_P=0.5, K_I=0.5/20, K_D=0.1)
        self.turn_control = PIDController(K_P=0.75, K_I=1.0/20, K_D=0.0)

        # Traffic lights
        self.current_traffic_light = None  # We want to detect when we passed at red/orange to kill the agent!
        self.hack_for_intersection = None  # Hack to detect when we are just after a traffic light, we want to be on intersection even if Carla doenst say tho!
        if "Town01" in self._map.name or "Town02" in self._map.name:
            print("We take a smaller distance cause we got EU style traffic light")
            self.distance_to_detect = self.distance_to_detect_EU
            self.max_angle_between_traffic_light_and_car = self.max_angle_between_traffic_light_and_car_EU
            self.US_style = False
        else:
            self.distance_to_detect = self.distance_to_detect_US
            self.max_angle_between_traffic_light_and_car = self.max_angle_between_traffic_light_and_car_US
            self.US_style = True

    def run_step(self, observations):
        self.steps += 1

        last_status = self.state
        num_steps, next_state = self.params[self.state]
        real_control = super().run_step(observations)
        real_control.throttle *= max((1.0 - abs(real_control.steer)), 0.25)

        control = carla.VehicleControl()
        control.manual_gear_shift = False

        if self.state == 'noise':
            control.steer = self.noise_steer
            control.throttle = self.last_throttle
        else:
            control.steer = real_control.steer
            control.throttle = real_control.throttle
            control.brake = real_control.brake

        if self.steps == num_steps:
            self.steps = 0
            self.state = next_state
            self.noise_steer = self.noise_func()
            self.last_throttle = control.throttle

        self.debug = {
                'waypoint': (self.waypoint.x, self.waypoint.y, self.waypoint.z),
                'vehicle': (self.vehicle.x, self.vehicle.y, self.vehicle.z)
                }

        return control, self.road_option, last_status, real_control

    def detect_traffic_light(self, traffic_light_list, waypoint=None):
        """
        This method is specialized to check traffic lights.

        :param traffic_light_list: list containing TrafficLight objects
        :return: a tuple given by (bool_flag, traffic_light), where
                 - bool_flag is True if there is a traffic light in RED
                   affecting us and False otherwise
                 - traffic_light is the object itself or None if there is no
                   red traffic light affecting us
        """

        # We may want to detect traffic light from a waypoint and not from the true position of the vehicule (
        if waypoint is None:
            # print("we detect traffic light from the true position of the vehicule")
            ego_vehicle_transform = self._vehicle.get_transform()
            ego_vehicle_rotation = ego_vehicle_transform.rotation.yaw
            ego_vehicle_location = ego_vehicle_transform.location
            ego_vehicle_waypoint = self._map.get_waypoint(ego_vehicle_location)
        else:
            # print("we detect traffic light from a waypoint (suppose to be the good position of the vehicule)")
            ego_vehicle_rotation = waypoint.transform.rotation.yaw
            ego_vehicle_location = waypoint.transform.location
            ego_vehicle_waypoint = waypoint

        if ego_vehicle_waypoint.is_junction:
            self.current_traffic_light = None
            self.hack_for_intersection = None
            # print("It is too late. Do not block the intersection! Keep going!")
            return (True, None, None, None)

        min_angle = 180.0
        sel_distance = 0.0
        sel_traffic_light = None
        for traffic_light in traffic_light_list:
            distance_tf_veh, d_angle_1, d_angle_2 = \
                self.compute_distance_angle_traffic_light_Marin(traffic_light.get_location(),
                                                                traffic_light.get_transform().rotation.yaw - 90, #In all town the traffic light "real" angle is 90 degre less than the angle in Ureal Engine
                                                                ego_vehicle_location,
                                                                ego_vehicle_rotation)

            if distance_tf_veh < self.distance_to_detect and (d_angle_1 + d_angle_2) < min(self.max_angle_between_traffic_light_and_car * 2, min_angle): # We multiply by 2 casue we sum 2 angles...
                # print("distance_tf_veh = ", distance_tf_veh)
                # print("d_angle_1 = ", d_angle_1)
                # print("d_angle_2 = ", d_angle_2)
                sel_distance = distance_tf_veh
                sel_traffic_light = traffic_light
                min_angle = d_angle_1 + d_angle_2

        if sel_traffic_light is not None:
            # print('=== Distance = {} | Angle = {} | ID = {}'.format(sel_distance, min_angle, sel_traffic_light.id))

            if self.current_traffic_light is None:
                self.current_traffic_light = sel_traffic_light
                self.hack_for_intersection = sel_traffic_light

            if self.current_traffic_light.id != sel_traffic_light.id:
                # print("WE CHANGE TRAFFIC LIGHT WITHOUT GOING ON INTERSECTION, THAT SHOULD NOT HAPPEN")
                self.current_traffic_light = sel_traffic_light
                self.hack_for_intersection = sel_traffic_light

            if self.US_style:
                # We need to adapt the dist_to_tl, we want it to be the distance to the actual position where he must stop!
                distance_UGLY = 0
                next_waypoint_UGLY_us = list(ego_vehicle_waypoint.next(1.0))
                while (len(next_waypoint_UGLY_us) == 1) and (not next_waypoint_UGLY_us[0].is_junction):
                    next_waypoint_UGLY_us = next_waypoint_UGLY_us[0]
                    next_waypoint_UGLY_us = list(next_waypoint_UGLY_us.next(1.0))
                    distance_UGLY += 1

                sel_distance = distance_UGLY

                if sel_distance >= HACK_MAX_DISTANCE_TO_TRAFFIC_LIGHT:
                    # print("THERE IS AN ISSUE THERE, lets say there were no traffic light at all")
                    self.current_traffic_light = None
                    self.hack_for_intersection = None
                    return (False, None, None, None)

            else:
                sel_distance = max(0, sel_distance - self.UGLY_HARDCODED_MIN_DIST_TO_TRAFFIC_LIGHT_EU)

            current_state = sel_traffic_light.state # I think a bug where coming when state changed EXACLTY between the next if elif statement!
            if current_state == carla.libcarla.TrafficLightState.Red:
                # print("il est rouge le soit disant feu")
                return (False, sel_traffic_light, self.LightState.Red.value, sel_distance)
            elif current_state == carla.libcarla.TrafficLightState.Green:
                # print("il est vert le soit disant feu")
                return (False, sel_traffic_light, self.LightState.Green.value, sel_distance)
            elif current_state == carla.libcarla.TrafficLightState.Yellow:
                # print("il est jaune le soit disant feu")
                return (False, sel_traffic_light, self.LightState.Yellow.value, sel_distance)
            else:
                print("PROBLEME IL EST RIEN DU TOUT LE SOIT DISANT FEU, DISONS QU'il était rouge...")
                return (False, sel_traffic_light, self.LightState.Red.value, sel_distance)
                # raise Exception

        else:
            self.current_traffic_light = None
            if self.hack_for_intersection is not None:
                return (True, None, None, None) # Hack to detect when we are just after a traffic light, we want to be on intersection even if Carla doenst say tho!
            else:
                return (False, None, None, None)

    @staticmethod
    def compute_distance_angle_traffic_light_Marin(tl_location, tl_orientation, veh_location, veh_orientation):
        """
        Compute relative angle and distance between a target_location and a current_location

        :param tl_location: location of the traffic light
        :param tl_orientation: orientation of the traffic light
        :param veh_location: location of the vehicule
        :param veh_orientation: orientation of the vehicule
        :return: a tuple composed by the distance to the object and the angle between both objects
        """
        tl_veh_vector = np.array([tl_location.x - veh_location.x, tl_location.y - veh_location.y])
        distance_tf_veh = np.linalg.norm(tl_veh_vector)

        veh_vector = np.array([math.cos(math.radians(veh_orientation)), math.sin(math.radians(veh_orientation))])
        tl_vector = np.array([math.cos(math.radians(tl_orientation)), math.sin(math.radians(tl_orientation))])
        d_angle_1 = math.degrees(math.acos(np.dot(tl_veh_vector, veh_vector) / distance_tf_veh))
        d_angle_2 = math.degrees(math.acos(np.dot(tl_veh_vector, tl_vector) / distance_tf_veh))

        return (distance_tf_veh, d_angle_1, d_angle_2)


def get_episode(env, params):
    data = list()
    progress = tqdm.tqdm(range(params.frames_per_episode), desc='Frame')
    start, target = env.pose_tasks[np.random.randint(len(env.pose_tasks))]
    env_params = {
            'weather': np.random.choice(list(cu.TRAIN_WEATHERS.keys())),
            'start': start,
            'target': target,
            'n_pedestrians': params.n_pedestrians,
            'n_vehicles': params.n_vehicles,
            }

    env.init(**env_params)
    env.success_dist = 5.0

    agent = NoisyAgent(env)
    agent.set_route(env._start_pose.location, env._target_pose.location)

    world = env._client.get_world()
    traffic_light_list = world.get_actors().filter('*traffic_light*')

    # Real loop.
    while len(data) < params.frames_per_episode and not env.is_success() and not env.collided:
        for _ in range(params.frame_skip):
            env.tick()

            observations = env.get_observations()
            control, command, last_status, real_control = agent.run_step(observations)
            agent_debug = agent.debug
            env.apply_control(control)

            observations['command'] = command
            observations['control'] = control
            observations['real_control'] = real_control

            # Traffic lights
            is_intersection_marin, traffic_light, light_state, distance_to_traffic_light = \
                agent.detect_traffic_light(traffic_light_list)
            if traffic_light:
                print("INTERSECTION MARIN = ", is_intersection_marin)
                print("incoming traffic light is at " + str(distance_to_traffic_light) + "meters and is of color ",
                      agent.LightState(light_state).name)
                green = carla.Color(0, 255, 0)
                size_plot_point = 0.1
                life_time_plot_point = 1.0
                world.debug.draw_point(traffic_light.get_location(), color=green, size=size_plot_point * 10,
                                       life_time=life_time_plot_point)

                observations['is_traffic_light'] = True
                observations['traffic_light_color'] = light_state - 1
                observations['traffic_light_distance'] = distance_to_traffic_light

            else:
                observations['is_traffic_light'] = False
                observations['traffic_light_color'] = 0
                observations['traffic_light_distance'] = 0
            print(f'is traffic light: {observations["is_traffic_light"]}')
            print(f'traffic color: {observations["traffic_light_color"]}')
            print(f'traffic distance: {observations["traffic_light_distance"]}')

            if not params.nodisplay:
                _debug(observations, agent_debug)

        observations['control'] = real_control
        processed = cu.process(observations)

        data.append(processed)

        progress.update(1)

    progress.close()

    if (not env.is_success() and not env.collided) or len(data) < 500:
        return None

    return data


def main(params):

    save_dir = Path(params.dataset_path)
    save_dir.mkdir(parents=True, exist_ok=True)

    total = 0

    for i in tqdm.tqdm(range(params.n_episodes), desc='Episode'):
        with make_suite('FullTown01-v1', port=params.port, planner=params.planner) as env:
            filepath = save_dir.joinpath('%03d' % i)

            if filepath.exists():
                continue

            data = None

            while data is None:
                data = get_episode(env, params)

            lmdb_env = lmdb.open(str(filepath), map_size=int(1e10))
            n = len(data)

            with lmdb_env.begin(write=True) as txn:
                txn.put('len'.encode(), str(n).encode())

                for i, x in enumerate(data):
                    txn.put(
                            ('rgb_%04d' % i).encode(),
                            np.ascontiguousarray(x['rgb']).astype(np.uint8))
                    txn.put(
                        ('depth_%04d' % i).encode(),
                        np.ascontiguousarray(x['depth']).astype(np.float32))
                    txn.put(
                        ('segmentation_%04d' % i).encode(),
                        np.ascontiguousarray(x['segmentation']).astype(np.uint8))
                    txn.put(
                            ('birdview_%04d' % i).encode(),
                            np.ascontiguousarray(x['birdview']).astype(np.uint8))
                    txn.put(
                            ('measurements_%04d' % i).encode(),
                            np.ascontiguousarray(x['measurements']).astype(np.float32))
                    txn.put(
                            ('control_%04d' % i).encode(),
                            np.ascontiguousarray(x['control']).astype(np.float32))

            total += len(data)

    print('Total frames: %d' % total)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--planner', type=str, choices=['old', 'new'], default='new')
    parser.add_argument('--dataset_path', required=True)
    parser.add_argument('--n_vehicles', type=int, default=100)
    parser.add_argument('--n_pedestrians', type=int, default=250)
    parser.add_argument('--n_episodes', type=int, default=50)
    parser.add_argument('--frames_per_episode', type=int, default=4000)
    parser.add_argument('--frame_skip', type=int, default=1)
    parser.add_argument('--nodisplay', action='store_true', default=False)
    parser.add_argument('--port', type=int, default=2000)

    params = parser.parse_args()

    main(params)
