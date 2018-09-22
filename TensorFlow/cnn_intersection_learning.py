#!/usr/bin/env python3

"""
This script allows you to manually control the simulator or Duckiebot
using the keyboard arrows.
"""

import sys
import argparse
from pyglet.window import key
import numpy as np
import gym
from gym_duckietown.envs import DuckietownEnv
from experiments.utils import save_img

import tensorflow as tf
import os
from image_preprocessing import *

parser = argparse.ArgumentParser()
parser.add_argument('--env-name', default=None)
parser.add_argument('--map-name', default='panos_map')
# parser.add_argument('--map-name', default='udem1')
parser.add_argument('--draw-curve', action='store_true', help='draw the lane following curve')
parser.add_argument('--draw-bbox', action='store_true', help='draw collision detection bounding boxes')
parser.add_argument('--domain-rand', action='store_true', help='enable domain randomization')
parser.add_argument('--frame-skip', default=1, type=int, help='number of frames to skip')
parser.add_argument('--max-steps', default=100000, type=int, help='number of maximum steps')

args = parser.parse_args()

if args.env_name is None:
    env = DuckietownEnv(
        map_name = args.map_name,
        draw_curve = args.draw_curve,
        draw_bbox = args.draw_bbox,
        domain_rand = args.domain_rand,
        frame_skip = args.frame_skip,
        max_steps = args.max_steps
    )
else:
    env = gym.make(args.env_name)

env.reset()
env.render()


@env.unwrapped.window.event
def on_key_press(symbol, modifiers):
    """
    This handler processes keyboard commands that
    control the simulation
    """

    if symbol == key.BACKSPACE or symbol == key.SLASH:
        print('RESET')
        env.reset()
        env.render()
    elif symbol == key.PAGEUP:
        env.unwrapped.cam_angle[0] = 0
    elif symbol == key.ESCAPE:
        env.close()
        sys.exit(0)

    # Take a screenshot
    elif symbol == key.RETURN:
        print('saving screenshot')
        img = env.render('rgb_array')
        save_img('screenshot.png', img)

# Register a keyboard handler
key_handler = key.KeyStateHandler()
env.unwrapped.window.push_handlers(key_handler)


# define lists to save images and velocities
left_img = []
right_img = []

left_velocities = []
right_velocities = []

#############################
#     Load frozen graph     #
#############################
# cnn_mode : regression, regression_grey, classification

cnn_type = 'regression_grey'

graph_path_lt = "working_models/models_left_turn/regression_grey/model_5_data_15_09_2018/frozen_graph/frozen_graph.pb"
graph_path_rt = "working_models/models_right_turn/regression_grey/model_5_data_15_09_2018/frozen_graph/frozen_graph.pb"
# graph_path_lf = "working_models/models_lane_following/frozen_graph/frozen_graph.pb" # original
graph_path_lf = "test_models/lane_following/48x96x1/frozen_graph.pb"
# graph_path_lf = "test_models/lane_following/120x160x1/frozen_graph.pb"
# graph_path_lf = "test_models/lane_following/48x96x3/frozen_graph/frozen_graph.pb"


##### import frozen graph for cnn left turn
with tf.gfile.GFile(graph_path_lt, "rb") as f:
    graph_def_lt = tf.GraphDef()
    graph_def_lt.ParseFromString(f.read())

# Then, we import the graph_def into a new Graph and returns it
with tf.Graph().as_default() as graph_lt:
    tf.import_graph_def(graph_def_lt, name="prefix")

# for left turn
x_lt = graph_lt.get_tensor_by_name('prefix/image:0')
y_lt = graph_lt.get_tensor_by_name('prefix/ConvNet/fc_layer_2/BiasAdd:0')


##### import frozen graph for cnn right turn
with tf.gfile.GFile(graph_path_rt, "rb") as f:
    graph_def_rt = tf.GraphDef()
    graph_def_rt.ParseFromString(f.read())

# Then, we import the graph_def into a new Graph and returns it
with tf.Graph().as_default() as graph_rt:
    tf.import_graph_def(graph_def_rt, name="prefix")

# for right turn
x_rt = graph_rt.get_tensor_by_name('prefix/image:0')
y_rt = graph_rt.get_tensor_by_name('prefix/ConvNet/fc_layer_2/BiasAdd:0')


# ##### import frozen graph for cnn lane following
# with tf.gfile.GFile(graph_path_lf, "rb") as f:
#     graph_def_lf = tf.GraphDef()
#     graph_def_lf.ParseFromString(f.read())
#
# # Then, we import the graph_def into a new Graph and returns it
# with tf.Graph().as_default() as graph_lf:
#     tf.import_graph_def(graph_def_lf, name="prefix")
#
# # for lane following
# x_lf = graph_lf.get_tensor_by_name('prefix/x:0')
# y_lf = graph_lf.get_tensor_by_name('prefix/fc_layer_2/BiasAdd:0')


##### import frozen graph for cnn lane following
with tf.gfile.GFile(graph_path_lf, "rb") as f:
    graph_def_lf = tf.GraphDef()
    graph_def_lf.ParseFromString(f.read())

# Then, we import the graph_def into a new Graph and returns it
with tf.Graph().as_default() as graph_lf:
    tf.import_graph_def(graph_def_lf, name="prefix")

# for lane following
x_lf = graph_lf.get_tensor_by_name('prefix/x:0')
y_lf = graph_lf.get_tensor_by_name('prefix/ConvNet/fc_layer_2/BiasAdd:0')

def update(state, action):


    # define if cnn_mode is on or not
    cnn_mode = False

    # obs, reward, done, info, omega, v = env.step(action)
    obs, reward, done, info, omega, v, vl, vr = env.step(action, cnn_mode)

    # # print('step_count = %s, reward=%.3f' % (env.unwrapped.step_count, reward))
    # print("omega = {}, v = {}, vL = {}, vR = {}".format(omega, v, vl, vr))

    if "CNN" in state:

        cnn_mode = True

        while True:

            if key_handler[key.SPACE]:
                break

            if "left_turn" in state:

                with tf.Session(graph=graph_lt) as sess:
                    [[vel_v, vel_w]] = sess.run(y_lt, feed_dict={x_lt: img_preprocess(obs, cnn_type)})


            if "right_turn" in state:

                with tf.Session(graph=graph_rt) as sess:
                    [[vel_v, vel_w]] = sess.run(y_rt, feed_dict={x_rt: img_preprocess(obs, cnn_type)})


            if "lane_following" in state:

                with tf.Session(graph=graph_lf) as sess:
                    [[vel_w]] = sess.run(y_lf, feed_dict={x_lf: img_preprocess_lane_following(obs)})

                    vel_v = 0.20

            adjust_vel = True

            if adjust_vel:

                v = 0.2
                w = vel_w * v / vel_v
                action = np.array([v, w])
            else:
                action = np.array([vel_v, vel_w])

            obs, reward, done, info, omega, v, vl, vr = env.step(action, cnn_mode)
            print("vel_v ={}, vel_w ={}".format(action[0], action[1]))
            print("cal_v ={}, cal_w ={}".format(v, omega))
            if done:
                print('done!')
                env.reset()
                env.render()

            env.render()


    if state == "Left":
        left_img.append(np.reshape(obs, (1, -1) ) )
        left_velocities.append([v, omega, vl, vr])

    if state == "Right":
        right_img.append(np.reshape(obs, (1, -1) ) )
        right_velocities.append([v, omega, vl, vr])

    if done:
        print('done!')
        env.reset()
        env.render()


    env.render()

while True:

    # define state for actions ("Empty", "Left", "Right")
    state = "Empty"

    # press escape to save the images and close the simulation
    if key_handler[key.U]:
        break

    action = np.array([0.0, 0.0])


    # Test CNN for left turns in simulation
    if key_handler[key.NUM_8]:
        state = "CNN_lane_following"
        update(state, action)

    if key_handler[key.NUM_4]:
        state = "CNN_left_turn"
        update(state, action)

    if key_handler[key.NUM_6]:
        state = "CNN_right_turn"
        update(state, action)


    if key_handler[key.A]:

        state = "Left"

        for i in range(45):

            if key_handler[key.SPACE]:
                break

            # go straight
            action = np.array([0.44, 0.0])
            update(state, action)

        for i in range(102):
            if key_handler[key.SPACE]:
                break

            # turn left
            action = np.array([0.35, +1])
            update(state, action)


        for i in range(53):

            if key_handler[key.SPACE]:
                break

            # go straight
            action = np.array([0.44, 0.0])
            update(state, action)

        print("Left images = {}".format(len(left_img) / 200))
        print("Right images = {}".format(len(right_img) / 200))


    elif key_handler[key.D]:
        state = "Right"

        for i in range(10):

            if key_handler[key.SPACE]:
                break

            # go straight
            action = np.array([0.44, 0.0])
            update(state, action)

        for i in range(102):
            if key_handler[key.SPACE]:
                break

            # turn right
            action = np.array([0.35, -1])
            update(state, action)


        # for i in range(20):
        for i in range(88):

            if key_handler[key.SPACE]:
                break

            # go straight
            action = np.array([0.44, 0.0])
            update(state, action)


        print("Left images = {}".format(len(left_img) / 200))
        print("Right images = {}".format(len(right_img) / 200))


    else:
        if key_handler[key.UP]:
            action = np.array([0.44, 0.0])
        if key_handler[key.DOWN]:
            action = np.array([-0.44, 0])
        if key_handler[key.LEFT]:
            action = np.array([0.35, +1])
        if key_handler[key.RIGHT]:
            action = np.array([0.35, -1])
        if key_handler[key.SPACE]:
            action = np.array([0, 0])

        # Speed boost
        if key_handler[key.LSHIFT]:
            # action *= 1.5
            action *= 5

        update(state, action)

        # Pressing H will delete the last left images and velocities
        if key_handler[key.H]:
            if len(left_img) > 0 and len(left_velocities) > 0:
                del left_img[-200:]
                del left_velocities[-200:]
            else:
                print("There is no left turn saved to delete.")

            print("Left images = {}".format(len(left_img) / 200))
            print("Right images = {}".format(len(right_img) / 200))

        # Pressing J will delete the last right images and velocities
        if key_handler[key.J]:
            if len(right_img) > 0 and len(right_velocities) > 0:
                del right_img[-200:]
                del right_velocities[-200:]
            else:
                print("There is no right turn saved to delete.")

            print("Left images = {}".format(len(left_img) / 200))
            print("Right images = {}".format(len(right_img) / 200))

left_img = np.reshape(np.array(left_img), (-1, 57600))
right_img = np.reshape(np.array(right_img), (-1, 57600))
left_velocities = np.array(left_velocities)
right_velocities = np.array(right_velocities)


print("\nleft_img.shape = {}".format(left_img.shape))
print("left_velocities.shape = {}".format(left_velocities.shape))
print("right_img.shape = {}".format(right_img.shape))
print("right_velocities.shape = {}".format(right_velocities.shape))
print("Closing")

# define the names of the train and test .h5 files
data_folder = os.path.join('..', 'data')
date = '15_09_2018'
save_folder = os.path.join( data_folder, date)

if not os.path.exists(save_folder):
    os.makedirs(save_folder)

left_img_file = os.path.join(save_folder, 'left_images.npy')
right_img_file = os.path.join(save_folder, 'right_images.npy')
left_vel_file = os.path.join(save_folder, 'left_velocities.npy')
right_vel_file = os.path.join(save_folder, 'right_velocities.npy')

np.save(left_img_file, left_img)
np.save(right_img_file, right_img)
np.save(left_vel_file, left_velocities)
np.save(right_vel_file, right_velocities)

print("\nTotal left turns = {} [images = {} , velocities = {}]\nTotal right turns = {} [images = {}, velocities = {}]"
      "\nAll saved in folder : {}.".format(int(left_img.shape[0]/200), left_img.shape[0], left_velocities.shape[0],
                                          int(right_img.shape[0]/200), right_img.shape[0], right_velocities.shape[0],
                                          save_folder))

env.close()


