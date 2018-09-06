#!/usr/bin/env python3

"""
This script allows you to manually control the simulator or Duckiebot
using the keyboard arrows.
"""

import sys
import argparse
import pyglet
from pyglet.window import key
import numpy as np
import gym
import gym_duckietown
from gym_duckietown.envs import DuckietownEnv
from experiments.utils import save_img

import pandas as pd
import os

parser = argparse.ArgumentParser()
parser.add_argument('--env-name', default=None)
parser.add_argument('--map-name', default='panos_map')
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


def update(state, action):

    # obs, reward, done, info, omega, v = env.step(action)
    obs, reward, done, info, omega, v, vl, vr = env.step(action)

    # print('step_count = %s, reward=%.3f' % (env.unwrapped.step_count, reward))

    # print("omega = {}, v = {}, vL = {}, vR = {}".format(omega, v, vl, vr))


    if state == "Left":
        left_img.append(np.reshape(obs, (1, -1) ) )
        left_velocities.append([omega, v, vl, vr])

    if state == "Right":
        right_img.append(np.reshape(obs, (1, -1) ) )
        right_velocities.append([omega, v, vl, vr])

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

    if key_handler[key.A]:

        state = "Left"

        for i in range(40):

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

        print("Left images = {}".format(len(left_img) / 142))
        print("Right images = {}".format(len(right_img) / 100))


    elif key_handler[key.D]:
        state = "Right"

        for i in range(100):
            if key_handler[key.SPACE]:
                break

            # turn right
            action = np.array([0.35, -1])
            update(state, action)

        print("Left images = {}".format(len(left_img) / 142))
        print("Right images = {}".format(len(right_img) / 100))


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
                del left_img[-142:]
                del left_velocities[-142:]
            else:
                print("There is no left turn saved to delete.")

            print("Left images = {}".format(len(left_img) / 142))
            print("Right images = {}".format(len(right_img) / 100))

        # Pressing J will delete the last right images and velocities
        if key_handler[key.J]:
            if len(right_img) > 0 and len(right_velocities) > 0:
                del right_img[-100:]
                del right_velocities[-100:]
            else:
                print("There is no right turn saved to delete.")

            print("Left images = {}".format(len(left_img) / 142))
            print("Right images = {}".format(len(right_img) / 100))

left_img = np.array(left_img)
right_img = np.array(right_img)
left_velocities = np.array(left_velocities)
right_velocities = np.array(right_velocities)

# print("left_img.shape = {}".format(left_img.shape))
# print("right_img.shape = {}".format(right_img.shape))
# print("left_velocities.shape = {}".format(left_velocities.shape))
# print("right_velocities.shape = {}".format(right_velocities.shape))
# print("Closing")

df_left_velocities = pd.DataFrame({
    'w': left_velocities[:, 0],
    'v': left_velocities[:, 1],
    'vl': left_velocities[:, 2],
    'vr': left_velocities[:, 3]
})

df_right_velocities = pd.DataFrame({
    'w': right_velocities[:, 0],
    'v': right_velocities[:, 1],
    'vl': right_velocities[:, 2],
    'vr': right_velocities[:, 3]
})

df_left_images = pd.DataFrame({
        'img': [left_img]
    })

df_right_images = pd.DataFrame({
        'img': [right_img]
    })


print("left_img.shape = {}".format(df_left_images['img'].shape))
print("right_img.shape = {}".format(df_right_images['img'].shape))
print("left_velocities.shape = {}".format(df_left_velocities.shape))
print("right_velocities.shape = {}".format(df_right_velocities.shape))
print("Closing")

# define the names of the train and test .h5 files
data = os.path.join(os.getcwd(), 'data', 'data.h5')

# check if the file exists in the data directory and if yes remove it before saving the new one
if os.path.isfile(data):
    os.remove(data)

df_left_velocities.to_hdf(data, key='left_velocities')
df_right_velocities.to_hdf(data, key='right_velocities')

df_left_images.to_hdf(data, key='left_images')
df_right_images.to_hdf(data, key='right_images')

print("\nTotal left turns = {} [images = {} , velocities = {}]\n Total right turns = {} [images = {}, velocities = {}]"
      "\n All saved in file : {}.".format(int(left_img.shape[0]/142), left_img.shape[0], df_left_velocities.shape[0],
                                          int(left_img.shape[0]/142), right_img.shape[0], df_right_velocities.shape[0],
                                          data))

env.close()

