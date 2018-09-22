# Gym-Duckietown

[![Build Status](https://circleci.com/gh/duckietown/gym-duckietown/tree/master.svg?style=shield)](https://circleci.com/gh/duckietown/gym-duckietown/tree/master) [![Docker Hub](https://img.shields.io/docker/pulls/duckietown/gym-duckietown.svg)](https://hub.docker.com/r/duckietown/gym-duckietown)


[Duckietown](http://duckietown.org/) self-driving car simulator environments for OpenAI Gym.

Please use this bibtex if you want to cite this repository in your publications:

```
@misc{gym_duckietown,
  author = {Maxime Chevalier-Boisvert, Florian Golemo, Yanjun Cao, Bhairav Mehta, Liam Paull},
  title = {Duckietown Environments for OpenAI Gym},
  year = {2018},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/duckietown/gym-duckietown}},
}
```

This simulator was created as part of work done at [Mila](https://mila.quebec/).

## Modifications in panos branch compared to master branch

Significant modifications in main scripts of simulator.

Created a new map in simulation full of intersections
- `Duckietown-panos_map`

Automate data extraction in intersections:
- Press `A` : automatically executes left turn in intersections and collects images and velocities in numpy arrays
- Press `D` : automatically executes right turn in intersections and collects images and velocities in numpy arrays
- Press `H` : delete images and velocities from last left turn 
- Press `J` : delete images and velocities from last right turn 
- Press `U` : close simulation and save all images and velocities in .npy files
- Press `ESC` : close simulation without saving the images and velocities

- Press `4 (numpad)` : run TensorFlow CNN graph for left turns in intersections
- Press `6 (numpad)` : run TensorFlow CNN graph for right turns in intersections
- Press `8 (numpad)` : run TensorFlow CNN graph for lane following

- Press `SPACE` : interrupt all the above actions and returns duckiebot to rest (v=0, omega=0)
- Press `BACKSPACE` : reset simulation and place duckiebot in random position in the map (all the collected data up to that point are not deleted and continue appending new data to them)
