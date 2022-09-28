# Advanced Starcraft2 Microtasks

This project is a reinforcement learning approach to the popular online RTS StarCraft2.
It uses the proximal policy optimization according to [keras implementation](https://keras.io/examples/rl/ppo_cartpole/).
We conducted the project as a part of the course "Deep reinforcement learning" course at the University of Osnabrück in the summer semester 2022 by Nick Lechtenbörger, Niklas Schemmer and Johannes Weißen.

## Install Requirements

```python
pip install -r requirements.txt
```

Please also follow the install guide for pysc2 on their [official github page](https://github.com/deepmind/pysc2#get-pysc2), to get pysc2.

## Usage

Before you start the program, define wether you want to test your model or if you want to start training.
There you can also change the training parameters.
Simply change the corresponding value in the FLAGS object in the main.py file.
Or pass them as command line parameters directly from the console.

## Start Program

```python
python main.py
```

## CUDA

To decrease training times it is highly recommended to use a GPU with CUDA.
You can find more information about installing CUDA for tensorflow at https://www.tensorflow.org/install/gpu.
