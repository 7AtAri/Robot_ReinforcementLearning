# Robotproject with Reinforcement Learning

## Introduction
The aim of this project was to train a 6-axis robot arm using reinforcement learning so that it learns to track a helix.
Therefore we implemented Deep Q-Learning with a Late Fusion Multimodal Model that uses a CNN Network for the spatial features and
gets the orientation features concatenated into the fully connected layers directly.
The replay memory works on basis of n-step-bootstrapping.

## Installation
Follow these steps to install our project

      git clone https://github.com/7AtAri/Robot_ReinforcementLearning.git

Install the required dependencies 

      python -m pip install --upgrade pip
      python -m pip install -r requirements.txt


**The only important files are in the folder "code":**
* n-step-bootstrapping.py
* Environment-cnn.py

if you want to run the project please run the file **n-step-bootstrapping.py**

## Different Branches
We have to different branches with different methods with what kind of Networks we are training our Agent

1. "main" branch

here we work with a Convolutional neural network (CNN) which can operate with 3D datas and the orientation of TCP 
will inserted just in the Fully Connected Layer (FC).

2. "working-secondbranch"

in that Version we also work with a CNN but we are not implementing the orientation of the TCP in the CNN.

## Usage
 In our environment we use the following functions based on the gymnasium environment

 - step(): updates the environment based on the actions of the agent

 - Reset(): resets the environment an set the enviroment to initial settings (TCP on Startposition, Reard to zero, Observation, Kinematics)

 - render(): visulize the Helix, current TCP position with current orientation (arrow) and the target voxel
 
<img src="./images_read_me/HelixVisu.PNG" alt="drawing" width="500"/>

Here you can see the model of our environment in which the robotarm has to be trained

## Overview with Class Diagramm
In the following figure you can see a class diagramm to give an overview of our Code...

<img src="./images_read_me/Classdiagram.PNG" alt="drawing" width="500"/>
