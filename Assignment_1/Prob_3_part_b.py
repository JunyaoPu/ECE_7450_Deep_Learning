'''
    File name: Prob_3_part_b.py
    Description: ECE-7650 Deep Learning HW#1 Problem #3.b
    Author: Junyao Pu
    Date created: Jan 30th, 2020
    Date last modified: Feb 4th, 2020
    Python Version: 3.6
'''
import numpy as np
import csv
import matplotlib.pyplot as plt

# plot weight image
def plot_10_class_weight(saved_weights):
    class_name=['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

    for i in range(10):
        grayscale = saved_weights[i].reshape((32, 32))

        f=[]
        f.append(plt.figure(i))

        plt.imshow(grayscale, cmap="gray")
        plt.xlabel('Pixels along x-axis')
        plt.ylabel('Pixels along y-axis')
        plt.title('Image version of weights for '+ class_name[i] +' class')
        plt.show()

#import weight from part a
saved_weights = []
saved_weights = np.loadtxt('weights.csv',delimiter=",", skiprows=0)

plot_10_class_weight(saved_weights)
