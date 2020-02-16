'''
    File name: Prob_3_part_c.py
    Description: ECE-7650 Deep Learning HW#1 Problem #3.c
    Author: Junyao Pu
    Date created: Jan 30th, 2020
    Date last modified: Feb 4th, 2020
    Python Version: 3.6
'''
import numpy as np
import csv
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd

REDUCE = 0

#import files
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

#normalization of vector
def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm

#sigmoid function
def test_sigmoid_vector(w, x, b):
    num = np.dot(w,x)+b
    return 1 / (1 + np.exp(-num))

#confusion matrix
def confusion_matrix(w,x,b ,testing_lable,prob_collection):
    for i in range (len(w)):
        class_result = test_sigmoid_vector(w[i], x, b[i])
        prob_collection.append(class_result.reshape((len(testing_lable),1)))

#import data files
home_path = '/home/junyao/Desktop/Graduate_Study/ECE-7450_Deep_Learning/Python_code/HW_1/cifar-10-batches-py/'
test_batch = []
label_name = []
test_batch.append(unpickle(home_path +'test_batch'))
label_name.append(unpickle(home_path +'batches.meta'))


print('LOADING DATA......')
print('..................')
print('..................')
#rearrange testing data to grayscale image
testing_lable = []
reorganized_testing_data=[]
for batch_num in range(len(test_batch)):
    for j in range (len(test_batch[batch_num][b'labels'])-REDUCE):
        i = 0
        vector_size=1024
        red = np.array(test_batch[batch_num][b'data'][j][(i*vector_size):(i+1)*vector_size])
        i+=1
        green = np.array(test_batch[batch_num][b'data'][j][(i*vector_size):(i+1)*vector_size])
        i+=1
        blue =np.array(test_batch[batch_num][b'data'][j][(i*vector_size):(i+1)*vector_size])

        grayscale = np.add(np.add(red*0.3, green*0.59),blue*0.11)
        lable = test_batch[batch_num][b'labels'][j]

        testing_lable.append(lable)
        reorganized_testing_data.append(normalize(grayscale).reshape((len(grayscale),1)))

x_testing=reorganized_testing_data[0]
for i in range (len(reorganized_testing_data)-1):
    x_testing = np.concatenate((x_testing, reorganized_testing_data[i+1]), axis=1)

#import weights and bias from part a)
saved_weights = []
saved_bias = []

saved_weights = np.loadtxt('weights.csv',delimiter=",", skiprows=0)
saved_bias = np.loadtxt('bias.csv',delimiter=",", skiprows=0)

print('CONSTRUCTIN FONFUSION MATRIX!!!')
print('CONSTRUCTIN FONFUSION MATRIX!!!')
print('CONSTRUCTIN FONFUSION MATRIX!!!')
#Calculate confusion matrix elements
prob_collection = []
confusion_array = np.zeros((10, 10))

confusion_matrix(saved_weights,x_testing,saved_bias ,testing_lable,prob_collection)

#rearrange the probability data
x_prob_collection = prob_collection[0]
for i in range (len(prob_collection)-1):
    x_prob_collection = np.concatenate((x_prob_collection, prob_collection[i+1]), axis=1)

for i in range(len(x_prob_collection)):
    confusion_array[testing_lable[i]][x_prob_collection[i].argmax()] += 1 ;

df_cm = pd.DataFrame(confusion_array, range(10), range(10))
sn.set(font_scale=1.4) # for label size
#sn.heatmap(df_cm, annot=True, annot_kws={"size": 8}) # font size
sn.heatmap(df_cm, annot=True,annot_kws={"size": 8}) # font size
plt.xlabel('Class airplane --> truck')
plt.ylabel('Class truck --> airplane')
plt.title('Confusion matrix')
plt.show()
