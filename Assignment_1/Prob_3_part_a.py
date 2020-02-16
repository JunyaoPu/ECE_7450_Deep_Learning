'''
    File name: Prob_3_part_a.py
    Description: ECE-7650 Deep Learning HW#1 Problem #3.a
    Author: Junyao Pu
    Date created: Jan 30th, 2020
    Date last modified: Feb 4th, 2020
    Python Version: 3.6
'''
import numpy as np
import csv
import matplotlib.pyplot as plt


LEARNING_RATR = 0.001
REDUCE = 0

#import data files
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

#normalize of the input array
def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm

#sigmoid function
def test_sigmoid_vector(w, x, b):
    num = np.dot(w,x)+b
    return 1 / (1 + np.exp(-num))

#calculate the cost function
def log_likehood_vector(y, w, x, b, reg_weight):

    M = len(y)

    reg = (reg_weight/2)*np.dot(w,w)

    sigm = test_sigmoid_vector(w, x, b)

    summation = y*np.log(sigm) + (1-y)*np.log(1-sigm)
    #print(summation)

    final =((-1/M)*np.sum(summation)) + reg

    return final

#calculate the gradient of cost function
def gradient_vector(y, w, x, b,dw,db):
    #N derivatives for weights
    M = len(y)

    sigm = test_sigmoid_vector(w, x, b)

    db[0] = (-1/M)*np.sum(y-sigm)

    for i in range(len(w)):
        dw[i] = ((-1/M)*np.sum((y-sigm)*x[i]))

#Training function
def training_fun(int_class, x, x_testing, training_lable, testing_lable, saved_weights, saved_bias):
    w = np.zeros(len(x))
    b = np.zeros(1)

    training_lable_class_num = []
    for i in range (len(training_lable)):
        if training_lable[i] == int_class:
            training_lable_class_num.append(1)
        else:
            training_lable_class_num.append(0)

    dw=np.zeros(len(w))
    db = [0]

    iteration=0
    while (iteration <= 200):

        if(iteration % 100 == 0):
            print(str(iteration) + str( ' steps --> cost function ')+str('Class # ')+str( ' :')+ str(log_likehood_vector(np.array(training_lable_class_num), np.array(w), x, b, 0.0)))

        gradient_vector(training_lable_class_num, w, x, b,dw,db)
        w-=dw
        b-=db
        iteration+=1
    print('TRAINING END!!! CLASS:' + str(int_class))

    saved_weights.append(w)
    saved_bias.append(b[0])

    acc = accuracy_report(int_class,w,x_testing,b ,testing_lable)
    print('TESTING ACCURACY FOR CLASS: '+ str(int_class) +' IS: {}' .format(round(acc,4)))

#Check accuracy
def accuracy_report(int_class,w,x,b ,testing_lable):
    class_result = test_sigmoid_vector(w, x, b)

    testing_result = []
    for i in range (len(testing_lable)):
        if class_result[i] >= 0.5:
            testing_result.append(1)
        else:
            testing_result.append(0)

    training_lable_class_num = []
    for i in range (len(testing_lable)):
        if testing_lable[i] == int_class:
            training_lable_class_num.append(1)
        else:
            training_lable_class_num.append(0)

    correct_num = 0
    for i in range (len(testing_lable)):
        if testing_result[i] == training_lable_class_num[i]:
            correct_num += 1

    accuracy_pre =    correct_num /  len(testing_lable)
    return accuracy_pre

#plot the weight image
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

#import data files
home_path = '/home/junyao/Desktop/Graduate_Study/ECE-7450_Deep_Learning/Python_code/HW_1/cifar-10-batches-py/'
lable_name = unpickle(home_path +'batches.meta')
data_batch = []
data_batch.append(unpickle(home_path +'data_batch_1'))
#data_batch.append(unpickle(home_path +'data_batch_2'))
#data_batch.append(unpickle(home_path +'data_batch_3'))
#data_batch.append(unpickle(home_path +'data_batch_4'))
#data_batch.append(unpickle(home_path +'data_batch_5'))

test_batch = []
label_name = []
test_batch.append(unpickle(home_path +'test_batch'))
label_name.append(unpickle(home_path +'batches.meta'))

print('LOADING DATA......')
print('..................')
print('..................')

#rearrange all training data to grayscale image
training_lable = []
reorganized_data=[]

for batch_num in range(len(data_batch)):
    #for j in range (len(data_batch[batch_num][b'labels'])):
    for j in range (len(data_batch[batch_num][b'labels'])-REDUCE):
        i = 0
        vector_size=1024
        red = np.array(data_batch[batch_num][b'data'][j][(i*vector_size):(i+1)*vector_size])
        i+=1
        green = np.array(data_batch[batch_num][b'data'][j][(i*vector_size):(i+1)*vector_size])
        i+=1
        blue =np.array(data_batch[batch_num][b'data'][j][(i*vector_size):(i+1)*vector_size])

        lable = data_batch[batch_num][b'labels'][j]

        grayscale = np.add(np.add(red*0.3, green*0.59),blue*0.11)

        training_lable.append(lable)
        reorganized_data.append(normalize(grayscale).reshape((len(grayscale),1)))

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

#rearrange the elements in the matrix
x=reorganized_data[0]
for i in range (len(reorganized_data)-1):
    x = np.concatenate((x, reorganized_data[i+1]), axis=1)

x_testing=reorganized_testing_data[0]
for i in range (len(reorganized_testing_data)-1):
    x_testing = np.concatenate((x_testing, reorganized_testing_data[i+1]), axis=1)

print('TRAINING START!')

saved_weights = []
saved_bias = []

#run 10 class training
for i in range (10):
    training_fun(i, x, x_testing, training_lable, testing_lable,saved_weights,saved_bias)

#save weights and bias for part b and part c
np.savetxt("weights.csv", saved_weights, delimiter=",")

np.savetxt("bias.csv", saved_bias, delimiter=",")
