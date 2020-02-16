'''
    File name: Prob_2_part_abc.py
    Description: ECE-7650 Deep Learning HW#1 Problem #2
    Author: Junyao Pu
    Date created: Jan 30th, 2020
    Date last modified: Feb 2nd, 2020
    Python Version: 3.6
'''
import numpy as np
import csv
import matplotlib.pyplot as plt
import random
import pylab

LEARNING_RATR = 0.1
'''
part c)--> write a sigmoid function
'''
def test_sigmoid_vector(w, x, b):
    num = np.dot(w,x)+b
    return 1 / (1 + np.exp(-num))
'''
#part a) --> write a cost function
'''
def log_likehood_vector(y, w, x, b, reg_weight):

    M = len(y)

    reg = (reg_weight/2)*np.dot(w,w)

    sigm = test_sigmoid_vector(w, x, b)

    summation = y*np.log(sigm) + (1-y)*np.log(1-sigm)
    #print(summation)

    final =((-1/M)*np.sum(summation)) + reg

    return final

'''
#part c) --> write a gradient of cost function
'''
def gradient_vector(y, w, x, b,dw,db):
    #N derivatives for weights
    M = len(y)

    sigm = test_sigmoid_vector(w, x, b)


    db[0] = (-1/M)*np.sum(y-sigm)

    for i in range(len(w)):
        dw[i] = ((-1/M)*np.sum((y-sigm)*x[i]))


#Read csv file
home_path = '/home/junyao/Desktop/Graduate_Study/ECE-7450_Deep_Learning/Python_code/HW_1/'
read_file = open(home_path +'KidsHeightData.csv', 'r', newline='\n', encoding="UTF-8-sig")
csv_read_file = csv.reader(read_file, dialect='excel')

shoe_size = []
parents_height = []
class_lable = []

for row in csv_read_file:
    shoe_size.append(float(row[0]))
    parents_height.append(float(row[1]))
    class_lable.append(float(row[2]))

shoe_size_plot = np.array(shoe_size)
parents_height_plot = np.array(parents_height)

shoe_size = np.array(shoe_size)
parents_height = np.array(parents_height)
class_lable = np.array(class_lable)


#organize training data set
w = np.zeros(2)
b = np.zeros(1)

x = []
x.append(shoe_size)
x.append(parents_height)
y = class_lable

dw=np.zeros(len(w))
db = [0]

dw[0]=1
iteration=0

#training with 50000 steps
while (iteration<=50000 and (np.linalg.norm(dw)+ np.linalg.norm(db))>= 0.0001):
    #print(iteration)
    gradient_vector(y, w, x, b, dw,db)

    w-=dw
    b-=db
    iteration+=1


    if(iteration%200 == 0):
        print(str('cost function:')+ str(log_likehood_vector(y, w, x, b, 0)))
print('--------------------------------------------')
print('Report our line parameters:')
print(str('cost function:')+ str(log_likehood_vector(y, w, x, b, 0)))
print(str('Learning_iteration:')+ str(iteration))
print(str('Variable w is:')+ str(w))
print(str('Variable b is:')+ str(b))

#Plot the data and separeting line
above_2m_shoe = []
above_2m_parents = []

below_2m_shoe = []
below_2m_parents = []


for i in range (len(y)):
    if y[i] == 1:
        above_2m_shoe.append(shoe_size_plot[i])
        above_2m_parents.append(parents_height_plot[i])
    else:
        below_2m_shoe.append(shoe_size_plot[i])
        below_2m_parents.append(parents_height_plot[i])

#Plot the training data and separeting line
plt.scatter(above_2m_shoe, above_2m_parents,color='black',marker="*",label='kids who grew up to be >= 2m')
plt.scatter(below_2m_shoe, below_2m_parents,color='green',marker = "2",label='kids who grew up to be <= 2m')
yfit = [-(w[0]*xi+b)/w[1] for xi in shoe_size_plot]
plt.plot(shoe_size_plot, yfit,color='red',label='Separeting Line with w1= ' + str(round(w[0],2)) +',w2= '+ str(round(w[1],2)) +',b = ' + str(round(b[0],2)))
plt.xlim(0,  8)
plt.ylim(4.5, 7)
plt.xlabel('Shoe Size at Age 3')
plt.ylabel('Average Height of Parents')
plt.title('2 sets of data and a separeting line')
plt.legend()
plt.show()
