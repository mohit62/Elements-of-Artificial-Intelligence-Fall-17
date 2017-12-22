#!/usr/bin/env python
# For k=11 we are getting 71.04% accuracy
#Reference
#[1] https://medium.com/dataholiks-distillery/l2-distance-matrix-vectorization-trick-26aa3247ac6c
#[2] https://en.wikipedia.org/wiki/Rectifier_(neural_networks)
#[3] https://stats.stackexchange.com/questions/47590/what-are-good-initial-weights-for-training-a-neural-network
#
'''./orient.py train train-data.txt model_file.txt nearest/adaboost/nnet/best'''
import sys
import adaboost
import numpy as np
from scipy import spatial
from collections import Counter
import math
#Converting training/test data to numpy array
def read_train_testTonumpy(train_test_file):
    data=[]
    file = open(train_test_file, 'r')
    file_list=[]
    for line in file:
        data.append([pixel for pixel in line.split()[1:]])
        file_list.append(line.split()[0])
            #print np.array(data)
            #print np.array(file_list)
    return np.array(data),np.array(file_list)

#Nearest Neighbour
def nearest(train_test,train_test_file,model_file,k,numpy_file_list):
    if train_test=='train':
        with open(model_file,'w') as f:
            np.savetxt(f,train_test_file,fmt="%3s")
        return "Model Trained"
    elif train_test=='test':
        train_dict=[]
        model=np.loadtxt(model_file)
        #calculating eucladian distance for all test data against all training data
        #euc_dists=spatial.distance.cdist(train_test_file[:,1:].astype(float), model[:,1:], metric='euclidean')
        euc_dists = np.sqrt(-2 * np.dot(train_test_file[:,1:].astype(float), model[:,1:].T) + np.sum(model[:,1:]**2,axis=1) + np.sum(train_test_file[:,1:].astype(float)**2,axis=1)[:, np.newaxis])
        #get index of the matrix sorted by row in the same shape of original matrix
        sorted_euc_dists=np.argsort(euc_dists, axis=1)
        k_sorted_euc_dist=sorted_euc_dists[:,:k]
        #print model[k_sorted_euc_dist,0].shape
        label_k=model[k_sorted_euc_dist,0]
        correct=0
        total=0
        f = open("output.txt",'w')
        for row_no in range(label_k.shape[0]):
            predict=max(Counter(label_k[row_no,]), key=Counter(label_k[row_no,]).get)
            f.write(numpy_file_list[row_no]+" "+str(int(predict))+"\n")
            if train_test_file[row_no,0].astype(float)==predict:
                correct+=1.0
            total+=1.0
        f.close()
        Accuracy=correct/total
        return Accuracy
    #print train_test_file

#Sigmoid function
def sigmoid(xvec):
    return 1/(1 + np.exp(-xvec))
# derivative for Sigmoid
def dsigmoid(xvec):
    return xvec * (1 - xvec)
# tanh function
def tanh(x):
    return np.tanh(x)
# derivative for tanh
def dtanh(y):
    return 1 - y**2
#softplus function
def softplus(y):
    return np.log(1+np.exp(y))
#derivative for softplus
def dsoftplus(y):
    return 1/(1 + np.exp(-y))
#Neural Network
def nnet(train_test,train_test_file,model_file,numpy_file_list,nh):
    if train_test=='train':
        alpha=0.1
        epochs=2000
        Xtrain=train_test_file[:,1:].astype(float)/255.0
        ytrain=train_test_file[:,0].astype(float)
        new_ytrain=[]
        for i in range(ytrain.shape[0]):
            if ytrain[i]==0:
                new_ytrain.append([1,0,0,0])
            elif ytrain[i]==90:
                new_ytrain.append([0,1,0,0])
            elif ytrain[i]==180:
                new_ytrain.append([0,0,1,0])
            else:
                new_ytrain.append([0,0,0,1])
        new_ytrain=np.array(new_ytrain)
        #print new_ytrain
        no_input_nodes=Xtrain.shape[1]
        #print no_input_nodes
        no_output_nodes=new_ytrain.shape[1]
        new_ytrain=np.array(new_ytrain)
        ri=math.sqrt(2.0/(no_input_nodes))
        ro=math.sqrt(2.0/(nh))
        np.random.seed(32687)
        wi=np.random.uniform(-ri,ri,size=(no_input_nodes, nh))
        wo=np.random.uniform(-ro,ro,size=(nh, no_output_nodes))
        #bh=np.random.uniform(size=(1,nh))
        #bout=np.random.uniform(size=(1,no_output_nodes))
        
        #Batch Gradient descent
        for iteration in range(epochs):
            a_i_1=Xtrain
            #input to hidden layer
            inp_i_1=np.dot(a_i_1,wi)#+bh
            #hidden layer activations
            a_j_1=softplus(inp_i_1)
            #input to Output layer
            inp_i_2=np.dot(a_j_1,wo)#+bout
            #Output layer activation
            a_j_2=softplus(inp_i_2)
            #Backward Propogation
            #Calculating gradient of error
            delta_j=np.subtract(new_ytrain,a_j_2)*dsoftplus(a_j_2)
            #gradient of error at hidden layer
            delta_i=np.dot(delta_j,wo.T)*dsoftplus(a_j_1)
            #update weights
            dot_output=np.dot(a_j_1[0].reshape((a_j_1[0].shape[0],1)),delta_j[0].reshape((delta_j[0].shape[0],1)).T)
            for sample in range(1,Xtrain.shape[0]):
                dot_output+=np.dot(a_j_1[sample].reshape((a_j_1[sample].shape[0],1)),delta_j[sample].reshape((delta_j[sample].shape[0],1)).T)
            dot_output/=float(Xtrain.shape[0])
            dot_input=np.dot(a_i_1[0].reshape((a_i_1[0].shape[0],1)),delta_i[0].reshape((delta_i[0].shape[0],1)).T)
            for sample in range(1,Xtrain.shape[0]):
                dot_input+=np.dot(a_i_1[sample].reshape((a_i_1[sample].shape[0],1)),delta_i[sample].reshape((delta_i[sample].shape[0],1)).T)
            dot_input/=float(Xtrain.shape[0])
            #break
            wo=wo+dot_output*alpha
            #bout += np.sum(delta_j, axis=0,keepdims=True) *alpha
            wi=wi+dot_input*alpha
            #bh += np.sum(delta_i, axis=0,keepdims=True) *alpha
        with open(model_file,'w') as f:
            np.savetxt(f,wo)
            np.savetxt(f,wi)
                

    if train_test=='test':
        data=[]
        file = open(model_file, 'r')
        file_list=[]
        for line in file:
            data.append([pixel for pixel in line.split()])
        wo=np.array(data[:nh]).astype(float)
        wi=np.array(data[nh:]).astype(float)
        orientation=[0,90,180,270]
        #forward propogation
        a_i_1=train_test_file[:,1:].astype(float)/255.0
        #input to hidden layer
        inp_i_1=np.dot(a_i_1,wi)
        #hidden layer activations
        a_j_1=softplus(inp_i_1)
        #input to Output layer
        inp_i_2=np.dot(a_j_1,wo)
        #Output layer activation
        a_j_2=softplus(inp_i_2)
        #Sorting output by index rowwise in matrix
        sorted_euc_dists=np.argsort(a_j_2, axis=1)
        f = open("output.txt",'w')
        correct=0
        total=0
        for row_no in range(len(sorted_euc_dists[:,-1])):
            predict=orientation[sorted_euc_dists[row_no,-1]]
            f.write(numpy_file_list[row_no]+" "+str(int(predict))+"\n")
            if train_test_file[row_no,0].astype(float)==predict:
                correct+=1.0
            total+=1.0
        f.close()
        Accuracy=correct/total
        return Accuracy

#Best Classifier
def best(train_test,train_test_file):
    return 0

#Getting arguments from the user
(train_test,train_test_file,model_file,model)=sys.argv[1:5]
#Value of K for k nearest neighbour
k=11
#no of hidden nodes
nh=32
#print numpy_train_test.shape
#Training or testing the model selected according to the arguments given by user
if model=='nearest':
    if train_test=='train':
        #Reading training/test file and converting it into a numpy array
        train_test_file_numpy=read_train_testTonumpy(train_test_file)
        numpy_train_test=train_test_file_numpy[0]
        numpy_file_list=train_test_file_numpy[1]
        print "Training model for nearest \n",nearest(train_test,numpy_train_test,model_file,k,numpy_file_list)
    elif train_test=='test':
        #Reading training/test file and converting it into a numpy array
        train_test_file_numpy=read_train_testTonumpy(train_test_file)
        numpy_train_test=train_test_file_numpy[0]
        numpy_file_list=train_test_file_numpy[1]
        print "Testing model for nearest\n Accuracy for the model is",nearest(train_test,numpy_train_test,model_file,k,numpy_file_list)
elif model=='adaboost':
    if train_test=='train':
        data = np.genfromtxt(train_test_file,dtype="string")
        print "Training model for adaboost",adaboost.adaboost_train(data,model_file)
    elif train_test=='test':
        print "Testing model for adaboost",adaboost.test_adaboost(train_test_file,model_file)
elif model=='nnet':
    if train_test=='train':
        train_test_file_numpy=read_train_testTonumpy(train_test_file)
        numpy_train_test=train_test_file_numpy[0]
        numpy_file_list=train_test_file_numpy[1]
        print "Training model for nnet",nnet(train_test,numpy_train_test,model_file,numpy_file_list,nh)
    elif train_test=='test':
        train_test_file_numpy=read_train_testTonumpy(train_test_file)
        numpy_train_test=train_test_file_numpy[0]
        numpy_file_list=train_test_file_numpy[1]
        print "Testing model for nnet\n Accuracy for the model is",nnet(train_test,numpy_train_test,model_file,numpy_file_list,nh)
elif model=='best':
    if train_test=='train':
        train_test_file_numpy=read_train_testTonumpy(train_test_file)
        numpy_train_test=train_test_file_numpy[0]
        numpy_file_list=train_test_file_numpy[1]
        print "Training model for best",nnet(train_test,numpy_train_test,model_file,numpy_file_list,nh)
    elif train_test=='test':
        train_test_file_numpy=read_train_testTonumpy(train_test_file)
        numpy_train_test=train_test_file_numpy[0]
        numpy_file_list=train_test_file_numpy[1]
        print "Testing model for best",nnet(train_test,numpy_train_test,model_file,numpy_file_list,nh)



