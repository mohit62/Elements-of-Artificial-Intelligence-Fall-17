# Each stump is comparison of each color in a image.
# Its been predicting 4 classes ie. 0 90 180 270.
# Hence total stumps are 192 * 192 * 4



from util_functions import eucl_distance,drop_col
import numpy as np
import math
import itertools
import adaboost_test
import random
from blue_classifier import classify_blue
import pickle as pkl
#---------Global----------------
stump={}
prediction_weight={}
stump_weights={}
trained_weights={}
orientation_score={}
label_dict = {}
train_label=[]
confusion_matrix={}
visited=[]
samples=[]
#-------------------------------



#---------Train Adaboost---------
def adaboost_train(data,model_filename):
    global train_label
    image_names, data= drop_col(data)
    train_label,data=drop_col(data)
    stump_count=1
    train_data={}
    global label_dict,samples
    #Assign labels and data to a dictionary
    for i in range(len(image_names)):
        train_data[image_names[i]]=data[i]
        label_dict[image_names[i]]=train_label[i]

    pool = list(itertools.permutations(range(192), 2))
    samples.append(random.sample(pool, 40))
    samples[0].append((999,999))
    """r=train_data[:,::3]
    g=train_data[:,1::3]
    b=train_data[:,2::3]"""
    #make_stumps(train_data)

    # Make stumps
    zero = data[::4]
    ninety = data[1::4]
    one_eighty = data[2::4]
    two_seventy = data[3::4]

    count_pixels(0, zero)
    count_pixels(90, ninety)
    count_pixels(180, one_eighty)
    count_pixels(270, two_seventy)

    stumps(data,image_names)
    initialize_weights(image_names)
    build_model(data,stump_count,image_names,train_label)
    pkl.dump( (trained_weights,orientation_score), open( model_filename, "wb" ) )


def initialize_weights(image_names):
    for index in range(len(image_names)):
        prediction_weight[index]= 1.0 / len(image_names)


# Count number of data points
def count_pixels(orientation, data):

    if orientation not in orientation_score:
        orientation_score[orientation]={}
    for image in data:

        for (i,j) in samples[0]:
            if i==999:
                continue
            if tuple([i, j]) not in orientation_score[orientation]:
                orientation_score[orientation][(i, j)] = 0

            if image[i]>image[j]:
                orientation_score[orientation][(i,j)] += 1


# Create hypothesis
def stumps(data,image_names):
    for (i,j) in samples[0]:
        count = 0.0
        for index in range(len(data)):
            if i== 999:
                guess=classify_blue(data[index])
            else:
                guess=predict_label(i,j)

                #name=image_names[index]

            if index not in confusion_matrix:
                confusion_matrix[index]={}

            if (i,j) not in confusion_matrix[index]:
                confusion_matrix[index][(i,j)]=False

            if guess==train_label[index]:
                confusion_matrix[index][(i,j)]=True
                count+=1

        error=(len(data)-count)/len(data)
        update_stump_weight((i, j), error)

# Updating weights of examples and selecting stumps
def build_model(data,stump_count,image_names,train_labels):
    hypothesis = max(stump_weights, key=stump_weights.get)
    for i in range(stump_count):
        error = 0.0

        for index in range(len(image_names)):
            if not confusion_matrix[index][hypothesis]:
                error+=prediction_weight[index]
        for index in range(len(image_names)):
            if confusion_matrix[index][hypothesis]:
                update_weight(index,error)

        normalize()
        trained_stump_wt(hypothesis, error)

        visited.append(hypothesis)
        hypothesis=next_hypothesis()


    """ordered=sorted(stump_weights,key=stump_weights.get,reverse=True)

    for (x,y) in ordered:
        for index in range(len(data)):
            if predict_label(x,y)!=train_label[index]:
                print "asd"""


# Select next best Hypothesis
def next_hypothesis():

    temp={}
    for hypothesis in samples[0]:
        if hypothesis in visited:
            continue

        temp[hypothesis]=hypothesis_score(hypothesis)

    return max(temp,key=temp.get)

# Calculate score of given hypothesis in
def hypothesis_score(hypothesis):
    score=0.0

    for index in range(len(confusion_matrix)):
        score += int(confusion_matrix[index][hypothesis])\
                *prediction_weight[index]
    return score


# Update weights of given example
def update_weight(index,error):
    b=prediction_weight[index]
    a=prediction_weight[index]*float(error)/(1-error)
    prediction_weight[index]=prediction_weight[index]*float(error)/(1-error)


# Normalize weights
def  normalize():
    total=sum(prediction_weight.values())
    for index in range(len(prediction_weight)):
        prediction_weight[index]*=1.0/(total)

# Update hypothesis weights
def update_stump_weight(hypothesis, error):
    stump_weights[hypothesis] = math.log((1 - error)*3 / error)
    #stump_weights[hypothesis] = math.log((error) /(1-error) )

# Update trained stump weights
def trained_stump_wt(hypothesis,error):
    #trained_weights[hypothesis]= math.log((error) /(1- error))
    trained_weights[hypothesis]= math.log((1 - error) *3/ error)


# Predict the label given the difference between pixel.
# Generate a hypothesis
def predict_label(x,y,minimum=False):
    temp={}
    for orientation in orientation_score:
        temp[orientation]=orientation_score[orientation][(x,y)]
    if minimum== False:
        return max(temp, key=temp.get)
    else:
        return min(temp,key=temp.get)
#---------------------------------------------------------------------------------------------------

#----------------------------------------------Test------------------------------------------------

def test_adaboost(train_test_file,model_file):
    test_data = np.genfromtxt(train_test_file, dtype="string")
    (trained_weights1,orientation_score1)=pkl.load( open( model_file, "rb" ) )
    adaboost_test.test_adaboost(test_data,trained_weights1,orientation_score1)
