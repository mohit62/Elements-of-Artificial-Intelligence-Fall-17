from util_functions import drop_col
from blue_classifier import classify_blue
import math

#----------globals-----------
# Keeps score of prediction label for given image:
# Prediction [image][orientation]
orientation_score={}

def test_adaboost(data, stump_weights, stump_label_count):
    print "inside testing adaboost"
    image_names, data = drop_col(data)
    test_label, data = drop_col(data)
    prediction = {}

    for index in range(len(data)):
        prediction[index]=predict_image_orientation(index, stump_weights, stump_label_count,data)
    test_accuracy(prediction,test_label)
    f = open("output.txt",'w')
    for index in range(len(data)):
        f.write(image_names[index]+" "+str(int(prediction[index]))+"\n")



def predict_image_orientation(index, stump_weights, stump_label_count,data):
    for stump in stump_weights:
        guess=predict_label(stump, stump_label_count,data[index])
        update_orientation_score(guess, index, stump_weights[stump])

    return max(orientation_score[index], key=orientation_score[index].get)

def update_orientation_score(guess, index, weight):

    if index not in orientation_score:
        orientation_score[index]={}
    for orientation in [0,90,180,270]:
        if orientation not in orientation_score[index]:
            orientation_score[index][orientation]=0.0

        if orientation==guess:
            #orientation_score[index][orientation]+=math.exp(weight)
            orientation_score[index][orientation]+=(weight)

        else:
            #orientation_score[index][orientation]-=math.exp(weight)
            orientation_score[index][orientation]-=(weight)


def predict_label(stump, stump_label_count,img):
    temp={}
    for orientation in stump_label_count:
        if stump==(999,999):

           return classify_blue(img)
        else:
            temp[orientation]=stump_label_count[orientation][stump]


    return max(temp, key=temp.get)

def test_accuracy(prediction,test_label):
    count=0.0
    for index in prediction:
        if prediction[index]==test_label[index]:
            count+=1
    print "accuracy is",count/len(test_label)
