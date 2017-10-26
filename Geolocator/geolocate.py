#!/usr/bin/env python
#Created by Mohit Saraf
#In this we have implemented NaiveBayes Classifier to classify tweets based on location.
#We have used laplace smoothening/add one smoothening to deal with unseen words we across in test data.
#Also we use logarithm of probabilities to prevent them from becoming zero.
#we have preprocessed train and test files to remove symbols as they act as irrelevant noise in the data.We have also removed certain words like articles etc which occur commomly in every sentence
#and cannot be considered an important feature set.
#The model gives an accuracy of 65.8 % for the given training and test data
#References:
#[1]https://nlp.stanford.edu/IR-book/html/htmledition/naive-bayes-text-classification-1.html

#read file and save the tweets corresponding to a location in a dictionary with key as location and the tweets
#as values.

from collections import Counter
import fileinput,math,sys

#preprocessing training data
def preprocessing(train_data):
    #merging all lists of words in tweet according to location
    newtrainData={}
    for location in train_data.keys():
        newtrainData[location]=[]
        for listno in train_data[location]:
            newtrainData[location]+=listno

    #Filtering data removing unwanted symbols
    filterData={}
    for location in train_data.keys():
        filterData[location]=[]
        for wordno in range(len(newtrainData[location])):
            filterData[location].append(newtrainData[location][wordno].replace("/",'').replace(":",'').replace("\n",'').replace("#",'').replace("_",'').replace(".",'').replace("-",'').replace(",",'').replace(")",'').replace("(",'').replace("*",'').replace("@",'').lower())
    # Counting frequency for each word w.r.t each location
    for location in train_data.keys():
        filterData[location]=Counter(filterData[location])
        for symb in ['&amp;','',"i'm",'in','at','a','and','the','an','as','i','to','for','of','this','my','you','our','with','so','on'] :
            del filterData[location][symb]
    return filterData

#preprocessing test data
def preprocessing_test(testData):
    newtestData={}
    for location in testData.keys():
        newtestData[location]=[]
        for listno in testData[location]:
            newtestData[location].append(listno)
    #Filtering data removing unwanted symbols
    filtertestData={}
    for location in newtestData.keys():
        filtertestData[location]=[]
        for tweet in range(len(newtestData[location])):
            filtertestData[location].append([])
            for word in range(len(newtestData[location][tweet])):
                filterword=newtestData[location][tweet][word].replace("/",'').replace(":",'').replace("\n",'').replace("#",'').replace("_",'').replace(".",'').replace("-",'').replace(",",'').replace(")",'').replace("(",'').replace("*",'').replace("@",'').lower()
                if  filterword not in ['&amp;','',"i'm",'in','at','a','and','the','an','as','i','to','for','of','this','my','you','our','with','so','on'] :
                    filtertestData[location][tweet].append(filterword)
    return filtertestData

#calculate probability of each location in the data set p(L)
def calculate_Loc_Prob(train_test_data):
    prob_Location={}
    for location in train_test_data.keys():
        prob_Location[location]=len(train_test_data[location])
    TotalnoOfTweets=sum(prob_Location.values())
    for location in train_test_data.keys():
        prob_Location[location]/=float(TotalnoOfTweets)
        prob_Location[location]=prob_Location[location]
    return prob_Location

#calculate probability p(w/L)
def calculate_word_loc_Prob(trainData):
    Vocabulary={}
    for location in trainData.keys():
        Vocabulary=Counter(Vocabulary)+Counter(trainData[location])
    V=len(Vocabulary)
    total_words_location={}
    prob_word_location={}
    for location in trainData.keys():
        total_words_location[location]=sum(trainData[location].values())
        prob_word_location[location]={}
        for word,freq in zip(trainData[location].keys(),trainData[location].values()):
            prob_word_location[location][word]=(freq+1)/float(total_words_location[location]+V)
    return prob_word_location,V,total_words_location

#read train/testdata
def readData(filename):
    traintest={}
    for line in fileinput.input(filename):
        temp=line.split(" ")
        if temp[0] not in traintest.keys():
            traintest[temp[0]]=[]
        traintest[temp[0]].append(temp[1:])
    return traintest

#calculating p(L/W) to predict location with Laplace Smoothing
def predictLocation(Traindata,Testdata,prob_loc,prob_loc_word,V,Total):
    Correct=0
    Incorrect=0
    Prediction=[]
    Actual=[]
    for location in Testdata.keys():
        for tweetno in range(len(Testdata[location])):
            tweet=Testdata[location][tweetno]
            prob_word_loc={}
            for loc in Testdata.keys():
                prob=math.log(prob_loc[loc])
                for word in tweet:
                    if word not in prob_loc_word[loc].keys():
                        prob_loc_word[loc][word]=1/float(V+Total[loc])
                    prob+=math.log(prob_loc_word[loc][word])
                prob_word_loc[loc]=prob
            predict=max(prob_word_loc, key=prob_word_loc.get)
            if location == predict:
                Correct+=1
            else:
                Incorrect+=1
            Prediction.append(predict)
    Accuracy=Correct/float(Correct+Incorrect)*100
    return Accuracy,Prediction
#Get trainingfile,testfile and outputfilename from the user
TrainingFile=sys.argv[1]
TestingFile=sys.argv[2]
OutputFile=sys.argv[3]
#Read training data from file
trainData=readData(TrainingFile)
#preprocess training data
preprocessedTrainData=preprocessing(trainData)
#calculate probability of each location
prob_loc=calculate_Loc_Prob(trainData)
#calculate probability of each word occuring w.r.t to that location
prob_loc_word=calculate_word_loc_Prob(preprocessedTrainData)
#get vocabulary count
V=prob_loc_word[1]
#get total words count
Total=prob_loc_word[2]
#Read test data from file
testData=readData(TestingFile)
#preprocess testing data
preprocessingtestdata=preprocessing_test(testData)
Output=predictLocation(preprocessedTrainData,preprocessingtestdata,prob_loc,prob_loc_word[0],V,Total)
#print accuracy for the model
print "Accuracy for the model is ",Output[0],"%"
#get 5 most common words for each location
for loc in preprocessedTrainData.keys():
    print "For Location ",loc," Most common 5 words are: ",preprocessedTrainData[loc].most_common(5)
#Writing output to the output.txt file with first column having predicted location and the second column with the actual location and the tweet after that
lineno=0
f = open(OutputFile,'w')
for line in fileinput.input(TestingFile):
    f.write(Output[1][lineno]+" "+line)
    lineno+=1
f.close()
