#!/usr/bin/python
# ./ocr.py : Perform optical character recognition, usage:
# Authors: (Mohit Saraf(sarafm),Paritosh Morparia(pmorpari),Uteerna Koul(ukoul))
# (based on skeleton code by D. Crandall, Oct 2017)
# 1)Naive Bayes Classifier
# ->For This we calculated the emission probability by calculating the noisy pixels for a letter  matching with the noise free image and the considering the leftover pixels as noise.We assigned a weight
# of 1/3 to each noisy pixel of image and 2/3 to each noise free pixel.
# 2)Variable elimination
# ->For this we calculated forward probabilities by marginalising the probabilities over each state in forward direction from start to end of the sentence.
# ->For backward we will start marginalising from the end of the sentence to the beginning of the sentence.We will take an extra observation t+1 next to the last observation in the sentence and assign
# an emission probability of 1 to that state and backward probability Beta of 1.
#-> After calculating the forward and backward probabilities we will be merging the two probabilities to get the best state sequence for the given image.
# 3)Vitterbi Algorithm
#->For this we will first calculate the values of vitterbi table for all states at time 0 and then multiply  it with maximum of the product of emission and transition probabilities and maintain a value of
# for previous state in a dictionary for which we get the maximum value of the product.After we calculate the vitterbi table for all the sequence we will get the state for which the probability for the
# last element in the sequence is maximum and then we bactrack from there using the previous values stored to get our desired sequence.
# [Results]:
# We get better performance for both Vitterbi and Variable elimination with vitterbi being the best in terms of prediction than Naive Bayes Classifier.For Noisy and sparse images the performance deteriorates for all three with Naive Bayes being the worst.
#Output
'''
[sarafm@silo part2]$ ./ocr.py courier-train.png bc.train.txt test-17-0.png
Simple: 1t 1s so orcerec.
HMM VE: It is so ordered.
HMM MAP: It is so ordered.
'''
#References:
#[1] http://www.cs.rochester.edu/u/james/CSC248/Lec11.pdf
#[2] https://web.stanford.edu/class/cs124/lec/naivebayes.pdf

from PIL import Image, ImageDraw, ImageFont
import sys,math
from collections import Counter

CHARACTER_WIDTH=14
CHARACTER_HEIGHT=25


def load_letters(fname):
    im = Image.open(fname)
    px = im.load()
    (x_size, y_size) = im.size
    #print im.size
    #print int(x_size / CHARACTER_WIDTH) * CHARACTER_WIDTH
    result = []
    for x_beg in range(0, int(x_size / CHARACTER_WIDTH) * CHARACTER_WIDTH, CHARACTER_WIDTH):
        result += [ [ "".join([ '*' if px[x, y] < 1 else ' ' for x in range(x_beg, x_beg+CHARACTER_WIDTH) ]) for y in range(0, CHARACTER_HEIGHT) ], ]
    return result

def load_training_letters(fname):
    TRAIN_LETTERS="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789(),.-!?\"' "
    letter_images = load_letters(fname)
    return { TRAIN_LETTERS[i]: letter_images[i] for i in range(0, len(TRAIN_LETTERS) ) }

#calculating emmision probabilities using naive bayes classifier matching training letters and test letters at pixel level and calculating probability of noise and match at each pixel
def emission_NaiveBayes(train_letters,test_letters):
    prob_emit={}
    for char in train_letters.keys():
        prob_emit[char]={}
        for pixels,index in zip(test_letters,range(0,len(test_letters))):
            count_match=1.0
            count_noise=1.0
            for i in range(25):
                for j in range(14):
                    if train_letters[char][i][j]==pixels[i][j]:
                        count_match*=2.0/3
                    else:
                        count_noise*=1.0/3
        
            prob_emit[char][index]=(count_match*count_noise)#*math.pow(10,65)
#print prob_emit
#print sum(prob_emit[char].values())
    return prob_emit

#calculating prior probability
def prior_probabilities(train_letters):
    prob_char={}
    for char in train_letters.keys():
        prob_char[char]=1.0/len(train_letters.keys())
    return prob_char

#calculating p(char/pixelvalueAtPosition) to predict character
def predict_test(test_letters,emm_prob,prob_charac,train_letters):
    finalpredict=""
    for index in range(len(test_letters)):
        prob_pixel_char={}
        for char in train_letters.keys():
            prob=math.log(prob_charac[char])+math.log(emm_prob[char][index])
            prob_pixel_char[char]=prob
        predict=max(prob_pixel_char, key=prob_pixel_char.get)
        finalpredict+=predict
    return finalpredict

#calculate initial probability and transisiton probability
def calculate_init_transition_prob(fname,train_letters):
    exemplars = []
    file = open(fname, 'r');
    list_line_char=[]
    for line in file:
        data = tuple([w for w in line.split()])
        list_char=[]
        corpus=data[0::2]
        #print corpus
        for word,word_index in zip(corpus,range(len(corpus))):
            if word_index!=0 and corpus[word_index] not in ['.',',','(',')','-','.','!','?','"',"'" ]:
                list_char.append(' ')
            for char in word:
                list_char.append(char)
    #del list_char[0]
        list_line_char.append(list_char)
    #print list_line_char

#calculate initial probability
    initial_letter=[]
    for line1 in list_line_char:
        #print line
        initial_letter.append(line1[0])
    freq_initial=Counter(initial_letter)
#print sum(freq_initial.values())#44204
#print len(freq_initial)
    prob_intial={}
    for initial in freq_initial.keys():
        prob_intial[initial]=float(freq_initial[initial])/(sum(freq_initial.values()))
#print sum(prob_intial.values())

#calculate transisiton probability
    freq_trans={}
    prob_trans={}
    uniq_trans=0.0
    for letter in train_letters.keys():
        corres_letters=[]
        for line2 in list_line_char:
            for char_index in range(len(line2)-1):
                if line2[char_index]==letter:
                    corres_letters.append(line2[char_index+1])
        freq_trans[letter]=Counter(corres_letters)
        uniq_trans+=len(freq_trans[letter])

#print freq_trans
    for s_current in freq_trans.keys():
        prob_trans[s_current]={}
        for s_next in freq_trans[s_current].keys():
            #print len(freq_trans[s_current].keys())
            prob_trans[s_current][s_next]=float(freq_trans[s_current][s_next])/(sum(freq_trans[s_current].values()))
#print sum(prob_trans[s_current].values())
    return prob_intial,prob_trans,freq_trans,freq_initial,uniq_trans

#prediction using vitterbi algorithm
def Vitterbi_Algorithm(initial_prob,em_prob,trans_prob,test_letters,frq_trans,frq_init,frq_unique):
    Vit={}
    #print initial_prob
    #calculating Vit[t][state] where t=0
    Vit={0:{},'prev':{}}
    for state in em_prob.keys():
        if state not in initial_prob:
            initial_prob[state]=1e-100
        Vit[0][state]=math.log(initial_prob[state])+math.log(em_prob[state][0])
        Vit['prev'][0]={}
        Vit['prev'][0][state]=None
            #print em_prob.keys()
    for t in range(1,len(test_letters)):
        Vit[t]={}
        Vit['prev'][t]={}
        for state1 in em_prob.keys():
            max_prob=max([(Vit[t-1][state_prev]+math.log(1e-100),state_prev) if (state1 not in trans_prob[state_prev].keys() or state_prev not in trans_prob.keys()) else (math.log(trans_prob[state_prev][state1])+Vit[t-1][state_prev],state_prev) for state_prev in em_prob.keys()])
            Vit['prev'][t][state1]=max_prob[1]
            Vit[t][state1]=math.log(em_prob[state1][t])+max_prob[0]
#print Vit['prev']
    sentence=[]
    charac=max(Vit[len(test_letters)-1],key=Vit[len(test_letters)-1].get)
    sentence.append(charac)
    for t_i in range(len(test_letters)-1,0,-1):
        sentence.append(Vit['prev'][t_i][charac])
        charac=Vit['prev'][t_i][charac]
#print sentence
    return ''.join(reversed(sentence))


#prediction using Variable elimination
def Variable_Elimination(initial_prob,em_probab,trans_probab,test_letters,frq_trans,frq_init,frq_unique):
#Implementation of Forward Algorithm
#initialization Step
    alpha={}
    alpha[0]={}
    for state in em_probab.keys():
        if state not in initial_prob:
            initial_prob[state]=1e-100
        alpha[0][state]=initial_prob[state]*em_probab[state][0]

    #iteration Step
    for t in range(1,len(test_letters)):
        alpha[t]={}
        for state1 in em_probab.keys():
            alpha[t][state1]=sum([math.pow(20,53)*alpha[t-1][state2]*1e-100*em_probab[state1][t] if (state1 not in trans_probab[state2].keys()) else math.pow(20,53)*alpha[t-1][state2]*trans_probab[state2][state1]*em_probab[state1][t] for state2 in em_probab.keys()])

#Implemetation of backward algorithm
    Beta={}
    Beta[len(test_letters)]={}
    for state_3 in em_probab.keys():
        Beta[len(test_letters)][state_3]=1.0
        em_probab[state_3][len(test_letters)]=1.0
    for t2 in range(len(test_letters)-1,-1,-1):
        Beta[t2]={}
        #print state3
        for state3 in em_probab.keys():
            Beta[t2][state3]=sum([math.pow(20,53)*Beta[t2+1][state4]*1e-100*em_probab[state4][t2+1] if (state4 not in trans_probab[state3].keys()) else math.pow(20,53)*Beta[t2+1][state4]*trans_probab[state3][state4]*em_probab[state4][t2+1]  for state4 in em_probab.keys()])
#print Beta
#print Beta
#Merging forward backward algorithms to predict the sequence of characters
    prod_alpha_beta={}
    for t3 in range(len(test_letters)):
        prod_alpha_beta[t3]={}
        for state5 in em_probab.keys():
            prod_alpha_beta[t3][state5]=(alpha[t3][state5]*Beta[t3][state5])
#print prod_alpha_beta
    sent=[]
    for t3 in range(len(test_letters)):
        sent.append(max(prod_alpha_beta[t3],key=prod_alpha_beta[t3].get))
    return ''.join(sent)

#####
# main program
(train_img_fname, train_txt_fname, test_img_fname) = sys.argv[1:]
train_letters = load_training_letters(train_img_fname)
test_letters = load_letters(test_img_fname)
Classifier=('Simple','VE','MAP')
#prior probabilities
prior_prob=prior_probabilities(train_letters)
#Emission probabbilities using naiveBayesModel
emission_prob=emission_NaiveBayes(train_letters,test_letters)
#initial and transition prob
cal_init_transit=calculate_init_transition_prob(train_txt_fname,train_letters)
#initial prob
initial_prob=cal_init_transit[0]
#transition prob
transition_prob=cal_init_transit[1]
#frequency of transitions
frq_trans=cal_init_transit[2]
#frequency of initial states
frq_init=cal_init_transit[3]
#frequency of unique transition
frq_unique=cal_init_transit[4]
for method in Classifier:
    if method=='Simple':
        prediction=predict_test(test_letters,emission_prob,prior_prob,train_letters)
        print 'Simple:\t',prediction
    elif method=='VE':
        print 'HMM VE:\t',Variable_Elimination(initial_prob,emission_prob,transition_prob,test_letters,frq_trans,frq_init,frq_unique)
    else:
        Vitterbi=Vitterbi_Algorithm(initial_prob,emission_prob,transition_prob,test_letters,frq_trans,frq_init,frq_unique)
        print 'HMM MAP:',Vitterbi
        
## Below is just some sample code to show you how the functions above work. 
# You can delete them and put your own code here!


# Each training letter is now stored as a list of characters, where black
#  dots are represented by *'s and white dots are spaces. For example,
#  here's what "a" looks like:
#print "\n".join([ r for r in train_letters['a'] ])

# Same with test letters. Here's what the third letter of the test data
#  looks like:
#print "\n".join([ r for r in test_letters[2] ])



