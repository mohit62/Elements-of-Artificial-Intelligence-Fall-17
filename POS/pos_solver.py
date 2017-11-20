#!/usr/bin/env python
##########################################################################
# CS B551 Fall 2017, Assignment #3
#
# Paritosh, Mohit, Uteerna
# pmorpari, sarafm, ukoul
# (Based on skeleton code by D. Crandall)
#
#
###########################################################################
#Training data steps:
#1. Make dictionaries which will be used eg: counting # of transitions, # words in pos
#2. Calculate required probabilities:
#   a.probability of state
#   b.Initial state probability
#   c.Transition probability
#   d.Emission probability
#
#Simplified HMM:
#   Used naive bayes algorithm for simplified prediction of words.
#   Probabilities used:
#           a. emission_probability---> P(W_i|S_i)
#           b. State probability------->P(S)
#
#Variable Elimination:
#   Marginalizing probabilities of states from forward and backward in the Markov chain
#   Probabilities used:
#       a) Initial factor for given state:----->initial_probability(state)
#       b) Factor in forward state:------------>p(previous_state)*transition_probability(state)
#       c) Factor in reverse state:------------>p(previous_state)*transition_probability(state)
#       d) Probability for given state:-------->factor_forward * factor_reverse * emission probability
#
#
#Viterbi
#The probabilities of the state 0 are calculated for every part of speech. The probability for each state is calculated
#by multiplying the emission probability with max value of product of the value of the last state and the transition
#probability of last to the current state and then backtracking to get the desired sequence.
#
#Results
#==> So far scored 1998 sentences with 29422 words.
#                   Words correct:     Sentences correct:
#   0. Ground truth:      100.00%              100.00%
#     1. Simplified:       93.95%               47.60%
#         2. HMM VE:       95.08%               54.35%
#        3. HMM MAP:       95.06%               54.45%
#DESIGN Desicions:
# We assigned a probability of 1e-10 if we dont encounter that state or word previously.We use logarithms to minimize the effect of probabilities getting small.
# We assigned Noun as the the state to the word which has not been encountered before and has a maximized value of zero.
#Conclusion:
#We conclude that Vitterbi Algorithm perfoms best followed by Variable elimination with Naive Bayes having the least accuracy amongst the three as it considers each word independently rather than
#the whole sentence.
#----
##################################################################################

import random
import math
from collections import Counter

# We've set up a suggested code structure, but feel free to change it. Just
# make sure your code still works with the label.py and pos_scorer.py code
# that we've supplied.
#
emmision_prob={}
prior_prob={}
total_unique_vocab=[]
new_emmision_freq={}
initial_prob={}
prob_trans={}
freq_trans={}
uniq_transit=[]
#calculating prior probabilities
def calculate_init_prior_prob(corpus):
    count_pos=[]
    count_initial_prob=[]
    for sentence in corpus:
        count_initial_prob.append(sentence[1][0])
        for pos in sentence[1]:
            count_pos.append(pos)
    freq_pos=Counter(count_pos)
    freq_init=Counter(count_initial_prob)
    for pos in freq_pos.keys():
        initial_prob[pos]=float(freq_init[pos])/sum(freq_init.values())
        prior_prob[pos]=float(freq_pos[pos])/sum(freq_pos.values())


#calculating emission probabilities
def calculate_emission_prob(corpus):
    emission_freq={}
    unique_freq=[]
    new_unique_freq={}
    for pos in prior_prob.keys():
        emission_freq[pos]=[]
        for sequence in corpus:
            for word,pos_tag in zip(sequence[0],sequence[1]):
                if pos==pos_tag:
                    emission_freq[pos].append(word)
                    unique_freq.append(word)
    new_unique_freq=len(Counter(unique_freq))

    total_unique_vocab.append(new_unique_freq)

    for pos in emission_freq.keys():
        new_emmision_freq[pos]={}
        new_emmision_freq[pos]=Counter(emission_freq[pos])

    for pos1 in new_emmision_freq.keys():
        emmision_prob[pos1]={}
        for word in new_emmision_freq[pos1].keys():
            emmision_prob[pos1][word]=float(new_emmision_freq[pos1][word])/(sum(new_emmision_freq[pos1].values()))


#Calculating transition probabilities
def calculate_transition_prob(corpus):
    corres=[]
    for pos in prior_prob.keys():
        corres_pos=[]
        for sequence in corpus:
            for pos_tag_index in range(len(sequence[1])-1):
                if sequence[1][pos_tag_index]==pos:
                    corres_pos.append(sequence[1][pos_tag_index+1])
                    corres.append(pos)
        freq_trans[pos]=Counter(corres_pos)
    uniq_trans=len(Counter(corres))
    uniq_transit.append(uniq_trans)
    for s_current in freq_trans.keys():
        prob_trans[s_current]={}
        for s_next in freq_trans[s_current].keys():
            prob_trans[s_current][s_next]=float(freq_trans[s_current][s_next])/(sum(freq_trans[s_current].values()))


class Solver:
    
    # Calculate the log of the posterior probability of a given sentence
    #  with a given part-of-speech labeling
    def posterior(self, sentence, label):
        return 0

    # Do the training!
    #data =((sentence),(pos))
    
    def train(self, data):
        calculate_init_prior_prob(data)
        calculate_emission_prob(data)
        calculate_transition_prob(data)

    # Functions for each algorithm.
    #
    def simplified(self, sentence):
        sequence=[]
        for word in sentence:
            prediction_prob={}
            for pos in prior_prob.keys():
                if word not in emmision_prob[pos].keys():
                    emmision_prob[pos][word]=1e-10
                prediction_prob[pos]=math.log(emmision_prob[pos][word])+math.log(prior_prob[pos])
            sequence.append(max(prediction_prob,key=prediction_prob.get))
        return sequence

    def hmm_ve(self, sentence):
        
        #Implementation of Forward Algorithm
        #initialization Step
        alpha={}
        alpha[0]={}
        for state in emmision_prob.keys():
            alpha[0][state]=initial_prob[state]*emmision_prob[state][sentence[0]]
    
        #iteration Step
        for t in range(1,len(sentence)):
            alpha[t]={}
            for state1 in emmision_prob.keys():
                alpha[t][state1]=sum([alpha[t-1][state2]*1e-10*emmision_prob[state1][sentence[t]] if (state1 not in prob_trans[state2].keys()) else alpha[t-1][state2]*prob_trans[state2][state1]*emmision_prob[state1][sentence[t]] for state2 in emmision_prob.keys()])
        #Implemetation of backward algorithm
        Beta={}
        #print len(sentence)
        Beta[len(sentence)]={}
        obs=list(sentence)
        obs.append(None)
        for state_6 in emmision_prob.keys():
            Beta[len(sentence)][state_6]=1.0
            emmision_prob[state_6][obs[-1]]=1.0
        for t2 in range(len(sentence)-1,-1,-1):
            Beta[t2]={}
          
            for state3 in emmision_prob.keys():
                Beta[t2][state3]=sum([Beta[t2+1][state4]*1e-10*emmision_prob[state4][obs[t2+1]] if (state4 not in prob_trans[state3].keys()) else Beta[t2+1][state4]*prob_trans[state3][state4]*emmision_prob[state4][obs[t2+1]]  for state4 in emmision_prob.keys()])
  
    #Merging forward backward algorithms to predict the sequence of pos tags
        prod_alpha_beta={}
        for t3 in range(len(sentence)):
            prod_alpha_beta[t3]={}
            for state5 in emmision_prob.keys():
                prod_alpha_beta[t3][state5]=(alpha[t3][state5]*Beta[t3][state5])
        sent=[]
        for t3 in range(len(sentence)):
            sent.append(max(prod_alpha_beta[t3],key=prod_alpha_beta[t3].get))
        return sent

    def hmm_viterbi(self, sentence):
        Vit={}
    #calculating Vit[t][state] where t=0
        Vit={0:{},'prev':{}}
        for state in prior_prob.keys():
            Vit[0][state]=initial_prob[state]*emmision_prob[state][sentence[0]]
            Vit['prev'][0]={}
            Vit['prev'][0][state]=None
        #calculating Vit[t][state] where t=1...n
        for t in range(1,len(sentence)):
            Vit[t]={}
            Vit['prev'][t]={}
            for state1 in prior_prob.keys():
                max_prob=max([(Vit[t-1][state_prev]*1e-10,state_prev) if (state1 not in prob_trans[state_prev].keys()) else (prob_trans[state_prev][state1]*Vit[t-1][state_prev],state_prev) for state_prev in prior_prob.keys()])
                if max_prob[0]==0:
                    max_prob[0]=0.01
                    max_prob[1]='noun'
                Vit['prev'][t][state1]=max_prob[1]
                Vit[t][state1]=emmision_prob[state1][sentence[t]]*max_prob[0]
        seq=[]
        charac=max(Vit[len(sentence)-1],key=Vit[len(sentence)-1].get)
        seq.append(charac)
        for t_i in range(len(sentence)-1,0,-1):
            seq.append(Vit['prev'][t_i][charac])
            charac=Vit['prev'][t_i][charac]
        seq.reverse()
        return seq


    # This solve() method is called by label.py, so you should keep the interface the
    #  same, but you can change the code itself. 
    # It should return a list of part-of-speech labelings of the sentence, one
    #  part of speech per word.
    #
    def solve(self, algo, sentence):
        if algo == "Simplified":
            return self.simplified(sentence)
        elif algo == "HMM VE":
            return self.hmm_ve(sentence)
        elif algo == "HMM MAP":
            return self.hmm_viterbi(sentence)
        else:
            print "Unknown algo!"

