#The code can be run using the following command e.g. python GameOfChance.py 6 6 5
#where 6,6 and 5 is the dice set from the user
#For the given game of chance we calculate all combinations of rerolls and then calculate maximum expectation
#from the list of expectations for each combination of reroll.
import sys,itertools
#calculate expectation for each reroll
def expected_Outcome(olddie,rerolls):
    return sum([(lambda roll:25 if roll[0]==roll[1] and roll[1]==roll[2] else sum(roll))([dieA,dieB,dieC]) for dieA in ([olddie[0]] if not rerolls[0] else range(1,7)) for dieB in ([olddie[1]] if not rerolls[1] else range(1,7)) for dieC in ([olddie[2]] if not rerolls[2] else range(1,7))])*1.0/6**sum(rerolls)
#get all combinations for rerolls and extract the one with maximum expectation
def rollAgain(die):
    return max([(expected_Outcome(die,rerolls),rerolls) for rerolls in itertools.product([True,False], repeat =3)])
#print Next Expected Die Roll
def printMove(initial_die):
    NextMove=rollAgain(initial_die)
    print "Roll Again die No" ,",".join([str(i+1) for i in range(len(NextMove[1])) if NextMove[1][i]]),"with expectation",NextMove[0]
#get dice set from the user
initial_die=[int(sys.argv[i+1]) for i in range(3)]
printMove(initial_die)
