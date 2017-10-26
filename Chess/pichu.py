#!/usr/bin/env python

#################################################################################################
# The code prints the next best possible move for the given configuration of the board          #
#                                                                                               #
# Initial State: The inital board configuration wherein the black and the white pieces are      #
# placed on two opposite sides  as given by the user.                                           #
#                                                                                               #
# Goal State: The next best possible move for the present configuration of the board using      #
#minimax alpha beta pruning algorithm                                                           #
#                                                                                               #
# Valid State: Any possible configuration of the board                                          #
#                                                                                               #
# Successor Function: The successor function gives us the successors by generating the next     #
# possible position of the pieces by moving them to the valid positions                         #
#                                                                                               #
# Evaluation cost:The cost we get from the evaluation function using material cost as the       #
# criteria.                                                                                     #
#                                                                                               #
# Evaluation Function: The evaluation function that we have considered here involves calculating#
# calculating the mechanincal cost involved with removing the pieces of the opponent with       #
# specific costs assigned to each piece.                                                        #
# The uppercase letters represent the white pieces and the lowercase letters represent the black#
# pieces                                                                                        #
#################################################################################################


# The following weights are arranged according to position of pieces on board
# Each piece has a different weight for different position.
# This information has been reproduced from websites
# https://jsfiddle.net/q76uzxwe/1/
# https://medium.freecodecamp.org/simple-chess-ai-step-by-step-1d55a9266977
# Second heuristic which gives score according to position
# Was supposed to be better, but took more time than one based on material space
import sys
from collections import Counter
from itertools import product, chain
import copy
import time

max_depth=[1]
best_move_at_level={}
successor_table={}
max_player = sys.argv[1]
state = sys.argv[2]
time_limit = sys.argv[3]
infinity=9999
end_time=time.time()+float(time_limit)
#The initial part of the code is concerned with finding successors and making valid moves.
def move(i,j,board):
    piece=board[i][j]
    if piece.lower() == "k":
        return move_kingfisher(i,j,board)
    elif piece.lower()=="p":
        return move_parakeet(i,j,board)
    elif piece.lower()=="r":
        return move_robin(i,j,board)
    elif piece.lower()=="b":
        return move_bluejay(i , j, board)
    elif piece.lower()=="q":
        return move_robin(i, j, board) + move_bluejay(i, j, board)
    elif piece.lower()=="n":
        return move_nighthawk(i, j, board)
    else:
        return []



def isopponent(a,b):
    if (a.islower()==b.islower):
        return False
    else: return True

#Shifts a piece from one position to another
#Replaces a . in empty position
def shift_piece(board, from_row, from_col, to_row, to_col):
    piece=board[from_row][from_col]
    board_prime = copy.deepcopy(board)
    board_prime[to_row][to_col] = piece
    board_prime[from_row][from_col] = "."
    return board_prime

def move_bluejay(i, j, board):
    successor = []
    piece = board[i][j]
    flag_top_right=0
    flag_top_left=0
    flag_bottom_left=0
    flag_bottom_right=0

    if piece.islower():
        for a in range(1, 8):
            if ((i + a) in range(0, 8)) and((j + a) in range(0, 8)) \
                    and (flag_bottom_right!=1):
                if (board[i + a][j + a].isupper()):
                    flag_bottom_right=1
                    successor.append(shift_piece(board,i,j,i + a,j + a))
                elif (board[i + a][j + a].islower()):
                    flag_bottom_right=1
                else: successor.append(shift_piece(board,i,j,i + a,j + a))

            if ((i + a)in range(0,8) and (j - a) in range(0, 8)\
                and flag_bottom_left!=1):
                if(board[i + a][j- a]).isupper():
                    flag_bottom_left=1
                    successor.append(shift_piece(board,i,j,i+a,j-a))
                elif (board[i + a][j - a].islower()):
                    flag_bottom_left=1
                else: successor.append((shift_piece(board,i,j,i+a,j-a)))

            if ((i - a) in range(0 ,8) and ((j-a) in range(0, 8))\
                and (flag_top_left!=1)):
                if (board[i - a][j - a].isupper()):
                    flag_top_left=1
                    successor.append(shift_piece(board,i,j,i-a,j-a))
                elif (board[i - a][j - a].islower()):
                    flag_top_left=1
                else: successor.append(shift_piece(board,i,j,i-a,j-a))

            if ((i-a) in range(0, 8) and (j + a) in range(0,8)\
                and (flag_top_right!=1)):
                if (board[i - a][j + a].isupper()):
                    flag_top_right=1
                    successor.append(shift_piece(board,i,j,i-a,j+a))
                elif (board[i - a][j + a].islower()):
                    flag_top_right=1
                else: successor.append(shift_piece(board, i , j,i - a,j + a))

    if piece.isupper():
        for a in range(1, 8):
            if ((i + a) in range(0, 8)) and ((j + a) in range(0, 8)) \
                    and (flag_bottom_right != 1):
                if (board[i + a][j + a].islower()):
                    flag_bottom_right = 1
                    successor.append(shift_piece(board, i, j, i + a, j + a))
                elif (board[i + a][j + a].isupper()):
                    flag_bottom_right = 1
                else:
                    successor.append(shift_piece(board, i, j, i + a, j + a))
            if ((i + a) in range(0, 8) and (j - a) in range(0, 8) \
                        and flag_bottom_left != 1):
                if (board[i + a][j - a]).islower():
                    flag_bottom_left = 1
                    successor.append(shift_piece(board, i, j, i + a, j - a))
                elif (board[i + a][j - a].isupper()):
                    flag_bottom_left = 1
                else:
                    successor.append((shift_piece(board, i, j, i + a, j - a)))
            if ((i - a) in range(0, 8) and ((j - a) in range(0, 8)) \
                        and (flag_top_left != 1)):
                if (board[i - a][j - a].islower()):
                    flag_top_left = 1
                    successor.append(shift_piece(board, i, j, i - a, j - a))
                elif (board[i - a][j - a].isupper()):
                    flag_top_left = 1
                else:
                    successor.append(shift_piece(board, i, j, i - a, j - a))
            if ((i - a) in range(0, 8) and (j + a) in range(0, 8) \
                        and (flag_top_right != 1)):
                if (board[i - a][j + a].islower()):
                    flag_top_right = 1
                    successor.append(shift_piece(board, i, j, i - a, j + a))
                elif (board[i - a][j + a].isupper()):
                    flag_top_right = 1
                else:
                    successor.append(shift_piece(board, i, j, i - a, j + a))

    return successor

def move_nighthawk(i, j, board):
    successor = []
    piece = board[i][j]
    mv = list(product([i-1,i+1],[j-2,j+2])) + list(product([i-2,i+2],[j-1,j+1]))
    moves = [(a,b) for a,b in mv if a>=0 and b>=0 and a<8 and b<8]
    for x in moves:
        p = x[0]
        q = x[1]
        if (piece.islower()!=board[p][q].islower()) or (not board[p][q].isalpha()):
                board_prime = shift_piece(board,i,j,p,q)
                if board_prime != board:
                        successor.append(board_prime)
    return successor

def move_robin(i, j, board):
    successor=[]
    piece =board[i][j]
    flag_up=0
    flag_down=0
    flag_left=0
    flag_right=0
    if piece.islower():
        for a in range(1, 8):
            if ((i + a)in range(0, 8) and flag_down!=1):
                if board[i + a][j].isupper():
                    successor.append(shift_piece(board, i, j, i + a, j))
                    flag_down=1
                elif (board [i + a][j].islower()):
                    flag_down=1
                else:
                    successor.append(shift_piece(board, i, j, i + a, j))

            if ( (i - a) in range (0, 8) and flag_up!=1):
                if board[i - a][j].isupper():
                    successor.append(shift_piece(board, i, j, i - a, j))
                    flag_up=1
                elif (board [i - a][j].islower()):
                    flag_up=1
                else:
                    successor.append(shift_piece(board, i, j, i - a, j))

            if (( j + a) in range (0, 8) and flag_right!=1):
                if board[i][j + a].isupper():
                    flag_right=1
                    successor.append(shift_piece(board, i, j, i, j + a))
                elif board[i][j + a].islower():
                    flag_right=1
                else :
                    successor.append(shift_piece(board, i, j, i, j + a))

            if ((j - a) in range(0, 8) and flag_left!=1):
                if board[i][j - a].isupper():
                    flag_left=1
                    successor.append(shift_piece(board, i, j, i ,j - a))
                elif board[i][j - a].islower():
                    flag_left=1
                else:
                    successor.append(shift_piece(board, i, j, i ,j - a))

    if piece.isupper():
        for a in range(1, 8):
            if ((i + a) in range(0, 8) and flag_down != 1):
                if board[i + a][j].islower():
                    successor.append(shift_piece(board, i, j, i + a, j))
                    flag_down = 1
                elif (board[i + a][j].isupper()):
                    flag_down = 1
                else:
                    successor.append(shift_piece(board, i, j, i + a, j))
            if ((i - a) in range(0, 8) and flag_up != 1):
                if board[i - a][j].islower():
                    successor.append(shift_piece(board, i, j, i - a, j))
                    flag_up = 1
                elif (board[i - a][j].isupper()):
                    flag_up = 1
                else:
                    successor.append(shift_piece(board, i, j, i - a, j))

            if ((j + a) in range(0, 8) and flag_right != 1):
                if board[i][j + a].islower():
                    flag_right = 1
                    successor.append(shift_piece(board, i, j, i, j + a))
                elif board[i][j + a].isupper():
                    flag_right = 1
                else:
                    successor.append(shift_piece(board, i, j, i, j + a))

            if ((j - a) in range(0, 8) and flag_left != 1):
                if board[i][j - a].islower():
                    flag_left = 1
                    successor.append(shift_piece(board, i, j, i, j - a))
                elif board[i][j - a].isupper():
                    flag_left = 1
                else:
                    successor.append(shift_piece(board, i, j, i, j - a))
    return successor

#Moves Kingfisher in all possible directions from
# given 8 directions
def move_kingfisher(i,j,board):
    successor=[]
    piece=board[i][j]
    moves = list(product([i-1,i+1],[j-1,j+1])) + \
            list(product([i-1,i+1],[j])) + list(product([i],[j-1,j+1]))
    moves = [(a,b) for a,b in moves if a>=0 and b>=0 and a<8 and b<8]
    for x in moves:
        p = x[0]
        q = x[1]
        if (piece.islower()!=board[p][q].islower()) or (not board[p][q].isalpha()):
                board_prime = shift_piece(board,i,j,p,q)
                if board_prime != board:
                        successor.append(board_prime)
    return successor

def move_parakeet(i,j,board):
    successor=[]
    piece =board[i][j]

    if piece.islower():
        if(i - 1 in range(0,8)):
            if not (board[i - 1][j].isalpha()):
                successor.append(shift_piece(board, i, j, i - 1, j))
                if (i - 1 == 0):
                    board[i - 1][j]='q'

                if (i - 2 in range(0, 8)):
                    if(i == 6 and not (board[i - 2][j].isalpha())):
                        successor.append(shift_piece(board, i, j, i - 2, j))
            if (j - 1 in range(0, 8)) and (board[i - 1][j - 1].isupper()):
                successor.append(shift_piece(board, i , j, i - 1,j - 1))
                if (i - 1 == 0):
                    board[i - 1][j - 1]=='q'

            if (j + 1 in range(0, 8)) and (board[i - 1][j + 1].isupper()):
                successor.append(shift_piece(board, i, j, i - 1, j + 1))
                if (i - 1 == 0):
                    board[i - 1][j + 1]=='q'

    elif piece.isupper():
        if(i + 1 in range(0,8)):
            if not(board[i+1][j].isalpha()):
                successor.append(shift_piece(board, i, j, i + 1, j))
                if (i + 1 == 7):
                    board[i + 1][j]=='Q'

                if (i + 2 in range(0, 8)):
                    if (i==1 and not (board[i+2][j].isalpha())):
                        successor.append(shift_piece(board, i, j, i + 2, j))
                if (j - 1 in range(0, 8) and board [i + 1][j - 1].islower()):
                    successor.append(shift_piece(board, i, j, i + 1, j - 1))
                    if (i + 1 == 7):
                        board[i + 1][j] == 'Q'
                if (j + 1 in range(0, 8) and board[i + 1][j + 1].islower()):
                    successor.append(shift_piece(board, i, j, i + 1, j + 1))
                    if (i + 1 == 7):
                        board[i + 1][j] == 'Q'
    return successor


#Successor returns the valid successors for given player
def successors(state,player):
    successor=[]
    if (to_tuple(state),player) in successor_table:
        return successor_table[(to_tuple(state),player)]
    for row in range(len(state)):
        for col in range (len(state[row])):
            if player=="b" and state[row][col].islower():
                s=move(row, col, state)
                if s != state:
                    successor +=s
            elif player=="w" and state[row][col].isupper():
                s=move(row, col, state)
                if s != state:
                    successor +=s
    successor_table[(to_tuple(state),player)]=successor
    return successor


#Alpha-beta-decision
#Reference : http://aima.cs.berkeley.edu/python/games.html
def alpha_beta_decision(state, player):
    maximum_value={}
    for s in successors(state,player):
        v=min_val(s, opponent(player), -infinity, infinity, 0)
        maximum_value[to_tuple(s)]=v


    a =max(maximum_value,key=maximum_value.get)
    print max_depth[0]
    return a

#Max-Value
def max_val(board, player, alpha, beta, depth):
    if is_goal(board,depth)or depth>max_depth[0]:
        return eval(board,player)
    alpha_dash=-infinity
    for s in successors(board,player):
        score_of_s=min_val(s, opponent(player), alpha, beta, depth + 1)
        alpha_dash=max(score_of_s, alpha_dash)
        if alpha_dash>=beta:
            return alpha_dash
        #alpha=max(alpha,v)
    return alpha

#Min-Value
def min_val(board, player, alpha, beta, depth):

    if is_goal(board,depth) or depth>max_depth[0]:
        return eval(board,player)
    beta_dash=infinity
    for s in successors(board,player):
        score_of_s=max_val(s, opponent(player), alpha, beta, depth + 1)
        #best_move_at_level[(to_tuple(s),player)]=score_of_s
        beta_dash = min(score_of_s, beta_dash)
        if alpha>=beta_dash:
            return beta_dash
        #beta = min(beta, v)
    return beta



#Evaluation Heuristic- weighted sum of features
#Weighted sum of pieces lost
#Reference of weights were obtained from https://chessprogramming.wikispaces.com/Evaluation
def eval(board,player):
    li = list(chain.from_iterable(board))
    count = Counter(li)
    score=0

    score= 200* (count['K']-count['k'])+\
                9 * (count['Q']-count['q'])+\
                5 * (count['R']-count['r'])+\
                3 * (count['B']-count['b'])+\
                3 * (count['N']-count['n'])+ \
                1 * (count['P'] - count['p'])
    return score

#Checks the goal state
def is_goal(board,depth):
    flag=0
    for r in board:
        if ("k" in r ):
            flag+=1
        if("K" in r):
            flag+=1
    if flag>1:
        return False
    else:
        max_depth[0]=depth-1
        return True

def to_tuple(s):
        tuple_of_tuple = tuple(tuple(x) for x in s)
        return tuple_of_tuple

def to_array(s):
    list_of_list= list( list(x) for x in s)
    return list_of_list

#Returns opponent player
def opponent(player):
    if player=="b":
        return "w"
    elif player=="w":
        return "b"

#Main
#Accepts board by commandline

def main():
    global state
    global max_depth
    print "initial board: ", state
    state=list(state)
    if len(state)!=64:
        print "The board does not have 64 tiles.."
        print "#tiles=",len(state)
    else:
        state=[state[i: i + 8] for i in range (0,64,8)]
        print state
        print "Welcome to Pichu move recommender"
        print "Fetching the next best move for your board..."
        print "Best move for now is :"

        for depth in range(2,10):
            if(time.time()>=end_time):
                break
            max_depth[0]=depth
            recommended_board = alpha_beta_decision(state, max_player)

            if is_goal(recommended_board,0):
                print "We won!!!"
                next_board = ''.join((item) for innerlist in recommended_board for item in innerlist)
                print next_board
                break
            next_board = ''.join((item) for innerlist in recommended_board for item in innerlist)
            print next_board

main()
