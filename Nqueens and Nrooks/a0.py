#!/usr/bin/env python
# a0.py : Solve the N-Rooks and N-Queens problem with an unavailable position on board!
# D. Crandall, 2016
# Updated by Zehua Zhang, 2017
# Updated by Mohit Saraf, 2017
# The N-rooks problem is: Given an empty NxN chessboard, place N rooks on the board so that no rooks
# can take any other, i.e. such that no two rooks share the same row or column.
#The N-queens problem is: Given an empty NxN chessboard, place N queens on the board so that no queens
# can take any other, i.e. such that no two rooks share the same row or column or diagonal.


import sys

# Count # of pieces in given row
def count_on_row(board, row):
    return sum( board[row] )
#Check # of pieces on left part of upper Diagonal
def check_Upperleftdiagonals(board, row, col):
    return sum([board[i][j] for i,j in zip(range(row,-1,-1),range(col,-1,-1))])
#Check # of pieces left part of lower Diagonal
def check_Lowerleftdiagonals(board, row, col):
    return sum([board[i][j] for i,j in zip(range(row,N,1),range(col,-1,-1))])
#Check # of pieces right part of upper Diagonal
def check_upperrightdiagonals(board, row, col):
    return sum([board[i][j] for i,j in zip(range(row,-1,-1),range(col,N,1))])
#Check # of pieces right part of lower Diagonal
def check_Lowerrightdiagonals(board, row, col):
    return sum([board[i][j] for i,j in zip(range(row,N,1),range(col,N,1))])
#Check # of pieces on the Diagonals
def count_diag(board, row, col):
    return check_Lowerrightdiagonals(board, row, col)+check_upperrightdiagonals(board, row, col)+check_Lowerleftdiagonals(board, row, col)+check_Upperleftdiagonals(board, row, col)
# Count # of pieces in given column
def count_on_col(board, col):
    return sum( [ row[col] for row in board ] )

# Count total # of pieces on board
def count_pieces(board):
    return sum([ sum(row) for row in board ] )

# Return a string with the board rendered in a human-friendly format
def printable_boardNRooks(board):
    return "\n".join([ " ".join([ "X" if col==.1 else "R" if col==1 else "_" for col in row ]) for row in board])
def printable_boardNQueens(board):
    return "\n".join([ " ".join([ "X" if col==.1 else "Q" if col==1 else "_" for col in row ]) for row in board])
# Add a piece to the board at the given position, and return a new board (doesn't change original)
def add_piece(board, row, col):
    return board[0:row] + [board[row][0:col] + [1,] + board[row][col+1:]] + board[row+1:]

# Efficient Successor Function to avoid generating successor states for states where NoOfPieces is already N and if a position already has a piece and for rows and columns already having a piece and for avoiding unavailable position
def successorsNrooks(board):
    #print "Successor starts \n",
    if count_pieces(board)<N:
        return [ add_piece(board, r, c) for r in range(0, N) for c in range(0,N) if board[r][c]!=1 and \
                count_on_row(board, r) < 1 and \
                count_on_col(board, c) < 1 and \
                board[r][c]!=.1 ]
    else:
        return []
# Efficient Successor Function to avoid generating successor states for states where NoOfPieces is already N and if a position already has a piece and for rows and columns already having a piece and for avoiding unavailable position.Also it checks for any queen piece  on diagonal attacking the newly placed queen piece
def successorsNqueen(board):
    #print "Successor starts \n",
    if count_pieces(board)<N+.1:
        return [ add_piece(board, r, c) for r in range(0, N) for c in range(0,N) if board[r][c]!=1 and \
                count_on_row(board, r) < 1 and \
                count_on_col(board, c) < 1 and \
                board[r][c]!=.1 and \
                count_diag(board, r, c)<1 ]
    else:
        return []

# check if board is a goal state for NQueens
def is_goalNqueen(board):
    return count_pieces(board) == N+.1 and \
        all( [ count_on_row(board, r) <= 1.1 for r in range(0, N) ] ) and \
        all( [ count_on_col(board, c) <= 1.1 for c in range(0, N) ] )
# check if board is a goal state for NQueens for no unavailable position
def is_goalNqueen2(board):
    return count_pieces(board) == N and \
        all( [ count_on_row(board, r) <= 1.1 for r in range(0, N) ] ) and \
        all( [ count_on_col(board, c) <= 1.1 for c in range(0, N) ] )
# check if board is a goal state for NRooks
def is_goal(board):
    return count_pieces(board) == N+.1 and \
        all( [ count_on_row(board, r) <= 1.1 for r in range(0, N) ] ) and \
        all( [ count_on_col(board, c) <= 1.1 for c in range(0, N) ] )
# check if board is a goal state for NRooks for no unavailable space
def is_goal2(board):
    return count_pieces(board) == N and \
        all( [ count_on_row(board, r) <= 1.1 for r in range(0, N) ] ) and \
        all( [ count_on_col(board, c) <= 1.1 for c in range(0, N) ] )
# Solve n-rooks!
def solveNrooks(initial_board):
    fringe = [initial_board]
    while len(fringe) > 0:
        for s in successorsNrooks( fringe.pop() ):
            if is_goal(s):
                return(s)
            fringe.append(s)
    return False
# Solve n-Queens!
def solveNQueens(initial_board):
    fringe = [initial_board]
    while len(fringe) > 0:
        for s in successorsNqueen( fringe.pop() ):
            if is_goal(s):
                return(s)
            fringe.append(s)
    return False

# Solve n-rooks for no unavailable space!
def solveNrooks2(initial_board):
    fringe = [initial_board]
    while len(fringe) > 0:
        for s in successorsNrooks( fringe.pop() ):
            if is_goal2(s):
                return(s)
            fringe.append(s)
    return False
# Solve n-Queens  for no unavailable space!
def solveNQueens2(initial_board):
    fringe = [initial_board]
    while len(fringe) > 0:
        for s in successorsNqueen( fringe.pop() ):
            if is_goalNqueen2(s):
                return(s)
            fringe.append(s)
    return False


#This is Options ,nrook or nqueen.It is passes through command line arguments.
Options = sys.argv[1]
# This is N, the size of the board. It is passed through command line arguments.
N = int(sys.argv[2])
#These are (xPos,yPos) ,the unavailable positions on the board.These are passed through command line arguments.
xPos = int(sys.argv[3])
yPos= int(sys.argv[4])


# The board is stored as a list-of-lists. Each inner list is a row of the board.
# A zero in a given square indicates no piece, and a 1 indicates a piece.
initial_board = [ [0]*N for r in range(N) ]
if xPos!=0 and yPos!=0:
    initial_board[xPos-1][yPos-1] = .1

if Options == "nrook":
    if xPos==0 and yPos==0:
        print ("Starting from initial board:\n" + printable_boardNRooks(initial_board) + "\n\nLooking for solution...\n")
        solution = solveNrooks2(initial_board)
        print (printable_boardNRooks(solution) if solution else "Sorry, no solution found. :(")
    else:
        print ("Starting from initial board:\n" + printable_boardNRooks(initial_board) + "\n\nLooking for solution...\n")
        solution = solveNrooks(initial_board)
        print (printable_boardNRooks(solution) if solution else "Sorry, no solution found. :(")
elif Options == "nqueen":
    if xPos==0 and yPos==0:
        print ("Starting from initial board:\n" + printable_boardNQueens(initial_board) + "\n\nLooking for solution...\n")
        solution = solveNQueens2(initial_board)
        print (printable_boardNQueens(solution) if solution else "Sorry, no solution found. :(")
    else:
        print ("Starting from initial board:\n" + printable_boardNQueens(initial_board) + "\n\nLooking for solution...\n")
        solution = solveNQueens(initial_board)
        print (printable_boardNQueens(solution) if solution else "Sorry, no solution found. :(")
else:
    print "Invalid option"

