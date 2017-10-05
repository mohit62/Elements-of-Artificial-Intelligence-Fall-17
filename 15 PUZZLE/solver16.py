	# put your 15 puzzle solver here!
#(A)---> State Space in this problem is any arrangement of the tiles in the puzzle
#(B)--->Goal State will be the tiles arranged with zero no of inversions and last tile is 0 .
#(C)--->The successor function will generate moves with either moving 1,2 or 3 times the blank tile to the right,left ,Up or Down.
#(D)--->The heuristic function used is the value of Manhattan Distance divided by 3 as it is admissible.We have also added a check for linear conflict in our heuristic function and for each linear conflict we add a heuristic value of 2 for each horizontal as well as vertical conflict.HeuristicValue=ManhattanDistance/3 + 2*NoOfLinearConflicts
#(E)--->We tested for other heuristic functions like normal manhattan distance,No of misplaced tiles and the number of inversions but they were not admissible in our case as they overestimated the no of steps to the goal state.E.G. State =[[1,2,3,4],[5,6,7,8],[9,10,11,12],[0,13,14,15]] is not admissible for neither manhattan distance nor misplaced tiles as the heuristic estimates the no of steps to be 3 whereas we can reach Goal state in one move by simply moving the 3 tiles left
#(F)--->We have used A* algorith#3 for our implementation to get the optimal path as it avoids revisting states and thus significantly reduces the state space giving us an optimal solution.
#(G)--->We consider uniform cost 1 here for each move.
#(J)--->We have also checked for the solvability of the puzzle in our code by considering the number of inversions and the distance of blank tile from the bottom.
#(K)--->We have also accounted for visited or closed states in our algorithmic implementation.
'''References:>
[1]Norvig and Russell, Artificial Intelligence: A Modern Approach,3rd ed., Prentice Hall, 2009.
[2]https://algorithmsinsight.wordpress.com/graph-theory-2/a-star-in-general/implementing-a-star-to-solve-n-puzzle/
[3]https://heuristicswiki.wikispaces.com/Linear+Conflict'''

import sys,copy,heapq,math
GoalState=[[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,0]]
parent={}
#print the puzzle board
def printPuzzle(PuzzleBoard):
    return "\n".join([ " ".join([ str(col) for col in row ]) for row in PuzzleBoard])

#Check no of permutation inversions
def noOfInversions(Board):
    Listoftiles=[Board[i][j] for i in range(4) for j in range(4) if Board[i][j] !=0]
    return sum([1 if Listoftiles[i]>Listoftiles[j+1] else 0 for i in range(len(Listoftiles)) for j in range(i,len(Listoftiles)-1) ])

#Check Distance of empty position from bottom
def distanceFromBottom(Board):
    return sum([4-row for row in range(len(Board)) for col in range(len(Board)) if Board[row][col]== 0])

#Check if solvability of puzzle
def isSolvable(Board):
    dist_fromBottom=distanceFromBottom(Board)
    no_Inversion=noOfInversions(Board)
    return sum([1 if (dist_fromBottom % 2 == 0 and no_Inversion % 2 !=0) or (dist_fromBottom % 2 != 0 and no_Inversion % 2 ==0)  else 0])

#Calculate Manhattan Distance for each tile
def get_Distance(Tile,current_row,current_col):
    actual_row=(Tile-1) / 4
    actual_col=(Tile-1) % 4
    return (abs(actual_row-current_row)+abs(actual_col-current_col))

#Heuristic function - Manhattan Distance/3
def heuristic_function(Board):
    return sum([get_Distance(Board[row][col],row,col) for row in range(4) for col in range(4) if Board[row][col]!=0])/3.0+2*linear_Conflict(Board)

#Check for Linear Conflict i.e. if both tiles are in the same row or column in current board and if their goal states are in the same row or column.We dont consider blank tile for linear conflict.We also check if goal position of one tile is to the left of other tile
def linear_Conflict(board):
    LC_heuristic_value = 0
    for i in range(4):
        for j in range(4):
            #Check for Linear Conflict horizontally
            for k in range(j+1, 4):
                if get_Position(GoalState,board[i][k])[0]==i and get_Position(GoalState,board[i][j])[0]==i and get_Position(GoalState,board[i][k])[1] < get_Position(GoalState,board[i][j])[1] and get_Position(board,board[i][k])[0]==i and get_Position(board,board[i][j])[0]==i and board[i][k]!=0 and board[i][j]!=0:
                    LC_heuristic_value += 1
            #Check for Linear Conflict vertically
            for k in range(i + 1, 4):
                if get_Position(GoalState,board[k][j])[1] == j and get_Position(GoalState,board[i][j])[1] == j and get_Position(GoalState,board[k][j])[0] < get_Position(GoalState,board[i][j])[0] and get_Position(board,board[k][j])[1]==j and get_Position(board,board[i][j])[1]==j and board[k][j]!=0 and board[i][j]!=0:
                    LC_heuristic_value += 1
    return LC_heuristic_value

#Swap Tiles
def swap(Board,row1,row2,col1,col2):
    temp=Board[row1][col1]
    Board[row1][col1]=Board[row2][col2]
    Board[row2][col2]=temp
    return Board

#determine blank tile position
def get_Position(Board,tile):
    for i in range(4):
        for j in range(4):
            if Board[i][j] == tile:
                return i,j

#Move blank piece Up,Down,Left,Right with 1,2 or 3 tiles at a time
def Moves(Board,move,noOfTilemoved):
    for i in range(noOfTilemoved):
        row=get_Position(Board,0)[0]
        col=get_Position(Board,0)[1]
        if move == "U" and row>0:
            transform_board=swap(Board,row,row-1,col,col)
        
        elif move == "D" and row<3:
            transform_board=swap(Board,row,row+1,col,col)

        elif move == "R" and col<3:
             transform_board=swap(Board,row,row,col,col+1)

        elif move == "L" and col>0:
            transform_board=swap(Board,row,row,col,col-1)

    return transform_board

#Successor Function
def Successor(Board):
    Move=["R","L","U","D"]
    row=get_Position(Board,0)[0]
    col=get_Position(Board,0)[1]
    Mover={"R":4-col,"L":col+1,"U":row+1,"D":4-row}
    return [Moves(copy.deepcopy(Board),move,noOfTilemoved) for move in Move for noOfTilemoved in range(1,Mover[move]) if move == "U" and row>0 or move == "D" and row<3 or move == "R" and col<3 or move == "L" and col>0 ]
#Get path of the node from initial node
def Path(parent,Children):
    path=[]
    path.append(Children)
    while parent[Children]!=0:
        path.append(parent[Children])
        Children=parent[Children]
    return path[::-1]
#Calculate cost for a particular path
def calculate_cost(path):
    return len(path) - 1
#Convert a list of list to tuple of tuple
def array_to_tuple(s):
    return tuple([tuple(i) for i in s])
#Convert tuple of tuple to List of list
def tuple_to_array(tup):
    return [[j for j in i] for i in tup]
#Generate the Sequence of Moves to Goal
def Seq_Moves(GoalPath):
    seq=""
    for i in range(len(GoalPath)-1):
        prev=tuple_to_array(GoalPath[i])
        next=tuple_to_array(GoalPath[i+1])
        next_row=get_Position(next,0)[0]
        next_col=get_Position(next,0)[1]
        prev_row=get_Position(prev,0)[0]
        prev_col=get_Position(prev,0)[1]
        row_move=next_row-prev_row
        col_move=next_col-prev_col
        if not row_move and col_move >0:
            seq+="L"+str(col_move)+str(prev_col+1)+" "
        if not row_move and col_move <0:
            seq+="R"+str(abs(col_move))+str(prev_col+1)+" "
        if not col_move and row_move >0:
            seq+="U"+str(row_move)+str(prev_row+1)+" "
        if not col_move and row_move <0:
             seq+="D"+str(abs(row_move))+str(prev_row+1)+" "
        
    return seq


#Solve puzzle using A* Algorithm 3 avoiding revisting the states
def solvePuzzle(InitialBoard):
    fringe = []
    heapq.heappush(fringe,(heuristic_function(InitialBoard), InitialBoard))
    Closed=[]
    t=array_to_tuple(InitialBoard)
    parent[t]=0
    while fringe:
        s=heapq.heappop(fringe)[1]
        Closed.append(s)
        Parent=array_to_tuple(s)
        if isGoal(s):
            path=Path(parent,Parent)
            return Seq_Moves(path)
        for s_dash in Successor(s):
            if s_dash in Closed:
                continue
            s_t_dash=array_to_tuple(s_dash)
            if s_dash in fringe:
                print s_dash
                NewCost=calculate_cost(Path(parent,Parent))+1+heuristic_function(s_dash)
                old_Cost=calculate_cost(Path(parent,s_t_dash))+heuristic_function(s_dash)
                if NewCost < old_Cost:
                    fringe.remove((old_Cost,s_dash))
                    heapq.heappush(fringe,( NewCost, s_dash))
                    parent[s_t_dash]=Parent
            else:
                parent[s_t_dash]=Parent
                heapq.heappush(fringe, (calculate_cost(Path(parent,s_t_dash))+heuristic_function(s_dash), s_dash))

#check for goal state
def isGoal(Board):
    if noOfInversions(Board) == 0 and Board[3][3] == 0:
        return 1
    else:
        return 0
#Get filename as Input
filename=sys.argv[1]
InitialBoard=[]
[InitialBoard.append(map(int,line.split())) for line in open(filename,'r')]
#Print Initial puzzle configuration
print ("Starting from initial board:\n" + printPuzzle(InitialBoard)+"\n\nLooking for solution...\n")
if isSolvable(InitialBoard):
    print solvePuzzle(InitialBoard)
else:
    print("Puzzle cannot be solved")
