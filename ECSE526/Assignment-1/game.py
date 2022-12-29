from logging.handlers import BaseRotatingHandler
import numpy as np
import copy
import time
# import socket

# SERVER = "156trlinux-1.ece.mcgill.ca"
# SERVER_IP = "132.206.44.21"
# PORT = 12345
# BUFFER_SIZE = 1024

# MESSAGE = 'game22 white\n'

# server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# server.connect((SERVER, PORT))
# server.send(bytes(MESSAGE, 'utf-8'))
# print("Message sent")
# data = server.recv(BUFFER_SIZE)
# print(data.decode("utf-8"))
def create_board(size):
        if size == '1':
            bc = np.ones((4,5),dtype=int)*(-1)
            bc[0][0], bc[2][0], bc[1][4], bc[3][4] = 0,0,0,0
            bc[1][0], bc[3][0], bc[0][4], bc[2][4] = 1,1,1,1
            return bc

        elif size == '2':
            bc = np.ones((6,7),dtype=int)*(-1)
            bc[1][1], bc[3][1], bc[2][5], bc[4][5] = 0,0,0,0
            bc[2][1], bc[4][1], bc[1][5], bc[3][5] = 1,1,1,1
            return bc

def show_board(sb):
    s = ''
    
    for i in range(sb.shape[0]):
        for j in range(sb.shape[1]):
            # if sb[i,j] == -1:
            #     sb[i,j] = 
            # if sb[i,j] == -1:
            #     s += " "
            s += str(sb[i][j])
            l = sb.shape[1]
            # if sb[i,j] == '-1':
            #     s += " "
            # for x in range(len(s)):
            #     for y in range(s.shape[1]):
            #         if s[x,y] == "-1":
            #             s[x,y] == " "
            if j != (l-1):
                # if s == "-1":
                #     s += " "
                s += ','         

        print(s)
        s = ''
def ask_move(choice,m): #ask human for a move
    while True:
        ask = input("What's your move?")
        j,i,dir = (int(ask[0])-1,int(ask[1])-1,ask[2])
        # print(row,col)
        while True: #check for right piece
            if choice.startswith('w'):
                if m[i][j] != 0:
                    print("That's a black piece, choose white piece")
                    ask = input("What's your move?")
                    j,i,dir = (int(ask[0])-1,int(ask[1])-1,ask[2])
                    continue
                if m[i][j] == 0:
                    #print("true")
                    return i,j,dir 
            if choice.startswith('b'):
                if m[i][j] != 1:    
                    print("That's a white piece, choose black piece")
                    ask = input("What's your move?")
                    j,i,dir = (int(ask[0])-1,ask(m[1])-1,ask[2])
                    continue
                if m[i][j] == 1:
                    return i,j,dir 

def move(i,j,dir,m,change):
    #print(i,j,dir)
    a = m.copy()
    if dir == "E":
        if a[i,j] == 0:
            a[i,j+1] = 0
            a[i,j] = -1      
        else:
            a[i,j+1] = 1
            a[i,j] = -1
    elif dir == "N":
        if a[i,j] == 0:
            a[i-1,j] = 0
            a[i,j] = -1        
        else:
            a[i-1,j] = 1
            a[i,j] = -1
    elif dir == 'S':
        if a[i,j] == 0:
            a[i+1,j] = 0
            a[i,j] = -1   
        else:
            a[i+1,j] = 1
            a[i,j] = -1
    elif dir == 'W':
        if a[i,j] == 0:
            a[i,j-1] = 0
            a[i,j] = -1      
        else:
            a[i,j-1] = 1
            a[i,j] = -1
    
    if change:
        m = a.copy()
        return m
    if not change:
        b = a.copy()
        a = m.copy()
        return b      
    
def check_win(cwb):
    #vertical win check
    #for white win check
   
    for i in range(cwb.shape[0]):
        for j in range(cwb.shape[1]):
                      
                
                v = i+2
                w = j+2
                if ((v <= cwb.shape[0]-1) and (w <= cwb.shape[1]-1)):
                   
                    if (cwb[i,j]== cwb[i,j+1]==cwb[i,j+2]==0):
                        
                        return 0
                    if (cwb[i,j]== cwb[i,j+1]==cwb[i,j+2]==1):
                        
                        return 1
                
                #vertical win
                    if (cwb[i,j]== cwb[i+1,j]==cwb[i+2,j]==0):
                        
                        return 0
                    if (cwb[i,j]== cwb[i+1,j]==cwb[i+2,j]==1):
                        
                        return 1
                
                #Diagonal win
               
                if ((i <= cwb.shape[0]-3) and (j <= cwb.shape[1]-3)):
                    #print(i,j)
                    
                    if (cwb[i,j] == cwb[i+1,j+1] == cwb[i+2,i+2] == 0):
                        return 0
                    if (cwb[i,j] == cwb[i+1,j+1] == cwb[i+2,i+2] == 1):
                        return 1 
                if ((i <= cwb.shape[0]-3) and ((j >= cwb.shape[1]-3) and (j < cwb.shape[1]))):
                    if (cwb[i,j] == cwb[i+1,j-1] == cwb[i+2,j-2] == 0):
                        return 0
                    if (cwb[i,j] == cwb[i+1,j-1] == cwb[i+2,j-2] == 1):
                        return 1 

                                  
                else:
                    continue
def possibleDirections(pmb,maximizing_player):
    possible_directions = []
    b =  copy.deepcopy(pmb)
    for i in range(pmb.shape[0]):
            for j in range(pmb.shape[1]):
                if maximizing_player:
                    if b[i,j] == 0 :
                    #check for right
                        if ((j <= pmb.shape[1]-2) and (b[i,j+1] == -1)):
                            #print('1')
                            m = str(i) +str(j)+'E'
                            possible_directions.append(m) 
                            
                    #check for left
                        if ((j >= 1) and (b[i,j-1] == -1)):
                            #print('2')
                            m =  str(i) + str(j) + 'W'
                            possible_directions.append(m) 
                    #check for up
                        if ( (i >= 1) and (b[i-1,j] == -1)):
                            #print('3')
                            m =str(i) + str(j) + 'N'
                            possible_directions.append(m) 
                    #check for down
                        if ((i <= b.shape[0]-2) and (b[i+1,j] == -1)):
                            #print('4')
                            m = str(i) + str(j) + 'S'
                            possible_directions.append(m) 

                if not maximizing_player:
                    if b[i,j] == 1:    
                        #check for right
                        if ((j <= pmb.shape[1]-2) and (b[i,j+1] == -1)):
                            #print('5')
                            m = str(i) + str(j) + 'E'
                            possible_directions.append(m) 
                            
                    #check for left
                        if ((j >= 1) and (b[i,j-1] == -1)):
                            #print('6')
                            m =  str(i) + str(j) + 'W'
                            possible_directions.append(m) 
                    #check for up
                        if ( (i >= 1) and (b[i-1,j] == -1)):
                            #print('7')
                            m = str(i) + str(j) + 'N'
                            possible_directions.append(m) 
                    #check for down
                        if ((i <= b.shape[0]-2) and (b[i+1,j] == -1)):
                            #print('8')
                            m =  str(i) + str(j) + 'S'
                            possible_directions.append(m) 
    return possible_directions
def minimax(board,depth,maximizing_player):
    if depth == 0 or check_win(board):
        temp_board = board.copy()
        return heuristic(temp_board),None
    if maximizing_player:
        children = possibleDirections(board,maximizing_player)
        #best_move = children[0]
        maxEval = -10000
        for child in children:
            i,j,dir = (int(child[0]), int(child[1]),child[2])
            temp_board = board.copy()
            temp_board = move(i,j,dir,temp_board,True)
            board = temp_board.copy()
            eval = minimax(temp_board,depth-1, False)[0]
            if eval > maxEval:
                maxEval = eval
                best_move = child
        return maxEval,best_move
    else:
        children = possibleDirections(board,maximizing_player)
        minEval = 10000
        for child in children:
            i,j,dir = (int(child[0]),int(child[1]),child[2])
            temp_board = board.copy()
            temp_board = move(i,j,dir,temp_board,True)
            board = temp_board.copy()
            eval = minimax(temp_board,depth-1,True)[0]
            if eval < minEval:
                minEval = eval
                best_move = child
        return minEval,best_move
def alpha_beta(board,depth,alpha,beta, maximizing_player):
    if depth == 0 or check_win(board):
        temp_board = board.copy()
        return heuristic(temp_board),None
    if maximizing_player:
        children = possibleDirections(board,maximizing_player)
        maxEval = -1000
        for child in children:
            i,j,dir = (int(child[0]), int(child[1]),child[2])
            temp_board = board.copy()
            temp_board = move(i,j,dir,temp_board,True)
            board = temp_board.copy()
            eval = alpha_beta(temp_board,depth-1,alpha,beta, False)[0]
            if eval > maxEval:
                maxEval = eval
                best_move = child 
            if eval > alpha:
                alpha = eval  
            if beta <= alpha:
                break  
        return maxEval,best_move
    else:
        children = possibleDirections(board,maximizing_player)
        minEval = 1000
        for child in children:
            i,j,dir = (int(child[0]), int(child[1]),child[2])
            temp_board = board.copy()
            temp_board = move(i,j,dir,temp_board,True)
            board = temp_board.copy()
            eval = alpha_beta(temp_board,depth-1,alpha,beta, False)[0]
            if eval < minEval:
                minEval = eval
                best_move = child
            if eval < beta:
                beta=eval
            if beta <= alpha:
                break
        return minEval,best_move
def heuristic(board):
    a = 0
    b = 0
    for i in range(board.shape[0]):
        for j in range(board.shape[1]):             
            if (j <= board.shape[0] -3):
                if (board[i,j] == board[i,j+1] == 0):
                    a += 2
                # if (board[i,j] == board[i,j+2] == 0):
                #     a += 1
                if (board[i,j] == board[i,j+1] == 1):
                    b += 2
                # if (board[i,j] ==  board[i,j+2] == 1):
                #     b += 1
            if (i <= board.shape[1]-3):
                if (board[i,j] == board[i+1,j] == 0):
                    a += 2
                # if (board[i,j] == board[i+2,j] == 0):
                #     a += 1
                if (board[i,j] == board[i+1,j] == 1):
                    b += 2
                # if (board[i,j] == board[i+1,j] == 1):
                #     b += 1
            if ((i <= board.shape[0]-2) and (j <= board.shape[1]-2)):
                if (board[i,j] == board[i+1,j+1] == 0):
                    a += 2
                # if (board[i,j] == board[i+2,j+2] == 0):
                #     a += 1 
                if (board[i,j] == board[i+1,j+1] == 1):
                    b += 2 
                # if (board[i,j] == board[i+2,j+2] == 1):
                #     b += 1 
            if (( i <= board.shape[0] -2  and j>=1)):
                if (board[i,j] == board[i+1,j-1] == 0):
                    a += 2
                # if (board[i,j] == board[i+2,j-2] == 0):
                #     a += 1
                if (board[i,j] == board[i+1,j-1] == 1):
                    b += 2
                # if (board[i,j] == board[i+2,j-2] == 1):
                #     b += 1                         
    return (a-b)             
choice = input("Which color do you choose to play? Choose 'white' or 'black':")
size = input("Size of the board:(type 1 for 5X4, type 2 for 7X6:")
AI = input('Which AI would you like to play agaist with? For Minimax enter 1 and for Alpha-Beta pruning enter 2:')
game_on = True
board = create_board(size)
print('Initial position of the board')
show_board(board)
if choice == "white":
    maximizing_player = False
else:
    maximizing_player = True
while game_on: 
    start = time.time()
    if maximizing_player:
        print("AI's move:")
        if AI == '1':
            best_move= minimax(board,6,maximizing_player)[1]
            print(best_move)
        else:
            best_move = alpha_beta(board,6,-100,100,maximizing_player)[1]
        i,j,dir = (int(best_move[0]),int(best_move[1]),best_move[2])
        print(i,j,dir)
        board = move(i,j,dir,board,True)
        show_board(board)
        i,j,dir = ask_move(choice,board)
        print(i,j,dir)
        board = move(i,j,dir,board,True)   
        show_board(board)  
        end = time.time()
        run = end - start
        print(run)
    else:
        i,j,dir = ask_move(choice,board)
        print(i,j,dir)
        board = move(i,j,dir,board,True)  
        show_board(board)
        if AI == '1':
            best_move= minimax(board,6,maximizing_player)[1]
        else:
            best_move = alpha_beta(board,6,-100,100,maximizing_player)[1]        
        print("AI's move:")
        i,j,dir = (int(best_move[0]),int(best_move[1]),best_move[2])
        print(i,j,dir)
        board = move(i,j,dir,board,True)
        show_board(board)
        end = time.time()
        run = end - start
        print(run)
    check_win(board)
    win = check_win(board)
    if win==0:
        print ('White wins')
        break
    elif win == 1:
        print("Black wins")
        break    
show_board(board)
