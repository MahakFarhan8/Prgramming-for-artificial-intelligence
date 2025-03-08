#!/usr/bin/env python
# coding: utf-8

# In[24]:


def position(board, row, col, n):

    for i in range(row):
        if board[i][col] == 'Q':
            return False

    # Check upper-left diagonal
    i, j = row, col
    while i >= 0 and j >= 0:
        if board[i][j] == 'Q':
            return False
        i -= 1
        j -= 1

    # Check upper-right diagonal
    i, j = row, col
    while i >= 0 and j < n:
        if board[i][j] == 'Q':
            return False
        i -= 1
        j += 1

    return True

def display_board(board,n):
    print("Solution:")
    for row in board:
        print(" ".join(row))
    print()
    
def solve_n_queens(board, row, n):
    if row == n:
        display_board(board, n)
        return
    
    for col in range(n):
        if is_safe(board, row, col, n):
            board[row][col] = 'Q' 
            solve_n_queens(board, row + 1, n)
            board[row][col] = '-'  

n = int(input("Enter the number of queens: "))
board = [['-' for _ in range(n)] for _ in range(n)]
solve_n_queens(board, 0,n)


# In[ ]:




