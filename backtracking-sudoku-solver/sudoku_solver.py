#!/usr/bin/env python

import copy

class sudoku_solver:
    '''
    This class implements a Sudoku solver using a backtracking algorithm.

    The solver is called using
    sol = Solution()
    sol.solve(board=board)

    where board is a list of lists of strings that contain the partially 
    filled Sudoku that should be solved. The input format follows the leetcode
    exercise 
        https://leetcode.com/problems/sudoku-solver/.
    In particular, sites that are not yet filled must be denoted by 
    a string ".".
    
    Example input:
    
    board = [
    ['.','5','.','.','7','.','.','8','3'],
    ['.','.','4','.','.','.','.','6','.'],
    ['.','.','.','.','5','.','.','.','.'],
    ['8','3','.','6','.','.','.','.','.'],
    ['.','.','.','9','.','.','1','.','.'],
    ['.','.','.','.','.','.','.','.','.'],
    ['5','.','7','.','.','.','4','.','.'],
    ['.','.','.','3','.','2','.','.','.'],
    ['1','.','.','.','.','.','.','.','.'],
    ]
    (This board is sudoku game number 71 from the website 
    http://mysudoku.com/game-sudoku/extremely-difficult_ga.html)


    The backtracking algorithm fills the board row-wise, starting from the 
    upper left corner of the Sudoku and ending at the lower right corner of
    the Sudoku.

    There are two options for solving:
    1. Search for all possible solutions
    2. Terminate once a single solution is found
    The behavior is set by the parameter "get_all_solutions" in the method
    "solve". By default we search for all possible solutions, i.e.
        get_all_solutions=True

    '''

    def __init__(self):
        self.solutions = []

    def move_is_valid(self,i,j,v):
        '''
        check if placing v at site (i,j) is a valid configuration for the
        current state of the board
        '''
        #
        # check if v appears in the current row
        for k in range(9):
            if self.board[i][k] == v:
                return False
        #
        # check if v appears in the current column
        for k in range(9):
            if self.board[k][j] == v:
                return False
        #
        # check if v appears in the current 3x3 square
        for k1 in range(3):
            for k2 in range(3):
                if self.board[3*(i//3) + k1][3*(j//3) + k2] == v:
                    return False 
        #
        return True 
            
    def backtrack(self,s):
        '''
        Backtracking step that tries filling a single empty site of the 
        Sudoku
        '''
        #
        # check whether we have arrived at the lower right corner of the 
        # baord
        if s == self.s_final:
            #
            self.found_solution = True
            # add current solution to list of solutions
            self.solutions.append(copy.deepcopy(self.board))
            #
            return
        #
        # convert the counting index s to a lattice position (i,j) = (row, column)
        i,j = divmod(s, self.n) 
        #
        if self.board[i][j] != ".": 
            # there is a number at the current site 
            # move on to the next site
            self.backtrack(s=s+1)
            #
        else:
            # if there is no number at the current site, try filling the site
            for v in range(1,10):
                v = str(v)
                #
                # check if filling the site (i,j) with value v is a valid move
                if self.move_is_valid(i=i,j=j,v=v): 
                    # if it is, make the change ..
                    self.board[i][j] = v                    
                    # .. and try filling the board further
                    self.backtrack(s=s+1)
                    #
                    if self.found_solution == True:
                        if not self.get_all_solutions:
                            return
                    # undo the change, so that we can try the next value
                    # at (i,j)
                    self.board[i][j] = "."
        return

    def solve(self, board, get_all_solutions=True):
        '''
        This function should be called to solve

        If get_all_solutions == False, then the algorithm terminates once a
        single solution has been found. If the Sudoku is known to have a 
        unique solution, then this will save some computation time.
        '''
        #
        self.solutions = []
        #
        self.board = board.copy()
        self.n = len(board)
        self.s_final = self.n*self.n
        #        
        self.get_all_solutions = get_all_solutions
        self.found_solution = False
        #
        self.backtrack(s=0)
        #
        return self.solutions
    
    def plot_sudoku(self,board):
        # the following code for visualizing a Sudoku 
        # is from https://stackoverflow.com/a/56581709
        # and slightly modified

        base = 3
        side = base*base

        def expandLine(line):
            return line[0]+line[5:9].join([line[1:5]*(base-1)]*base)+line[9:13]
        line0  = expandLine("╔═══╤═══╦═══╗")
        line1  = expandLine("║ . │ . ║ . ║")
        line2  = expandLine("╟───┼───╫───╢")
        line3  = expandLine("╠═══╪═══╬═══╣")
        line4  = expandLine("╚═══╧═══╩═══╝")

        nums   = [ [""]+[n.replace('.',' ') for n in row] for row in board ]
        print(line0)
        for r in range(1,side+1):
            print( "".join(n+s for n,s in zip(nums[r-1],line1.split("."))) )
            print([line2,line3,line4][(r%side==0)+(r%base==0)])