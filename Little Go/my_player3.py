# 4876010998: XU KANGYAN
# CSCI561 HW 2 Little-Go
# 2020.3.27

import sys
import random
import math
from copy import deepcopy
ALPHA = -math.inf

# ---------- Read Input ---------- #

def read_input():
    with open("input.txt", 'r') as f:
        lines = f.readlines()
        # 1-Black-X   2-White-O
        stone_type = int(lines[0])
        # The Board After I Played Last Round
        board_previous = [[int(x) for x in line.rstrip('\n')] for line in lines[1:6]]
        # The Board After Opponent Played
        board = [[int(x) for x in line.rstrip('\n')] for line in lines[6:11]]
        return stone_type, board_previous, board


# ---------- Write Output ---------- #

def write_output(move):
    if move == "PASS":
        result = "PASS"
    else:
        result = str(move[0]) + ',' + str(move[1])
    with open("output.txt", 'w') as f:
        f.write(result)


# ----------- MyPlayer ----------- #

class MyPlayer:
    def __init__(self):
        pass

    # --------------------------------- #
    # AI Method Random

    def ai_method_random(self, feasible):
        return random.choice(feasible)

    # --------------------------------- #
    # AI Method Alpha-Beta Pruning

    def ai_method_alpha_beta_pruning(self, m, n, stone_type, board):
        global ALPHA
        flag = True
        my_score_list = []
        oppo_stone_type = 3 - stone_type  # Opponent Stone Type
        for i in range(5):
            for j in range(5):
                board_next = deepcopy(board)  # The Board After I Played With Removing
                go_next = GoRule(oppo_stone_type, [], board_next)
                if board_next[i][j] == 0:  # Empty
                    board_next[i][j] = go_next.stone_type  # Board After Opponent Played Without Removing Dead Stones
                    board_dd_removed = go_next.dead_opponent_stone_remove(board_next)
                    if go_next.liberty_check(i, j, board_dd_removed):
                        my_count = 0
                        for ii in range(5):
                            for jj in range(5):
                                if board_dd_removed[ii][jj] == stone_type:
                                    my_count += 1
                        my_score_list.append((i, j, my_count))

        beta = math.inf

        for item in my_score_list:
            if item[2] < beta:
                beta = item[2]
            if ALPHA >= beta:
                flag = False
                break

        if flag:
            ALPHA = beta
            return m, n

    # --------------------------------- #
    # Move Operation

    def move(self, go):
        flag = False
        place_feasible = []  # Store Feasible Place (i,j)
        stone_count = 0
        for i in range(5):
            for j in range(5):
                if go.board[i][j] != 0:
                    stone_count += 1

        # Add Feasible Place (i,j) under Liberty Rule KO Rule
        for i in range(5):
            for j in range(5):
                board = deepcopy(go.board)
                if go.place_occupation_check(i, j):
                    board[i][j] = go.stone_type  # The Board After I Played Without Removing Dead Stones
                    my_stone_type = deepcopy(board[i][j])
                    board_opdd_removed = go.dead_opponent_stone_remove(board)  # With Dead Opponent Stones Removed
                    if not go.ko_check(board_opdd_removed) and go.liberty_check(i, j, board_opdd_removed):
                        place_feasible.append((i, j))
                        flag = True
                if flag:
                    my_move = self.ai_method_alpha_beta_pruning(i, j, my_stone_type, board_opdd_removed)
                    if my_move:
                        result = my_move

        if not place_feasible:
            return "PASS"
        elif stone_count < 7:
            return self.ai_method_random(place_feasible)
        else:
            return result


# ------------ GoRule ------------ #

class GoRule:
    def __init__(self, stone_type, board_previous, board):
        self.max_move = 24
        self.board_previous = board_previous  # The Board After I Played Last Round
        self.board = board  # The Board After Opponent Played
        self.stone_type = stone_type

    # --------------------------------- #
    # Get (i, j), Return Empty or Not T/F

    def place_occupation_check(self, i, j):
        board = self.board
        if board[i][j] != 0:
            return False
        return True

    # --------------------------------- #
    # Get Board, (i, j), Check (i, j) liberty, Return T/F

    def liberty_check(self, i, j, board):
        connected_stones = self.stone_connected_dfs(i, j, board)
        for stone in connected_stones:
            for neighbor in self.neighbor_location(stone[0], stone[1]):
                if board[neighbor[0]][neighbor[1]] == 0:
                    return True
        return False

    # --------------------------------- #
    # Get Board, (i, j), DFS Find Stones Connected, Return Their Location

    def stone_connected_dfs(self, i, j, board):
        frontier = [(i, j)]
        explored = []
        while frontier:
            node = frontier.pop()
            explored.append(node)
            group = self.neighbor_stone_connected_check(node[0], node[1], board)
            for neighbor in group:
                if (neighbor not in frontier) and (neighbor not in explored):
                    frontier.append(neighbor)
        return explored

    # --------------------------------- #
    # Get Board, (i, j), Check (i, j) Neighbor Stone Type, Return Same_Stone_Type Neighbor Location Group

    def neighbor_stone_connected_check(self, i, j, board):
        same_stone_type_neighbor_location_group = []
        neighbor = self.neighbor_location(i, j)
        for location in neighbor:
            if board[location[0]][location[1]] == board[i][j]:
                same_stone_type_neighbor_location_group.append(location)
        return same_stone_type_neighbor_location_group

    # --------------------------------- #
    # Get Board, Return T/F

    def ko_check(self, board):
        board_previous = self.board_previous  # The Board After I Played Last Round
        if self.board_compare(board_previous, board):
            return True  # Which Means State Repeating
        return False

    # --------------------------------- #
    # Get Board, Remove All Dead Stones, Return New Board ! BOARD IS CHANGED !

    def dead_opponent_stone_remove(self, board):
        opponent_stone_type = 3 - self.stone_type
        remove_list = []
        for i in range(5):
            for j in range(5):
                if board[i][j] == opponent_stone_type:
                    if not self.liberty_check(i, j, board):
                        remove_list.append((i, j))
        for stone in remove_list:
            board[stone[0]][stone[1]] = 0
        return board

    # --------------------------------- #
    # Get (i, j), Return All its Neighbor (x, y)

    def neighbor_location(self, i, j):
        stone_neighbor = []  # Store Neighbor's Location (x, y)
        if i > 0: stone_neighbor.append((i-1, j))  # Stone on  Left Edge or Not
        if i < 4: stone_neighbor.append((i+1, j))  # Stone on Right Edge or Not
        if j > 0: stone_neighbor.append((i, j-1))  # Stone on    Up Edge or Not
        if j < 4: stone_neighbor.append((i, j+1))  # Stone on  Down Edge or Not
        return stone_neighbor

    # --------------------------------- #
    # Get Two Board, Return T/F

    def board_compare(self, board_1, board_2):
        for i in range(5):
            for j in range(5):
                if board_1[i][j] != board_2[i][j]:
                    return False
        return True


# ------------- main ------------- #

if __name__ == "__main__":
    stone_type_N, board_previous_N, board_N = read_input()
    go_N = GoRule(stone_type_N, board_previous_N, board_N)
    player = MyPlayer()
    move_N = player.move(go_N)
    write_output(move_N)
