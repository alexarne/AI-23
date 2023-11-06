#!/usr/bin/env python3
import random
import numpy as np

from fishing_game_core.game_tree import Node
from fishing_game_core.player_utils import PlayerController
from fishing_game_core.shared import ACTION_TO_STR

from time import time

class PlayerControllerHuman(PlayerController):
    def player_loop(self):
        """
        Function that generates the loop of the game. In each iteration
        the human plays through the keyboard and send
        this to the game through the sender. Then it receives an
        update of the game through receiver, with this it computes the
        next movement.
        :return:
        """

        while True:
            # send message to game that you are ready
            msg = self.receiver()
            if msg["game_over"]:
                return


class PlayerControllerMinimax(PlayerController):

    def __init__(self):
        super(PlayerControllerMinimax, self).__init__()
        self.TIME_LIMIT = 0.072
        self.MAX_DEPTH = 3

    def player_loop(self):
        """
        Main loop for the minimax next move search.
        :return:
        """

        # Generate first message (Do not remove this line!)
        first_msg = self.receiver()

        while True:
            msg = self.receiver()

            # Create the root node of the game tree
            node = Node(message=msg, player=0)

            # Possible next moves: "stay", "left", "right", "up", "down"
            best_move = self.search_best_next_move(initial_tree_node=node)

            # Execute next action
            self.sender({"action": best_move, "search_time": None})

    def search_best_next_move(self, initial_tree_node):
        """
        Use minimax (and extensions) to find best possible next move for player 0 (green boat)
        :param initial_tree_node: Initial game tree node
        :type initial_tree_node: game_tree.Node
            (see the Node class in game_tree.py for more information!)
        :return: either "stay", "left", "right", "up" or "down"
        :rtype: str
        """

        # EDIT THIS METHOD TO RETURN BEST NEXT POSSIBLE MODE USING MINIMAX ###

        # NOTE: Don't forget to initialize the children of the current node
        #       with its compute_and_get_children() method!

        self.previous_time = time()
        best_move, value, timeout = self.minimax(initial_tree_node, self.MAX_DEPTH, -np.inf, np.inf, 0)
        print("move chosen:", best_move)
        print("value?", value)
        print("timeout?", timeout)

        # random_move = random.randrange(5)
        return ACTION_TO_STR[best_move]
    
    # Manhattan distance
    def manhattan(self, hook, fish):
        x = abs(fish[0]-hook[0])
        return min(x, 20-x) + abs(fish[1]-hook[1])

    # Euclidian distance
    def euclidian(self, hook, fish):
        return np.sqrt(abs(fish[0]-hook[0])**2 + abs(fish[1]-hook[1])**2)

    # ν(A, s) = Score(Green boat) − Score(Red boat) + tiebreaker, where
    #   Score() takes fish-on-hook into account and
    #   tiebreaker is a value in [0, 1) which gives proximity to highest potential (score per distance) fish
    def heuristic(self, node):
        # p1 = player
        # p2 = opponent
        p1_score, p2_score = node.state.get_player_scores()
        hooks = node.state.get_hook_positions()
        p1_hook = hooks[0]
        p2_hook = hooks[1]
        fish_scores = node.state.get_fish_scores() #list(node.state.get_fish_scores().values())
        fishes = node.state.get_fish_positions()
        value_diff = p1_score - p2_score

        # Consider all fish, see if on either hook, if not then approximate highest potential fish
        best_fish_value = 0
        for fish in fishes:
            # print("fish pos:",fishes[fish][0],",",fishes[fish][1],"fish score:",scores[fish])
            p1_distance = self.manhattan(p1_hook, fishes[fish])
            p2_distance = self.manhattan(p2_hook, fishes[fish])
            if p1_distance == 0:
                value_diff += fish_scores[fish]
            elif p2_distance == 0:
                value_diff -= fish_scores[fish]
            elif fish_scores[fish] > 0:
                fish_value = fish_scores[fish] / p1_distance
                if fish_value > best_fish_value:
                    best_fish_value = fish_value

        max_fish_value = 15     # Highest score fish at 1 distance + some margin
        tiebreaker_value = best_fish_value / max_fish_value

        value = value_diff + tiebreaker_value
        # print("heuristic value:", value)
        return value
    
    def sort_nodes(self, nodes, player):
        return sorted(nodes, key=lambda x: self.minimax(x, 0, -np.inf, np.inf, 1 if player==0 else 0), reverse=player==1)

    def minimax(self, node, depth, alpha, beta, player):
        if time() - self.previous_time > self.TIME_LIMIT:
            return 0, 0, True

        if depth == 0:
            return node.move, self.heuristic(node), False
        
        children = node.compute_and_get_children()
        # children = self.sort_nodes(children, player)
        if len(children) == 0:
            return node.move, self.heuristic(node), False
        
        moves = []
        for child in children: 
            if player == 0: # maximizing player
                value = -np.inf
                for child in children:
                    move, temp, timeout = self.minimax(child, depth-1, alpha, beta, 1)
                    if (depth == self.MAX_DEPTH):
                        print("move", child.move, "value", temp)
                    if timeout:
                        return 0, 0, True
                    if (temp > value):
                        moves = []
                        moves.append(child.move)
                    elif (temp == value):
                        moves.append(child.move)

                    value = max(value, temp)
                    alpha = max(alpha, value)
                    if beta <= alpha:
                        break
            else: # player 1, minimizing player
                value = np.inf
                moves = []
                for child in children:
                    move, temp, timeout = self.minimax(child, depth-1, alpha, beta, 0)
                    if timeout:
                        return 0, 0, True
                    if (temp < value):
                        moves = []
                        moves.append(child.move)
                    elif (temp == value):
                        moves.append(child.move)
                        
                    value = min(value, temp)
                    beta = min(beta, value)
                    if beta <= alpha:
                        break

        return moves[0], value, False

    # ... based on this from wikipedia:

# function minimax(node, depth, maximizingPlayer) is
    # if depth = 0 or node is a terminal node then
    #     return the heuristic value of node
    # if maximizingPlayer then
    #     value := −∞
    #     for each child of node do
    #         value := max(value, minimax(child, depth − 1, FALSE))
    #     return value
    # else (* minimizing player *)
    #     value := +∞
    #     for each child of node do
    #         value := min(value, minimax(child, depth − 1, TRUE))
    #     return value