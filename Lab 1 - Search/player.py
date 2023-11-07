#!/usr/bin/env python3
import numpy as np
from time import time

from fishing_game_core.game_tree import Node
from fishing_game_core.player_utils import PlayerController
from fishing_game_core.shared import ACTION_TO_STR

DEBUG = True

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
        self.repeated_states = {}
        self.initial_time = time()
        # time limit 0.015 is enough to beat all but test_2.json
        # time limit 0.035 is enough to beat all test cases
        self.time_limit = 0.055
        self.max_depth = 10000
        super(PlayerControllerMinimax, self).__init__()

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

        best_move = self.iterative_deepening(initial_tree_node)

        return ACTION_TO_STR[best_move]

    def iterative_deepening(self, initial_tree_node):
        self.initial_time = time()
        # self.repeated_states = {}
        depth = 1
        best_move = self.run_minimax(initial_tree_node, 1, -np.inf, np.inf)
        if DEBUG:
            print("depth", depth, "move:", best_move)
        depth += 1
        while depth < self.max_depth:
            try:
                alt_move = self.run_minimax(initial_tree_node, depth, -np.inf, np.inf)
                if DEBUG:
                    print("depth", depth, "move:", best_move)
                if alt_move[1] > best_move[1]:
                    best_move = alt_move
                depth += 1
            except:
                if DEBUG:
                    print("couldn't do depth", depth)
                break
        return best_move[0]
    
    def run_minimax(self, node, depth, alpha, beta):
        children = node.compute_and_get_children()
        moves = []
        for child in children:
            moves.append((child.move, self.minimax(child, depth, alpha, beta, 1)))

        # get index of best move
        best_move = max(moves, key=lambda x: x[1])

        if DEBUG:
            print("best move is", best_move)
        return best_move
    
    # Manhattan distance
    def manhattan(self, hook, fish):
        x = abs(fish[0]-hook[0])
        return min(x, 20-x) + abs(fish[1]-hook[1])

    # Euclidian distance
    def euclidian(self, hook, fish):
        x = abs(fish[0]-hook[0])
        return np.sqrt((min(x, 20-x))**2 + abs(fish[1]-hook[1])**2)

    def hashish(self, state):
        hooks = state.get_hook_positions()
        h = str(hooks[0])+str(hooks[1])
        fishes = state.get_fish_positions()
        fish_scores = state.get_fish_scores()
        for idx, fish in enumerate(fishes):
            h += str(idx)+str(fishes[fish])+str(fish_scores[fish])
        return h.replace(" ", "")

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

        # closest = np.inf # can be included for alternative final heuristic value, makes it worse in some cases though

        # Consider all fish, see if on either hook, if not then approximate highest potential fish
        best_fish_value = 0
        for fish in fishes:
            # print("fish pos:",fishes[fish][0],",",fishes[fish][1],"has score:",fish_scores[fish])
            # print("currently best_fish_value =", best_fish_value)
            p1_distance = self.manhattan(p1_hook, fishes[fish])
            p2_distance = self.manhattan(p2_hook, fishes[fish])
            if p1_distance == 0:
                value_diff += fish_scores[fish]
            elif p2_distance == 0:
                value_diff -= fish_scores[fish]
            elif fish_scores[fish] > 0:
                fish_value = fish_scores[fish] / (p1_distance+0.01)
                if fish_value > best_fish_value:
                    best_fish_value = fish_value
            
        #     # the following makes boat miss the last fish in first test case because it's one move slower to the second-last fish (I think that's why at least):
        #     if closest > p1_distance:
        #         closest = p1_distance
        # if closest == np.inf:
        #     closest = 0

        tiebreaker_value = best_fish_value

        value = 5*value_diff + 4*tiebreaker_value # - closest
        return value
    
    def sort_nodes(self, nodes):
        return sorted(nodes, key=self.heuristic, reverse=True)

    def minimax(self, node, depth, alpha, beta, player):

        if time() - self.initial_time > self.time_limit:
            raise TimeoutError

        # key = self.hashish(node.state)
        # if key in self.repeated_states and depth <= self.repeated_states[key][0]:
        #     if DEBUG:
        #         print("repeating states avoided")
        #     return self.repeated_states[key][1]

        children = node.compute_and_get_children()
        if depth == 0 or len(children) == 0:
            return self.heuristic(node)
        if player == 0: # maximizing player
            value = -np.inf
            for child in sorted(children, key=self.heuristic, reverse=True):
                value = max(value, self.minimax(child, depth-1, alpha, beta, 1))
                alpha = max(alpha, value)
                if beta <= alpha:
                    break
        else: # player 1, minimizing player
            value = np.inf
            for child in sorted(children, key=self.heuristic, reverse=False):
                value = min(value, self.minimax(child, depth-1, alpha, beta, 0))
                beta = min(beta, value)
                if beta <= alpha:
                    break

        # if DEBUG:
        #     print("new state found")
        # self.repeated_states[key] = [depth, value]

        return value
