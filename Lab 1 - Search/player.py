#!/usr/bin/env python3
import random
import numpy as np
from time import time

from fishing_game_core.game_tree import Node
from fishing_game_core.player_utils import PlayerController
from fishing_game_core.shared import ACTION_TO_STR

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
        self.limit = 0.045
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

        # DEPTH = 4  # idk what is reasonable

        # children = initial_tree_node.compute_and_get_children()
        # moves = []
        # for child in children:
        #     moves.append(self.iterative_deepening(child))
        #     # moves.append(self.minimax(child, DEPTH, -np.inf, np.inf, 1))

        # # get index of best move
        # best_move = max(enumerate(moves), key=lambda x: x[1])[0]

        print("move chosen:", best_move)

        # random_move = random.randrange(5)
        return ACTION_TO_STR[best_move]

    def iterative_deepening(self, initial_tree_node):
        self.initial_time = time()
        # self.repeated_states = {}
        depth = 1
        best_move = self.run_minimax(initial_tree_node, 1, -np.inf, np.inf)
        print("depth", depth, "move:", best_move)
        depth += 1
        while depth < 8:
            try:
                alt_move = self.run_minimax(initial_tree_node, depth, -np.inf, np.inf)
                print("depth", depth, "move:", best_move)
                if alt_move[1] > best_move[1]:
                    best_move = alt_move
                depth += 1
            except:
                print("couldn't do depth", depth)
                break
        print("ITERATIVE RETURN:",best_move)
        return best_move[0]
    
    def run_minimax(self, node, depth, alpha, beta):
        children = node.compute_and_get_children()
        moves = []
        for child in children:
            moves.append((child.move, self.minimax(child, depth, alpha, beta, 1)))
        
        best_value = max(moves)
        best_moves = []
        for move, value in enumerate(moves):
            if value > best_value:
                best_value = value
                best_moves = []
            if value >= best_value:
                best_moves.append(move)
        for idx, move in enumerate(moves):
            print("move",idx, "("+ACTION_TO_STR[idx]+") minimax score:", move)
        print("best_moves =", best_moves)

        # get index of best move
        best_move = max(moves, key=lambda x: x[1])

        print("RETURNING",best_move)

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

        # Consider all fish, see if on either hook, if not then approximate highest potential fish
        best_fish_value = 0
        closest = np.inf
        for fish in fishes:
            # print("fish pos:",fishes[fish][0],",",fishes[fish][1],"has score:",fish_scores[fish])
            # print("currently best_fish_value =", best_fish_value)
            p1_distance = self.euclidian(p1_hook, fishes[fish])
            p2_distance = self.euclidian(p2_hook, fishes[fish])
            if p1_distance == 0:
                value_diff += fish_scores[fish]
            elif p2_distance == 0:
                value_diff -= fish_scores[fish]
            elif fish_scores[fish] > 0:
                fish_value = fish_scores[fish] / (p1_distance+0.01)
                # print("checking fish_value =", fish_value)
                if fish_value > best_fish_value:
                    best_fish_value = fish_value
            if closest > p1_distance:
                closest = p1_distance
        # print("AFTER THE FACT, BEST_FISH_VALUE:", best_fish_value,"AND CLOSEST=",closest)
        if closest == np.inf:
            closest = 0

        tiebreaker_value = best_fish_value

        value = 5*value_diff + 4*tiebreaker_value - closest
        # print("heuristic value:", value)
        return value
    
    def sort_nodes(self, nodes):
        return sorted(nodes, key=self.heuristic, reverse=True)

    def minimax(self, node, depth, alpha, beta, player):
        move = node.move

        if time() - self.initial_time > self.limit:
            raise TimeoutError

        # key = self.hashish(node.state)
        # if key in self.repeated_states:
        #     return self.repeated_states[key]

        children = node.compute_and_get_children()
        if depth == 0 or len(children) == 0:
            return self.heuristic(node)
        if player == 0: # maximizing player
            value = -np.inf
            # for child in sorted(children, key=self.heuristic, reverse=False):
            for child in children:
                alt_move = self.minimax(child, depth-1, alpha, beta, 1)
                # if alt_move[1] > value:
                if alt_move > value:
                    # move = alt_move
                    # value = alt_move[1]
                    value = alt_move
                alpha = max(alpha, value)
                if beta <= alpha:
                    break
            # return value
        else: # player 1, minimizing player
            value = np.inf
            # for child in sorted(children, key=self.heuristic, reverse=True):
            for child in children:
                alt_move = self.minimax(child, depth-1, alpha, beta, 0)
                # if alt_move[1] < value:
                if alt_move < value:
                    # move = alt_move
                    # value = alt_move[1]
                    value = alt_move
                beta = min(beta, value)
                if beta <= alpha:
                    break
            # return value
        
        # if key not in self.repeated_states or value > self.repeated_states[key]:
        #     self.repeated_states[key] = value

        return value
        # return (move, value)

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