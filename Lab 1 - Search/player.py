#!/usr/bin/env python3
import random
import numpy as np

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

        DEPTH = 5  # idk what is reasonable

        children = initial_tree_node.compute_and_get_children()
        moves = []
        for child in children:
            moves.append(self.minimax(child, DEPTH, 0))

        # get index of best move
        best_move = max(enumerate(moves), key=lambda x: x[1])[0]

        # random_move = random.randrange(5)
        return ACTION_TO_STR[best_move]
    
    # Manhattan distance
    def manhattan(self, hook, fish):
        return abs(fish[0]-hook[0]) + abs(fish[1]-hook[1])

    # Euclidian distance
    def euclidian(self, hook, fish):
        return np.sqrt(abs(fish[0]-hook[0])**2 + abs(fish[1]-hook[1])**2)

    # ν(A, s) = Score(Green boat) − Score(Red boat) from instructions, idk if or how player id should be accounted for, making player1 score negative makes sense to me
    def heuristic(self, node, player):
        score0, score1 = node.state.get_player_scores()
        value = score0 - score1
        if player == 1:
            value *= -1
        return value

    def minimax(self, node, depth, player):
        children = node.compute_and_get_children()
        if depth == 0 or len(children) == 0:
            return self.heuristic(node, player)
        if player == 0: # maximuzing player
            value = -np.inf
            for child in children:
                value = max(value, self.minimax(child, depth-1, 1))
            return value
        else: # player 1, minimizing player
            value = np.inf
            for child in children:
                value = min(value, self.minimax(child, depth-1, 0))
            return value

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