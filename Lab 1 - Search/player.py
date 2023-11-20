#!/usr/bin/env python3
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
	
# -------------------- MINIMAX --------------------
	def run_minimax(self, node, depth, alpha, beta):
		children = node.compute_and_get_children()
		moves = []
		for child in children:
			val = self.minimax(child, depth, alpha, beta, 1)
			moves.append((child.move, val, self.heuristic(child)))

		# get index of best move, primarily sort by minimax value and tiebreak using immediate heuristic value
		best_move = max(moves, key=lambda x: (x[1], x[2]))

		return best_move

	def minimax(self, node, depth, alpha, beta, player):
		# iterative deepening timeout	
		if time() - self.initial_time > self.time_limit:
			raise TimeoutError

		# -------------------- REPEATED STATES CHECKING --------------------
		key = self.hashish(node.state)
		if key in self.repeated_states and depth <= self.repeated_states[key][0]:
			self.repeated_states[key] = (depth, self.repeated_states[key][1])
			return self.repeated_states[key][1]

		# -------------------- MINIMAX ALGORITHM --------------------
		children = node.compute_and_get_children()
		p0caught, p1caught = node.state.get_caught()
		if depth == 0 or (len(children) == 1 and (p0caught or p1caught)) or len(children) == 0:
			return self.quiet_search(node, 2, player)
		if player == 0: # maximizing player
			value = -np.inf
			for child in self.sort_nodes(children, reverse=True):
				value = max(value, self.minimax(child, depth-1, alpha, beta, 1))
				alpha = max(alpha, value)
				if beta <= alpha:
					break
		else: # player 1, minimizing player
			value = np.inf
			for child in self.sort_nodes(children, reverse=False):
				value = min(value, self.minimax(child, depth-1, alpha, beta, 0))
				beta = min(beta, value)
				if beta <= alpha:
					break

		# -------------------- REPEATED STATES INSERTION --------------------
		if key not in self.repeated_states:
			self.repeated_states[key] = (depth, value)
		elif self.repeated_states[key][1] < value or (self.repeated_states[key][1] == value and self.repeated_states[key][0] > depth):
			self.repeated_states[key] = (depth, value)

		return value

# -------------------- HEURISTIC --------------------
	# ν(A, s) = Score(Green boat) − Score(Red boat) + tiebreaker, where
	#   Score() takes fish-on-hook into account and
	#   tiebreaker is the proximity to highest potential (score per distance) fish
	def heuristic(self, node):
		# p1 = player
		# p2 = opponent
		p1_score, p2_score = node.state.get_player_scores()
		hooks = node.state.get_hook_positions()
		p1_hook = hooks[0]
		p2_hook = hooks[1]
		fish_scores = node.state.get_fish_scores()
		fishes = node.state.get_fish_positions()
		value_diff = p1_score - p2_score

		# Consider all fish, see if on either hook, if not then approximate highest potential fish
		MAX_FISH_DIST = 40
		best_fish_value = 0
		for fish in fishes:
			p1_distance = self.manhattan(p1_hook, fishes[fish], p2_hook)
			p2_distance = self.manhattan(p2_hook, fishes[fish], p1_hook)
			if p1_distance == 0:
				value_diff += fish_scores[fish]
			elif p2_distance == 0:
				value_diff -= fish_scores[fish]
			elif fish_scores[fish] > 0:
				fish_value = (fish_scores[fish])*((MAX_FISH_DIST - p1_distance) / MAX_FISH_DIST)
				if fish_value > best_fish_value:
					best_fish_value = fish_value
		value = value_diff + best_fish_value
		return value

	# Manhattan distance
	def manhattan(self, hook, fish, enemyHook):
		fd = hook[0]-fish[0]        # fish-delta
		hd = hook[0]-enemyHook[0]   # hook-delta
		yd = abs(fish[1]-hook[1])

		# Go left or right depending on if enemy is blocking
		xStraight = abs(fd)
		xWrapped = 20-abs(fd)
		if (fd*hd >= 0):    # Fish on same side as enemy hook
			# Check if enemy hook is closer on right side
			if (abs(hd) < abs(fd)):
				return xWrapped + yd
			if (abs(hd) > abs(fd)):
				return xStraight + yd
			# Fish is on top of enemy boat, unreachable
			return 99
			
		# Fish and enemy hook on opposite sides
		return xStraight + yd

	# Euclidian distance
	def euclidian(self, hook, fish):
		x = abs(fish[0]-hook[0])
		return np.sqrt((min(x, 20-x))**2 + abs(fish[1]-hook[1])**2)

# -------------------- OPTIMIZATIONS BELOW --------------------

# -------------------- MOVE ORDERING --------------------
	def sort_nodes(self, nodes, reverse):
		return sorted(nodes, key=self.heuristic, reverse=reverse)

# -------------------- ITERATIVE DEEPENING --------------------
	def iterative_deepening(self, initial_tree_node):
		self.initial_time = time()
		self.repeated_states = {}
		depth = 1
		best_move = self.run_minimax(initial_tree_node, 1, -np.inf, np.inf)

		depth += 1
		while depth < self.max_depth:
			try:
				best_move = self.run_minimax(initial_tree_node, depth, -np.inf, np.inf)
				depth += 1
			except:
				break
		return best_move[0]

# -------------------- REPEATED STATES HASH FUNCTION --------------------
	def hashish(self, state):
		h = ""
		hooks = state.get_hook_positions()
		h += str(hooks[0])+str(hooks[1])
		fishes = state.get_fish_positions()
		fish_scores = state.get_fish_scores()
		for fish in fishes:
			h += ";"+str(fishes[fish])+str(fish_scores[fish])
		return h.replace(" ","")

# -------------------- QUISSENCE SEARCH --------------------
	def get_capturing_children(self, node):
		children = node.compute_and_get_children()
		p0caught1, p1caught1 = node.state.get_caught()
		capture_children = []
		for child in children:
			p0caught2, p1caught2 = node.state.get_caught()
			if (p0caught1 != p0caught2 or p1caught1 != p1caught2):
				capture_children.append(child)
		return capture_children

	def quiet(self, node):
		return len(self.get_capturing_children(node)) == 0
	
	def quiet_search(self, node, depth, player):
		if self.quiet(node) or depth == 0:
			return self.heuristic(node)
		
		children = self.get_capturing_children(node)
		value = self.quiet_search(children[0], depth-1, 1-player)
		for child in children[1:]:
			if player == 1:
				value = max(value, self.quiet_search(child, depth-1, 0))
			elif player == 0:
				value = min(value, self.quiet_search(child, depth-1, 1))
		return value
