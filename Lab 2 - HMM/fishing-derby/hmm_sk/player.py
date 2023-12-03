#!/usr/bin/env python3

from player_controller_hmm import PlayerControllerHMMAbstract
from constants import *
import random
import math
import sys
EPSILON = sys.float_info.epsilon



# forward algorithm - alpha-pass algorithm
# A - Transition matrix
# B - Emission matrix
# pi - Initial state vector
# emissions - Sequence of observations
# Returns - Probability of observing the observation sequence
def forward(A, B, pi, emissions):
    N = len(A) # num states
    alpha = [[0 for _ in range(N)] for _ in range(len(emissions))] # zeros matrix
    alpha[0] = [pi[0][i] * B[i][emissions[0]] for i in range(N)]
    for t in range(1, len(emissions)):
        for i in range(N):
            alpha[t][i] = sum([alpha[t-1][j] * A[j][i] for j in range(N)]) * B[i][emissions[t]]
    return sum(alpha[-1])

# backward algorithm - beta-pass algorithm
# A - Transition matrix
# B - Emission matrix
# pi - Initial state vector
# emissions - Sequence of observations
# Returns - New estimated transition matrix, emissions matrix, and initial state vector
DEBUG = False
def estimate_model(A, B, pi, emissions):
    if DEBUG:
        print("estimating -------------")
    N = len(A)          # num states
    T = len(emissions)  # num observations

    oldLogProb = 0
    logProb = 1
    MAX_ITER = 100
    iter = 0
    while iter < MAX_ITER and abs(oldLogProb - logProb) > 1e-2:
        if DEBUG:
            print("new pass----------")
        if DEBUG:
            print("A:", A)
        if DEBUG:
            print("B:", B)
        if DEBUG:
            print("pi:", pi)
        oldLogProb = logProb
        # Compute all (with alpha and beta normalized)
        norm = [1 for _ in range(T)]

        # --------- alpha (normalized) ---------
        alpha = [[0 for _ in range(N)] for _ in range(T)] # zeros matrix
        alpha[0] = [pi[0][i] * B[i][emissions[0]] for i in range(N)]
        norm[0] = sum(alpha[0])
        alpha[0] = [pi[0][i] * B[i][emissions[0]] / (norm[0]+EPSILON) for i in range(N)]
        for t in range(1, T):
            for i in range(N):
                alpha[t][i] = sum([alpha[t-1][j] * A[j][i] for j in range(N)]) * B[i][emissions[t]]
            norm[t] = sum(alpha[t])
            if DEBUG:
                print("norm", norm[t])
            if DEBUG:
                print("alpha", alpha)
            alpha[t] = [alpha[t][i] / (norm[t]+EPSILON) for i in range(N)]


        # --------- beta (normalized) ---------
        beta = [[0 for _ in range(N)] for _ in range(T)] # zeros matrix
        beta[-1] = [1 / (norm[-1]+EPSILON) for _ in range(N)]
        for t in range(T-2, -1, -1):
            for i in range(N):
                beta[t][i] = sum([beta[t+1][j] * B[j][emissions[t+1]] * A[i][j] for j in range(N)])
                beta[t][i] = beta[t][i] / (norm[t]+EPSILON)

        # --------- di-gamma ---------
        # Doesn't need normalization because alpha & beta are normalized
        # print(alpha[-1])
        dg = [[[alpha[t][i]*A[i][j]*B[j][emissions[t+1]]*beta[t+1][j] 
                for j in range(N)] 
                for i in range(N)] 
                for t in range(T-1)]
        
        # --------- gamma ---------
        g = [[sum([dg[t][i][j] for j in range(N)]) for i in range(N)] for t in range(T-1)]
        if DEBUG:
            print(g)
        # Re-estimate A, B, pi
        A = [[sum([dg[t][i][j] for t in range(T-1)]) / (sum([g[t][i] for t in range(T-1)])+EPSILON)
            for j in range(len(A[0]))] 
            for i in range(len(A))]
        B = [[sum([(1 if emissions[t] == k else 0)*g[t][j] for t in range(T-1)]) / (sum([g[t][j] for t in range(T-1)])+EPSILON)
            for k in range(len(B[0]))] 
            for j in range(len(B))]
        pi = [g[0]]

        if DEBUG:
            print("NORM:", norm)

        # Repeat until convergence
        logProb = sum([math.log(norm[i]+EPSILON) for i in range(len(norm))])

    return A, B, pi

def init_matrix(size_y, size_x):
    matrix = [[random.random() for _ in range(size_x)] for _ in range(size_y)]
    for i in range(size_y):
        rowsum = sum(matrix[i])
        matrix[i] = [v / rowsum for v in matrix[i]]
    return matrix

class PlayerControllerHMM(PlayerControllerHMMAbstract):
    def init_parameters(self):
        """
        In this function you should initialize the parameters you will need,
        such as the initialization of models, or fishes, among others.
        """

        self.obs = [[] for _ in range(N_FISH)]
        self.guesses = 0

        # Models stored A, B, pi
        N = 1
        self.models = [(init_matrix(N, N),
                        init_matrix(N, N_EMISSIONS), # N_EMISSIONS = K
                        init_matrix(N, N)) 
                        for _ in range(N_SPECIES)]

        pass

    def guess(self, step, observations):
        """
        This method gets called on every iteration, providing observations.
        Here the player should process and store this information,
        and optionally make a guess by returning a tuple containing the fish index and the guess.
        :param step: iteration number
        :param observations: a list of N_FISH observations, encoded as integers
        :return: None or a tuple (fish_id, fish_type)
        """
        
        # Store observations
        for i in range(len(observations)):
            self.obs[i].append(observations[i])

        # Only start guessing when we have to
        if (step <= N_STEPS-N_FISH):
            return None
        
        # Guess fish in order
        # Use forward algorithm to get most probable model 
        # for the observations
        current_fish = self.guesses
        highest = 0
        index = 0
        for i in range(N_SPECIES):
            A, B, pi = self.models[i]
            emissions = self.obs[current_fish]
            val = forward(A, B, pi, emissions)
            if val > highest:
                highest = val
                index = i
        return (current_fish, index)

    def reveal(self, correct, fish_id, true_type):
        """
        This methods gets called whenever a guess was made.
        It informs the player about the guess result
        and reveals the correct type of that fish.
        :param correct: tells if the guess was correct
        :param fish_id: fish's index
        :param true_type: the correct type of the fish
        :return:
        """

        if DEBUG:
            print("On fish", fish_id, "guessed", correct)

        # Tweak the model based on the observations for that fish
        A, B, pi = self.models[true_type]
        observations = self.obs[fish_id]
        self.models[true_type] = estimate_model(A, B, pi, observations)
        self.guesses = self.guesses + 1

        pass
