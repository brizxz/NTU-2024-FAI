# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        food_pos = newFood.asList()
        food_cnt = len(food_pos)
        min_dis = 1e6
        for i in range(food_cnt):
            dis = manhattanDistance(newPos, food_pos[i]) + 100*food_cnt
            min_dis = min(min_dis, dis)
        if food_cnt == 0:
            min_dis = 0
        score = -min_dis

        for i in range(len(newGhostStates)):
            ghostPos = successorGameState.getGhostPosition(i+1)
            if manhattanDistance(newPos,ghostPos) <= 1:
                score -= 1e6
        return score

def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        num_agent = gameState.getNumAgents()
        action_score = []

        def next_actlist(list_now):
            return [i for i in list_now if i != 'Stop']

        def minimax(gameState, cnt):
            if cnt >= self.depth*num_agent or gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)
            if cnt%num_agent != 0: # ghost run (min)
                result = 1e9
                for act in next_actlist(gameState.getLegalActions(cnt%num_agent)):
                    sdot = gameState.generateSuccessor(cnt%num_agent,act)
                    result = min(result, minimax(sdot, cnt+1))
                return result
            else: # pacman run (max)
                result = -1e9
                for act in next_actlist(gameState.getLegalActions(cnt%num_agent)):
                    sdot = gameState.generateSuccessor(cnt%num_agent,act)
                    result = max(result, minimax(sdot, cnt+1))
                    if cnt == 0: # highest node (now) 
                        action_score.append(result)
                return result
        minimax(gameState, 0)
        pacman_nextact_list = next_actlist(gameState.getLegalActions(0))
        # return the action makes pacman have highest action score
        return pacman_nextact_list[action_score.index(max(action_score))]

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        num_agent = gameState.getNumAgents()
        action_score = []

        def next_actlist(list_now):
            return [i for i in list_now if i != 'Stop']

        def alpha_beta_pruning(gameState, cnt, alpha, beta):
            if cnt >= self.depth*num_agent or gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)
            if cnt%num_agent != 0: # ghost run (min)
                result = 1e9
                for act in next_actlist(gameState.getLegalActions(cnt%num_agent)):
                    sdot = gameState.generateSuccessor(cnt%num_agent,act)
                    result = min(result, alpha_beta_pruning(sdot, cnt+1, alpha, beta))
                    beta = min(beta, result)
                    if result < alpha:
                        break
                return result
            else: # pacman run (max)
                result = -1e9
                for act in next_actlist(gameState.getLegalActions(cnt%num_agent)):
                    sdot = gameState.generateSuccessor(cnt%num_agent,act)
                    result = max(result, alpha_beta_pruning(sdot, cnt+1, alpha, beta))
                    alpha = max(alpha, result)
                    if result > beta:
                        break
                    if cnt == 0: # highest node (now) 
                        action_score.append(result)
                return result
        alpha_beta_pruning(gameState, 0, -1e15, 1e15)
        pacman_nextact_list = next_actlist(gameState.getLegalActions(0))
        # return the action makes pacman have highest action score
        return pacman_nextact_list[action_score.index(max(action_score))]


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        num_agent = gameState.getNumAgents()
        action_score = []

        def next_actlist(list_now):
            return [i for i in list_now if i != 'Stop']

        def expect_minimax(gameState, cnt):
            if cnt >= self.depth*num_agent or gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)
            if cnt%num_agent != 0: # ghost run (min)
                score_sum = 0 
                for act in next_actlist(gameState.getLegalActions(cnt%num_agent)):
                    sdot = gameState.generateSuccessor(cnt%num_agent,act)
                    score_sum += expect_minimax(sdot, cnt+1)
                return float(score_sum) / len(next_actlist(gameState.getLegalActions(cnt%num_agent)))
            else: # pacman run (max)
                result = -1e9
                for act in next_actlist(gameState.getLegalActions(cnt%num_agent)):
                    sdot = gameState.generateSuccessor(cnt%num_agent,act)
                    result = max(result, expect_minimax(sdot, cnt+1))
                    if cnt == 0: # highest node (now) 
                        action_score.append(result)
                return result
        expect_minimax(gameState, 0)
        pacman_nextact_list = next_actlist(gameState.getLegalActions(0))
        # return the action makes pacman have highest action score
        return pacman_nextact_list[action_score.index(max(action_score))]

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    def score_ghost(gameState):
        score = 0
        for ghost in gameState.getGhostStates():
            dis_ghost = manhattanDistance(gameState.getPacmanPosition(), ghost.getPosition())
            if ghost.scaredTimer > 0:
                score += max(8 - dis_ghost, 0)**2
            else:
                score -= max(7 - dis_ghost, 0)**2
        return score

    disFood = [0]
    for food in currentGameState.getFood().asList():
        disFood.append(1.0/manhattanDistance(currentGameState.getPacmanPosition(), food))
    score_Food = max(disFood)

    score_cap = [0]
    for cap in currentGameState.getCapsules():
        score_cap.append(49.0/manhattanDistance(currentGameState.getPacmanPosition(), cap))
    score_Capsules = max(score_cap)

    score = currentGameState.getScore()
    score_Ghosts = score_ghost(currentGameState)
    return score + score_Ghosts + score_Food + score_Capsules

# Abbreviation
better = betterEvaluationFunction
