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
        from util import manhattanDistance

        # Generate the successor game state after taking the action.
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        # Start with the game's in-built score for the successor state.

        gamescore = successorGameState.getScore()

        # Food Heuristic: Calculate the distance to the closest food pellet.

        foodlist = newFood.asList()

        if len(foodlist) > 0:
            closest_food_distance = min([manhattanDistance(newPos, food) for food in foodlist])

            # Providing the reward for getting closer to the food!

            gamescore += 10.0 / (closest_food_distance + 1)

        # Encourage eating every bit of food

        gamescore -= 2 * len(foodlist)

        # Ghost Heuristic: Calculating the distance to the closest ghost.

        for ghoststate in newGhostStates:

            ghostposition = ghoststate.getPosition()
            distance_to_ghost = manhattanDistance(newPos, ghostposition)

            if ghoststate.scaredTimer == 0: #Dangerous Ghost

                if distance_to_ghost == 0:
                    return -float('inf')  # Pacman is caught by the ghost
                
                if distance_to_ghost <= 1:
                    return -float('inf')  # Pacman is too close to the ghost
                
                # Penalty for staying close to the ghost

                gamescore -= 4.0 / distance_to_ghost

            else:
                # Reward for moving towards the scared ghosts
                gamescore += 3.0 / (distance_to_ghost + 1)

        if action == Directions.STOP:
            gamescore -= 5  # Discourage stopping

        return gamescore

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
        bestAction = None
        bestValue = float('-inf')

        for action in gameState.getLegalActions(0):  # Pacman's turn

            successor = gameState.generateSuccessor(0, action)

            value = self.value(successor, 1, 0)  # Start with the first ghost (index 1)

            if value > bestValue:

                bestValue = value

                bestAction = action

        return bestAction
    
    def value(self, gameState, agentIndex, currentDepth): #Returns the minimax value of the state.

        if gameState.isWin() or gameState.isLose() or currentDepth == self.depth:

            return self.evaluationFunction(gameState)

        if agentIndex == 0:  # Pacman's turn (Maximizer)

            return self.maxValue(gameState, agentIndex, currentDepth)

        else:  # Ghosts' turn (Minimizer)

            return self.minValue(gameState, agentIndex, currentDepth)
        
    def maxValue(self, gameState, agentIndex, currentDepth):

        actions = gameState.getLegalActions(agentIndex)

        if len(actions) == 0:

            return self.evaluationFunction(gameState)
        
        v = float('-inf')

        for action in actions:

            successor = gameState.generateSuccessor(agentIndex, action)

            nextAgent = 1
            nextDepth = currentDepth

            v = max(v, self.value(successor, nextAgent, nextDepth))

        return v
    
    def minValue(self, gameState, agentIndex, currentDepth):

        actions = gameState.getLegalActions(agentIndex)

        if len(actions) == 0:

            return self.evaluationFunction(gameState)
        
        v = float('inf')

        numAgents = gameState.getNumAgents()

        for action in actions:

            successor = gameState.generateSuccessor(agentIndex, action)

            # Last ghost has moved, so we go back to Pacman and increase the depth
            if agentIndex == numAgents - 1:

                nextAgent = 0
                nextDepth = currentDepth + 1

            else:

                nextAgent = agentIndex + 1
                nextDepth = currentDepth

            v = min(v, self.value(successor, nextAgent, nextDepth))

        return v


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

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
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
