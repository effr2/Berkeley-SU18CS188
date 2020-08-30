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

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
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

    def evaluationFunction(self, currentGameState, action):
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

        newScore = successorGameState.getScore()
        '''
        if successorGameState.getCapsules():
            distancesToCapsules = [manhattanDistance(newPos, capsule) for capsule in successorGameState.getCapsules()]
            newScore+=1.0/min(distancesToCapsules)**2
        '''

        # Substract 1/distance^2 to the closest ghost from the score. The closer you are to a ghost the less score you
        # obtain from reaching this state.
        minDistanceFromGhost = min([manhattanDistance(newPos,ghostState.getPosition()) for ghostState in newGhostStates])
        if minDistanceFromGhost > 0:
            newScore -= 1.0 / minDistanceFromGhost**2

        # T
        foodDistances = [manhattanDistance(newPos, foodPos) for foodPos in newFood.asList()]
        if foodDistances:
            newScore += 1.0 / min(foodDistances)**2

        return newScore

def scoreEvaluationFunction(currentGameState):
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

    def getAction(self, gameState):
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
        # initialize variables
        currentBestValue = float("-inf")
        currentBestAction = Directions.STOP

        # evaluate which is the best action from the current set of legal actions
        for action in gameState.getLegalActions(0):
            successorState = gameState.generateSuccessor(0, action)
            successorValue = self.getValue(successorState, 0, 1)
            # change value to successor value if the successor value is better than the current value
            if successorValue > currentBestValue:
                currentBestValue = successorValue
                currentBestAction = action
        return currentBestAction

    def getValue(self, gameState, currentDepth, agentIndex):
        """
        Helper function that determines the value of a state. If the agent index is 0, it will use the maximizer
        algorithm. If the agent is a ghost, it will use the minimizer algorithm.
        :param gameState: current state of the game
        :param currentDepth: current depth of the minimax tree
        :param agentIndex: pacman or ghosts
        :return: evaluation of actions, max value for pacman or min value for ghosts
        """
        # if game is at terminal state(win or lose) or maximum ply depth, evaluate state
        if currentDepth == self.depth or gameState.isWin() or gameState.isLose() or gameState.getLegalActions(agentIndex) == 0:
            return self.evaluationFunction(gameState)
        # if agent is pacman, use maximizer algorithm
        elif agentIndex == 0:
            maxValue = float("-inf")
            for action in gameState.getLegalActions(0):
                maxValue = max(maxValue, self.getValue(gameState.generateSuccessor(0, action), currentDepth, 1))
            return maxValue
        # if agent is ghost, use minimizer algorithm
        else:
            minValue = float("inf")
            for action in gameState.getLegalActions(agentIndex):
                if agentIndex == gameState.getNumAgents() - 1:
                    minValue = min(minValue,
                                   self.getValue(gameState.generateSuccessor(agentIndex, action), currentDepth + 1,
                                                 0))
                else:
                    minValue = min(minValue,
                                   self.getValue(gameState.generateSuccessor(agentIndex, action), currentDepth,
                                                 agentIndex + 1))
            return minValue

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        # initialize variables
        currentBestActionValue = float("-inf")
        currentBestAction = Directions.STOP
        alpha = float("-inf")
        beta = float("inf")

        # evaluate which is the best action from the current set of legal actions
        for action in gameState.getLegalActions(0):
            successorState = gameState.generateSuccessor(0, action)
            successorValue = self.getValue(successorState, 0, 1, alpha, beta)
            # change value to successor value if the successor value is better than the current value
            if successorValue > currentBestActionValue:
                currentBestActionValue = successorValue
                currentBestAction = action
            alpha = max(alpha, currentBestActionValue) # update alpha
        return currentBestAction

    def getValue(self, gameState, currentDepth, agentIndex, alpha, beta):
        """
        Helper function that determines the value of a state using minimax with alphabeta pruning. If the agent index
        is 0, it will use the maximizer algorithm. If the agent is a ghost, it will use the minimizer algorithm. When
        using the minimizer and maximizer algorithm, it will check if it can prune values from the tree. The minimizer
        algorithm will prune the remaining values if the current value is less than alpha. The maximizer algorithm will
        prune the remaining values if the current value is greater than beta.
        :param gameState: current state of the game
        :param currentDepth: current depth of the minimax tree
        :param agentIndex: pacman or ghosts
        :param alpha: used for minimizer pruning
        :param beta: used for maximizer pruning
        :return: evaluation of actions, best action value for pacman or min value for ghosts
        """
        # if game is at terminal state(win or lose) or maximum ply depth, evaluate state
        if currentDepth == self.depth or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
        # if agent is pacman, use maximizer algorithm
        elif agentIndex == 0:
            maxValue = float("-inf")
            for action in gameState.getLegalActions(0):
                maxValue = max(maxValue,
                               self.getValue(gameState.generateSuccessor(0, action), currentDepth, 1, alpha, beta))
                # if maximizer value is greater than beta, return value and prune the rest
                if maxValue > beta:
                    return maxValue
                alpha = max(alpha, maxValue) # update alpha value
            return maxValue
        # if agent is ghost, use minimizer algorithm
        else:
            minValue = float("inf")
            for action in gameState.getLegalActions(agentIndex):
                if agentIndex == gameState.getNumAgents() - 1:
                    minValue = min(minValue,
                                   self.getValue(gameState.generateSuccessor(agentIndex, action), currentDepth + 1, 0,
                                                 alpha, beta))
                else:
                    minValue = min(minValue,
                                   self.getValue(gameState.generateSuccessor(agentIndex, action), currentDepth,
                                                 agentIndex + 1, alpha, beta))
                # if minimizer value is less than alpha, return value and prune the rest
                if minValue < alpha:
                    return minValue
                beta = min(beta, minValue) # update beta value
            return minValue

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        bestActionValue = float("-inf")
        bestAction = Directions.STOP
        # evaluate which is the best action of the current set of legal actions
        for action in gameState.getLegalActions(0):
            successorState = gameState.generateSuccessor(0, action)
            succesorValue = self.getValue(successorState, 0, 1)
            if succesorValue > bestActionValue:
                bestActionValue = succesorValue
                bestAction = action
        return bestAction

    def getValue(self, gameState, currentDepth, agentIndex):
        """
            Helper function that determines the value of a state using expectimax algorithm. If the agent index
            is 0, it will use the maximizer algorithm. If the agent is a ghost, it will use calculate an average of the
            values for each possible action.
            :param gameState: current state of the game
            :param currentDepth: current depth of the minimax tree
            :param agentIndex: pacman or ghosts
            :return: evaluation of actions, best action value for pacman or average value for ghosts
        """
        # if game is at terminal state(win or lose) or maximum ply depth, evaluate state
        if currentDepth == self.depth or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
        # if agent is pacman, use maximizer algorithm
        elif agentIndex == 0:
            maxValue = float("-inf")
            for action in gameState.getLegalActions(0):
                maxValue = max(maxValue, self.getValue(gameState.generateSuccessor(0, action), currentDepth, 1))
            return maxValue
        # if agent is ghost, find the average of the possible values for each action
        else:
            avgValue = 0.0
            possibleActions = gameState.getLegalActions(agentIndex)
            for action in gameState.getLegalActions(agentIndex):
                if agentIndex == gameState.getNumAgents() - 1:
                    avgValue += self.getValue(gameState.generateSuccessor(agentIndex, action),
                                                        currentDepth + 1, 0)
                else:
                    avgValue += self.getValue(gameState.generateSuccessor(agentIndex, action), currentDepth,
                                                        agentIndex + 1)
            return avgValue/len(possibleActions)

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION:
      The evaluation function will reward Pacman if he takes actions that(in increasing reward order):
      1)get him closer to food
      2)eat power pellets
      It will punish Pacman if:
      1)get close to ghosts that are not scared
    """
    pos = currentGameState.getPacmanPosition()
    #list of remaining food
    foodRemaining = [food for food in currentGameState.getFood().asList() if food]
    ghostStates = currentGameState.getGhostStates()
    scaredTimes = [ghostState.scaredTimer for ghostState in ghostStates]

    newScore = currentGameState.getScore()

    minDistanceToGhost = min([manhattanDistance(pos, ghostStates[0].getPosition()) for ghost in ghostStates])
    ghostScaredTime = min(scaredTimes)

    if ghostScaredTime == 0:
        newScore -= 1/ (minDistanceToGhost + 1)
    else:
        newScore += 1 / (minDistanceToGhost + 1)

    # distance to closest food
    minDistanceToFood=0
    if foodRemaining:
        minDistanceToFood = min([manhattanDistance(pos, foodPos) for foodPos in foodRemaining])

    if minDistanceToFood:
        newScore += 1 / minDistanceToFood
        newScore -= len(foodRemaining)

    newScore += ghostScaredTime * 2


    return newScore

# Abbreviation
better = betterEvaluationFunction

