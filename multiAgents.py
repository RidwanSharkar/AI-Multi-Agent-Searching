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
        newFood = successorGameState.getFood().asList()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        score = successorGameState.getScore()

        nearestFood = 0 if len(newFood) == 0 else min([manhattanDistance(foodPos, newPos) for foodPos in newFood])

        score += 1 / (nearestFood + 1)

        for ghostState in newGhostStates:
            
            ghostPos = ghostState.getPosition()
            distance = manhattanDistance(ghostPos, newPos)


            if(ghostState.scaredTimer > 0):
                score +=  500 / (distance + 1)  
            else:
                score -= 10 / (distance + 1)

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

        generateSuccessor(self, agentIndex, action):

        def getNumAgents(self):
        """

        """for each in range(self.depth):
        
            for action in gameState.getLegalActions(0):

                GameState succState = gameState.generateSuccessor(0,action)

                if gameState.isLose() or gameState.isWin(): 
                    return action
                


    def minimizer(GameState succState, int ghosts): "recursive helper function"

        for each in range(ghosts):

            for each in gameState.getLegalActions(ghosts):

                score = max(score, minimizer(succState, ghosts+1))
"""
        return self.minmax(0, 0, gameState)
    
    def minmax(self, agent, depth, gameState: GameState):
            
            if gameState.isWin() or gameState.isLose() or depth == self.depth:

                return self.evaluationFunction(gameState)

            if agent == 0:

                return self.maximizer(0, depth, gameState)
            
            return self.minimizer(agent, depth, gameState)

    def minimizer(self, agent, depth, gameState: GameState):

        minScore = float('inf')

        for action in gameState.getLegalActions(agent):
            
            succState = gameState.generateSuccessor(agent, action)

            if agent == gameState.getNumAgents() - 1:

                score = self.minmax(0, depth + 1, succState)  # moving to next agent - pacman

            else:

                score = self.minmax(agent + 1, depth, succState)  # moving to next agent - next ghost
            
            minScore = min(score, minScore)

        return minScore


    def maximizer(self, agent, depth, gameState: GameState):
        
        maxScore = -float('inf')

        bestAction = None

        for action in gameState.getLegalActions():

            succState = gameState.generateSuccessor(agent,action)
            score = self.minmax(agent+1,depth,succState)

            if (score > maxScore):
                bestAction = action
                maxScore = score

        if (depth == 0):

            return bestAction
        
        return maxScore
        
class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """
    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        alpha = -float('inf')
        beta = float('inf')
        return self.minmax(0, 0, alpha, beta, gameState)
    
    def minmax(self, agent, depth, alpha, beta, gameState: GameState):
            
            if gameState.isWin() or gameState.isLose() or depth == self.depth:

                #print ("value found: ", self.evaluationFunction(gameState))
                return self.evaluationFunction(gameState)

            if agent == 0:

                return self.maximizer(0, depth, alpha, beta, gameState)
            
            return self.minimizer(agent, depth, alpha, beta, gameState)

    def minimizer(self, agent, depth, alpha, beta, gameState: GameState):

        minScore = float('inf')

        for action in gameState.getLegalActions(agent):
            
            succState = gameState.generateSuccessor(agent, action)

            if agent == gameState.getNumAgents() - 1:

                minScore = min(minScore, self.minmax(0, depth + 1, alpha, beta, succState))  # moving to next agent - pacman

            else:

                minScore = min(minScore, self.minmax(agent + 1, depth, alpha, beta, succState))  # moving to next agent - next ghost
            
            if(minScore < alpha):

                return minScore
            
            beta = min(minScore, beta)

        return minScore


    def maximizer(self, agent, depth, alpha, beta, gameState: GameState):
        
        maxScore = -float('inf')

        bestAction = None

        for action in gameState.getLegalActions():

            succState = gameState.generateSuccessor(agent,action)

            score = self.minmax(agent+1, depth, alpha, beta, succState)

            if(score > maxScore):
                
                bestAction = action
            
            maxScore = max(score,maxScore)

            if (maxScore > beta):

                return maxScore
            
            alpha = max(maxScore, alpha)

        if (depth == 0):

            return bestAction
        
        return maxScore

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
        return self.minmax(0, 0, gameState)
    
    def minmax(self, agent, depth, gameState: GameState):
            
            if gameState.isWin() or gameState.isLose() or depth == self.depth:

                #print ("value found: ", self.evaluationFunction(gameState))
                return self.evaluationFunction(gameState)

            if agent == 0:

                return self.maximizer(0, depth, gameState)
            
            return self.expected(agent, depth, gameState)
    
    def maximizer(self, agent, depth, gameState: GameState):

        maxValue = -float('inf')

        bestAction = None

        for action in gameState.getLegalActions():

            succState = gameState.generateSuccessor(agent,action)

            score = self.minmax(agent+1, depth, succState)

            if(score > maxValue):

                bestAction = action

            maxValue = max(maxValue, score)

        if(depth == 0):

            return bestAction

        return maxValue

    def expected(self, agent, depth, gameState: GameState):

        expValue = 0

        for action in gameState.getLegalActions(agent):
            
            succState = gameState.generateSuccessor(agent, action)

            if agent == gameState.getNumAgents() - 1:
                
                prob = 1/len(gameState.getLegalActions(agent))
                expValue += prob*self.minmax(0, depth+1, succState)

            else:
                prob = 1/len(gameState.getLegalActions(agent))
                expValue += prob*self.minmax(agent+1, depth, succState)

        return expValue
        

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <Honestly I just used the same evaluation function I did for Q1. I deducted value for ghosts based on distance, and added if scaredTimer was active.
        The ghost distance had a much larger impact that the food pellets, and I had to initially tinker with the ghost's score values.>
    """
    "*** YOUR CODE HERE ***"

    score = currentGameState.getScore()
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood().asList()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    nearestFood = 0 if len(newFood) == 0 else min([manhattanDistance(foodPos, newPos) for foodPos in newFood])

    score += 1 / (nearestFood + 1)

    for ghostState in newGhostStates:
        
        ghostPos = ghostState.getPosition()
        distance = manhattanDistance(ghostPos, newPos)

        if(ghostState.scaredTimer > 0):
            score +=  500 / (distance + 1)  
        else:
            score -= 10 / (distance + 1)

    return score

# Abbreviation
better = betterEvaluationFunction
