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


from util import manhattanDistance, PriorityQueue
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
        oldPos = currentGameState.getPacmanPosition()
        oldFood = currentGameState.getFood()
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        oldScore = currentGameState.getScore() # current score
        newScore = successorGameState.getScore() # new score
        if newScore > oldScore:
            return newScore
        # if pacman is approaching the closest food
        oldMinDist = min([manhattanDistance(oldPos, foodpos) for foodpos in oldFood.asList()])
        newMinDist = min([manhattanDistance(newPos, foodpos) for foodpos in newFood.asList()])
        return newScore + (oldMinDist - newMinDist)

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
        # Simplified version of minimax algorithm
        # def minimax(state, agentNum, depth, evalFn, agentIdx):
        #     if depth == 0 or state.isWin() or state.isLose():
        #         return evalFn(state), None
        #     isMax = agentIdx == 0
        #     value = float('-inf') if isMax else float('inf')
        #     action = None
        #     for act in state.getLegalActions(agentIdx):
        #         nextState = state.generateSuccessor(agentIdx, act) # state and nextState have same agent list
        #         tmp, _ = minimax(nextState, agentNum, depth-((agentIdx+1) // agentNum), evalFn, (agentIdx+1) % agentNum)
        #         if isMax:
        #             if tmp > value:
        #                 value, action = tmp, act
        #         else:
        #             if tmp < value:
        #                 value, action = tmp, act
        #     return value, action

        # 
        # agentIdx: 0=maximizer, else=minimizer
        def minimax(state, agentNum, depth, evalFn, agentIdx):
            if depth == 0 or state.isWin() or state.isLose():
                return evalFn(state), None
            
            value, action = 0, None
            if agentIdx == 0:
                # maximizer
                value = float('-inf')
                for act in state.getLegalActions(agentIdx):
                    nextState = state.generateSuccessor(agentIdx, act) # state and nextState have same agent list
                    tmp, _ = minimax(nextState, agentNum, depth, evalFn, agentIdx+1)
                    if tmp > value:
                        value, action = tmp, act
            else:
                # minimizer
                value = float('inf')
                for act in state.getLegalActions(agentIdx):
                    nextState = state.generateSuccessor(agentIdx, act) # state and nextState have same agent list
                    tmp = 0
                    if agentIdx == agentNum - 1:
                        tmp, _ = minimax(nextState, agentNum, depth-1, evalFn, 0)
                    else:
                        tmp, _ = minimax(nextState, agentNum, depth, evalFn, agentIdx+1)
                    if tmp < value:
                        value, action = tmp, act
            return value, action
        
        _, ret = minimax(gameState, gameState.getNumAgents(), self.depth, self.evaluationFunction, 0)
        return ret


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        # alpha: max value of maximizer on the path
        # beta:  min value of maximizer on the path
        def minimax(state, agentNum, depth, evalFn, agentIdx, alpha, beta):
            if depth == 0 or state.isWin() or state.isLose():
                return evalFn(state), None
            
            isMax = agentIdx == 0
            value = float('-inf') if isMax else float('inf')
            action = None
            for act in state.getLegalActions(agentIdx):
                nextState = state.generateSuccessor(agentIdx, act) # state and nextState have same agent list
                tmp, _ = minimax(nextState, agentNum, depth-((agentIdx+1) // agentNum), evalFn, (agentIdx+1) % agentNum, alpha, beta)
                if isMax:
                    if tmp > value:
                        value, action = tmp, act
                        alpha = max(alpha, value)
                    if tmp > beta: 
                        # maximizer's optimal value >= tmp ==>> maximizer cannot be chosen by minimizer on upper level
                        break
                else:
                    if tmp < value:
                        value, action = tmp, act
                        beta = min(beta, value)
                    if tmp < alpha:
                        # minimizer's optimal value must less or equal than tmp ==>> minimizer cannot be chosen
                        break
            return value, action
        
        _, ret = minimax(gameState, gameState.getNumAgents(), self.depth, self.evaluationFunction, 0, float("-inf"), float("inf"))
        return ret

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
        def expectimax(state, agentNum, depth, evalFn, agentIdx):
            if depth == 0 or state.isWin() or state.isLose():
                return evalFn(state), None
            
            value, action = 0, None
            if agentIdx == 0:
                # maximizer
                value = float('-inf')
                for act in state.getLegalActions(agentIdx):
                    nextState = state.generateSuccessor(agentIdx, act) # state and nextState have same agent list
                    tmp, _ = expectimax(nextState, agentNum, depth, evalFn, agentIdx+1)
                    if tmp > value:
                        value, action = tmp, act
            else:
                # random adversary
                children = []
                for act in state.getLegalActions(agentIdx):
                    nextState = state.generateSuccessor(agentIdx, act) # state and nextState have same agent list
                    tmp, _ = expectimax(nextState, agentNum, depth-((agentIdx+1) // agentNum), evalFn, (agentIdx+1) % agentNum)
                    children.append((tmp, act))
                value = sum(child[0] for child in children) / len(children)
                # all actions of random adversary has the same weight
                action = None  # Action is irrelevant for the random adversary
            return value, action
        
        _, ret = expectimax(gameState, gameState.getNumAgents(), self.depth, self.evaluationFunction, 0)
        return ret

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    return betterEvaluationFunction1(currentGameState)
    # return betterEvaluationFunction2(currentGameState)

def betterEvaluationFunction1(currentGameState: GameState):
    score = currentGameState.getScore()
    pos = currentGameState.getPacmanPosition()
    foods = currentGameState.getFood().asList()
    ghostStates = currentGameState.getGhostStates()
    scaredTimes = [ghostState.scaredTimer for ghostState in ghostStates]

    # Calculate the distance to the closest food
    foodDistances = [manhattanDistance(pos, food) for food in foods]
    closestFoodDist = min(foodDistances) if foodDistances else 0

    # Calculate the distance to the ghosts
    ghostDistances = [manhattanDistance(pos, ghost.getPosition()) for ghost in ghostStates]

    # Avoid ghosts unless they are scared
    ghostPenalty = 0
    for i, dist in enumerate(ghostDistances):
        if scaredTimes[i] == 0 and dist < 2:  # Penalize being too close to active ghosts
            ghostPenalty -= 100 / (dist + 1)

    # Reward eating scared ghosts
    scaredGhostReward = 0
    for i, dist in enumerate(ghostDistances):
        if scaredTimes[i] > 0:  # Reward approaching scared ghosts
            scaredGhostReward += 200 / (dist + 1)

    # Combine the features into a single evaluation score
    return score - closestFoodDist + ghostPenalty + scaredGhostReward

def betterEvaluationFunction2(currentGameState: GameState):
    score = currentGameState.getScore()
    pos = currentGameState.getPacmanPosition()
    foods = currentGameState.getFood().asList()
    ghostStates = currentGameState.getGhostStates()
    scaredTimes = [ghostState.scaredTimer for ghostState in ghostStates]

    # MST
    def lazyPrim(nodes):
        if not nodes:
            return 0
        total = 0
        sz = len(nodes)             
        mst = set()                 
        pq = PriorityQueue()        
        pq.push((nodes[0], 0), 0)   
        while len(mst) < sz:
            node, cost = pq.pop()
            if node in mst:
                continue
            mst.add(node)
            total += cost
            for next in nodes:
                if next in mst:
                    continue
                cost = manhattanDistance(node, next)
                pq.push((next, cost), cost)
        return total
    
    if not foods:
        return score
    
    if currentGameState.isWin():
        return 999999
    if currentGameState.isLose():
        return -999999

    mstCost = lazyPrim(foods)
    closestFoodDist = min(manhattanDistance(pos, food) for food in foods)

    # Avoid ghosts unless they are scared
    ghostPenalty = 0
    ghostDistances = [manhattanDistance(pos, ghost.getPosition()) for ghost in ghostStates]
    for i, dist in enumerate(ghostDistances):
        if scaredTimes[i] == 0 and dist < 2:  # Penalize being too close to active ghosts
            ghostPenalty -= 100 / (dist + 1)

    # Reward eating scared ghosts
    scaredGhostReward = 0
    for i, dist in enumerate(ghostDistances):
        if scaredTimes[i] > 0:  # Reward approaching scared ghosts
            scaredGhostReward += 200 / (dist + 1)

    return score - closestFoodDist - mstCost + ghostPenalty + scaredGhostReward

# Abbreviation
better = betterEvaluationFunction
