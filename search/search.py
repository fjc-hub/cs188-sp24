# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util
from game import Directions
from typing import List
from util import Queue, PriorityQueue

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()




def tinyMazeSearch(problem: SearchProblem) -> List[Directions]:
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem: SearchProblem) -> List[Directions]:
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"
    actions = []
    isVis = {}
    def dfs(node):
        nonlocal actions, isVis
        if problem.isGoalState(node):
            return node
        isVis[node] = True
        for lst in problem.getSuccessors(node):
            suc = lst[0]
            act = lst[1]
            if suc in isVis:
                continue
            actions += [act]
            target = dfs(suc)
            if target != None:
                return target
            actions = actions[:len(actions)-1]
        isVis[node] = False
        return None
    
    goal = dfs(problem.getStartState())
    if goal == None: # Goal not found
        return []
    # Goal exists
    return actions


def breadthFirstSearch(problem: SearchProblem) -> List[Directions]:
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    start, goal = problem.getStartState(), None
    isQueued = {start: True}
    edge = {}
    queue = Queue()
    queue.push(start)
    while not queue.isEmpty():
        node = queue.pop()
        if problem.isGoalState(node):
            goal = node
            break
        for lst in problem.getSuccessors(node):
            suc = lst[0]
            act = lst[1]
            if suc in isQueued:
                continue
            isQueued[suc] = True
            if suc not in edge:
                edge[suc] = [node, act]  # record the first preceded node
            queue.push(suc)
    
    if goal is None:
        return []
    # reconstruct optimal path backforward
    ret = []
    while goal in edge:
        ret = [edge[goal][1]] + ret
        goal = edge[goal][0]
    return ret

def uniformCostSearch(problem: SearchProblem) -> List[Directions]:
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    start, goal = problem.getStartState(), None
    # isExpand = {} # record if node has been popped by pq. no need, positive circle path works
    edges = {}
    costs = {start: 0}
    pq = PriorityQueue()
    pq.update(start, 0)
    while not pq.isEmpty():
        node = pq.pop()
        cost = costs[node]
        if problem.isGoalState(node):
            goal = node
            break
        # if node in isExpand:
        #     continue
        # isExpand[node] = True
        for lst in problem.getSuccessors(node):
            succes = lst[0]
            action = lst[1]
            weight = lst[2]
            if (succes not in costs) or (cost + weight < costs[succes]):
                costs[succes] = cost + weight
                edges[succes] = [node, action]  # record the lowest cost preceded node
                pq.update(succes, cost + weight)

    # reconstruct optimal path backforward
    return reconstruct(edges, goal)

# backtrack from end, to construct a path
def reconstruct(edges: dict, end):
    if end is None:
        return []
    ret = []
    while end in edges:
        ret = [edges[end][1]] + ret
        end = edges[end][0]
    return ret

def nullHeuristic(state, problem=None) -> float:
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem: SearchProblem, heuristic=nullHeuristic) -> List[Directions]:
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    start, goal = problem.getStartState(), None
    edges = {}
    costs = {start: 0}
    pq = PriorityQueue()
    pq.update(start, 0)
    while not pq.isEmpty():
        node = pq.pop()
        cost = costs[node]
        if problem.isGoalState(node):
            goal = node
            break
        for lst in problem.getSuccessors(node):
            succes = lst[0]
            action = lst[1]
            weight = lst[2]
            if (succes not in costs) or (cost + weight < costs[succes]):
                costs[succes] = cost + weight
                edges[succes] = [node, action]  # record the lowest cost preceded node
                pq.update(succes, costs[succes] + heuristic(succes, problem))

    # reconstruct optimal path backforward
    return reconstruct(edges, goal)

# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
