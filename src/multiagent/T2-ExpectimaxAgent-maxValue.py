def maxValue(self, gameState, agentIndex, depth):
    v = {'value': float('-inf'), 'action': Directions.STOP}
    legalMoves = gameState.getLegalActions(agentIndex)        

    for action in legalMoves:
        if action == Directions.STOP: continue

        successorGameState = gameState.generateSuccessor(agentIndex, action) 
        successorExpectiMax = self.expectimax(successorGameState, agentIndex+1, depth, action)

        if v['value'] < successorExpectiMax['value']:
            v['value'] = successorExpectiMax['value']
            v['action'] = action

    return v