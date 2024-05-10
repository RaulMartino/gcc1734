def minValue(self, gameState, agentIndex, depth, alpha, beta):
    v = {'value':float('inf'), 'action': Directions.STOP}
    legalMoves = gameState.getLegalActions(agentIndex)

    for action in legalMoves:

        if action == Directions.STOP: continue

        successorGameState = gameState.generateSuccessor(agentIndex, action)
        successorMinMax = self.minimax(successorGameState, agentIndex + 1, depth, action, alpha, beta)

        if v['value'] > successorMinMax['value']:
            v['value'] = successorMinMax['value']
            v['action'] = action

        if v['value'] <= alpha: return v

        beta = min(beta, v['value'])

    return v