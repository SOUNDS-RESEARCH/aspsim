import numpy as np
import itertools as it


def getCenterOfCircle(edgePoints, maxTolerance=np.inf):
    numPoints = edgePoints.shape[0]
    pointsInterval = numPoints // 3
    pointIdx = [0, pointsInterval, 2*pointsInterval]
    
    chosenPoints = edgePoints[pointIdx,0:2]
    #print(chosenPoints)
    x = chosenPoints[:,0]
    y = chosenPoints[:,1]

    deltaA, mA = lineInfo(chosenPoints[0,:], chosenPoints[1,:])
    deltaB, mB = lineInfo(chosenPoints[1,:], chosenPoints[2,:])

    xCenter = (mA*mB*(y[0]-y[2]) + mB*(x[0]+x[1]) - mA*(x[1]+x[2])) / (2*(mB-mA))
    yCenter = (1/mA) * (xCenter - (x[0] + x[1])/2) + (y[0] + y[1])/2
    center = np.array([[xCenter, yCenter]])

    radia = np.sqrt(np.sum((edgePoints - center)**2, axis=-1))
    radius = np.mean(radia)
    maxDifference = 0
    for i,j in it.combinations(range(numPoints), 2):
        dif = np.abs(radia[i] - radia[j])
        if dif > maxDifference:
            maxDifference = dif

    if maxDifference < maxTolerance:
        return center, radius
    raise ValueError

def lineInfo (point1, point2):
    delta = point2 - point1
    print(delta)
    m = delta[1] / delta[0]
    #print(m)
    return delta, m



# deltaPos = np.zeros((,2))

# for i,j in it.combinations(range(numPoints), 2):
#       deltaPos = edgePoints[i,:] - edgePoints[j,:]