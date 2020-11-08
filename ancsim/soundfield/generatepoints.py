import numpy as np
import ancsim.utilities as util


def concentricalCircles(
    numPoints, numCircles, radius, zDistance=None, angleOffset="distribute"
):
    a = numCircles // 2
    if zDistance is not None:
        zValues = [i * zDistance for i in range(-a, -a + numCircles)]
        if numCircles % 2 == 0:
            zValues = [zVal + zDistance / 2 for zVal in zValues]
    else:
        zValues = [None for i in range(numCircles)]

    pointsPerCircle = [numPoints // numCircles for _ in range(numCircles)]
    for i in range(numPoints - numCircles * (numPoints // numCircles)):
        pointsPerCircle[i] += 1

    if angleOffset == "random":
        startAngles = np.random.rand(numCircles) * np.pi * 2
    elif angleOffset == "same":
        startAngles = np.zeros(numCircles)
    elif angleOffset == "almostSame":
        angleSection = 2 * np.pi / pointsPerCircle[0]
        startAngles = np.random.rand(numCircles) * (angleSection / 10)
    elif angleOffset == "distribute":
        angleSection = 2 * np.pi / pointsPerCircle[0]
        startAngles = np.arange(numCircles) * angleSection / numCircles

    coords = [
        equiangularCircle(circlePoints, radius, angle, zVal)
        for i, (circlePoints, angle, zVal) in enumerate(
            zip(pointsPerCircle, startAngles, zValues)
        )
    ]

    coords = np.concatenate(coords)
    return coords


def equiangularCircle(numPoints, radius, startAngle=0, z=None):
    angleStep = 2 * np.pi / numPoints

    angles = startAngle + np.arange(numPoints) * angleStep
    angles = np.mod(angles, 2 * np.pi)
    radia = np.random.uniform(radius[0], radius[1], size=numPoints)
    [x, y] = util.pol2cart(radia, angles)

    if z is not None:
        coords = np.zeros((numPoints, 3))
        coords[:, 2] = z
    else:
        coords = np.zeros((numPoints, 2))
    coords[:, 0:2] = np.stack((x, y)).T
    return coords


def uniformCylinder(numPoints, radius, height):
    numPlanes = 4
    zVals = np.linspace(-height / 2, height / 2, numPlanes + 2)
    zVals = zVals[1:-1]

    pointsPerPlane = numPoints // numPlanes
    allPoints = np.zeros((pointsPerPlane * numPlanes, 3))

    for n in range(numPlanes):
        xyPoints = sunflowerPattern(
            pointsPerPlane, radius, np.random.rand() * 2 * np.pi
        )
        allPoints[n * pointsPerPlane : (n + 1) * pointsPerPlane :, 0:2] = xyPoints
        allPoints[n * pointsPerPlane : (n + 1) * pointsPerPlane, 2] = zVals[n]
    return allPoints


# translated from user3717023's MATLAB code from stackoverflow
def sunflowerPattern(N, radius, offsetAngle=0):
    phisq = np.square((np.sqrt(5) + 1) / 2)
    # golden ratio
    k = np.arange(1, N + 1)
    r = radius * np.sqrt(k - (1 / 2)) / np.sqrt(N - 1 / 2)
    theta = k * 2 * np.pi / phisq
    [x, y] = util.pol2cart(r, theta)
    return np.stack((x, y)).T


def uniformDisc(pointDistance, radius, zAxis=None):
    lim = (-radius, radius)
    numPoints = int(2 * radius / pointDistance)
    x = np.linspace(lim[0], lim[1], numPoints)
    y = np.linspace(lim[0], lim[1], numPoints)
    [xGrid, yGrid] = np.meshgrid(x, y)
    xGrid = xGrid.flatten()
    yGrid = yGrid.flatten()

    coords = np.vstack((xGrid, yGrid))
    dist = np.linalg.norm(coords, axis=0)
    idxs2 = dist <= radius
    coordsCircle = coords[:, idxs2].T

    if zAxis is not None:
        coordsCircle = np.concatenate(
            (coordsCircle, np.full((coordsCircle.shape[0], 1), zAxis)), axis=-1
        )
    return coordsCircle


def uniformFilledRectangle(numPoints, lim=(-2.4, 2.4), zAxis=None):
    pointsPerAxis = int(np.sqrt(numPoints))
    assert np.isclose(pointsPerAxis ** 2, numPoints)
    if len(lim) == 2:
        x = np.linspace(lim[0], lim[1], pointsPerAxis)
        y = np.linspace(lim[0], lim[1], pointsPerAxis)
    elif len(lim) == 4:
        x = np.linspace(lim[0], lim[2], pointsPerAxis)
        y = np.linspace(lim[1], lim[3], pointsPerAxis)

    [xGrid, yGrid] = np.meshgrid(x, y)
    evalPoints = np.vstack((xGrid.flatten(), yGrid.flatten())).T

    if zAxis is not None:
        evalPoints = np.concatenate(
            (evalPoints, np.full((pointsPerAxis ** 2, 1), zAxis)), axis=-1
        )
    return evalPoints


def uniformFilledCuboid(numPoints, dims, zNumPoints=4):
    pointsPerAxis = int(np.sqrt(numPoints / zNumPoints))
    assert np.isclose(pointsPerAxis ** 2 * zNumPoints, numPoints)
    x = np.linspace(-dims[0] / 2, dims[0] / 2, pointsPerAxis)
    y = np.linspace(-dims[1] / 2, dims[1] / 2, pointsPerAxis)
    z = np.linspace(-dims[2] / 2, dims[2] / 2, zNumPoints)
    [xGrid, yGrid, zGrid] = np.meshgrid(x, y, z)
    evalPoints = np.vstack((xGrid.flatten(), yGrid.flatten(), zGrid.flatten())).T

    return evalPoints


def uniformFilledCuboid_better(numPoints, dims, zNumPoints=4):
    pointsPerAxis = int(np.sqrt(numPoints / zNumPoints))
    assert np.isclose(pointsPerAxis ** 2 * zNumPoints, numPoints)
    x = np.linspace(-dims[0] / 2, dims[0] / 2, 2 * pointsPerAxis + 1)[1::2]
    y = np.linspace(-dims[1] / 2, dims[1] / 2, 2 * pointsPerAxis + 1)[1::2]
    z = np.linspace(-dims[2] / 2, dims[2] / 2, 2 * zNumPoints + 1)[1::2]
    [xGrid, yGrid, zGrid] = np.meshgrid(x, y, z)
    evalPoints = np.vstack((xGrid.flatten(), yGrid.flatten(), zGrid.flatten())).T

    return evalPoints


def FourEquidistantRectangles(
    numPoints, sideLength, sideOffset, zLow, zHigh, offset="distributed"
):
    if offset != "distributed":
        raise NotImplementedError
    points = np.zeros((numPoints, 3))
    points[:, 0:2] = equidistantRectangle(numPoints, (sideLength, sideLength))
    # points[0::2,2] = zLow
    # points[1::2,2] = zHigh

    # idxSet = np.sort(np.concatenate((np.arange(numPoints)[2::4], np.arange(numPoints)[3::4])))
    # for i in idxSet:
    #     if np.isclose(points[i,0], sideLength/2):
    #         points[i,0] += sideOffset
    #     elif np.isclose(points[i,0], -sideLength/2):
    #         points[i,0] -= sideOffset
    #     elif np.isclose(points[i,1], sideLength/2):
    #         points[i,1] += sideOffset
    #     elif np.isclose(points[i,1], -sideLength/2):
    #         points[i,1] -= sideOffset
    #     else:
    #         raise ValueError
    idxSet = np.sort(
        np.concatenate((np.arange(numPoints)[2::4], np.arange(numPoints)[3::4]))
    )
    points[:, 2] = zLow
    points[idxSet, 2] = zHigh
    # points[]
    # points[0::2,2] = zLow
    # points[1::2,2] = zHigh

    for i in np.arange(numPoints)[1::2]:
        if np.isclose(points[i, 0], sideLength / 2):
            points[i, 0] += sideOffset
        elif np.isclose(points[i, 0], -sideLength / 2):
            points[i, 0] -= sideOffset
        elif np.isclose(points[i, 1], sideLength / 2):
            points[i, 1] += sideOffset
        elif np.isclose(points[i, 1], -sideLength / 2):
            points[i, 1] -= sideOffset
        else:
            raise ValueError
    return points


def stackedEquidistantRectangles(
    numPoints, numRect, dims, zDistance, offset="distributed"
):
    a = numRect // 2
    zValues = [i * zDistance for i in range(-a, -a + numRect)]
    if numRect % 2 == 0:
        zValues = [zVal + zDistance / 2 for zVal in zValues]

    pointsPerRect = [numPoints // numRect for _ in range(numRect)]
    for i in range(numPoints - numRect * (numPoints // numRect)):
        pointsPerRect[i] += 1

    offsets = np.linspace(0, 1, 2 * numRect + 1)[1::2]

    points = np.zeros((numPoints, 3))
    idxCount = 0
    for i in range(numRect):
        points[idxCount : idxCount + pointsPerRect[i], 0:2] = equidistantRectangle(
            pointsPerRect[i], dims, offset=offsets[i]
        )
        points[idxCount : idxCount + pointsPerRect[i], 2] = zValues[i]
        idxCount += pointsPerRect[i]
    return points


def equidistantRectangle(numPoints, dims, offset=0.5):
    if numPoints == 0:
        return np.zeros((0, 2))
    totalLength = 2 * (dims[0] + dims[1])
    pointDist = totalLength / numPoints

    points = np.zeros((numPoints, 2))
    if numPoints < 4:
        points = equidistantRectangle(4, dims)
        pointChoices = np.random.choice(4, numPoints, replace=False)
        points = points[pointChoices, :]
    else:
        lengths = [dims[0], dims[1], dims[0], dims[1]]
        xVal = [-dims[0] / 2, dims[0] / 2, dims[0] / 2, -dims[0] / 2]
        yVal = [-dims[1] / 2, -dims[1] / 2, dims[1] / 2, dims[1] / 2]

        startPos = pointDist * offset
        xFac = [1, 0, -1, 0]
        yFac = [0, 1, 0, -1]
        numCounter = 0

        for i in range(4):
            numAxisPoints = 1 + int((lengths[i] - startPos) / pointDist)
            axisPoints = startPos + np.arange(numAxisPoints) * pointDist
            distLeft = lengths[i] - axisPoints[-1]
            points[numCounter : numCounter + numAxisPoints, 0] = (
                xVal[i] + xFac[i] * axisPoints
            )
            points[numCounter : numCounter + numAxisPoints, 1] = (
                yVal[i] + yFac[i] * axisPoints
            )
            numCounter += numAxisPoints
            startPos = pointDist - distLeft

    return points


def stackedEquidistantRectangles_old(numPoints, numRect, dims, zDistance):
    a = numRect // 2
    zValues = [i * zDistance for i in range(-a, a + numRect)]
    if numRect % 2 == 0:
        zValues = [zVal + zDistance / 2 for zVal in zValues]

    pointsPerRect = [numPoints // numRect for _ in range(numRect)]
    for i in range(numPoints - numRect * (numPoints // numRect)):
        pointsPerRect[i] += 1

    points = np.zeros((numPoints, 3))
    for i in range(numRect):
        points[
            i * pointsPerRect[i] : (i + 1) * pointsPerRect[i], 0:2
        ] = equidistantRectangle(pointsPerRect[i], dims)
        points[i * pointsPerRect[i] : (i + 1) * pointsPerRect[i], 2] = zValues[i]
    return points


def equidistantRectangle_old(numPoints, dims):
    totalLength = 2 * (dims[0] + dims[1])
    pointDist = totalLength / numPoints

    points = np.zeros((numPoints, 2))
    if numPoints < 4:
        points = equidistantRectangle(4, dims)
        pointChoices = np.random.choice(4, numPoints, replace=False)
        points = points[pointChoices, :]
    else:
        lengths = [dims[0], dims[1], dims[0], dims[1]]
        xVal = [-dims[0] / 2, dims[0] / 2, dims[0] / 2, -dims[0] / 2]
        yVal = [-dims[1] / 2, -dims[1] / 2, dims[1] / 2, dims[1] / 2]
        startPos = pointDist / 2 + np.clip(np.random.randn() * 0.3, -1, 1) * (
            pointDist / 2
        )
        xFac = [1, 0, -1, 0]
        yFac = [0, 1, 0, -1]
        numCounter = 0

        for i in range(np.min((4, numPoints))):
            numAxisPoints = 1 + int((lengths[i] - startPos) / pointDist)
            axisPoints = startPos + np.arange(numAxisPoints) * pointDist
            distLeft = lengths[i] - axisPoints[-1]
            points[numCounter : numCounter + numAxisPoints, 0] = (
                xVal[i] + xFac[i] * axisPoints
            )
            points[numCounter : numCounter + numAxisPoints, 1] = (
                yVal[i] + yFac[i] * axisPoints
            )
            numCounter += numAxisPoints
            startPos = pointDist - distLeft

    return points


def equidistantRectangle_forfewer(numPoints, dims):
    totalLength = 2 * (dims[0] + dims[1])
    pointDist = totalLength / numPoints

    points = np.zeros((numPoints, 2))
    if numPoints == 1:
        points[0, 0] = np.random.rand() * dims[0] / 4 - dims[0] / 4
        points[0, 1] = np.random.choice([-1, 1]) * dims[1] / 2
    elif numPoints == 2:
        points[:, 0] = np.random.rand(2) * dims[0] / 4 - dims[0] / 4
        points[0, 1] = dims[1] / 2
        points[1, 1] = dims[1] / 2
    elif numPoints == 3:
        points[0:2, 0] = np.random.rand(2) * dims[0] / 4 - dims[0] / 4
        points[0, 1] = dims[1] / 2
        points[1, 1] = dims[1] / 2
        points[2, 0] = np.random.choice([-1, 1]) * dims[0] / 2
        points[2, 1] = np.random.rand() * dims[1] / 4 - dims[1] / 4
    else:
        lengths = [dims[0], dims[1], dims[0], dims[1]]
        xVal = [-dims[0] / 2, dims[0] / 2, dims[0] / 2, -dims[0] / 2]
        yVal = [-dims[1] / 2, -dims[1] / 2, dims[1] / 2, dims[1] / 2]
        startPos = np.random.rand() * pointDist
        xFac = [1, 0, -1, 0]
        yFac = [0, 1, 0, -1]
        numCounter = 0

        for i in range(np.min((4, numPoints))):
            numAxisPoints = 1 + int((lengths[i] - startPos) / pointDist)
            axisPoints = startPos + np.arange(numAxisPoints) * pointDist
            distLeft = lengths[i] - axisPoints[-1]
            points[numCounter : numCounter + numAxisPoints, 0] = (
                xVal[i] + xFac[i] * axisPoints
            )
            points[numCounter : numCounter + numAxisPoints, 1] = (
                yVal[i] + yFac[i] * axisPoints
            )
            numCounter += numAxisPoints
            startPos = pointDist - distLeft

    return points


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    points = stackedEquidistantRectangles(2, 40, [4, 5], 0.4)

    plt.plot(points[0:20, 0], points[0:20, 1], "x")
    plt.plot(points[20:, 0], points[20:, 1], "x")
    plt.show()
