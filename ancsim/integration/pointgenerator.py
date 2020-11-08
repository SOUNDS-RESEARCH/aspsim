import numpy as np


def selectPointGen(config, randomState=None):
    if config["ARRAYSHAPES"] == "circle":
        return cylinder(config["TARGETWIDTH"], config["TARGETHEIGHT"], randomState)
    elif config["ARRAYSHAPES"] in  ("cuboid", "rectangle", "doublerectangle"):
        return block(
            [config["TARGETWIDTH"], config["TARGETWIDTH"]],
            config["TARGETHEIGHT"],
            randomState,
        )
    else:
        raise NotImplementedError


def getRng(randomState):
    if randomState is None:
        return np.random.RandomState()
    else:
        return randomState


def cylinder(radius, height, randomState=None):
    rng = getRng(randomState)
    totVol = radius ** 2 * height

    def pointGen(numSamples):
        r = np.sqrt(rng.rand(numSamples)) * radius
        angle = rng.rand(numSamples) * 2 * np.pi
        x = r * np.cos(angle)
        y = r * np.sin(angle)
        z = rng.uniform(-height / 2, height / 2, size=numSamples)
        points = np.stack((x, y, z))
        return points

    return pointGen, totVol


def block(rectDims, height, randomState=None):
    rng = getRng(randomState)
    totVol = rectDims[0] * rectDims[1] * height

    def pointGenerator(numSamples):
        x = rng.uniform(-rectDims[0] / 2, rectDims[0] / 2, numSamples)
        y = rng.uniform(-rectDims[1] / 2, rectDims[1] / 2, numSamples)
        z = rng.uniform(-height / 2, height / 2, numSamples)
        points = np.stack((x, y, z))
        return points

    return pointGenerator, totVol
