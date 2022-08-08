import numpy as np

# np.random.seed(1)
import copy

import ancsim.room.generatepoints as gp
import ancsim.array as ar
import ancsim.room.geometry as geo
import ancsim.signal.sources as src

def debug (config):
    arrays = ar.ArrayCollection()
    arrays.add_array(
        ar.FreeSourceArray("source", np.zeros((1,3)), 
            src.Counter(num_channels=1))
    )

    arrays.add_array(
        ar.ControllableSourceArray("loudspeaker", np.zeros((1,3)))
    )

    arrays.add_array(
        ar.MicArray("mic", np.zeros((1,3)))
    )
    propPaths = {"source":{"mic":"isolated"}, "loudspeaker" : {"mic":"none"}}
    return arrays, propPaths


def audioProcessing(config, 
                     numInput=1,
                     numOutput=1):
    if numInput > 1 or numOutput > 1:
        raise NotImplementedError

    arrays = ar.ArrayCollection()
    arrays.add_array(
        ar.FreeSourceArray("source", np.zeros((1,3)), 
            src.SineSource(num_channels=numInput, power=1,freq=100, samplerate=config["samplerate"]))
    )
    arrays.add_array(
        ar.ControllableSourceArray("output", np.zeros((numOutput,3)))
    )
    arrays.add_array(
        ar.MicArray("input", np.zeros((numInput,3)))
    )

    propPaths = {"source":{"input":"isolated"}, "output" : {"input":"none"}}
    return arrays, propPaths

def signalEstimation(config, 
                     numInput=1,
                     numOutput=1):

    arrays = ar.ArrayCollection()
    arrays.add_array(
        ar.FreeSourceArray("source", np.zeros((numInput,3)), 
            #src.SineSource(num_channels=numInput, power=1,freq=100, samplerate=config["samplerate"]))
            src.WhiteNoiseSource(num_channels=numInput, power=1))
    )
    arrays.add_array(
        ar.MicArray("input", np.zeros((numInput,3)))
    )

    arrays.add_array(
        ar.MicArray("desired", np.zeros((numOutput, 3)))
    )
    propPaths = {"source":{"input":"isolated", "desired" : "random"}}

    return arrays, propPaths

def ancMultiPoint(config, 
                    numError=4, 
                    numSpeaker=4, 
                    targetWidth=1.0,
                    targetHeight=0.2,
                    speakerWidth=2.5,
                    speakerHeight=0.2):
    arrays = ar.ArrayCollection()

    arrays.add_array(ar.MicArray("error",
        gp.FourEquidistantRectangles(
            numError,
            targetWidth,
            0.03,
            -targetHeight / 2,
            targetHeight / 2,
    )))

    arrays.add_array(ar.ControllableSourceArray("speaker",
        gp.stackedEquidistantRectangles(
            numSpeaker,
            2,
            [speakerWidth, speakerWidth],
            speakerHeight,
    )))

    source = src.BandlimitedNoiseSource(1, 1, (50,100), config["samplerate"])

    arrays.add_array(ar.FreeSourceArray("source",
        np.array([[-3.5, 0.4, 0.3]], dtype=np.float64), source))
    arrays.add_array(ar.MicArray("ref",
        np.array([[-3.5, 0.4, 0.3]], dtype=np.float64)))

    propPaths = {"speaker":{"ref":"none"}, "source" : {"ref":"isolated"}}

    return arrays, propPaths
