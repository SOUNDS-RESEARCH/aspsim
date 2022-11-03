import numpy as np

# np.random.seed(1)
import copy

import ancsim.room.generatepoints as gp
import ancsim.array as ar
import ancsim.room.region as reg
import ancsim.signal.sources as src

def debug (sim_info):
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
    prop_paths = {"source":{"mic":"isolated"}, "loudspeaker" : {"mic":"none"}}
    return arrays, prop_paths


def audio_processing(sim_info, 
                     num_input=1,
                     num_output=1):
    if num_input > 1 or num_output > 1:
        raise NotImplementedError

    arrays = ar.ArrayCollection()
    arrays.add_array(
        ar.FreeSourceArray("source", np.zeros((1,3)), 
            src.SineSource(num_channels=num_input, power=1,freq=100, samplerate=sim_info.samplerate))
    )
    arrays.add_array(
        ar.ControllableSourceArray("output", np.zeros((num_output,3)))
    )
    arrays.add_array(
        ar.MicArray("input", np.zeros((num_input,3)))
    )

    prop_paths = {"source":{"input":"isolated"}, "output" : {"input":"none"}}
    return arrays, prop_paths

def signal_estimation(sim_info, 
                     num_input=1,
                     num_output=1):

    arrays = ar.ArrayCollection()
    arrays.add_array(
        ar.FreeSourceArray("source", np.zeros((num_input,3)), 
            #src.SineSource(num_channels=numInput, power=1,freq=100, samplerate=config["samplerate"]))
            src.WhiteNoiseSource(num_channels=num_input, power=1))
    )
    arrays.add_array(
        ar.MicArray("input", np.zeros((num_input,3)))
    )

    arrays.add_array(
        ar.MicArray("desired", np.zeros((num_output, 3)))
    )
    prop_paths = {"source":{"input":"isolated", "desired" : "random"}}

    return arrays, prop_paths

def anc_multi_point(sim_info, 
                    num_error=4, 
                    num_speaker=4, 
                    target_width=1.0,
                    target_height=0.2,
                    speaker_width=2.5,
                    speaker_height=0.2):
    arrays = ar.ArrayCollection()

    arrays.add_array(ar.MicArray("error",
        gp.four_equidistant_rectangles(
            num_error,
            target_width,
            0.03,
            -target_height / 2,
            target_height / 2,
    )))

    arrays.add_array(ar.ControllableSourceArray("speaker",
        gp.stacked_equidistant_rectangles(
            num_speaker,
            2,
            [speaker_width, speaker_width],
            speaker_height,
    )))

    source = src.BandlimitedNoiseSource(1, 1, (50,100), sim_info.samplerate)

    arrays.add_array(ar.FreeSourceArray("source",
        np.array([[-3.5, 0.4, 0.3]], dtype=np.float64), source))
    arrays.add_array(ar.MicArray("ref",
        np.array([[-3.5, 0.4, 0.3]], dtype=np.float64)))

    prop_paths = {"speaker":{"ref":"none"}, "source" : {"ref":"isolated"}}

    return arrays, prop_paths
