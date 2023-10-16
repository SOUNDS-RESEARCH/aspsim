import numpy as np
import scipy.spatial.distance as distfuncs
import pyroomacoustics as pra

class PathGenerator:
    def __init__(self, sim_info, arrays):
        self.sim_info = sim_info
        self.arrays = arrays

        self.calc_rir_parameters(self.sim_info)

    def calc_rir_parameters(self, sim_info):
        pra.constants.set("c", sim_info.c)

        room_center = np.array(sim_info.room_center)
        room_size = np.array(sim_info.room_size)

        rt60 = sim_info.rt60
        if rt60 > 0:
            self.e_absorbtion, self.max_order = pra.inverse_sabine(rt60, room_size)
            self.max_order += 8
        else:
            self.e_absorbtion = 0.9
            self.max_order = 0

        shortest_distance = np.inf
        for src, mic in self.arrays.mic_src_combos():
            shortest_distance = np.min((shortest_distance, np.min(distfuncs.cdist(src.pos, mic.pos))))
       # shortest_distance = np.min(distfuncs.cdist(pos_from, pos_to))
        #min_dly = int(np.ceil(shortest_distance * samplerate / c))
        self.min_dly = 0
        frac_dly_len = 2*(self.min_dly + sim_info.extra_delay) + 1
        pra.constants.set("frac_delay_length",frac_dly_len)
        #if verbose:
        if frac_dly_len < 20:
            print("WARNING: fractional delay length: ",frac_dly_len)



    def create_path (self, src, mic, reverb, sim_info, return_path_info=False, verbose=False):
        """Generate the impulse response between a source and a microphone array
        
        Parameters
        ----------
        src : Array 
            The source array
        mic : Array
            The microphone array
        reverb : str
            The type of propagation between the source and the microphone array
            Possible values are "none", "identity", "isolated", "random", "ism", "freespace"
        sim_info : SimInfo
            The simulation info object
        return_path_info : bool, optional
            If True, the method will return a tuple of the path and a dict containing metadata
            about the path. The default is False.
        verbose : bool, optional
            If True, the method will print information about the path generation. The default is False.
        """
        path_info = {}
        if reverb == "none": 
            path = np.zeros((src.num, mic.num, 1))
        elif reverb == "direct":
            assert src.num == mic.num
            path = np.eye(src.num, mic.num)[...,None]
        elif reverb == "random":
            path = np.random.normal(size=(src.num, mic.num, sim_info.max_room_ir_length))
        elif reverb == "ism":
            if sim_info.spatial_dims == 3:
                path = ir_room_image_source_3d(
                        src.pos, 
                        mic.pos, 
                        sim_info.room_size, 
                        sim_info.room_center, 
                        sim_info.max_room_ir_length, 
                        sim_info.rt60, 
                        sim_info.samplerate, 
                        self.e_absorbtion, 
                        self.max_order, 
                        self.min_dly,
                        randomized_ism = sim_info.randomized_ism,
                        calculate_metadata=return_path_info,
                        verbose = verbose)
                if return_path_info:
                    path, path_info["ism_info"] = path
            else:
                raise ValueError
        #elif reverb == "modified":
        #    pass
        else:
            raise ValueError
        if return_path_info:
            return path, path_info
        return path





def ir_room_image_source_3d(
    pos_from,
    pos_to,
    room_size,
    room_center,
    ir_len,
    rt60,
    samplerate,
    e_absorbtion,
    max_order,
    min_dly, 
    randomized_ism = True,
    calculate_metadata=False,
    verbose=False,
    #extra_delay = 0, # this is multiplied by two since frac_dly must be even
):

    num_from = pos_from.shape[0]
    num_to = pos_to.shape[0]
    ir = np.zeros((num_from, num_to, ir_len))

    room_center = np.array(room_center)
    room_size = np.array(room_size)
    pos_offset = room_size / 2 - room_center

    max_trunc_error = np.NINF
    max_trunc_value = np.NINF
    max_num_ir_at_once = 500
    num_computed = 0
    while num_computed < num_to:
        room = pra.ShoeBox(
            room_size,
            materials=pra.Material(e_absorbtion),
            fs=samplerate,
            max_order=max_order,
            use_rand_ism = randomized_ism, 
            max_rand_disp = 0.05
        )

        for src_idx in range(num_from):
            room.add_source((pos_from[src_idx, :] + pos_offset).T)

        block_size = np.min((max_num_ir_at_once, num_to - num_computed))
        mics = pra.MicrophoneArray(
            (pos_to[num_computed : num_computed + block_size, :] + pos_offset[None, :]).T,
            samplerate,
        )
        room.add_microphone_array(mics)

        if verbose:
            print(
                "Computing RIR {} - {} of {}".format(
                    num_computed * num_from + 1,
                    (num_computed + block_size) * num_from,
                    num_to * num_from,
                )
            )
        room.compute_rir()
        for to_idx, receiver in enumerate(room.rir):
            for from_idx, single_rir in enumerate(receiver):
                ir_len_to_use = np.min((len(single_rir), ir_len)) - min_dly
                ir[from_idx, num_computed + to_idx, :ir_len_to_use] = np.array(single_rir)[
                    min_dly:ir_len_to_use+min_dly
                ]
        num_computed += block_size

        if calculate_metadata:
            truncError, truncValue = calc_truncation_info(room.rir, ir_len)
            max_trunc_error = np.max((max_trunc_error, truncError))
            max_trunc_value = np.max((max_trunc_value, truncValue))

    if calculate_metadata:
        metadata = {}
        metadata["Max Normalized Truncation Error (dB)"] = max_trunc_error
        metadata["Max Normalized Truncated Value (dB)"] = max_trunc_value
        if rt60 > 0:
            try:
                metadata["Measured RT60 (min)"] = np.min(room.measure_rt60())
                metadata["Measured RT60 (max)"] = np.max(room.measure_rt60())
            except ValueError:
                metadata["Measured RT60 (min)"] = "failed"
                metadata["Measured RT60 (max)"] = "failed"
        else:
            metadata["Measured RT60 (min)"] = 0
            metadata["Measured RT60 (max)"] = 0
        metadata["Max ISM order"] = max_order
        metadata["Energy Absorption"] = e_absorbtion
        return ir, metadata
    return ir




def ir_room_image_source_3d_orig(
    pos_from,
    pos_to,
    room_size,
    room_center,
    ir_len,
    rt60,
    samplerate,
    c,
    randomized_ism = True,
    calculate_metadata=False,
    verbose=False,
    extra_delay = 0, # this is multiplied by two since frac_dly must be even
):

    num_from = pos_from.shape[0]
    num_to = pos_to.shape[0]
    ir = np.zeros((num_from, num_to, ir_len))
    room_center = np.array(room_center)
    room_size = np.array(room_size)

    pos_offset = room_size / 2 - room_center

    if rt60 > 0:
        e_absorbtion, max_order = pra.inverse_sabine(rt60, room_size)
        max_order += 8
    else:
        e_absorbtion = 0.9
        max_order = 0
    # print("Energy Absorption: ", eAbsorption)
    # print("Max Order: ", maxOrder)

    pra.constants.set("c", c)

    shortest_distance = np.min(distfuncs.cdist(pos_from, pos_to))
    #min_dly = int(np.ceil(shortest_distance * samplerate / c))
    min_dly = 0
    frac_dly_len = 2*(min_dly + extra_delay) + 1
    pra.constants.set("frac_delay_length",frac_dly_len)
    if verbose:
        if frac_dly_len < 20:
            print("WARNING: fractional delay length: ",frac_dly_len)

    max_trunc_error = np.NINF
    max_trunc_value = np.NINF
    max_num_ir_at_once = 500
    num_computed = 0
    while num_computed < num_to:
        room = pra.ShoeBox(
            room_size,
            materials=pra.Material(e_absorbtion),
            fs=samplerate,
            max_order=max_order,
            use_rand_ism = randomized_ism, 
            max_rand_disp = 0.05
        )
        # room = pra.ShoeBox(roomSim, materials=pra.Material(e_absorption), fs=sampleRate, max_order=max_order)

        for src_idx in range(num_from):
            room.add_source((pos_from[src_idx, :] + pos_offset).T)

        block_size = np.min((max_num_ir_at_once, num_to - num_computed))
        mics = pra.MicrophoneArray(
            (pos_to[num_computed : num_computed + block_size, :] + pos_offset[None, :]).T,
            room.fs,
        )
        room.add_microphone_array(mics)

        if verbose:
            print(
                "Computing RIR {} - {} of {}".format(
                    num_computed * num_from + 1,
                    (num_computed + block_size) * num_from,
                    num_to * num_from,
                )
            )
        room.compute_rir()
        for to_idx, receiver in enumerate(room.rir):
            for from_idx, single_rir in enumerate(receiver):
                ir_len_to_use = np.min((len(single_rir), ir_len)) - min_dly
                ir[from_idx, num_computed + to_idx, :ir_len_to_use] = np.array(single_rir)[
                    min_dly:ir_len_to_use+min_dly
                ]
        num_computed += block_size

        if calculate_metadata:
            truncError, truncValue = calc_truncation_info(room.rir, ir_len)
            max_trunc_error = np.max((max_trunc_error, truncError))
            max_trunc_value = np.max((max_trunc_value, truncValue))

    if calculate_metadata:
        metadata = {}
        metadata["Max Normalized Truncation Error (dB)"] = max_trunc_error
        metadata["Max Normalized Truncated Value (dB)"] = max_trunc_value
        if rt60 > 0:
            try:
                metadata["Measured RT60 (min)"] = np.min(room.measure_rt60())
                metadata["Measured RT60 (max)"] = np.max(room.measure_rt60())
            except ValueError:
                metadata["Measured RT60 (min)"] = "failed"
                metadata["Measured RT60 (max)"] = "failed"
        else:
            metadata["Measured RT60 (min)"] = 0
            metadata["Measured RT60 (max)"] = 0
        metadata["Max ISM order"] = max_order
        metadata["Energy Absorption"] = e_absorbtion
        return ir, metadata
    return ir

    
def calc_truncation_info(all_rir, trunc_len):
    max_trunc_error = np.NINF
    max_trunc_value = np.NINF
    for to_idx, receiver in enumerate(all_rir):
        for from_idx, single_rir in enumerate(receiver):
            ir_len = np.min((len(single_rir), trunc_len))
            trunc_error = calc_truncation_error(single_rir, ir_len)
            trunc_value = calc_truncation_value(single_rir, ir_len)
            max_trunc_error = np.max((max_trunc_error, trunc_error))
            max_trunc_value = np.max((max_trunc_value, trunc_value))
    return max_trunc_error, max_trunc_value


def calc_truncation_error(rir, trunc_len):
    tot_power = np.sum(rir ** 2)
    trunc_power = np.sum(rir[trunc_len:] ** 2)
    trunc_error = trunc_power / tot_power
    return 10 * np.log10(trunc_error)


def calc_truncation_value(rir, trunc_len):
    if len(rir) <= trunc_len:
        return -np.inf
    max_trunc_value = np.max(np.abs(rir[trunc_len:]))
    max_value = np.max(np.abs(rir))
    return 20 * np.log10(max_trunc_value / max_value)


def show_rt60(multi_channel_ir):
    for i in range(multi_channel_ir.shape[0]):
        for j in range(multi_channel_ir.shape[1]):
            single_ir = multi_channel_ir[i, j, :]
            rt = pra.experimental.rt60.measure_rt60(single_ir)
            print("rt60 is: ", rt)
