import numpy as np
import scipy.spatial.distance as distfuncs
import pyroomacoustics as pra




def ir_room_image_source_3d(
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
