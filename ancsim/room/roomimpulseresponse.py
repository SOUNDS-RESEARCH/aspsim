import numpy as np
import scipy.spatial.distance as distfuncs
import scipy.special as special
import pyroomacoustics as pra


def tf_point_source_2d(pos_from, pos_to, freq, c):
    assert (pos_from.shape[1] == 2) and (pos_to.shape[1] == 2)
    dist = distfuncs.cdist(pos_from, pos_to)[None, :, :]
    k = 2 * np.pi * freq[:, None, None] / c
    tf = (1j / 4) * special.hankel2(0, k * dist)
    return np.transpose(tf, (0, 2, 1))


def tf_point_source_3d(pos_from, pos_to, freqs, c):
    assert (pos_from.shape[1] == 3) and (pos_to.shape[1] == 3)
    dist = distfuncs.cdist(pos_from, pos_to)[None, :, :]
    k = 2 * np.pi * freqs[:, None, None] / c
    tf = np.exp(-1j * dist * k) / (4 * np.pi * dist)
    return np.moveaxis(tf,1,2)


def ir_point_source_3d(pos_from, pos_to, samplerate, c, max_order=25):
    dist = distfuncs.cdist(pos_from, pos_to)
    sample_delay = (samplerate * dist) / c
    max_filt_len = int(np.max(sample_delay)) + 1 + (max_order // 2) + 1
    ir = np.zeros((dist.shape[0], dist.shape[1], max_filt_len))

    attenuation = lambda d: 1 / (4 * np.pi * np.max((d, 1e-2)))

    for row in range(dist.shape[0]):
        for col in range(dist.shape[1]):
            ir[row, col, :] = _frac_delay_lagrange_filter(
                sample_delay[row, col], max_order, max_filt_len
            )
            ir[row, col, :] *= attenuation(dist[row, col])
    return ir


def ir_point_source_2d(fromPos, toPos, samplerate, c, extraLen=100, maxOrder=25):
    dist = distfuncs.cdist(fromPos, toPos)
    sampleDelay = (samplerate * dist) / c

    delayFiltLen = int(np.max(sampleDelay)) + 1 + (maxOrder // 2) + 1
    maxFiltLen = delayFiltLen + extraLen - 1
    ir = np.zeros((dist.shape[0], dist.shape[1], maxFiltLen))
    for row in range(dist.shape[0]):
        for col in range(dist.shape[1]):
            irFromImpact = 1 / (
                2
                * np.pi
                * np.sqrt(
                    c ** 2
                    * ((sampleDelay[row, col] + 0.5 + np.arange(extraLen)) / samplerate)
                    ** 2
                    - (dist[row, col] ** 2)
                )
            )
            correctDelay = _frac_delay_lagrange_filter(
                sampleDelay[row, col], maxOrder, delayFiltLen
            )
            fullIr = np.convolve(irFromImpact, correctDelay, "full")

            ir[row, col, :] = fullIr
    return ir



def _frac_delay_lagrange_filter(delay, maxOrder, maxFiltLen):
    """Generates fractional delay filter using lagrange interpolation
    if maxOrder would require a longer filter than maxFiltLen,
    maxFiltLen will will override and set a lower interpolation order"""
    ir = np.zeros(maxFiltLen)
    fracdly, dly = np.modf(delay)
    order = int(np.min((maxOrder, 2 * dly, 2 * (maxFiltLen - 1 - dly))))

    delta = np.floor(order / 2) + fracdly
    h = _lagrange_interpol(order, delta)

    diff = delay - delta
    startIdx = int(np.floor(diff))
    ir[startIdx : startIdx + order + 1] = h
    return ir


def _lagrange_interpol(N, delta):
    irLen = int(N + 1)
    h = np.zeros(irLen)

    for n in range(irLen):
        k = np.arange(irLen)
        k = k[np.arange(irLen) != n]

        h[n] = np.prod((delta - k) / (n - k))
    return h





def ir_room_image_source_3d(
    pos_from,
    pos_to,
    room_size,
    room_center,
    ir_len,
    rt60,
    samplerate,
    c, 
    calculate_metadata=False,
    verbose=False,
):

    numFrom = pos_from.shape[0]
    numTo = pos_to.shape[0]
    ir = np.zeros((numFrom, numTo, ir_len))
    room_center = np.array(room_center)
    room_size = np.array(room_size)

    posOffset = room_size / 2 - room_center

    if rt60 > 0:
        eAbsorption, maxOrder = pra.inverse_sabine(rt60, room_size)
        maxOrder += 3
    else:
        eAbsorption = 0.9
        maxOrder = 0
    # print("Energy Absorption: ", eAbsorption)
    # print("Max Order: ", maxOrder)

    pra.constants.set("c", c)

    shortest_distance = np.min(distfuncs.cdist(pos_from, pos_to))
    min_dly = int(np.ceil(shortest_distance * samplerate / c))
    frac_dly_len = 2*min_dly+1
    pra.constants.set("frac_delay_length",frac_dly_len)
    if verbose:
        if frac_dly_len < 20:
            print("WARNING: fractional delay length: ",frac_dly_len)

    maxTruncError = np.NINF
    maxTruncValue = np.NINF
    maxNumIRAtOnce = 500
    numComputed = 0
    while numComputed < numTo:
        room = pra.ShoeBox(
            room_size,
            materials=pra.Material(eAbsorption),
            fs=samplerate,
            max_order=maxOrder,
            use_rand_ism = True, 
            max_rand_disp = 0.05
        )
        # room = pra.ShoeBox(roomSim, materials=pra.Material(e_absorption), fs=sampleRate, max_order=max_order)

        for srcIdx in range(numFrom):
            room.add_source((pos_from[srcIdx, :] + posOffset).T)

        block_size = np.min((maxNumIRAtOnce, numTo - numComputed))
        mics = pra.MicrophoneArray(
            (pos_to[numComputed : numComputed + block_size, :] + posOffset[None, :]).T,
            room.fs,
        )
        room.add_microphone_array(mics)

        if verbose:
            print(
                "Computing RIR {} - {} of {}".format(
                    numComputed * numFrom + 1,
                    (numComputed + block_size) * numFrom,
                    numTo * numFrom,
                )
            )
        room.compute_rir()
        for toIdx, receiver in enumerate(room.rir):
            for fromIdx, singleRIR in enumerate(receiver):
                irLenToUse = np.min((len(singleRIR), ir_len)) - min_dly
                #print(irLenToUse)
                ir[fromIdx, numComputed + toIdx, :irLenToUse] = np.array(singleRIR)[
                    min_dly:irLenToUse+min_dly
                ]
        numComputed += block_size

        # print("RT 60: ", room.measure_rt60())
        if calculate_metadata:
            truncError, truncValue = calc_truncation_info(room.rir, ir_len)
            maxTruncError = np.max((maxTruncError, truncError))
            maxTruncValue = np.max((maxTruncValue, truncValue))

    if calculate_metadata:
        metadata = {}
        metadata["Max Normalized Truncation Error (dB)"] = maxTruncError
        metadata["Max Normalized Truncated Value (dB)"] = maxTruncValue
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
        metadata["Max ISM order"] = maxOrder
        metadata["Energy Absorption"] = eAbsorption
        return ir, metadata
    return ir

    
def calc_truncation_info(allRIR, truncLen):
    maxTruncError = np.NINF
    maxTruncValue = np.NINF
    for toIdx, receiver in enumerate(allRIR):
        for fromIdx, singleRIR in enumerate(receiver):
            irLen = np.min((len(singleRIR), truncLen))
            truncError = calc_truncation_error(singleRIR, irLen)
            truncValue = calc_truncation_value(singleRIR, irLen)
            maxTruncError = np.max((maxTruncError, truncError))
            maxTruncValue = np.max((maxTruncValue, truncValue))
    return maxTruncError, maxTruncValue


def calc_truncation_error(RIR, truncLen):
    totPower = np.sum(RIR ** 2)
    truncPower = np.sum(RIR[truncLen:] ** 2)
    truncError = truncPower / totPower
    return 10 * np.log10(truncError)


def calc_truncation_value(RIR, truncLen):
    if len(RIR) <= truncLen:
        return -np.inf
    maxTruncValue = np.max(np.abs(RIR[truncLen:]))
    maxValue = np.max(np.abs(RIR))
    return 20 * np.log10(maxTruncValue / maxValue)


def show_rt60(multiChannelIR):
    for i in range(multiChannelIR.shape[0]):
        for j in range(multiChannelIR.shape[1]):
            singleIR = multiChannelIR[i, j, :]
            rt = pra.experimental.rt60.measure_rt60(singleIR)
            print("rt60 is: ", rt)
