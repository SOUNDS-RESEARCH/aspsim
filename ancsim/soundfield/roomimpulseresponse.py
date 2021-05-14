import numpy as np
import scipy.spatial.distance as distfuncs
import scipy.special as special
import pyroomacoustics as pra

import ancsim.signal.filterdesign as fd
import ancsim.utilities as util


def tfPointSource2d(fromPos, toPos, freq, c):
    assert (fromPos.shape[1] == 2) and (toPos.shape[1] == 2)
    dist = distfuncs.cdist(fromPos, toPos)
    k = 2 * np.pi * freq / c

    tf = (1j / 4) * special.hankel2(0, k * dist)
    return np.transpose(tf, (0, 2, 1))


def tfPointSource3d(fromPos, toPos, freqs, c):
    assert (fromPos.shape[1] == 3) and (toPos.shape[1] == 3)
    dist = distfuncs.cdist(fromPos, toPos)[None, :, :]
    k = freqs[:, None, None] * 2 * np.pi / c
    tf = np.exp(-1j * dist * k) / (4 * np.pi * dist)
    return tf


def irPointSource3d(fromPos, toPos, samplerate, c, maxOrder=25):
    dist = distfuncs.cdist(fromPos, toPos)
    sampleDelay = (samplerate * dist) / c
    maxFiltLen = int(np.max(sampleDelay)) + 1 + (maxOrder // 2) + 1
    ir = np.zeros((dist.shape[0], dist.shape[1], maxFiltLen))

    attenuation = lambda d: 1 / (4 * np.pi * np.max((d, 1e-2)))

    for row in range(dist.shape[0]):
        for col in range(dist.shape[1]):
            ir[row, col, :] = fd.fracDelayLagrangeFilter(
                sampleDelay[row, col], maxOrder, maxFiltLen
            )
            ir[row, col, :] *= attenuation(dist[row, col])
    return ir


def irPointSource2d(fromPos, toPos, samplerate, c, extraLen=100, maxOrder=25):
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
            correctDelay = fd.fracDelayLagrangeFilter(
                sampleDelay[row, col], maxOrder, delayFiltLen
            )
            fullIr = np.convolve(irFromImpact, correctDelay, "full")

            ir[row, col, :] = fullIr
    return ir

def irRoomImageSource3d(
    fromPos,
    toPos,
    roomSize,
    roomCenter,
    irLen,
    rt60,
    samplerate,
    c, 
    calculateMetadata=False,
):

    numFrom = fromPos.shape[0]
    numTo = toPos.shape[0]
    ir = np.zeros((numFrom, numTo, irLen))
    roomCenter = np.array(roomCenter)
    roomSize = np.array(roomSize)

    posOffset = roomSize / 2 - roomCenter

    if rt60 > 0:
        eAbsorption, maxOrder = pra.inverse_sabine(rt60, roomSize)
        maxOrder += 10
    else:
        eAbsorption = 0.9
        maxOrder = 0
    # print("Energy Absorption: ", eAbsorption)
    # print("Max Order: ", maxOrder)

    pra.constants.set("c", c)

    shortest_distance = np.min(distfuncs.cdist(fromPos, toPos))
    min_dly = int(np.ceil(shortest_distance * samplerate / c))
    frac_dly_len = 2*min_dly+1
    pra.constants.set("frac_delay_length",frac_dly_len)
    if frac_dly_len < 20:
        print("WARNING: fractional delay length: ",frac_dly_len)

    maxTruncError = np.NINF
    maxTruncValue = np.NINF
    maxNumIRAtOnce = 500
    numComputed = 0
    while numComputed < numTo:
        room = pra.ShoeBox(
            roomSize,
            materials=pra.Material(eAbsorption),
            fs=samplerate,
            max_order=maxOrder,
        )
        # room = pra.ShoeBox(roomSim, materials=pra.Material(e_absorption), fs=sampleRate, max_order=max_order)

        for srcIdx in range(numFrom):
            room.add_source((fromPos[srcIdx, :] + posOffset).T)

        blockSize = np.min((maxNumIRAtOnce, numTo - numComputed))
        mics = pra.MicrophoneArray(
            (toPos[numComputed : numComputed + blockSize, :] + posOffset[None, :]).T,
            room.fs,
        )
        room.add_microphone_array(mics)

        print(
            "Computing RIR {} - {} of {}".format(
                numComputed * numFrom + 1,
                (numComputed + blockSize) * numFrom,
                numTo * numFrom,
            )
        )
        room.compute_rir()
        for toIdx, receiver in enumerate(room.rir):
            for fromIdx, singleRIR in enumerate(receiver):
                irLenToUse = np.min((len(singleRIR), irLen)) - min_dly
                #print(irLenToUse)
                ir[fromIdx, numComputed + toIdx, :irLenToUse] = np.array(singleRIR)[
                    min_dly:irLenToUse+min_dly
                ]
        numComputed += blockSize

        # print("RT 60: ", room.measure_rt60())
        if calculateMetadata:
            truncError, truncValue = calculateTruncationInfo(room.rir, irLen)
            maxTruncError = np.max((maxTruncError, truncError))
            maxTruncValue = np.max((maxTruncValue, truncValue))

    if calculateMetadata:
        metadata = {}
        metadata["Max Normalized Truncation Error (dB)"] = maxTruncError
        metadata["Max Normalized Truncated Value (dB)"] = maxTruncValue
        metadata["Measured RT60 (min)"] = np.min(room.measure_rt60())
        metadata["Measured RT60 (max)"] = np.max(room.measure_rt60())
        metadata["Max ISM order"] = maxOrder
        metadata["Energy Absorption"] = eAbsorption
        return ir, metadata
    return ir

    
def calculateTruncationInfo(allRIR, truncLen):
    maxTruncError = np.NINF
    maxTruncValue = np.NINF
    for toIdx, receiver in enumerate(allRIR):
        for fromIdx, singleRIR in enumerate(receiver):
            irLen = np.min((len(singleRIR), truncLen))
            truncError = calculateTruncationError(singleRIR, irLen)
            truncValue = calculateTruncationValue(singleRIR, irLen)
            maxTruncError = np.max((maxTruncError, truncError))
            maxTruncValue = np.max((maxTruncValue, truncValue))
    return maxTruncError, maxTruncValue


def calculateTruncationError(RIR, truncLen):
    totPower = np.sum(RIR ** 2)
    truncPower = np.sum(RIR[truncLen:] ** 2)
    truncError = truncPower / totPower
    return 10 * np.log10(truncError)


def calculateTruncationValue(RIR, truncLen):
    maxTruncValue = np.max(np.abs(RIR[truncLen:]))
    maxValue = np.max(np.abs(RIR))
    return 20 * np.log10(maxTruncValue / maxValue)


def irSimulatedRoom3d_matlab(fromPos, toPos, samplerate, c, irLen=1024, reverbTime=0.4):
    eng = matlab.engine.start_matlab()
    roomDim = [6, 5, 4]  # Room dimensions [x y z] (m)
    center = np.array([x / 2 for x in roomDim], dtype=np.float64)

    roomFromPos = fromPos + center[None, :]
    roomToPos = toPos + center[None, :]

    mtype = "omnidirectional"  # Type of microphone
    order = -1  # -1 equals maximum reflection order!
    numDim = 3  # Room dimension
    orientation = 0  # Microphone orientation (rad)
    hp_filter = 1  # Enable high-pass filter

    print("Computing Room IR...")
    ir = np.array(
        eng.rir_gen_wrapper(
            c,
            matlab.double([samplerate]),
            matlab.double(roomToPos.tolist()),
            matlab.double(roomFromPos.tolist()),
            matlab.double([roomDim]),
            reverbTime,
            irLen,
            mtype,
            order,
            numDim,
            orientation,
            hp_filter,
        )
    )
    return ir


def showrt60(multiChannelIR):
    for i in range(multiChannelIR.shape[0]):
        for j in range(multiChannelIR.shape[1]):
            singleIR = multiChannelIR[i, j, :]
            rt = pra.experimental.rt60.measure_rt60(singleIR)
            print("rt60 is: ", rt)
