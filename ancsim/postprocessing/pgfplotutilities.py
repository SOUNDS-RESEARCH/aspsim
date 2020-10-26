import numpy as np
import scipy.signal as spsig
import ancsim.utilities as util


def loadPGFPlotData(filePath):
    return np.genfromtxt(filePath)


def savePGFPlotData(filePath, data, suffix="_processed", fmt="%f %f"):
    np.savetxt(
        filePath.parent.joinpath(filePath.stem + suffix + filePath.suffix),
        data,
        fmt=fmt,
    )


def findColumnTypes(data, axis=1):
    """Only finds int or float"""
    assert data.ndim == 2
    numTypes = data.shape[axis]

    types = []
    for i in range(numTypes):
        if np.allclose(np.mod(data.take(indices=i, axis=axis), 1), 0):
            types.append(int)
        else:
            types.append(float)
    return types


def createFormatString(types, precision):
    # if isinstance(precision, (list, np.ndarray)):
    #     precisionCount = 0

    fmt = ""
    for i, t in enumerate(types):
        if t == int:
            fmt += "%i"
        elif t == float:
            # if isinstance(precision, int):
            fmt += "%." + str(precision) + "e"
            # elif isinstance(precision, (list, np.ndarray)):
            #     fmt +=
            #     pass
        fmt += " "
    fmt = fmt[:-1]
    return fmt


def sortPGFPlotData(filePath, axisToSortBy=0, idxToSortBy=0):
    data = loadPGFPlotData(filePath)
    args = np.argsort(data[:, 0], axisToSortBy)
    sortedData = data[args, :]

    savePGFPlotData(filePath, sortedData)


def reduceSize(filePath, decimation=1, precision=3, decimationSections=tuple()):
    data = loadPGFPlotData(filePath)
    numDataPoints = data.shape[0]

    if isinstance(decimation, int):
        reducedData = data[::decimation, :]
    if isinstance(decimation, (tuple, list, np.ndarray)):
        assert len(decimation) == len(decimationSections) + 1
        decSections = np.concatenate(
            ((0,), np.array(decimationSections), (numDataPoints,)), axis=-1
        )
        reducedData = np.concatenate(
            [
                data[decSections[i] : decSections[i + 1] : decFactor, :]
                for i, decFactor in enumerate(decimation)
            ],
            axis=0,
        )

    fmt = createFormatString(findColumnTypes(data), precision)
    savePGFPlotData(filePath, reducedData, fmt=fmt)


def periodical_mean(filePath, calc_at_time, mean_len, decibel=False):
    orig_data = loadPGFPlotData(filePath)
    # assert np.all(orig_data[:-1,0] <= orig_data[1:,0]), "Time indices must be sorted"
    assert np.all(orig_data[:-1, 0] == orig_data[1:, 0] - 1), "One sample per entry"
    # assert np.all(np.isin(calc_at_time, orig_data[:,0])), "Time indices must exist"

    calc_at_idx = np.searchsorted(orig_data[:, 0], calc_at_time)
    new_data = np.zeros(len(calc_at_idx), 2)
    for n, idx in enumerate(calc_at_idx):
        new_data[n, 0] = idx
        new_data[n, 1] = util.pow2db(
            np.mean(util.db2pow(orig_data[idx - mean_len + 1 : idx + 1, 1]))
        )

    fmt = createFormatString(findColumnTypes(data), precision)
    savePGFPlotData(filePath, new_data, suffix="mean_line", fmt=fmt)
