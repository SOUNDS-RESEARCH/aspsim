import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
import tikzplotlib
import os
import pathlib
import ancsim.utilities as util
import ancsim.fileutilities as fu


def deleteEarlierTikzPlot(folder, name):
    currentIdx = fu.findIndexInName(name)
    if currentIdx is None:
        return

    earlierFiles = fu.findAllEarlierFiles(folder, name, currentIdx)
    for f in earlierFiles:
        if f.is_dir():
            for plotFile in f.iterdir():
                # assert(plotFile.stem.startswith(startName))
                try:
                    if plotFile.suffix == ".pdf":
                        plotFile.rename(
                            folder.joinpath(plotFile.stem + plotFile.suffix)
                        )
                    else:
                        assert plotFile.suffix == ".tsv" or plotFile.suffix == ".tex"
                        plotFile.unlink()
                except PermissionError:
                    pass
            try:
                f.rmdir()
            except PermissionError:
                pass

def output_plot(printMethod, folder, name="", keepOnlyLatestTikz=True):
    if printMethod == "show":
        plt.show()
    elif printMethod == "tikz":
        if folder is not None:
            nestedFolder = folder.joinpath(name)
            try:
                nestedFolder.mkdir()
            except FileExistsError:
                pass
            tikzplotlib.save(
                str(nestedFolder.joinpath(name + ".tex")),
                externalize_tables=True,
                tex_relative_path_to_data="../figs/" + name + "/",
                float_format=".8g",
            )
            plt.savefig(
                str(nestedFolder.joinpath(name + ".pdf")),
                dpi=300,
                facecolor="w",
                edgecolor="w",
                orientation="portrait",
                format="pdf",
                transparent=True,
                bbox_inches=None,
                pad_inches=0.2,
            )
            if keepOnlyLatestTikz:
                deleteEarlierTikzPlot(folder, name)
    elif printMethod == "pdf":
        if folder is not None:
            plt.savefig(
                str(folder.joinpath(name + ".pdf")),
                dpi=300,
                facecolor="w",
                edgecolor="w",
                orientation="portrait",
                format="pdf",
                transparent=True,
                #bbox_inches=None,
                bbox_inches="tight",
                pad_inches=0.2,
            )
    elif printMethod == "svg":
        if folder is not None:
            plt.savefig(
                str(folder.joinpath(name + ".svg")),
                dpi=300,
                format="svg",
                transparent=True,
                bbox_inches="tight",
                pad_inches=0.2,
            )
    elif printMethod == "none":
        pass
    else:
        raise ValueError
    plt.close("all")



def setBasicPlotLook(ax):
    ax.grid(True)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

def remove_axes_and_labels(ax):
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

def plot3Din2D(ax, posData, symbol="x", name=""):
    uniqueZValues = np.unique(posData[:,2].round(decimals=4))
    
    # if len(uniqueZValues) == 1:
    #     alpha = np.array([1])
    # else:
    #     alpha = np.linspace(0.4, 1, len(uniqueZValues))

    for i, zVal in enumerate(uniqueZValues):
        idx = np.where(np.around(posData[:,2],4) == zVal)[0]

        ax.plot(posData[idx,0], posData[idx,1], symbol, label=f"{name}: z = {zVal}")


def plotPos(pos, folder, config, printMethod="pdf"):
    [fig, ax] = plt.subplots(1,1, figsize=(8,8))
    ax.set_title("Positions")

    plot3Din2D(ax, pos["speaker"], "o", "loudspeaker")
    plot3Din2D(ax, pos["source"], "o", "source")
    plot3Din2D(ax, pos["error"], "x", "error mic")
    if "ref" in pos:
        plot3Din2D(ax, pos["ref"], "x", "reference mic")

    ax.autoscale()
    output_plot(printMethod, folder, "positions")

    

def plotPos2dDisc(pos, folder, config, printMethod="pdf"):
    def setLook(ax, config):
        ax.grid(True)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.axis("equal")
        ax.set_title("Positions")
        ax.set_xlabel("Sources: o, Microphones: x \n Both axes show meter")

        circle = Circle(
            (0, 0),
            config["TARGETWIDTH"],
            facecolor=(0.7, 0.7, 0.7),
            edgecolor="none",
            linewidth=3,
            alpha=0.5,
        )
        ax.add_patch(circle)

    # Plot with sources
    [fig, ax] = plt.subplots(1, 1, figsize=(8, 8))
    ax.plot(pos["speaker"][:, 0], pos["speaker"][:, 1], "o")
    ax.plot(pos["error"][:, 0], pos["error"][:, 1], "x")
    ax.plot(pos["source"][:, 0], pos["source"][:, 1], "o")
    if "ref" in pos:
        ax.plot(pos["ref"][:, 0], pos["ref"][:, 1], "x")

    setLook(ax, config)

    output_plot(printMethod, folder, "positions")

    # Plot without sources
    [fig, ax] = plt.subplots(1, 1, figsize=(8, 8))
    ax.plot(pos["speaker"][:, 0], pos["speaker"][:, 1], "o")
    ax.plot(pos["error"][:, 0], pos["error"][:, 1], "x")
    if "ref" in pos:
        ax.plot(pos["ref"][:, 0], pos["ref"][:, 1], "x")
    setLook(ax, config)
    output_plot(printMethod, folder, "positions_close")


def plotPos3dDisc(pos, folder, config, printMethod="pdf"):
    def setLook(ax, config):
        ax.grid(True)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.axis("equal")
        ax.set_title("Positions")
        ax.set_xlabel("Sources: o, Microphones: x \n Both axes show meter")
        plt.legend()
        circle = Circle(
            (0, 0),
            config["TARGETWIDTH"],
            facecolor=(0.7, 0.7, 0.7),
            edgecolor="none",
            linewidth=3,
            alpha=0.5,
        )
        ax.add_patch(circle)

    def plotMultipleCircles(ax, posArray, symbol="x", name=""):
        currentz = posArray[0, -1]
        circleIdxs = [0]
        for i in range(posArray.shape[0]):
            if not np.isclose(posArray[i, -1], currentz):
                currentz = posArray[i, -1]
                circleIdxs.append(i)
        circleIdxs.append(posArray.shape[0])

        for i in range(len(circleIdxs) - 1):
            ax.plot(
                posArray[circleIdxs[i] : circleIdxs[i + 1], 0],
                posArray[circleIdxs[i] : circleIdxs[i + 1], 1],
                symbol,
                label=name + ", " + "z = " + str(posArray[circleIdxs[i], 2]),
            )

    # Plot with primary sources
    [fig, ax] = plt.subplots(1, 1, figsize=(8, 8))
    plotMultipleCircles(ax, pos["error"], "x", "Error mic")
    plotMultipleCircles(ax, pos["speaker"], "o", "Sec Speaker")
    ax.plot(pos["source"][:, 0], pos["source"][:, 1], "o")
    if "ref" in pos:
        ax.plot(pos["ref"][:, 0], pos["ref"][:, 1], "x")
    setLook(ax, config)

    output_plot(printMethod, folder, "positions")

    # Plot without primary sources
    [fig, ax] = plt.subplots(1, 1, figsize=(8, 8))
    plotMultipleCircles(ax, pos["error"], "x", "Error mic")
    plotMultipleCircles(ax, pos["speaker"], "o", "Sec Speaker")
    setLook(ax, config)
    output_plot(printMethod, folder, "positions_close")


def plotPos3dRect(
    pos, folder, config, roomSize=None, roomCenter=None, printMethod="pdf"
):
    def setLook(ax, config):
        ax.grid(True)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.axis("equal")
        ax.set_title("Positions")
        ax.set_xlabel("Sources: o, Microphones: x \n Both axes show meter")
        plt.legend()
        targetzone = Rectangle(
            [-config["TARGETWIDTH"] / 2, -config["TARGETWIDTH"] / 2],
            config["TARGETWIDTH"],
            config["TARGETWIDTH"],
            facecolor=(0.7, 0.7, 0.7),
            edgecolor="none",
            linewidth=3,
            alpha=0.5,
        )
        ax.add_patch(targetzone)
        if roomSize is not None and roomCenter is not None:
            roomOutline = Rectangle(
                [roomCenter[0] - roomSize[0] / 2, roomCenter[1] - roomSize[1] / 2],
                roomSize[0],
                roomSize[1],
                facecolor=(0.5, 0.5, 0.5),
                edgecolor="black",
                fill=False,
                linewidth=10,
            )
            ax.add_patch(roomOutline)

    def plotMultipleRects(ax, posArray, symbol="x", name=""):
        zValues = np.unique(posArray[:,2])
        for z in zValues:
            idx = np.where(posArray[:,2] == z)[0]
            ax.plot(posArray[idx,0], posArray[idx,1], symbol, 
                    label=f"{name}, height = {z}")

    # def plotMultipleRects(ax, posArray, symbol="x", name=""):
    #     currentz = posArray[0, -1]
    #     rectIdx = [0]
    #     for i in range(posArray.shape[0]):
    #         if not np.isclose(posArray[i, -1], currentz):
    #             currentz = posArray[i, -1]
    #             rectIdx.append(i)
    #     rectIdx.append(posArray.shape[0])

    #     for i in range(len(rectIdx) - 1):
    #         ax.plot(
    #             posArray[rectIdx[i] : rectIdx[i + 1], 0],
    #             posArray[rectIdx[i] : rectIdx[i + 1], 1],
    #             symbol,
    #             label=name + ", " + "z = " + str(posArray[rectIdx[i], 2]),
    #         )

    # Plot with primary sources
    [fig, ax] = plt.subplots(1, 1, figsize=(8, 8))
    plotMultipleRects(ax, pos["error"], "x", "Error mic")
    plotMultipleRects(ax, pos["speaker"], "o", "Sec Speaker")
    ax.plot(pos["source"][:, 0], pos["source"][:, 1], "o")

    if "ref" in pos:
        ax.plot(pos["ref"][:, 0], pos["ref"][:, 1], "x")
    setLook(ax, config)
    ax.autoscale()
    output_plot(printMethod, folder, "positions")

    # Plot without primary sources
    [fig, ax] = plt.subplots(1, 1, figsize=(8, 8))
    plotMultipleRects(ax, pos["error"], "x", "Error mic")
    plotMultipleRects(ax, pos["speaker"], "o", "Sec Speaker")
    setLook(ax, config)
    margin = 0.1
    ax.set_xlim(
        pos["speaker"][:, 0].min() - margin, pos["speaker"][:, 0].max() + margin
    )
    ax.set_ylim(
        pos["speaker"][:, 1].min() - margin, pos["speaker"][:, 1].max() + margin
    )
    output_plot(printMethod, folder, "positions_close")


def compareReduction(regRed, pointRed, legend, timeIdx, folder, printMethod="pdf"):
    fig, axes = plt.subplots(2, 1, figsize=(16, 10))
    fig.tight_layout(pad=2)

    for i, red in enumerate(pointRed):
        axes[0].plot(red[:timeIdx], alpha=0.8)

    for i, red in enumerate(regRed):
        axes[1].plot(red[:timeIdx], alpha=0.8)

    # for l in loss:
    #    axes[2].plot(l[:timeIdx], alpha=0.8)

    for ax in axes:
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.grid(True)
        ax.legend(legend, loc="upper right")

    axes[0].set_title("Reduction at Error Points")
    axes[1].set_title("Regional Reduction")
    # axes[2].set_title("Loss")

    output_plot(printMethod, folder, "reduction_" + str(timeIdx))


def showSoundfield(
    sound,
    noise,
    pixelPos,
    pos,
    algoNames,
    timeIdx,
    folder,
    numSamples=1024,
    extraName="",
    printMethod="pdf",
):
    numAlgos = len(sound)
    plotTypes = 3
    imAxisLen = int(np.sqrt(pixelPos.shape[0]))

    fig, axes = plt.subplots(
        numAlgos, plotTypes, figsize=(15, plotTypes * numAlgos + 3)
    )
    axes = axes.reshape(numAlgos, plotTypes)
    fig.tight_layout(pad=2.5)

    noise = noise[:, -numSamples:]

    dataToPlot = [[] for _ in range(numAlgos)]
    minVals = np.full(plotTypes, np.inf)
    maxVals = np.full(plotTypes, -np.inf)
    for algo in range(numAlgos):
        e = sound[algo]
        e = e[:, -numSamples:]

        powRed = util.pow2db(np.mean(e ** 2, axis=-1) / np.mean(noise ** 2, axis=-1))
        minVals[0] = np.min((minVals[0], np.min(powRed)))
        maxVals[0] = np.max((maxVals[0], np.max(powRed)))
        dataToPlot[algo].append(powRed)

        noisePow = util.pow2db(np.mean(e ** 2, axis=-1))
        minVals[1] = np.min((minVals[1], np.min(noisePow)))
        maxVals[1] = np.max((maxVals[1], np.max(noisePow)))
        dataToPlot[algo].append(noisePow)

        primNoisePow = util.pow2db(np.mean(noise ** 2, axis=-1))
        minVals[2] = np.min((minVals[2], np.min(primNoisePow)))
        maxVals[2] = np.max((maxVals[2], np.max(primNoisePow)))
        dataToPlot[algo].append(primNoisePow)

    for i in range(numAlgos):
        for j in range(plotTypes):
            clr = axes[i, j].pcolormesh(
                pixelPos[:, 0].reshape(imAxisLen, imAxisLen),
                pixelPos[:, 1].reshape(imAxisLen, imAxisLen),
                dataToPlot[i][j].reshape(imAxisLen, imAxisLen),
                vmin=minVals[j],
                vmax=maxVals[j],
                cmap="magma",
            )
            fig.colorbar(clr, ax=axes[i, j])

            axes[i, j].set_title("Power Reduction (dB)")
            axes[i, j].plot(pos["speaker"][:, 0], pos["speaker"][:, 1], "o")
            axes[i, j].plot(pos["source"][:, 0], pos["source"][:, 1], "o", color="m")
            axes[i, j].plot(pos["error"][:, 0], pos["error"][:, 1], "x", color="y")
            if "ref" in pos:
                axes[i, j].plot(pos["ref"][:, 0], pos["ref"][:, 1], "x", color="y")
            axes[i, j].axis("equal")
            axes[i, j].spines["right"].set_visible(False)
            axes[i, j].spines["top"].set_visible(False)
            axes[i, j].spines["left"].set_visible(False)
            axes[i, j].spines["bottom"].set_visible(False)
            axes[i, j].set_xlim(pixelPos[:, 0].min(), pixelPos[:, 0].max())
            axes[i, j].set_ylim(pixelPos[:, 1].min(), pixelPos[:, 1].max())

    pad = 25
    colTitles = ["Noise Reduction", "Total Noise Power", "Primary noise Power"]
    for ax, col in zip(axes[0, :], colTitles):
        ax.annotate(
            col,
            xy=(0.5, 1),
            xytext=(0, pad),
            xycoords="axes fraction",
            textcoords="offset points",
            size="large",
            ha="center",
            va="baseline",
            fontsize=18,
        )

    for ax, rowTitle in zip(axes[:, 0], algoNames):
        ax.annotate(
            rowTitle,
            xy=(0, 0.5),
            xytext=(-ax.yaxis.labelpad, 0),
            xycoords=ax.yaxis.label,
            textcoords="offset points",
            size="large",
            ha="right",
            va="center",
            rotation=90,
            fontsize=15,
        )

    plt.subplots_adjust(top=0.92, right=1)

    output_plot(printMethod, folder, "soundfield_" + extraName + str(timeIdx))


# def printFilterInfo(A, Afreq, pos, freqs, numMics):
#     freqidx = int(s.FREQSTART * s.KERNFILTLEN / s.samplerate)
#     for a in range(numMics):
#         for b in range(numMics):
#             print("Amp Original: ", np.abs(Afreq[[freqidx], a,b]))
#             print("Amp New     : ", np.abs(np.fft.fft(A[a,b,:])[freqidx]))
#             print("Angle Original: ", np.angle(Afreq[freqidx,a,b] ))
#             print("Angle New     : ", np.angle(np.fft.fft(A[a,b,:])[freqidx]))


def analyzeFilter(A, Afreq, pos, freqs, numMics):
    for a in range(numMics):
        for b in range(numMics):
            fig, axes = plt.subplots(2, 1)
            axes[0].plot(np.abs(np.fft.fft(A[a, b, :])))
            axes[0].plot(Afreq[:, a, b])
            # axes[0].plot(np.abs(np.fft.fft(irls)))
            axes[0].legend(("freqsampling", "original"))
            axes[0].set_title("amplitude")

            axes[1].plot(np.unwrap(np.angle(np.fft.fft(A[a, b, :]))))
            axes[1].plot(np.unwrap(np.angle(Afreq[:, a, b])))
            # axes[1].plot(np.unwrap(np.angle(np.fft.fft(irls))))
            axes[1].legend(("freqsampling", "original"))
            axes[1].set_title("phase")
            plt.show()
