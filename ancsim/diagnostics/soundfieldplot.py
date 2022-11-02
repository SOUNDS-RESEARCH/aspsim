import numpy as np
import matplotlib.pyplot as plt

import ancsim.diagnostics.diagnosticplots as dplt

def compare_soundfields(pos, sig, algo_labels, row_labels, arrays_to_plot={}):
    """single pos array of shape (numPoints, spatialDim)
        sig and labels are lists of the same length, numAlgos
        each entry in sig is an array of shape (numAxes, numPoints), 
        so for example multiple frequencies can be plotted at the same time.
        
        The resulting plot is multiple axes, numAlgos rows high, and numAxes columns wide"""
    #num_rows, num_cols = sfp.get_num_pixels(pos)
    pos, sig = sort_for_imshow(pos, sig)
    if sig.ndim == 2:
        sig = sig[None,...]
    if sig.ndim == 3:
        sig = sig[None,...]

    numAlgos = sig.shape[0]
    numEachAlgo = sig.shape[1]
    extent = size_of_2d_plot([pos] + list(arrays_to_plot.values()))
    extent /= np.max(extent)
    scaling = 5

    fig, axes = plt.subplots(numAlgos, numEachAlgo, figsize=(numEachAlgo*extent[0]*scaling, numAlgos*extent[1]*scaling))
    fig.tight_layout(pad=2.5)

    


    for i, rows in enumerate(np.atleast_2d(axes)):
        for j, ax in enumerate(rows):
            sf_plot(ax, pos, sig[i,j,:,:], f"{algo_labels[i]}, {row_labels[j]}", arrays_to_plot)
    

def size_of_2d_plot(array_pos):
    """array_pos is a list/tuple of ndarrays of shape(any, spatialDim)
        describing the positions of all objects to be plotted. First axis 
        can be any value for the arrays, second axis must be 2 or more. Only
        the first two are considered. 
        
        returns np.array([x_size, y_size])"""
    array_pos = [ap[...,:2].reshape(-1,2) for ap in array_pos]
    all_pos = np.concatenate(array_pos, axis=0)
    extent = np.max(all_pos, axis=0) - np.min(all_pos, axis=0)  
    return extent[:2]
        

def sf_plot(ax, pos, sig, title="", arrays_to_plot = {}):
    """takes a single ax, pos and sig and creates a decent looking soundfield plot
        Assumes that pos and sig are already correctly sorted"""

    im = ax.imshow(sig, interpolation="none", extent=(pos[...,0].min(), pos[...,0].max(), pos[...,1].min(), pos[...,1].max()))
    
    # if isinstance(arrays_to_plot, ar.ArrayCollection):
    #     for array in arrays_to_plot:
    #         array.plot(ax)
    if isinstance(arrays_to_plot, dict):
        for arName, array in arrays_to_plot.items():
            ax.plot(array[:,0], array[:,1], "x", label=arName)
    else:
        raise ValueError

    ax.legend()
    ax.set_title(title)
    dplt.set_basic_plot_look(ax)
    ax.axis("equal")
    plt.colorbar(im, ax=ax, orientation='vertical')
    #plt.colorbar(ax=ax)

def get_num_pixels(pos, pos_decimals=5):
    pos_cols = np.unique(pos[:,0].round(pos_decimals))
    pos_rows = np.unique(pos[:,1].round(pos_decimals))
    num_rows = len(pos_rows)
    num_cols = len(pos_cols)
    return num_rows, num_cols


def sort_for_imshow(pos, sig, pos_decimals=5):
    """ pos must be of shape (numPos, spatialDims) placed on a rectangular grid, 
        but can be in any order.
        sig can be a single signal or a list of signals of shape (numPos, signalDim), where each
        entry on first axis is the value for pos[0,:] """
    if pos.shape[1] == 3:
        assert np.allclose(pos[:,2], np.ones_like(pos[:,2])*pos[0,2])

    num_rows, num_cols = get_num_pixels(pos, pos_decimals)
    unique_x = np.unique(pos[:,0].round(pos_decimals))
    unique_y = np.unique(pos[:,1].round(pos_decimals))

    sortIndices = np.zeros((num_rows, num_cols), dtype=int)
    for i, y in enumerate(unique_y):
        rowIndices = np.where(np.abs(pos[:,1] - y) < 10**(-pos_decimals))[0]
        rowPermutation = np.argsort(pos[rowIndices,0])
        sortIndices[i,:] = rowIndices[rowPermutation]

    pos = pos[sortIndices,:]

    sig = np.moveaxis(np.atleast_3d(sig),1,2)
    dims = sig.shape[:2]
    sig_sorted = np.zeros((dims[0], dims[1], num_rows, num_cols), dtype=sig.dtype)
    for i in range(dims[0]):
        for j in range(dims[1]):
            single_sig = sig[i,j,:]
            sig_sorted[i,j,:,:] = np.flip(single_sig[sortIndices],axis=0)
    sig_sorted = np.squeeze(sig_sorted)
    
    #sig = [np.flip(s[sortIndices],axis=0) for s in sig]
    
    return pos, sig_sorted