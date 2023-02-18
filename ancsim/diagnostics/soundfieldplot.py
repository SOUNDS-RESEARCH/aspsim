import numpy as np
import matplotlib.pyplot as plt

import ancsim.diagnostics.plot as dplt
import ancsim.array as ar

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

    num_algos = sig.shape[0]
    num_each_algo = sig.shape[1]
    extent = size_of_2d_plot([pos] + list(arrays_to_plot.values()))
    extent /= np.max(extent)
    scaling = 5

    fig, axes = plt.subplots(num_algos, num_each_algo, figsize=(num_each_algo*extent[0]*scaling, num_algos*extent[1]*scaling))
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
        

def sf_plot(ax, pos, sig, title="", arrays_to_plot = None, vminmax=None):
    """
        Makes a soundfield plot using the supplied matplotlib.pyplot axis

        Parameters
        ---------
        ax : axis obtained from fig, axes = plt.subplots()
        pos : ndarray of shape (num_pos, spatial_dim)
            must be sorted, use function sort_for_imshow()
        sig : ndarray of shape (num_pos, signal_dim)
            signal value for each position of the sound field
            must also be sorted the same way as pos
        title : str
            optional title for the plot
        arrays_to_plot : Array, ArrayCollection, list of Arrays, or dict of Arrays
            will be plotted along with the soundfield image. Can be used to see
            where microphones or loudspeakers are in relation to the soundfield
        vminmax : length-2 tuple of float
            the tuple should be (vmin, vmax), which sets the minimum and maximum value
            that the colors get assigned to. Use this in particular when plotting 
            multiple soundfields next to each other, so that the colors mean the
            same values in the different plots
        """

    if vminmax is None:
        vminmax = (np.min(sig), np.max(sig))

    im = ax.imshow(sig, interpolation="none", 
                    extent=(pos[...,0].min(), pos[...,0].max(), pos[...,1].min(), pos[...,1].max()), 
                    vmin=vminmax[0], vmax=vminmax[1],
                    cmap="magma",)
    
    if arrays_to_plot is not None:
        if isinstance(arrays_to_plot, ar.Array):
            arrays_to_plot = [arrays_to_plot]
        elif isinstance(arrays_to_plot, dict):
            arrays_to_plot = arrays_to_plot.values()
        for plot_ar in arrays_to_plot:
            plot_ar.plot(ax)
            #for arName, array in arrays_to_plot.items():
            #    ax.plot(array[:,0], array[:,1], "x", label=arName)

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
    """
        Sorts the position and signal values to display correctly when
        imshow is used to plot the sound field image

        Parameters
        ---------
        pos : ndarray of shape (num_pos, spatial_dim)
            must represent a rectangular grid, but can be in any order.
        sig : ndarray of shape (num_pos, signal_dim)
            signal value for each position of the sound field
        pos_decimals : int
            selects how many decimals the position values are rounded to when 
            calculating all the unique position values
        
        Returns
        -------
        pos_sorted : ndarray of shape (num_rows, num_cols, spatial_dim)
        sig_sorted : ndarray of shape (num_rows, num_cols, signal_dim)
    """
    if pos.shape[1] == 3:
        assert np.allclose(pos[:,2], np.ones_like(pos[:,2])*pos[0,2])

    num_rows, num_cols = get_num_pixels(pos, pos_decimals)
    unique_x = np.unique(pos[:,0].round(pos_decimals))
    unique_y = np.unique(pos[:,1].round(pos_decimals))

    sort_indices = np.zeros((num_rows, num_cols), dtype=int)
    for i, y in enumerate(unique_y):
        row_indices = np.where(np.abs(pos[:,1] - y) < 10**(-pos_decimals))[0]
        row_permutation = np.argsort(pos[row_indices,0])
        sort_indices[i,:] = row_indices[row_permutation]

    pos = pos[sort_indices,:]

    #sig = np.moveaxis(np.atleast_3d(sig),1,2)
    #dims = sig.shape[:2]
    signal_dim = sig.shape[-1]
    sig_sorted = np.zeros((num_rows, num_cols, signal_dim), dtype=sig.dtype)
    sig_sorted = sig[sort_indices,:]
    #for i in range(dims[0]):
     #   for j in range(dims[1]):
     #       single_sig = sig[i,j,:]
     #       sig_sorted[i,j,:,:] = np.flip(single_sig[sort_indices],axis=0)
    # sig_sorted = np.squeeze(sig_sorted)
    
    #sig = [np.flip(s[sortIndices],axis=0) for s in sig]
    
    return pos, sig_sorted






# def sort_for_imshow(pos, sig, pos_decimals=5):
#     """ 
#         Parameters
#         ---------
#         pos : ndarray of shape (num_pos, spatial_dim)
#         sig : ndarray of shape (num_pos, signal_dim)
#         pos_decimals : int
#             selects how many decimals the position values are rounded to when 
#             calculating all the unique position values

#         pos must be of shape (numPos, spatialDims) placed on a rectangular grid, 
#         but can be in any order.
#         sig can be a single signal or a list of signals of shape (numPos, signalDim), where each
#         entry on first axis is the value for pos[0,:] 
        
#         Returns
#         -------
#         pos_sorted : ndarray of shape (num_rows, num_cols, spatial_dim)
#         sig_sorted : ndarray of shape (num_rows, num_cols, signal_dim)
#     """
#     if pos.shape[1] == 3:
#         assert np.allclose(pos[:,2], np.ones_like(pos[:,2])*pos[0,2])

#     num_rows, num_cols = get_num_pixels(pos, pos_decimals)
#     unique_x = np.unique(pos[:,0].round(pos_decimals))
#     unique_y = np.unique(pos[:,1].round(pos_decimals))

#     sort_indices = np.zeros((num_rows, num_cols), dtype=int)
#     for i, y in enumerate(unique_y):
#         row_indices = np.where(np.abs(pos[:,1] - y) < 10**(-pos_decimals))[0]
#         row_permutation = np.argsort(pos[row_indices,0])
#         sort_indices[i,:] = row_indices[row_permutation]

#     pos = pos[sort_indices,:]

#     sig = np.moveaxis(np.atleast_3d(sig),1,2)
#     dims = sig.shape[:2]
#     sig_sorted = np.zeros((dims[0], dims[1], num_rows, num_cols), dtype=sig.dtype)
#     for i in range(dims[0]):
#         for j in range(dims[1]):
#             single_sig = sig[i,j,:]
#             sig_sorted[i,j,:,:] = np.flip(single_sig[sort_indices],axis=0)
#     sig_sorted = np.squeeze(sig_sorted)
    
#     #sig = [np.flip(s[sortIndices],axis=0) for s in sig]
    
#     return pos, sig_sorted