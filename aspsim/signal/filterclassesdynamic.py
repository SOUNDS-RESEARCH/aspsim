"""Contains implementations of time-varying FIR filters. 

This module should not have to be referenced directly. Instead, use the
function create_filter in the module filterclasses.py and specify
dyanmic=True as keyword argument. 
"""
import numpy as np
import numba as nb





SPEC_FILTERSUMDYNAMIC = [
    ("ir", nb.float64[:,:,:]),
    ("num_in", nb.int32),
    ("num_out", nb.int32),
    ("ir_len", nb.int32),
    ("buffer", nb.float64[:,:]),
    ("ir_new", nb.float64[:,:,:]),
]
@nb.experimental.jitclass(SPEC_FILTERSUMDYNAMIC)
class FilterSumDynamic:
    """
    Implements a time-varying convolution y(n) = sum_{i=0}^{I-1} h(i, n) x(n-i)

    Use the method update_ir to change h(i,n) between using the process method to
    filter the signal. If update_ir is not called, the last ir is used. 
    If update_ir is called twice, the first ir is forgotten.

    Parameters
    ----------
    ir : ndarray of shape (num_in, num_out, ir_len)
    """
    def __init__(self, ir):
        self.ir = ir
        self.num_in = ir.shape[0]
        self.num_out = ir.shape[1]
        self.ir_len = ir.shape[2]
        self.buffer = np.zeros((self.num_in, self.ir_len - 1))

        self.ir_new = ir

    def update_ir(self, ir_new):
        assert ir_new.ndim == 3
        assert ir_new.shape == (self.num_in, self.num_out, self.ir_len)
        self.ir_new = ir_new

    def process(self, in_sig):
        assert in_sig.ndim == 2
        assert in_sig.shape[0] == self.num_in #maybe shape is 1 is okay also for implicit broadcasting?
        num_samples = in_sig.shape[-1]

        buffered_input = np.concatenate((self.buffer, in_sig), axis=-1)
        num_buf = buffered_input.shape[-1]
        self.ir = self.ir_new 
        
        filtered = np.zeros((self.num_out, num_samples))

        for n in range(num_samples):
            for in_ch in range(self.num_in):
                for out_ch in range(self.num_out):
                    for j in range(self.ir_len):
                        filtered[out_ch, n] += self.ir[in_ch, out_ch, j] * buffered_input[in_ch,num_buf-num_samples+n-j]
                    
        self.buffer[:, :] = buffered_input[:, buffered_input.shape[-1] - self.ir_len + 1 :]
        return filtered



class FilterSumDynamic_python:
    """
    Implements a time-varying convolution y(n) = sum_{i=0}^{I-1} h(i, n) x(n-i)

    Use the method update_ir to change h(i,n) between using the process method to
    filter the signal. If update_ir is not called, the last ir is used. 
    If update_ir is called twice, the first ir is forgotten.

    Parameters
    ----------
    ir : ndarray of shape (num_in, num_out, ir_len)
    """
    def __init__(self, ir):
        self.ir = ir
        self.num_in = ir.shape[0]
        self.num_out = ir.shape[1]
        self.ir_len = ir.shape[2]
        self.buffer = np.zeros((self.num_in, self.ir_len - 1))

        self.ir_new = ir

    def update_ir(self, ir_new):
        assert ir_new.ndim == 3
        assert np.allclose(ir_new.shape, (self.num_in, self.num_out, self.ir_len))
        self.ir_new = ir_new

    def process(self, in_sig):
        assert in_sig.ndim == 2
        assert in_sig.shape[0] == self.num_in #maybe shape is 1 is okay also for implicit broadcasting?
        num_samples = in_sig.shape[-1]

        buffered_input = np.concatenate((self.buffer, in_sig), axis=-1)
        num_buf = buffered_input.shape[-1]
        self.ir = self.ir_new 
        
        filtered = np.zeros((self.num_out, num_samples))

        for n in range(num_samples):
            for in_ch in range(self.num_in):
                for out_ch in range(self.num_out):
                    for j in range(self.ir_len):
                        filtered[out_ch, n] += self.ir[in_ch, out_ch, j] * buffered_input[in_ch,num_buf-num_samples+n-j]
                    
        self.buffer[:, :] = buffered_input[:, buffered_input.shape[-1] - self.ir_len + 1 :]
        return filtered