import numpy as np
import hypothesis as hyp
import hypothesis.strategies as st
from timeit import default_timer as timer

import aspsim.signal.filterclasses as fc
import aspsim.signal.filterclassesdynamic as fcd


@hyp.settings(deadline=None)
@hyp.given(
    ir_len = st.integers(min_value=1, max_value=8),
    num_in = st.integers(min_value=1, max_value=4),
    num_out = st.integers(min_value=1, max_value=4),
    num_samples = st.integers(min_value=1, max_value=32),
)
def test_filtersum_dynamic_with_static_ir_equals_normal_filtersum(ir_len, num_in, num_out, num_samples):
    rng = np.random.default_rng()
    ir = rng.normal(0, 1, size=(num_in,num_out,ir_len))

    filt_dyn = fcd.FilterSumDynamic(ir)
    filt = fc.create_filter(ir)

    out = np.zeros((num_out, num_samples))
    out_dyn = np.zeros((num_out, num_samples))

    sig = rng.normal(0, 1, size=(num_in, num_samples))
    out_dyn = filt_dyn.process(sig)
    out = filt.process(sig)
    # for i in range(num_samples):
    #     out[:,i:i+1] = filt.process(sig[:,i:i+1])
    #     out_dyn[:,i:i+1] = filt_dyn.process(sig[:,i:i+1])
    assert np.allclose(out, out_dyn)


@hyp.settings(deadline=None)
@hyp.given(
    num_in = st.integers(min_value=1, max_value=4),
    num_out = st.integers(min_value=1, max_value=4),
    num_samples = st.integers(min_value=1, max_value=32),
)
def test_filtersum_dynamic_no_output_with_increasing_delay(num_in, num_out, num_samples):
    ir_len = num_samples + 1
    rng = np.random.default_rng()
    
    ir = np.zeros((num_in,num_out, ir_len))
    filt_dyn = fcd.FilterSumDynamic(ir)

    out = np.zeros((num_out, num_samples))

    sig = rng.normal(0, 1, size=(num_in, num_samples))
    for i in range(num_samples):
        ir = np.zeros((num_in, num_out, ir_len))
        ir[:,:,i+1:] = 1
        filt_dyn.update_ir(ir)
        out[:,i:i+1] = filt_dyn.process(sig[:,i:i+1])

    assert np.allclose(np.sum(np.abs(out)), 0)




@hyp.settings(deadline=None)
@hyp.given(
    ir_len = st.integers(min_value=1, max_value=8),
    num_in = st.integers(min_value=1, max_value=4),
    num_out = st.integers(min_value=1, max_value=4),
    num_samples = st.integers(min_value=2, max_value=32),
)
def test_filtersum_dynamic_piecewise_static_ir_equals_normal_filtersum(ir_len, num_in, num_out, num_samples):
    rng = np.random.default_rng()
    ir1 = rng.normal(0, 1, size=(num_in,num_out,ir_len))
    ir2 = rng.normal(0, 1, size=(num_in,num_out,ir_len))
    change_idx = num_samples // 2

    filt_dyn = fcd.FilterSumDynamic(ir1)
    filt1 = fc.create_filter(ir1)
    filt2 = fc.create_filter(ir2)

    out = np.zeros((num_out, num_samples))
    out_dyn = np.zeros((num_out, num_samples))

    sig = rng.normal(0, 1, size=(num_in, num_samples))
    out[:,:change_idx] = filt1.process(sig[:,:change_idx])
    filt2.process(sig[:,:change_idx])
    out[:,change_idx:] = filt2.process(sig[:,change_idx:])

    for i in range(num_samples):
        out_dyn[:,i:i+1] = filt_dyn.process(sig[:,i:i+1])
        if i == change_idx-1:
            filt_dyn.update_ir(ir2)

    assert np.allclose(out, out_dyn)



@hyp.settings(deadline=None)
@hyp.given(
    ir_len = st.integers(min_value=1, max_value=8),
    num_in = st.integers(min_value=1, max_value=1),
    num_out = st.integers(min_value=1, max_value=1),
    num_samples = st.integers(min_value=2, max_value=32),
)
def test_dynamic_filtering_equals_static_filtering_and_then_selecting_samples(ir_len, num_in, num_out, num_samples):
    rng = np.random.default_rng()
    irs = rng.normal(0, 1, size=(1,num_samples,ir_len))
    in_sig = rng.normal(0, 1, size=(1,num_samples))

    static_filt = fc.create_filter(irs)
    static_sig_out = static_filt.process(in_sig)
    static_sig_select = np.array([static_sig_out[i,i] for i in range(num_samples)])[None,:]

    dyn_filt = fc.create_filter(irs[:,0:1,:], dynamic=True)
    dyn_out = np.zeros((1,num_samples))
    for i in range(num_samples):
        dyn_filt.update_ir(irs[:,i:i+1,:])
        dyn_out[:,i] = dyn_filt.process(in_sig[:,i:i+1])

    assert np.allclose(dyn_out, static_sig_select)


#test_filtersum_dynamic_zero_ir_equals_static_filter_with_shorter_input_signal

#def test_dynamic_filtering_equals_static_filtering_and_then_selecting_sampled()



@hyp.settings(deadline=None)
@hyp.given(
    ir_len = st.integers(min_value=1, max_value=8),
    num_in = st.integers(min_value=1, max_value=4),
    num_out = st.integers(min_value=1, max_value=4),
    num_samples = st.integers(min_value=2, max_value=32),
)
def test_filtersum_dynamic_equals_direct_implementation_of_definition(ir_len, num_in, num_out, num_samples):
    rng = np.random.default_rng()
    ir = rng.normal(0, 1, size = (num_samples, num_in,num_out,ir_len))
    sig = rng.normal(0, 1, size = (num_in, num_samples))
    buf_sig = np.concatenate((np.zeros((num_in, ir_len-1)), sig), axis=-1)

    filt = fcd.FilterSumDynamic(ir[0,...])
    out = np.zeros((num_out, num_samples))
    out_dyn = np.zeros((num_out, num_samples))

    for n in range(num_samples):
        filt.update_ir(ir[n,...])
        out_dyn[:,n:n+1] = filt.process(sig[:,n:n+1])
        for out_ch in range(num_out):
            for in_ch in range(num_in):
                for i in range(ir_len):
                    out[out_ch,n] += ir[n, in_ch, out_ch, i] * buf_sig[in_ch,ir_len+n-i-1]
    assert np.allclose(out, out_dyn)


@hyp.settings(deadline=None)
@hyp.given(
    ir_len = st.integers(min_value=1, max_value=8),
    num_in = st.integers(min_value=1, max_value=4),
    num_out = st.integers(min_value=1, max_value=4),
    num_blocks = st.integers(min_value=2, max_value=32),
    block_size = st.integers(min_value=1, max_value=5)
)
def test_filtersum_dynamic_equals_block_based_direct_implementation_of_definition(ir_len, num_in, num_out, num_blocks, block_size):
    rng = np.random.default_rng()
    num_samples = num_blocks * block_size
    ir = rng.normal(0, 1, size = (num_blocks, num_in,num_out,ir_len))
    sig = rng.normal(0, 1, size = (num_in, num_samples))
    buf_sig = np.concatenate((np.zeros((num_in, ir_len-1)), sig), axis=-1)

    filt = fcd.FilterSumDynamic(ir[0,...])
    out = np.zeros((num_out, num_samples))
    out_dyn = np.zeros((num_out, num_samples))

    for b in range(num_blocks):
        filt.update_ir(ir[b,...])
        out_dyn[:,b*block_size:(b+1)*block_size] = filt.process(sig[:,b*block_size:(b+1)*block_size])
        for out_ch in range(num_out):
            for in_ch in range(num_in):
                for n in range(block_size):
                    for i in range(ir_len):
                        out[out_ch,b*block_size+n] += ir[b, in_ch, out_ch, i] * buf_sig[in_ch,ir_len+b*block_size+n-i-1]
    assert np.allclose(out, out_dyn)




@hyp.settings(deadline=None)
@hyp.given(
    ir_len = st.integers(min_value=1, max_value=8),
    num_in = st.integers(min_value=1, max_value=4),
    num_out = st.integers(min_value=1, max_value=4),
    num_blocks = st.integers(min_value=1, max_value=32),
    block_size = st.integers(min_value=1, max_value=32),
)
def test_filtersum_dynamic_numba_and_python_implementations_are_equal(ir_len, num_in, num_out, num_blocks, block_size):
    rng = np.random.default_rng()
    num_samples = num_blocks * block_size
    ir = rng.normal(0, 1, size = (num_blocks, num_in,num_out,ir_len))
    sig = rng.normal(0, 1, size = (num_in, num_samples))

    filt = fcd.FilterSumDynamic(ir[0,...])
    filtpy = fcd.FilterSumDynamic_python(ir[0,...])
    out = np.zeros((num_out, num_samples))
    outpy = np.zeros((num_out, num_samples))

    for b in range(num_blocks):
        filt.update_ir(ir[b,...])
        filtpy.update_ir(ir[b,...])
        out[:,b*block_size:(b+1)*block_size] = filt.process(sig[:,b*block_size:(b+1)*block_size])
        outpy[:,b*block_size:(b+1)*block_size] = filtpy.process(sig[:,b*block_size:(b+1)*block_size])
    assert np.allclose(out, outpy)



@hyp.settings(deadline=None)
@hyp.given(
    ir_len = st.integers(min_value=128, max_value=128),
    num_in = st.integers(min_value=4, max_value=4),
    num_out = st.integers(min_value=5, max_value=5),
    num_blocks = st.integers(min_value=128, max_value=128),
    block_size = st.integers(min_value=16, max_value=16),
)
def test_filtersum_dynamic_numba_is_faster_than_python_implementation(ir_len, num_in, num_out, num_blocks, block_size):
    rng = np.random.default_rng()
    num_samples = num_blocks * block_size
    ir = rng.normal(0, 1, size = (num_blocks, num_in,num_out,ir_len))
    sig = rng.normal(0, 1, size = (num_in, num_samples))

    filt = fcd.FilterSumDynamic(ir[0,...])
    filtpy = fcd.FilterSumDynamic_python(ir[0,...])
    out = np.zeros((num_out, num_samples))
    outpy = np.zeros((num_out, num_samples))

    start = timer()
    for b in range(num_blocks):
        filt.update_ir(ir[b,...])
        out[:,b*block_size:(b+1)*block_size] = filt.process(sig[:,b*block_size:(b+1)*block_size])
    end = timer()
    time = end - start

    start = timer()
    for b in range(num_blocks):
        filtpy.update_ir(ir[b,...])
        outpy[:,b*block_size:(b+1)*block_size] = filtpy.process(sig[:,b*block_size:(b+1)*block_size])
    end = timer()
    timepy = end - start
    assert time < timepy