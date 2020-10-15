from __future__ import division
import numpy as np
import scipy as sp
from numpy.lib.stride_tricks import as_strided
import scipy.signal as sg
from scipy.interpolate import interp1d
import wave
from scipy.cluster.vq import vq
from scipy import linalg, fftpack
from numpy.testing import assert_almost_equal
from scipy.linalg import svd
from scipy.io import wavfile
from scipy.signal import firwin
import zipfile
import tarfile
import os
import copy
import multiprocessing
from multiprocessing import Pool
import functools
import time
try:
    import urllib.request as urllib  # for backwards compatibility
except ImportError:
    import urllib2 as urllib

import tensorflow as tf


def fetch_audio(wav_path):
    fs, d = wavfile.read(wav_path)
    d = d.astype('float32') / (2 ** 15)
    return fs, d

def voiced_unvoiced(X, window_size=256, window_step=128, copy=True):
    """
    Voiced unvoiced detection from a raw signal

    Based on code from:
        https://www.clear.rice.edu/elec532/PROJECTS96/lpc/code.html

    Other references:
        http://www.seas.ucla.edu/spapl/code/harmfreq_MOLRT_VAD.m

    Parameters
    ----------
    X : ndarray
        Raw input signal

    window_size : int, optional (default=256)
        The window size to use, in samples.

    window_step : int, optional (default=128)
        How far the window steps after each calculation, in samples.

    copy : bool, optional (default=True)
        Whether to make a copy of the input array or allow in place changes.
    """
    X = np.array(X, copy=copy)
    if len(X.shape) < 2:
        X = X[None]
    n_points = X.shape[1]
    n_windows = n_points // window_step
    # Padding
    pad_sizes = [(window_size - window_step) // 2,
                 window_size - window_step // 2]
    # TODO: Handling for odd window sizes / steps
    X = np.hstack((np.zeros((X.shape[0], pad_sizes[0])), X,
                   np.zeros((X.shape[0], pad_sizes[1]))))

    clipping_factor = 0.6
    b, a = sg.butter(10, np.pi * 9 / 40)
    voiced_unvoiced = np.zeros((n_windows, 1))
    period = np.zeros((n_windows, 1))
    for window in range(max(n_windows - 1, 1)):
        XX = X.ravel()[window * window_step + np.arange(window_size)]
        XX *= sg.hamming(len(XX))
        XX = sg.lfilter(b, a, XX)
        left_max = np.max(np.abs(XX[:len(XX) // 3]))
        right_max = np.max(np.abs(XX[-len(XX) // 3:]))
        clip_value = clipping_factor * np.min([left_max, right_max])
        XX_clip = np.clip(XX, clip_value, -clip_value)
        XX_corr = np.correlate(XX_clip, XX_clip, mode='full')
        center = np.argmax(XX_corr)
        right_XX_corr = XX_corr[center:]
        prev_window = max([window - 1, 0])
        if voiced_unvoiced[prev_window] > 0:
            # Want it to be harder to turn off than turn on
            strength_factor = .29
        else:
            strength_factor = .3
        start = np.where(right_XX_corr < .3 * XX_corr[center])[0]
        # 20 is hardcoded but should depend on samplerate?
        try:
            start = np.max([20, start[0]])
        except IndexError:
            start = 20
        search_corr = right_XX_corr[start:]
        index = np.argmax(search_corr)
        second_max = search_corr[index]
        if (second_max > strength_factor * XX_corr[center]):
            voiced_unvoiced[window] = 1
            period[window] = start + index - 1
        else:
            voiced_unvoiced[window] = 0
            period[window] = 0
    return np.array(voiced_unvoiced), np.array(period)


def lpc_analysis(X, order=8, window_step=128, window_size=2 * 128,
                 emphasis=0.9, voiced_start_threshold=.9,
                 voiced_stop_threshold=.6, truncate=False, copy=True):
    """
    Extract LPC coefficients from a signal

    Based on code from:
        http://labrosa.ee.columbia.edu/matlab/sws/

    _rParameters
    ----------
    X : ndarray
        Signals to extract LPC coefficients from

    order : int, optional (default=8)
        Order of the LPC coefficients. For speech, use the general rule that the
        order is two times the expected number of formants plus 2.
        This can be formulated as 2 + 2 * (fs // 2000). For approx. signals
        with fs = 7000, this is 8 coefficients - 2 + 2 * (7000 // 2000).

    window_step : int, optional (default=128)
        The size (in samples) of the space between each window

    window_size : int, optional (default=2 * 128)
        The size of each window (in samples) to extract coefficients over

    emphasis : float, optional (default=0.9)
        The emphasis coefficient to use for filtering

    voiced_start_threshold : float, optional (default=0.9)
        Upper power threshold for estimating when speech has started

    voiced_stop_threshold : float, optional (default=0.6)
        Lower power threshold for estimating when speech has stopped

    truncate : bool, optional (default=False)
        Whether to cut the data at the last window or do zero padding.

    copy : bool, optional (default=True)
        Whether to copy the input X or modify in place

    Returns
    -------
    lp_coefficients : ndarray
        lp coefficients to describe the frame

    per_frame_gain : ndarray
        calculated gain for each frame

    residual_excitation : ndarray
        leftover energy which is not described by lp coefficents and gain

    voiced_frames : ndarray
        array of [0, 1] values which holds voiced/unvoiced decision for each
        frame.

    References
    ----------
    D. P. W. Ellis (2004), "Sinewave Speech Analysis/Synthesis in Matlab",
    Web resource, available: http://www.ee.columbia.edu/ln/labrosa/matlab/sws/
    """
    X = np.array(X, copy=copy)
    if len(X.shape) < 2:
        X = X[None]

    n_points = X.shape[1]
    n_windows = int(n_points // window_step)
    if not truncate:
        pad_sizes = [(window_size - window_step) // 2,
                     window_size - window_step // 2]
        # TODO: Handling for odd window sizes / steps
        X = np.hstack((np.zeros((X.shape[0], int(pad_sizes[0]))), X,
                       np.zeros((X.shape[0], int(pad_sizes[1])))))
    else:
        pad_sizes = [0, 0]
        X = X[0, :n_windows * window_step]

    lp_coefficients = np.zeros((n_windows, order + 1))
    per_frame_gain = np.zeros((n_windows, 1))
    residual_excitation = np.zeros(
        int(((n_windows - 1) * window_step + window_size)))
    
    X = sg.lfilter([1, -emphasis], 1, X)
    autocorr_X = np.zeros((n_windows, int(2 * window_size - 1)))
    for window in range(max(n_windows - 1, 1)):
        wtws = int(window * window_step)
        XX = X.ravel()[wtws + np.arange(window_size, dtype="int32")]
        WXX = XX * sg.hanning(window_size)
        autocorr_X[window] = np.correlate(WXX, WXX, mode='full')
        center = np.argmax(autocorr_X[window])
        RXX = autocorr_X[window,
                         np.arange(center, window_size + order, dtype="int32")]
        R = linalg.toeplitz(RXX[:-1])
        solved_R = linalg.pinv(R).dot(RXX[1:])
        filter_coefs = np.hstack((1, -solved_R))
        residual_signal = sg.lfilter(filter_coefs, 1, WXX)
        gain = np.sqrt(np.mean(residual_signal ** 2))
        lp_coefficients[window] = filter_coefs[:lp_coefficients[window].shape[0]]

        per_frame_gain[window] = gain
        assign_range = wtws + np.arange(window_size, dtype="int32")
        residual_excitation[assign_range] += residual_signal / gain
    # Throw away first part in overlap mode for proper synthesis
    residual_excitation = residual_excitation[int(pad_sizes[0]):]
    return lp_coefficients, per_frame_gain, residual_excitation


def lpc_to_frequency(lp_coefficients, per_frame_gain):
    """
    Extract resonant frequencies and magnitudes from LPC coefficients and gains.
    Parameters
    ----------
    lp_coefficients : ndarray
        LPC coefficients, such as those calculated by ``lpc_analysis``

    per_frame_gain : ndarray
       Gain calculated for each frame, such as those calculated
       by ``lpc_analysis``

    Returns
    -------
    frequencies : ndarray
       Resonant frequencies calculated from LPC coefficients and gain. Returned
       frequencies are from 0 to 2 * pi

    magnitudes : ndarray
       Magnitudes of resonant frequencies

    References
    ----------
    D. P. W. Ellis (2004), "Sinewave Speech Analysis/Synthesis in Matlab",
    Web resource, available: http://www.ee.columbia.edu/ln/labrosa/matlab/sws/
    """
    n_windows, order = lp_coefficients.shape

    frame_frequencies = np.zeros((n_windows, (order - 1) // 2))
    frame_magnitudes = np.zeros_like(frame_frequencies)

    for window in range(n_windows):
        w_coefs = lp_coefficients[window]
        g_coefs = per_frame_gain[window]
        roots = np.roots(np.hstack(([1], w_coefs[1:])))
        # Roots doesn't return the same thing as MATLAB... agh
        frequencies, index = np.unique(
            np.abs(np.angle(roots)), return_index=True)
        # Make sure 0 doesn't show up...
        gtz = np.where(frequencies > 0)[0]
        frequencies = frequencies[gtz]
        index = index[gtz]
        magnitudes = g_coefs / (1. - np.abs(roots))
        sort_index = np.argsort(frequencies)
        frame_frequencies[window, :len(sort_index)] = frequencies[sort_index]
        frame_magnitudes[window, :len(sort_index)] = magnitudes[sort_index]
    return frame_frequencies, frame_magnitudes


def lpc_synthesis(lp_coefficients, per_frame_gain, residual_excitation=None,
                  voiced_frames=None, window_step=128, emphasis=0.9):
    """
    Synthesize a signal from LPC coefficients

    Based on code from:
        http://labrosa.ee.columbia.edu/matlab/sws/
        http://web.uvic.ca/~tyoon/resource/auditorytoolbox/auditorytoolbox/synlpc.html

    Parameters
    ----------
    lp_coefficients : ndarray
        Linear prediction coefficients

    per_frame_gain : ndarray
        Gain coefficients

    residual_excitation : ndarray or None, optional (default=None)
        Residual excitations. If None, this will be synthesized with white noise

    voiced_frames : ndarray or None, optional (default=None)
        Voiced frames. If None, all frames assumed to be voiced.

    window_step : int, optional (default=128)
        The size (in samples) of the space between each window

    emphasis : float, optional (default=0.9)
        The emphasis coefficient to use for filtering

    overlap_add : bool, optional (default=True)
        What type of processing to use when joining windows

    copy : bool, optional (default=True)
       Whether to copy the input X or modify in place

    Returns
    -------
    synthesized : ndarray
        Sound vector synthesized from input arguments

    References
    ----------
    D. P. W. Ellis (2004), "Sinewave Speech Analysis/Synthesis in Matlab",
    Web resource, available: http://www.ee.columbia.edu/ln/labrosa/matlab/sws/
    """
    # TODO: Incorporate better synthesis from
    # http://eecs.oregonstate.edu/education/docs/ece352/CompleteManual.pdf
    window_size = 2 * window_step
    [n_windows, order] = lp_coefficients.shape

    n_points = (n_windows + 1) * window_step
    n_excitation_points = n_points + window_step + window_step // 2

    random_state = np.random.RandomState(1999)
    if residual_excitation is None:
        # Need to generate excitation
        if voiced_frames is None:
            # No voiced/unvoiced info
            voiced_frames = np.ones((lp_coefficients.shape[0], 1))
        residual_excitation = np.zeros((n_excitation_points))
        f, m = lpc_to_frequency(lp_coefficients, per_frame_gain)
        t = np.linspace(0, 1, window_size, endpoint=False)
        hanning = sg.hanning(window_size)
        for window in range(n_windows):
            window_base = window * window_step
            index = window_base + np.arange(window_size)
            if voiced_frames[window]:
                sig = np.zeros_like(t)
                cycles = np.cumsum(f[window][0] * t)
                sig += sg.sawtooth(cycles, 0.001)
                residual_excitation[index] += hanning * sig
            residual_excitation[index] += hanning * 0.01 * random_state.randn(
                window_size)
    else:
        n_excitation_points = residual_excitation.shape[0]
        n_points = n_excitation_points + window_step + window_step // 2
    residual_excitation = np.hstack((residual_excitation,
                                     np.zeros(window_size)))
    if voiced_frames is None:
        voiced_frames = np.ones_like(per_frame_gain)

    synthesized = np.zeros((n_points))
    for window in range(n_windows):
        window_base = window * window_step
        oldbit = synthesized[window_base + np.arange(window_step)]
        w_coefs = lp_coefficients[window]
        if not np.all(w_coefs):
            # Hack to make lfilter avoid
            # ValueError: BUG: filter coefficient a[0] == 0 not supported yet
            # when all coeffs are 0
            w_coefs = [1]
        g_coefs = voiced_frames[window] * per_frame_gain[window]
        index = window_base + np.arange(window_size)
        newbit = g_coefs * sg.lfilter([1], w_coefs,
                                      residual_excitation[index])
        synthesized[index] = np.hstack((oldbit, np.zeros(
            (window_size - window_step))))
        synthesized[index] += sg.hanning(window_size) * newbit
    
    return synthesized


def lpc_compress_decompress(X, samplerate = 16000, lpc_order = 10):
    window_size = 400
    dct_components = 200
    lpc_order = lpc_order

    window_step = window_size // 2
    a, g, e = lpc_analysis(X, order=lpc_order,
                           window_step=window_step,
                           window_size=window_size, emphasis=0.9,
                           copy=True)

    
    v, p = voiced_unvoiced(X, window_size=window_size,
                                       window_step=window_step)

    
    

    noisy_lpc = lpc_synthesis(a, g, voiced_frames=v,
                                          emphasis=0.9,
                                          window_step=window_step)
    
    return a, np.array(noisy_lpc, dtype= 'float32')


def lpc_analysis_tf(_X, order=8, window_step=128, window_size=2 * 128,
                 emphasis=0.9, voiced_start_threshold=.9,
                 voiced_stop_threshold=.6, truncate=False, copy=True):
    """
    Extract LPC coefficients from a signal

    Based on code from:
        http://labrosa.ee.columbia.edu/matlab/sws/

    _rParameters
    ----------
    X : ndarray
        Signals to extract LPC coefficients from

    order : int, optional (default=8)
        Order of the LPC coefficients. For speech, use the general rule that the
        order is two times the expected number of formants plus 2.
        This can be formulated as 2 + 2 * (fs // 2000). For approx. signals
        with fs = 7000, this is 8 coefficients - 2 + 2 * (7000 // 2000).

    window_step : int, optional (default=128)
        The size (in samples) of the space between each window

    window_size : int, optional (default=2 * 128)
        The size of each window (in samples) to extract coefficients over

    emphasis : float, optional (default=0.9)
        The emphasis coefficient to use for filtering

    voiced_start_threshold : float, optional (default=0.9)
        Upper power threshold for estimating when speech has started

    voiced_stop_threshold : float, optional (default=0.6)
        Lower power threshold for estimating when speech has stopped

    truncate : bool, optional (default=False)
        Whether to cut the data at the last window or do zero padding.

    copy : bool, optional (default=True)
        Whether to copy the input X or modify in place

    Returns
    -------
    lp_coefficients : ndarray
        lp coefficients to describe the frame

    per_frame_gain : ndarray
        calculated gain for each frame

    residual_excitation : ndarray
        leftover energy which is not described by lp coefficents and gain

    voiced_frames : ndarray
        array of [0, 1] values which holds voiced/unvoiced decision for each
        frame.

    References
    ----------
    D. P. W. Ellis (2004), "Sinewave Speech Analysis/Synthesis in Matlab",
    Web resource, available: http://www.ee.columbia.edu/ln/labrosa/matlab/sws/
    """
    X = tf.identity(_X)
    if len(X.shape) < 2:
        X = X[None]

    n_points = int(X.shape[1])
    n_windows = int(n_points // window_step)
    if not truncate:
        pad_sizes = [(window_size - window_step) // 2,
                     window_size - window_step // 2]
        # TODO: Handling for odd window sizes / steps
        X = tf.concat((tf.zeros((X.shape[0], int(pad_sizes[0])), dtype= 'float32'), X,
                       tf.zeros((X.shape[0], int(pad_sizes[1])), dtype= 'float32')), axis = 1)
    else:
        pad_sizes = [0, 0]
        X = X[0, :n_windows * window_step]
    X = X[0]

    lp_coefficients = []
    per_frame_gains = []
    residual_excitation = tf.zeros(
        int(((n_windows - 1) * window_step + window_size)), dtype = 'float32')
    
    # Pre-emphasis high-pass filter
    _filter = tf.constant([-emphasis, 1], dtype='float32')
    X = tf.nn.conv1d(
        X[None,:,None], _filter[:,None,None], stride=1, padding='SAME'
    )[0,:,0]
    
    tf_hanning = tf.constant(sg.hanning(window_size), dtype = 'float32')
    for window in range(max(n_windows - 1, 1)):
        wtws = int(window * window_step)
        XX = X[wtws:wtws + window_size]
        WXX = XX * tf_hanning
        WXX_padded = tf.pad(WXX, [[WXX.shape[0]-1, WXX.shape[0]-1]])
        _correlate = tf.nn.conv1d(WXX_padded[None,:,None], WXX[:,None,None], stride=1, padding='VALID')
        autocorr_X_window = _correlate[0,:,0]
        center = window_size-1
        RXX = autocorr_X_window[center: window_size + order]
        R = tf.linalg.LinearOperatorToeplitz(RXX[:-1], RXX[:-1]).to_dense()
        solved_R = tf.tensordot( tf.linalg.pinv(R), (RXX[1:]), axes = 1 )
        filter_coefs = tf.concat((tf.ones(1), -solved_R), axis = 0)
        residual_signal = tf.nn.conv1d(WXX[None,:,None], filter_coefs[::-1][:,None,None], stride=1, padding='SAME')[0,:,0]
        gain = tf.math.sqrt(tf.math.reduce_mean(residual_signal ** 2))
        per_frame_gains.append(gain)
        lp_coefficients.append( filter_coefs[:order + 1] )

    lp_coefficients.append(tf.zeros([order + 1]))
    per_frame_gains.append(tf.constant(0, dtype = 'float32'))
    return lp_coefficients, per_frame_gains, residual_excitation



def lfilter_tf_batched(input_signal_tf, w_coeffs_tf, window_size, lpc_order = 20):
    
    y = []
    x = input_signal_tf
    for idx in range(window_size):
        _len_filter = min(int(lpc_order + 1), len(y))
        if _len_filter > 1:
            _dot_product = tf.math.reduce_sum(w_coeffs_tf[:,1:_len_filter]* tf.transpose( y_tensor[::-1][:_len_filter-1,:] ), axis = 1)
            y.append( x[:,idx] - _dot_product )
        else:
            y.append( x[:,idx] )
        y_tensor = tf.stack(y)
    return y_tensor

def generate_residual_excitation(voiced_frames, lp_coefficients_np, per_frame_gain_np, window_size, window_step, n_windows, n_excitation_points):
    residual_excitation = np.zeros((n_excitation_points))
    f, m = lpc_to_frequency(lp_coefficients_np, per_frame_gain_np)
    t = np.linspace(0, 1, window_size, endpoint=False)
    hanning = sg.hanning(window_size)
    
    random_state = np.random.RandomState(1999)
    
    for window in range(n_windows):
        window_base = window * window_step
        index = window_base + np.arange(window_size)
        if voiced_frames[window]:
            sig = np.zeros_like(t)
            cycles = np.cumsum(f[window][0] * t)
            sig += sg.sawtooth(cycles, 0.001)
            residual_excitation[index] += hanning * sig
        residual_excitation[index] += hanning * 0.01 * random_state.randn(
            window_size)
    residual_excitation = np.hstack((residual_excitation,
                                     np.zeros(window_size)))
    return residual_excitation.astype('float32')

def lpc_synthesis_tf(lp_coefficients, per_frame_gain, residual_excitation=None,
                  voiced_frames=None, window_step=128, emphasis=0.9, lpc_order = 20):
    
    window_size = 2 * window_step
    [n_windows, order] = lp_coefficients.shape
    n_windows = int(n_windows)
    order = int(order)
    
    n_points = (n_windows + 1) * window_step
    n_excitation_points = n_points + window_step + window_step // 2
    _generate_residual_excitation = lambda x, y, z: generate_residual_excitation(x, y, z, window_size, window_step, n_windows, n_excitation_points)
    residual_excitation_tf = tf.numpy_function(_generate_residual_excitation, [voiced_frames, lp_coefficients, per_frame_gain], Tout=tf.float32)

    synthesized = tf.zeros([n_points], dtype = 'float32')
    tf_hanning = tf.constant( sg.hanning(window_size), dtype = 'float32' )
    filtered_outs = []
    new_bits = []
    
    residual_excitation_tf_batches = []
    for window in range(n_windows):
        window_base = window * window_step
        residual_excitation_tf_batches.append( residual_excitation_tf[window_base:window_base + window_size] )
        window_base = window_base + window_step
    residual_excitation_tf_batches = tf.stack(residual_excitation_tf_batches)
    filtered_out_batches = lfilter_tf_batched(residual_excitation_tf_batches, lp_coefficients, window_size, lpc_order)
    updates = []
    for window in range(n_windows):
        window_base = window * window_step
        
        oldbit = synthesized[window_base: window_base + window_step]
        w_coefs = lp_coefficients[window]
        g_coefs = voiced_frames[window] * per_frame_gain[window]
        filtered_out = filtered_out_batches[:, window]
        filtered_outs.append(filtered_out)
        newbit = g_coefs * filtered_out
        new_bits.append(newbit * tf_hanning)
        
        si = window_base
        ei = min(window_base + window_size, n_points)
        index_tf = tf.constant( np.arange(si, ei), dtype = 'int32' )
        _add_tensor = tf.pad(tf_hanning[0:ei-si] * newbit[0:ei-si], [[si, n_points - ei]])
        synthesized = synthesized + _add_tensor
        
    return synthesized

def voiced_unvoiced_wrapper(X, window_size=256, window_step=128):
    v,_ =  voiced_unvoiced(X, window_size, window_step)
    return v.astype('float32')
    
def lpc_compress_decompress_tf(X_tf, samplerate = 16000, lpc_order = 10):
    window_size = 400
    dct_components = 200
    lpc_order = lpc_order

    window_step = window_size // 2
    a, g, _ = lpc_analysis_tf(X_tf, order=lpc_order,
                           window_step=window_step,
                           window_size=window_size, emphasis=0.9,
                           copy=True)
    
    a = tf.stack(a)
    g = tf.stack(g)
    
    _voiced_unvoiced_np_func = lambda _x: voiced_unvoiced_wrapper(_x, window_size=window_size,
                                       window_step=window_step)
    
    # voiced frames only used for generating the excitation signal, therefore reusing the exact numpy implementation
    v = tf.numpy_function(_voiced_unvoiced_np_func, [X_tf], Tout=tf.float32)

    
    noisy_lpc = lpc_synthesis_tf(a, g, voiced_frames=v,
                                          emphasis=0.9,
                                          window_step=window_step, 
                                          lpc_order = lpc_order)
    
    return a, noisy_lpc





