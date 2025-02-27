from typing import Union, Optional, Dict, List, Tuple, Callable
from numpy import iscomplex, ndarray, linspace, pi, sin, cos, expand_dims, concatenate
import math
import numpy as np
import scipy
import scipy.linalg as slin
import scipy.io as sio
from sklearn.metrics import accuracy_score
from scipy import signal
from scipy.signal import cheb1ord, filtfilt, cheby1
from itertools import combinations
import matplotlib.pyplot as plt
import mne

def loadmat(filename):
    '''
    this function should be called instead of direct sio.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    Notes: only works for mat before matlab v7.3
    '''
    def _check_keys(d):
        '''
        checks if entries in dictionary are mat-objects. If yes
        todict is called to change them to nested dictionaries
        '''
        for key in d:
            if isinstance(d[key], sio.matlab.mio5_params.mat_struct):
                d[key] = _todict(d[key])
            elif isinstance(d[key], ndarray):
                d[key] = _tolist(d[key])
        return d

    def _todict(matobj):
        '''
        A recursive function which constructs from matobjects nested dictionaries
        '''
        d = {}
        for strg in matobj._fieldnames:
            elem = matobj.__dict__[strg]
            if isinstance(elem, sio.matlab.mio5_params.mat_struct):
                d[strg] = _todict(elem)
            elif isinstance(elem, ndarray):
                d[strg] = _tolist(elem)
            else:
                d[strg] = elem
        return d

    def _tolist(elem):
        '''
        A recursive function which constructs lists from cellarrays
        (which are loaded as numpy ndarrays), recursing into the elements
        if they contain matobjects.
        '''
        if elem.dtype == object:
            elem_list = []
            for sub_elem in elem:
                if isinstance(sub_elem, sio.matlab.mio5_params.mat_struct):
                    elem_list.append(_todict(sub_elem))
                elif isinstance(sub_elem, ndarray):
                    elem_list.append(_tolist(sub_elem))
                else:
                    elem_list.append(sub_elem)
            return elem_list
        else:
            return elem

    data = sio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)

def suggested_weights_filterbank(num_subbands: Optional[int] = 5, data_type='') -> List[float]:
    """
    Provide suggested weights of filterbank for benchmark dataset

    Returns
    -------
    weights_filterbank : List[float]
        Suggested weights of filterbank
    """
    return [i**(-1.25)+0.25 for i in range(1,num_subbands+1,1)]

def filter(x, wp=[6, 90], ws=[4, 100], fs=250):
    nyq = 0.5 * fs
    Wp = [wp[0]/nyq, wp[1]/nyq]
    Ws = [ws[0]/nyq, ws[1]/nyq]
    N, Wn=cheb1ord(Wp, Ws, 3, 40)
    b, a = cheby1(N, 0.5, Wn,'bandpass')
    x = filtfilt(b,a,x,padlen=3*(max(len(b),len(a))-1)) # apply filter
    return x

def notch_filter(data, srate):
    """

    Parameters
    ----------
    data: numpy array of any shape, the last dimension will be filtered.
    fs: sampling rate
    Q: f_remove/bandwidth. A larger Q indicate a narrower range of filtering.
    f_remove: the frequency that needs to be removed.

    Returns
    -------

    """
    b1, a1 = signal.cheby1(4, 2, [47.0 / (srate / 2), 53.0 / (srate / 2)], 'bandstop')
    return signal.filtfilt(b1, a1, data, axis=1, padtype='odd', padlen=3 * (max(len(b1), len(a1)) - 1))

def lowpass(data: np.ndarray, cutoff: float, sample_rate: float, poles: int = 5):
    sos = signal.butter(poles, cutoff, 'lowpass', fs=sample_rate, output='sos')
    filtered_data = signal.sosfiltfilt(sos, data)
    return filtered_data

def highpass(data: np.ndarray, cutoff: float, sample_rate: float, poles: int = 5):
    sos = signal.butter(poles, cutoff, 'highpass', fs=sample_rate, output='sos')
    filtered_data = signal.sosfiltfilt(sos, data)
    return filtered_data

def bandpass(data: np.ndarray, edges: list[float], sample_rate: float, poles: int = 5):
    sos = scipy.signal.butter(poles, edges, 'bandpass', fs=sample_rate, output='sos')
    filtered_data = scipy.signal.sosfiltfilt(sos, data)
    return filtered_data


def get_ref_sig(freqs: list, phases: list, srate: float,
                sig_len: float,
                N: int,
                ignore_stim_phase: bool = False) -> List[ndarray]:
    """
    Construct sine-cosine-based reference signals  for all stimuli

    Parameters
    ----------
    freqs : List[float]

    phases : List[float]

    srate: float
        sampling rate of the signal

    sig_len : float
        signal length (in second)
    N : int
        Number of harmonics

    Returns
    -------
    ref_sig : List[ndarray]
        List of reference signals
    """
    if ignore_stim_phase:
        phases = [0 for _ in range(len(freqs))]

    L = int(np.floor(sig_len * srate))  # data length
    ref_sig = [gen_ref_sin(freq, srate, L, N, phase) for freq, phase in
               zip(freqs, phases)]

    return ref_sig   # list: (class_num, ), array: (2*N, L)




def gen_ref_sin(freq: float,
                srate: int,
                L: int,
                N: int,
                phase: float) -> ndarray:
    """
    Generate sine-cosine-based reference signals of one stimulus

    Parameters
    ----------
    freq : float
        Stimulus frequency
    srate : int
        Sampling rate
    L : int
        Signal length
    N : int
        Number of harmonics
    phase : float
        Stimulus phase

    Returns
    -------
    ref_sig: ndarray
        Sine-cosine-based reference signal
        2N * L
    """

    t = linspace(0, (L - 1) / srate, L)
    t = expand_dims(t, 0)

    y = []
    for n in range(1, N + 1, 1):
        y.append(sin(2 * pi * n * freq * t + n * phase))
        y.append(cos(2 * pi * n * freq * t + n * phase))
    y = concatenate(y, axis=0)

    return y


def filterbank(X: ndarray, srate,
               num_subbands: Optional[int] = 5) -> ndarray:
    """
    Suggested filterbank function for benchmark dataset

    Parameters:
    -----------
    -X : ndarray
        shape: (channel, time)

    return:
    -----------
    -filterbank_X: ndarray
        shape: (num_subbands, channel, time)

    """

    filterbank_X = np.zeros((num_subbands, X.shape[0], X.shape[1]))

    for k in range(1, num_subbands + 1, 1):
        Wp = [(8 * k) / (srate / 2), 90 / (srate / 2)]
        Ws = [(8 * k - 2) / (srate / 2), 100 / (srate / 2)]

        gstop = 40
        while gstop >= 20:
            try:
                N, Wn = signal.cheb1ord(Wp, Ws, 3, gstop)
                bpB, bpA = signal.cheby1(N, 0.5, Wn, btype='bandpass')
                filterbank_X[k - 1, :, :] = signal.filtfilt(bpB, bpA, X, axis=1, padtype='odd',
                                                            padlen=3 * (max(len(bpB), len(bpA)) - 1))
                break
            except:
                gstop -= 1
        if gstop < 20:
            raise ValueError("""
Filterbank cannot be processed. You may try longer signal lengths.
Filterbank order: {:n}
gstop: {:n}
bpB: {:s}
bpA: {:s}
Required signal length: {:n}
Signal length: {:n}""".format(k,
                              gstop,
                              str(bpB),
                              str(bpA),
                              3 * (max(len(bpB), len(bpA)) - 1),
                              X.shape[1]))

    return filterbank_X



def ITR(N, P, winBIN):
    '''
    Calculate the ITR.

    Parameters:
    -----------
    - N: number of classes
    - P: the accuracy
    - winBIN: the time of the signal (in seconds).
    :return:
    '''
    winBIN = winBIN + 0.5

    if P == 1:
        ITR = math.log2(N) * 60 / winBIN
    elif P == 0:
        ITR = (math.log2(N) + 0 + (1 - P) * math.log2((1 - P) / (N - 1))) * 60 / winBIN
        ITR = 0
    else:
        ITR = (math.log2(N) + P * math.log2(P) + (1 - P)
               * math.log2((1 - P) / (N - 1))) * 60 / winBIN

    return ITR

def qr_list(X : List[ndarray]) -> Tuple[List[ndarray], List[ndarray], List[ndarray]]:
    """
    QR decomposition of list X
    Note: Elements in X will be transposed first and then decomposed

    Parameters
    ----------
    X : List[ndarray]

    Returns
    -------
    Q : List[ndarray]
    R : List[ndarray]
    P : List[ndarray]
    """
    Q = []
    R = []
    P = []
    for el in X:
        if len(el.shape) == 2: # reference signal
            Q_tmp, R_tmp, P_tmp = qr_remove_mean(el.T)
            Q.append(Q_tmp)
            R.append(R_tmp)
            P.append(P_tmp)
        elif len(el.shape) == 3: # template signal
            Q_tmp = []
            R_tmp = []
            P_tmp = []
            for k in range(el.shape[0]):
                Q_tmp_tmp, R_tmp_tmp, P_tmp_tmp = qr_remove_mean(el[k,:,:].T)
                Q_tmp.append(np.expand_dims(Q_tmp_tmp, axis=0))
                R_tmp.append(np.expand_dims(R_tmp_tmp, axis=0))
                P_tmp.append(np.expand_dims(P_tmp_tmp, axis=0))
            Q.append(np.concatenate(Q_tmp,axis=0))
            R.append(np.concatenate(R_tmp,axis=0))
            P.append(np.concatenate(P_tmp,axis=0))
        else:
            raise ValueError('Unknown data type')
    return Q, R, P


def sum_list(X: list) -> ndarray:
    """
    Calculate sum of a list

    Parameters
    ------------
    X: list

    Returns
    -------------
    sum_X: ndarray
    """
    sum_X = None
    for x in X:
        if type(x) is list:
            x = sum_list(x)
        if sum_X is None:
            sum_X = x
        else:
            sum_X = sum_X + x
    return sum_X

def mean_list(X: list) -> ndarray:
    """
    Calculate mean of a list

    Parameters
    -----------
    X: list

    Returns
    ----------
    mean_X: ndarray
    """

    # X: list: (trial_num, ), array: (n_delay*channel, 2*time)
    tmp = []
    for X_single_trial in X:
        if type(X_single_trial) is list:
            X_single_trial = mean_list(X_single_trial)
        tmp.append(np.expand_dims(X_single_trial, axis = 0))
    tmp = np.concatenate(tmp, axis = 0)
    return np.mean(tmp, axis=0)


def eigvec(X : ndarray,
           Y : Optional[ndarray] = None):
    """
    Calculate eigenvectors

    Parameters
    -----------------
    X : ndarray
        A complex or real matrix whose eigenvalues and eigenvectors will be computed.
    Y : ndarray
        If Y is given, eig(Y\X), or say  eig(X, Y), will be computed

    Returns
    ---------------
    eig_vec : ndarray
        Eigenvectors. The order follows the corresponding eigenvalues (from high to low values)
    """
    if Y is None:
        eig_d1, eig_v1 = slin.eig(X) #eig(X)
    else:
        eig_d1, eig_v1 = slin.eig(X, Y) #eig(Y\X)

    if len(eig_d1.shape) == 2:
        eig_d1 = np.diagonal(eig_d1)

    sort_idx = np.argsort(eig_d1)[::-1]
    eig_vec = eig_v1[:,sort_idx]

    if Y is not None:
        square_val = np.diag(eig_vec.T @ Y @ eig_vec)
        norm_v = np.sqrt(square_val)
        eig_vec = eig_vec/norm_v

    if np.iscomplex(eig_vec).any():
        eig_vec = np.real(eig_vec)

    return eig_vec


def qr_remove_mean(X: ndarray) -> Tuple[ndarray, ndarray, ndarray]:
    """
    Remove column mean and QR decomposition

    Parameters
    ----------
    X : ndarray
        (M * N)

    Returns
    -------
    Q : ndarray
        (M * K)
    R : ndarray
        (K * N)
    P : ndarray
        (N,)
    """

    X_remove_mean = X - np.mean(X, 0)

    Q, R, P = slin.qr(X_remove_mean, mode='economic', pivoting=True)

    return Q, R, P

def nextpow2(n):
    '''
    Retrun the first P such that 2 ** P >= abs(n).
    '''
    return np.ceil(np.log2(np.abs(n)))


def fft(X: ndarray,
        fs: float,
        detrend_flag: bool = True,
        NFFT: Optional[int] = None):
    """
    Calculate FFT

    Parameters
    -----------
    X : ndarray
        Input signal. The shape is (1*N) where N is the sampling number.
    fs : float
        Sampling freqeuncy.
    detrend_flag : bool
        Whether detrend. If True, X will be detrended first. Default is True.
    NFFT : Optional[int]
        Number of FFT. If None, NFFT is equal to 2^nextpow2(X.shape[1]). Default is None.

    Returns
    -------------
    freqs : ndarray
        Corresponding frequencies
    fft_res : ndarray
        FFT result
    """
    X_raw, X_col = X.shape
    if X_raw != 1:
        raise ValueError('The row number of the input signal for the FFT must be 1.')
    if X_col == 1:
        raise ValueError('The column number of the input signal for the FFT cannot be 1.')
    if NFFT is None:
        NFFT = 2 ** nextpow2(X_col)
    if type(NFFT) is not int:
        NFFT = int(NFFT)

    if detrend_flag:
        X = signal.detrend(X, axis=1)

    fft_res = np.fft.fft(X, NFFT, axis=1)
    freqs = np.fft.fftfreq(NFFT, 1 / fs)
    freqs = np.expand_dims(freqs, 0)
    if NFFT & 0x1:
        fft_res = fft_res[:, :int((NFFT + 1) / 2)]
        freqs = freqs[:, :int((NFFT + 1) / 2)]
    else:
        fft_res = fft_res[:, :int(NFFT / 2)]
        freqs = freqs[:, :int(NFFT / 2)]
    # fft_res = fft_res/X_col

    return freqs, fft_res

def freqs_snr(X : ndarray,
              target_fre : float,
              srate : float,
              Nh : int,
              detrend_flag : bool = True,
              NFFT : Optional[int] = None):
    """
    Calculate FFT and then calculate SNR
    """
    freq, fft_res = fft(X, srate, detrend_flag = detrend_flag, NFFT = NFFT)
    abs_fft_res = np.abs(fft_res)

    stim_amp = 0
    for n in range(Nh):
        freLoc = np.argmin(np.abs(freq - (target_fre*(n+1))))
        stim_amp += abs_fft_res[0,freLoc]
    snr = 10*np.log10(stim_amp/(np.sum(abs_fft_res)-stim_amp))
    return snr

def sine_snr(X : ndarray,
             ref : ndarray):
    """
    Calculate SNR using reference
    """
    ref = ref/ref.shape[1]
    ref_mul = ref.T @ ref
    X_ref = np.trace(X @ ref.T @ ref @ X.T)
    X_non_ref = np.trace(X @ (np.eye(ref.shape[1]) - ref_mul) @ X.T)
    return 10*np.log10(X_ref/X_non_ref)


def get_snr_single_trial(X, freq, phase, channel_idx=0,
                         sig_len : float = None,
                         Nh : int = 1,
                         srate : Optional[float] = None,
                         detrend_flag : bool = True,
                         NFFT : Optional[int] = None,
                         type : str = 'sine',
                         harmonic_num : int = 5,
                         ignore_stim_phase : bool = False):
    """
    Calculate the SNR of one single trial

    Parameters:
    -----------
    -X: ndarray
       (channel, time), data of a single trial
    -freq: float
        Stimulus frequency of this trial
    -phase: float
        Stimulus phase of this trial
    -channel_idx: int
        Choose a single channel to calculate the SNR.
    -type: str
        'fft' or 'sine'
    -sig_len: float
        signal length (in seconds)
    """
    X = X[channel_idx:channel_idx+1, :]

    if sig_len is None:
        sig_len = X.shape[1] / srate

    if type.lower() == 'fft':
        return freqs_snr(X, freq, srate, Nh,
                        detrend_flag = detrend_flag,
                        NFFT = NFFT)
    elif type.lower() == 'sine':
        ref_sig = get_ref_sig([freq], [phase], srate, sig_len, harmonic_num, ignore_stim_phase)[0]
        return sine_snr(X, ref_sig)
    else:
        raise ValueError('Unknown SNR type')

def freqs_phase(X : ndarray,
                target_fre : float,
                target_phase : float,
                srate : float,
                detrend_flag : bool = True,
                NFFT : Optional[int] = None):
    """
    Calculate FFT and then calculate phase
    """
    freq, fft_res = fft(X, srate, detrend_flag = detrend_flag, NFFT = NFFT)
    angle_fft_res = np.angle(fft_res)
    freLoc = np.argmin(np.abs(freq - target_fre))
    stim_phase = angle_fft_res[0,freLoc]
    if stim_phase != target_phase:
        k1 = np.floor((target_phase - stim_phase)/(2*np.pi))
        k2 = -k1
        k3 = np.floor((stim_phase - target_phase)/(2*np.pi))
        k4 = -k3
        k = np.array([k1,k2,k3,k4])
        k_loc = np.argmin(np.abs(stim_phase + 2*np.pi*k - target_phase))
        stim_phase = stim_phase + 2*np.pi*k[k_loc]
    return stim_phase

def get_phase_single_trial(X : ndarray,
                         freq: float,
                         phase : float,
                         channel_idx : int,
                         srate : int = 250,
                         detrend_flag : bool = True,
                         NFFT : Optional[int] = None,
                         remove_target_phase : bool = False):
    """
    Calculate the phase of one single trial

    Parameters:
    -----------
    -X: ndarray
       (channel, time), data of a single trial
    -freq: float
        Stimulus frequency of this trial
    -phase: float
        Stimulus phase of this trial
    -channel_idx: int
        Choose a single channel to calculate the SNR.
    """

    X = X[channel_idx:channel_idx + 1, :]

    trial_phase = freqs_phase(X, freq, phase,
                       srate,
                       detrend_flag = detrend_flag,
                       NFFT = NFFT)
    if remove_target_phase:
        trial_phase -= phase
    return trial_phase



def zero_channels(X_origin, channel_bug):
    """
    将channel_bug中包含的通道中的值赋0。

    参数:
    X -- numpy数组, 维度为(channel_num, time_points, sample_num)
    channel_bug -- 需要赋值为0的通道索引数组

    返回:
    X_zeroed -- 赋值后的numpy数组
    """
    
    X_attacked = X_origin.copy()  
    for channel in channel_bug:
        if channel < X_origin.shape[0]:  
            
            X_attacker = np.load('S1_attacker.pickle', allow_pickle=True)
            
            X_attacker = X_attacker.squeeze().transpose((1, 0, 2))
           
            X_attacker = X_attacker[:, :X_origin.shape[1], :]
            
            X_attacked[channel, :, :] = X_attacker[channel, :, :]
            
            print(np.all(X_attacked[0, :, :] == X_origin[0, :, :]))
            print(np.all(X_attacked[1, :, :] == X_origin[1, :, :]))
            print(np.all(X_attacked[2, :, :] == X_origin[2, :, :]))

        else:
            print(f"通道 {channel} 超出范围，有效范围为0到{X_origin.shape[0] - 1}。")
    return X_attacked


def bug_channel_select(num_bug_channel, channel_num):
    """
    生成包含所有可能的不同通道组合的列表。

    参数:
    num_bug_channel -- 需要赋值为0的通道数量
    channel_num -- 总通道数量

    返回:
    channel_combinations -- 包含所有可能的通道组合的列表
    """

    channel_combinations = list(combinations(range(channel_num), num_bug_channel))
    return channel_combinations


# Create a custom montage
# Provided text data for each electrode, split into lines
electrode_text_data = """
1 -18 0.51111 FP1
2 0 0.51111 FPZ
3 18 0.51111 FP2
4 -23 0.41111 AF3
5 23 0.41111 AF4
6 -54 0.51111 F7
7 -49 0.41667 F5
8 -39 0.33333 F3
9 -22 0.27778 F1
10 0 0.25556 FZ
11 22 0.27778 F2
12 39 0.33333 F4
13 49 0.41667 F6
14 54 0.51111 F8
15 -72 0.51111 FT7
16 -69 0.39444 FC5
17 -62 0.27778 FC3
18 -45 0.17778 FC1
19 0 0.12778 FCz
20 45 0.17778 FC2
21 62 0.27778 FC4
22 69 0.39444 FC6
23 72 0.51111 FT8
24 -90 0.51111 T7
25 -90 0.38333 C5
26 -90 0.25556 C3
27 -90 0.12778 C1
28 90 0 CZ
29 90 0.12778 C2
30 90 0.25556 C4
31 90 0.38333 C6
32 90 0.51111 T8
33 90 0.71111 M1
34 -108 0.51111 TP7
35 -111 0.39444 CP5
36 -118 0.27778 CP3
37 -135 0.17778 CP1
38 180 0.12778 CPZ
39 135 0.17778 CP2
40 118 0.27778 CP4
41 111 0.39444 CP6
42 108 0.51111 TP8
43 -90 0.71111 M2
44 -126 0.51111 P7
45 -131 0.41667 P5
46 -141 0.33333 P3
47 -158 0.27778 P1
48 180 0.25556 PZ
49 158 0.27778 P2
50 141 0.33333 P4
51 131 0.41667 P6
52 126 0.51111 P8
53 -144 0.51111 PO7
54 -144 0.44722 PO5
55 -157 0.41111 PO3
56 180 0.38333 POz
57 157 0.41111 PO4
58 144 0.44722 PO6
59 144 0.51111 PO8
60 -135 0.72222 CB1
61 -162 0.51111 O1
62 180 0.51111 Oz
63 162 0.51111 O2
64 135 0.72222 CB2
"""

def plv(x, freq, phase, srate):
    '''
    Calculate the phase locking value (PLV) using multiple trials corresponding to a
    specific stimulus frequency.

    Parameters:
    -----------
    - x : list: (trials, ), ndarray: (filterbank, channel, time)
    - freq: float
    - phase: float

    Return:
    --------
    - plvs: ndarray
        shape: (filterbank, channel), the PLV for each channel calculated using all trials.

    References:
    ------------
    Sharon, O., & Nir, Y. (2018). Attenuated Fast Steady-State Visual Evoked Potentials During Human Sleep.
    https://doi.org/10.1093/cercor/bhx043
    https://praneethnamburi.com/2011/08/10/plv/
    https://dsp.stackexchange.com/questions/25165/phase-locking-value-phase-synchronization
    '''
    trial_num = len(x)
    filterbank_num, channel_num, sig_len = x[0].shape
    phases = [np.zeros((filterbank_num, channel_num)) for _ in range(trial_num)]
    # calculate the phase of channels in each trial corresponding to a specific freq.
    for fb_idx in range(filterbank_num):
        fb_x = [trial_data[fb_idx, :, :] for trial_data in x]
        # list: (trial, ), ndarray: (channel, sig_len)
        for trial_idx in range(trial_num):
            fb_trial_x = fb_x[trial_idx]
            for ch_idx in range(channel_num):
                trial_ch_phase = get_phase_single_trial(fb_trial_x, freq, phase, channel_idx=ch_idx, srate=srate)
                phases[trial_idx][fb_idx, ch_idx] = trial_ch_phase

    plvs = np.zeros((filterbank_num, channel_num))

    for fb_idx in range(filterbank_num):
        fb_phases = [trial_phases[fb_idx, :] for trial_phases in phases]
        fb_phases = np.stack(fb_phases, axis=0).T   # (channel, trial)
        fb_plv = np.abs(np.sum(np.exp(np.complex(0,1)*fb_phases), axis=1))/trial_num  # (channel, )
        plvs[fb_idx, :] = fb_plv

    return plvs

def getTrialData(x, y, idx):
    '''
    Get trials corresponding to a stimulus specified by idx.

    Parameters:
    -----------
    - x: list: (trials, ), ndarray: any shape
    - y: list: (trials, )
    - idx: index of the specified frequency

    Return:
    -----------
    - trial_x: list: (specified_trial_num, ), ndarray: same shape as x
    '''

    trial_x = []
    for trial_idx, trial_y in enumerate(y):
        if trial_y == idx:
            trial_x.append(x[trial_idx])

    return trial_x


def cal_correlations_with_template(multi_ch_data, template):
    '''
    Calculate the correlations between each channel of the multi-channel data and the
    template.

    Parameters:
    -----------
    - multi_ch_data : ndarray
        shape: (channel, time)
    - template : ndarray
        shape: (time, ), same time length with multi_ch_data

    Return:
    --------
    - correlations : ndarray
        shape: (channel, )
    '''

    ch_num = multi_ch_data.shape[0]

    correlations = []
    for ch_idx in range(ch_num):
        r = np.corrcoef(multi_ch_data[ch_idx, :], template)[0, 1]
        correlations.append(r)

    correlations = np.array(correlations)

    return correlations


def get_channel_information():
    electrode_text_data = """
            1 -18 0.51111 FP1
            2 0 0.51111 FPZ
            3 18 0.51111 FP2
            4 -23 0.41111 AF3
            5 23 0.41111 AF4
            6 -54 0.51111 F7
            7 -49 0.41667 F5
            8 -39 0.33333 F3
            9 -22 0.27778 F1
            10 0 0.25556 FZ
            11 22 0.27778 F2
            12 39 0.33333 F4
            13 49 0.41667 F6
            14 54 0.51111 F8
            15 -72 0.51111 FT7
            16 -69 0.39444 FC5
            17 -62 0.27778 FC3
            18 -45 0.17778 FC1
            19 0 0.12778 FCZ
            20 45 0.17778 FC2
            21 62 0.27778 FC4
            22 69 0.39444 FC6
            23 72 0.51111 FT8
            24 -90 0.51111 T7
            25 -90 0.38333 C5
            26 -90 0.25556 C3
            27 -90 0.12778 C1
            28 90 0 CZ
            29 90 0.12778 C2
            30 90 0.25556 C4
            31 90 0.38333 C6
            32 90 0.51111 T8
            33 90 0.71111 M1
            34 -108 0.51111 TP7
            35 -111 0.39444 CP5
            36 -118 0.27778 CP3
            37 -135 0.17778 CP1
            38 180 0.12778 CPZ
            39 135 0.17778 CP2
            40 118 0.27778 CP4
            41 111 0.39444 CP6
            42 108 0.51111 TP8
            43 -90 0.71111 M2
            44 -126 0.51111 P7
            45 -131 0.41667 P5
            46 -141 0.33333 P3
            47 -158 0.27778 P1
            48 180 0.25556 PZ
            49 158 0.27778 P2
            50 141 0.33333 P4
            51 131 0.41667 P6
            52 126 0.51111 P8
            53 -144 0.51111 PO7
            54 -144 0.44722 PO5
            55 -157 0.41111 PO3
            56 180 0.38333 POZ
            57 157 0.41111 PO4
            58 144 0.44722 PO6
            59 144 0.51111 PO8
            60 -135 0.72222 CB1
            61 -162 0.51111 O1
            62 180 0.51111 OZ
            63 162 0.51111 O2
            64 135 0.72222 CB2
            """


    # Initialize lists to hold the electrode positions and channel names
    all_electrode_positions = []
    all_channel_names = []

    # Split the provided data into lines
    electrode_lines = electrode_text_data.strip().split('\n')

    # Complete the list appending process and exclude the M1, M2, CB1, and CB2 electrodes
    for line in electrode_lines:
        # Split by whitespace and extract the data
        _, theta, radius, label = line.split()

        # Skip the electrodes that are not needed for plotting
        # 这几个电极不除掉画图的时候topomap会超出大脑边界
        if label in ['M1', 'M2', 'CB1', 'CB2']:
            continue

        theta = float(theta)  # Convert to float
        radius = float(radius)  # Convert to float

        # Convert polar coordinates (theta, radius) to Cartesian coordinates (x, y, z)
        theta_rad = np.deg2rad(theta)  # Convert angle from degrees to radians
        y = radius * np.cos(theta_rad)  # Calculate x-coordinate
        x = radius * np.sin(theta_rad)  # Calculate y-coordinate
        z = np.sqrt(1 - radius ** 2) if radius <= 1 else 0  # Protect against invalid values

        # Append the calculated Cartesian coordinates to the positions list
        all_electrode_positions.append([x, y, z])
        # Append the channel name to the channel names list
        all_channel_names.append(label)

    return all_electrode_positions, all_channel_names


def create_raw_eeg(X, srate, ch_names):

    if X.shape[0] != len(ch_names):
        raise ValueError('The length of the first dimension of the data and channel list must be the same!')

    all_electrode_positions, all_channel_names = get_channel_information()

    # Create a montage from the electrode positions. The order must be consistent in correlations.
    montage_pos = dict()
    exclude_ch_idx = []
    for ch_idx, ch in enumerate(ch_names):
        if ch in all_channel_names:
            montage_pos[ch] = all_electrode_positions[all_channel_names.index(ch)]
        else:
            exclude_ch_idx.append(ch_idx)

    if len(exclude_ch_idx) > 0:
        raise ValueError('Assigned channel is not found in our montange system!')

    montage = mne.channels.make_dig_montage(ch_pos=montage_pos, coord_frame='head')

    info = mne.create_info(ch_names=montage.ch_names, sfreq=srate,
                                ch_types='eeg')

    raw = mne.io.RawArray(X, info, verbose=False)
    raw.set_montage(montage)

    return raw

def create_eeg_info(ch_names, srate):
    all_electrode_positions, all_channel_names = get_channel_information()

    # Create a montage from the electrode positions. The order must be consistent in correlations.
    montage_pos = dict()
    exclude_ch_idx = []
    for ch_idx, ch in enumerate(ch_names):
        if ch in all_channel_names:
            montage_pos[ch] = all_electrode_positions[all_channel_names.index(ch)]
        else:
            exclude_ch_idx.append(ch_idx)

    if len(exclude_ch_idx) > 0:
        raise ValueError('Assigned channel is not found in our montange system!')

    montage = mne.channels.make_dig_montage(ch_pos=montage_pos, coord_frame='head')

    info = mne.create_info(ch_names=montage.ch_names, sfreq=srate,
                           ch_types='eeg')

    fake_data = np.zeros((len(montage.ch_names), 1))
    raw = mne.io.RawArray(fake_data, info, verbose=False)
    raw.set_montage(montage)

    return raw.info



def are_lists_equal(list1, list2):
    if len(list1) != len(list2):
        return False
    for arr1, arr2 in zip(list1, list2):
        if not np.array_equal(arr1, arr2):
            return False
    return True





