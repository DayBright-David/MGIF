"""
SSVEP recognition methods
"""
import os
from typing import Union, Optional, Dict, List, Tuple, Callable
from numpy import ndarray
from joblib import Parallel, delayed
from functools import partial
from copy import deepcopy
import warnings
from sklearn.metrics import accuracy_score
from utils import create_raw_eeg, create_eeg_info
from processing_toolkit import add_gaussian_white_noise
import scipy.special as scs
from utils import are_lists_equal

import numpy as np

#from .basemodel import BaseModel
from utils import qr_list, mean_list, sum_list, eigvec
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset

def _covariance(X: ndarray,
                     X_mean: ndarray, 
                     num: int, 
                     division_num: int) -> ndarray:
    # X, X_mean: array: (n_delay*channel, 2*time)
    if num == 1:
        X_tmp = X
    else:
        X_tmp = X - X_mean
    return X_tmp @ X_tmp.T / division_num

def _r_canoncorr_withUV(X: ndarray,
                            Y: List[ndarray],
                            P: List[ndarray],
                            U: ndarray,
                            V: ndarray) -> ndarray:
    """
    Calculate correlation of CCA based on canoncorr for single trial data using existing U and V

    Parameters
    ----------
    X : ndarray
        Single trial EEG data
        EEG shape: (filterbank_num, channel_num*n_delay, signal_len)
    Y : List[ndarray]
        List of reference signals
    P : List[ndarray]
        List of P
    U : ndarray
        Spatial filter
        shape: (filterbank_num * stimulus_num * channel_num * n_component)
    V : ndarray
        Weights of harmonics
        shape: (filterbank_num * stimulus_num * harmonic_num * n_component)

    Returns
    -------
    R : ndarray
        Correlation
        shape: (filterbank_num * stimulus_num)

    trial_spatial_filtered_data : ndarray

    """
    # X: (filterbank, n_delay*channel, time)
    # Y: List: (class_num, ), array: (filterbank_num, channel_num * n_delay, signal_len + ref_sig_Q[0].shape[0])
    # P: List: (class_num, ), array: (time, time)
    filterbank_num, _, signal_len = X.shape
    n_component = U.shape[-1]

    if len(Y[0].shape)==2:
        harmonic_num = Y[0].shape[0]
    elif len(Y[0].shape)==3:
        harmonic_num = Y[0].shape[1]
    else:
        raise ValueError('Unknown data type')
    stimulus_num = len(Y)

    R = np.zeros((filterbank_num, stimulus_num))

    # 一个trial中多个filterbank频带数据经过n_component个空域滤波器的结果
    # ndarray: (filterbank, stimulus_num, n_component, 2*time)
    trial_spatial_filtered_data = np.zeros((filterbank_num, stimulus_num, n_component, signal_len+P[0].shape[-1]))
    for k in range(filterbank_num):
        tmp_X = X[k,:,:]
        for i in range(stimulus_num):
            tmp = np.concatenate([tmp_X, tmp_X @ P[i]], axis=-1)  # (n_delay*channel, 2*time)

            if len(Y[i].shape)==2:
                Y_tmp = Y[i]
            elif len(Y[i].shape)==3:
                Y_tmp = Y[i][k,:,:]
            else:
                raise ValueError('Unknown data type')
            

            A_r = U[k,i,:,:]  # (n_delay*channel, n_component)
            B_r = V[k,i,:,:]
            
            a = A_r.T @ tmp   # (n_component, 2*time)，待分类trial的空域滤波结果
            b = B_r.T @ Y_tmp  # 第i类的template的空域滤波结果

            trial_spatial_filtered_data[k, i, :, :] = a

            a = np.reshape(a, (-1))
            b = np.reshape(b, (-1))
            

            # r2 = stats.pearsonr(a, b)[0]
            # r = stats.pearsonr(a, b)[0]
            r = np.corrcoef(a, b)[0,1]
            R[k,i] = r



    return R, trial_spatial_filtered_data

def _gen_delay_X(X: List[ndarray],
                 n_delay: int) -> List[ndarray]:
    """
    Generate delayed signal

    Parameters
    -----------
    X: List[ndarray]
        Original EEG signals
    n_delay: int
        Number of delayed signals
        0 means no delay

    Returns
    -------------
    X_delay: List[ndarray]
        Combine original signals and delayed signals along channel axis
    """
    X_delay = []
    for X_single_trial in X:
        if len(X_single_trial.shape) == 2:
            ch_num, sig_len = X_single_trial.shape
            X_delay_single_trial = [np.concatenate([X_single_trial[:,dn:sig_len],np.zeros((ch_num,dn))],axis=-1)
                                    for dn in range(n_delay)]
            X_delay.append(np.concatenate(X_delay_single_trial,axis=0))
        elif len(X_single_trial.shape) == 3:
            filterbank_num, ch_num, sig_len = X_single_trial.shape
            X_delay_single_trial = []
            for filterbank_idx in range(filterbank_num):
                tmp = [np.concatenate([X_single_trial[filterbank_idx,:,dn:sig_len],np.zeros((ch_num,dn))],axis=-1)
                       for dn in range(n_delay)]
                tmp = np.concatenate(tmp,axis=0)
                X_delay_single_trial.append(np.expand_dims(tmp, axis=0))
            X_delay_single_trial = np.concatenate(X_delay_single_trial, axis=0)
            X_delay.append(X_delay_single_trial)
        else:
            raise ValueError("Shapes of X have error")
    return X_delay

def _gen_P_combine_X(X: List[ndarray],
                     P: ndarray) -> List[ndarray]:
    """
    Combine signal and signal * P

    Parameters
    --------------
    X: List[ndarray]
        Original signal
    P: ndarray
        P = Q @ Q.T

    Returns
    -------------
    P_combine_X: List[ndarray]
        Combine X and X @ P along time axis
    """

    # X: list: (trial_num,), array(channel_num * n_delay, time)
    # P: array shape: (signal_len, signal_len)

    # list: (trial_num, ), array: (n_delay*channel, 2*time)
    return [np.concatenate([X_single_trial, X_single_trial @ P], axis=-1) for X_single_trial in X]


def _cal_ref_sig_P(ref_sig):
    '''
    Calculate P = Q @ Q.T where Q is the orthogonal matrix of the reference signals.

    Parameters:
    -----------
    - ref_sig: list: (class, ), ndarray: (2*harmonic_num, time)

    Return:
    ----------
    - ref_sig_P: list: (class, ), ndarray: (time, time)
    '''

    ######## calculate matrix P for each stimulus frequency using the reference signal ##############
    ref_sig_Q, _, _ = qr_list(
        ref_sig)  # List shape: (class_num,), array shape: (signal_len, min(signal_len, harmonic_num))
    ref_sig_P = [Q @ Q.T for Q in
                 ref_sig_Q]  # P = Q @ Q.T  # List shape: (class_num,), array shape: (signal_len, signal_len)

    return ref_sig_P

def _augmentData(X, ref_sig_P, n_delay, n_jobs=None):

    '''
    Augment trials in a filterbank by delaying and projection.
    Step 1: For a single trial, delay and concatenate all delayed data along the channel dimension.
    Step 2: For the augmented data derived in step 1, get a projection of it using the matrix P=Q @ Q.T, where Q is the
            orthogonal matrix of the reference signal corresponding to that trial. Then concatenate the augmented data
            and its projection to get the new augmented data.

    Parameters:
    -----------
    - X: list: (class_num, ), list: (trial, ), ndarray: (channel, time)
    - ref_sig_P: list: (class_num, ), ndarray: (time, time)
    - n_delay: int
        number of delayed signal + 1
    - n_jobs:

    Return:
    -----------

    '''


    if n_jobs is not None:
            X_train_delay = Parallel(n_jobs = n_jobs)(delayed(partial(_gen_delay_X, n_delay = n_delay))(X = X_single_class) for X_single_class in X)
            P_combine_X_train = Parallel(n_jobs = n_jobs)(delayed(_gen_P_combine_X)(X = X_single_class, P = P_single_class) for X_single_class, P_single_class in zip(X_train_delay, ref_sig_P))
    else:
        ######################## data augmentation ############################

        # 1. calculate delayed signal
        X_train_delay = []
        for X_single_class in X:
            X_train_delay.append(
                _gen_delay_X(X = X_single_class, n_delay = n_delay)
            )

        # X_train_delay: list: (class_num, ), list: (trial_num, ), array(channel_num*n_delay, time)

        # 2. calculate projection and concatenate
        P_combine_X_train = []
        for X_single_class, P_single_class in zip(X_train_delay, ref_sig_P):
            P_combine_X_train.append(
                _gen_P_combine_X(X = X_single_class, P = P_single_class)
            )

        # P_combine_X_train: list: (class_num, ), list: (trial_num, ), array: (n_delay*channel, 2*time)



    # Calculate template
    if n_jobs is not None:
        P_combine_X_train_mean = Parallel(n_jobs=n_jobs)(delayed(mean_list)(X = P_combine_X_train_single_class) for P_combine_X_train_single_class in P_combine_X_train)
    else:
        P_combine_X_train_mean = []
        for P_combine_X_train_single_class in P_combine_X_train:
            P_combine_X_train_mean.append(
                mean_list(X = P_combine_X_train_single_class)
            )
        # P_combine_X_train_mean: list: (class_num, ), array: (n_delay*channel, 2*time)


    return P_combine_X_train, P_combine_X_train_mean



def _replace_with_weighted_average(X, normalized_adj_matrix, problematic_sensors):
    '''
    Interpolate bad channels using other normal channels according to the adjacent matrix.

    Parameters:
    -----------
    - X: (channel, time) or (filterbank, channel, time)
    - normalized_adj_matrix:
    :param problematic_sensors:
    :return:
    '''
    X_copy = X.copy()
    # 把自身的权重赋为0
    for i in range(normalized_adj_matrix.shape[0]):
        normalized_adj_matrix[i, i] = 0

    if problematic_sensors is not None:
        for i in problematic_sensors:  # 遍历所有有问题的传感器
            # 找到与当前传感器相邻的传感器
            neighboring_sensors_indices = np.where(normalized_adj_matrix[i, :] > 0)[0]
            # 计算相邻传感器的加权平均值
            if neighboring_sensors_indices.size > 0:  # 确保有相邻传感器
                if X_copy.ndim == 2:
                    weighted_average = np.average(X_copy[neighboring_sensors_indices, :], axis=0,
                                                  weights=normalized_adj_matrix[i, neighboring_sensors_indices])
                    X_copy[i, :] = weighted_average  # 用加权平均值替换整个有问题的传感器的数据
                elif X_copy.ndim == 3:
                    weighted_average = np.average(X_copy[:, neighboring_sensors_indices, :], axis=1,
                                                  weights=normalized_adj_matrix[i, neighboring_sensors_indices])
                    X_copy[:, i, :] = weighted_average  # 用加权平均值替换整个有问题的传感器的数据

        return X_copy
    else:
        raise ValueError('problematic_sensors must be assigned!')


def _cal_ch_correlation_matrix(X):
    '''
    Calculate the adjacent matrix based on correlations between channels.

    Parameters:
    -----------
    - X : ndarray
        shape: (channel, time)

    Return:
    -------
    - correlation_matrix : ndarray
        shape: (channel, channel)
    '''
    ch_num = X.shape[0]
    correlation_matrix = np.corrcoef(X)
    np.fill_diagonal(correlation_matrix, 0)

    # correlation_matrix = np.zeros((ch_num, ch_num))
    # for i in range(ch_num):
    #     for j in range(i+1, ch_num):
    #         cor_ij = np.corrcoef(X[i, :], X[j, :])[0,1]
    #         correlation_matrix[i, j] = cor_ij
    #         correlation_matrix[j, i] = cor_ij


    selected_correlation_matrix = np.zeros(correlation_matrix.shape)
    # 取前三相似的相关系数值，其他全部赋为0

    max_idx = np.argsort(correlation_matrix, axis=1)[:, -3:][:, ::-1]
    for i in range(ch_num):
        selected_correlation_matrix[i, max_idx[i]] = correlation_matrix[i, max_idx[i]]

    return selected_correlation_matrix

class SSVEP_TDCA():
    """
    SSVEP TDCA method
    """
    def __init__(self,
                 n_component: int = 1,
                 n_jobs: Optional[int] = None,
                 weights_filterbank: Optional[List[float]] = None,
                 n_delay: int = 0):
        """
        Special parameter
        -----------------
        n_delay: int
            Number of delayed signals
            Default is 0 (no delay)
        """

        if n_component < 0:
            raise ValueError('n_component must be larger than 0')

        self.ID = 'TDCA'
        self.n_component = n_component
        self.n_jobs = n_jobs
        self.n_delay = n_delay + 1

        self.model = {}
        self.model['weights_filterbank'] = weights_filterbank
        self.model['U'] = None # Spatial filter of EEG
        self.mode = 'normal'
        self.ch_num = None
        self.stimulus_num = None
        self.test_trials_correlations = None  # 最后计算出每个测试数据与每个类别模板的相关系数

    def __copy__(self):
        copy_model = SSVEP_TDCA(n_component = self.n_component,
                          n_jobs = self.n_jobs,
                          weights_filterbank = self.model['weights_filterbank'],
                          n_delay = self.n_delay)
        copy_model.model = deepcopy(self.model)
        return copy_model

    def fit(self,
            X: Optional[List[ndarray]] = None,
            Y: Optional[List[int]] = None,
            train_ref_sig: Optional[List[ndarray]] = None,
            *argv, **kwargv):
        """
        Parameters
        -------------
        X : Optional[List[ndarray]], optional
            List of training EEG data. The default is None.
            List shape: (trial_num,)
            EEG shape: (filterbank_num, channel_num, signal_len)
        Y : Optional[List[int]], optional
            List of labels (stimulus indices). The default is None.
            List shape: (trial_num,)
        ref_sig : Optional[List[ndarray]], optional
            Sine-cosine-based reference signals. The default is None.
            List of shape: (stimulus_num,)
            Reference signal shape: (harmonic_num, signal_len)
        """
        if Y is None:
            raise ValueError('TDCA requires training label')
        if X is None:
            raise ValueError('TDCA requires training data')
        if train_ref_sig is None:
            raise ValueError("TDCA requires reference signals")


        self.train_data = X.copy()
        self.train_label = Y.copy()
        # template signals and spatial filters
        #   U: (filterbank_num * stimulus_num * channel_num * n_component)
        #   X or template_sig: (filterbank_num, channel_num, signal_len)
        filterbank_num, channel_num, signal_len = X[0].shape
        self.ch_num = channel_num

        stimulus_num = len(train_ref_sig)
        self.stimulus_num = stimulus_num

        n_component = self.n_component
        n_delay = self.n_delay
        possible_class = list(set(Y))
        possible_class.sort(reverse = False)

        train_ref_sig_P = _cal_ref_sig_P(train_ref_sig)

        U = np.zeros((filterbank_num, 1, channel_num * n_delay, n_component))
        for filterbank_idx in range(filterbank_num):
            # list[list]: (class_num, ), list: (trial_num, ), array: (channel, time)
            X_train = [[X[i][filterbank_idx,:,:] for i in np.where(np.array(Y) == class_val)[0]] for class_val in possible_class]
            trial_num = len(X_train[0])

            P_combine_X_train, P_combine_X_train_mean = _augmentData(X_train, train_ref_sig_P, n_delay, self.n_jobs)
            # Calulcate spatial filter
            P_combine_X_train_all_mean = mean_list(P_combine_X_train_mean)  # (n_delay*channel, 2*time) 所有类别的trial的平均
            X_tmp = []
            X_mean = []
            # 把所有类的所有trial按照类别分别放到X_tmp这个list中, 每个trial对应的类的平均值放在X_mean中
            for P_combine_X_train_single_class, P_combine_X_train_mean_single_class in zip(P_combine_X_train, P_combine_X_train_mean):
                for X_tmp_tmp in P_combine_X_train_single_class:
                    X_tmp.append(X_tmp_tmp)
                    X_mean.append(P_combine_X_train_mean_single_class)
            # X_tmp: list: (class_num*trial_num, ), array: (n_delay*channel, 2*time)
            # X_mean: same as X_tmp

            if self.n_jobs is not None:
                Sw_list = Parallel(n_jobs=self.n_jobs)(delayed(partial(_covariance, num = trial_num,
                                                                                        division_num = trial_num))(X = X_tmp_tmp, X_mean = X_mean_tmp)
                                                                                        for X_tmp_tmp, X_mean_tmp in zip(X_tmp, X_mean))
                Sb_list = Parallel(n_jobs=self.n_jobs)(delayed(partial(_covariance, X_mean = P_combine_X_train_all_mean,
                                                                                        num = stimulus_num,
                                                                                        division_num = stimulus_num))(X = P_combine_X_train_mean_single_class)
                                                                                        for P_combine_X_train_mean_single_class in P_combine_X_train_mean)
            else:
                Sw_list = []
                for X_tmp_tmp, X_mean_tmp in zip(X_tmp, X_mean):
                    Sw_list.append(
                        _covariance(X = X_tmp_tmp, X_mean = X_mean_tmp, num = trial_num, division_num = trial_num)
                    )
                # Sw_list: list: (class_num*trial_num, ), array: (n_delay*channel, n_delay*channel)
                Sb_list = []
                for P_combine_X_train_mean_single_class in P_combine_X_train_mean:
                    Sb_list.append(
                        _covariance(X = P_combine_X_train_mean_single_class, X_mean = P_combine_X_train_all_mean, num = stimulus_num, division_num = stimulus_num)
                    )
                # Sb_list: list: (class_num, ), array: (n_delay*channel, n_delay*channel)

            Sw = sum_list(Sw_list)
            Sb = sum_list(Sb_list)
            eig_vec = eigvec(Sb, Sw)
            U[filterbank_idx,0,:,:] = eig_vec[:,:n_component]
        U = np.repeat(U, repeats = stimulus_num, axis = 1)
        self.model['U'] = U
        # self.model['template_sig'] = template_sig
        

    def get_template_sig(self, train_data, train_label, test_ref_sig_P):
        '''
        Get the template signal for each stimulus frequency based on the signal time of the test set.
        Step 1: Segment the training set to be of equal signal length with the test set.
        Step 2: Calculate the augmented data of the training set and average across class to get an evoked
                augmented data for each class.

        :return:
        '''
        train_data_copy = train_data.copy()
        test_winLEN = test_ref_sig_P[0].shape[-1]
        # segment train_data to get the same data length as test data
        train_data_copy = [trial[:, :, :test_winLEN] for trial in train_data_copy]

        filterbank_num, channel_num, signal_len = train_data_copy[0].shape
        n_delay = self.n_delay
        possible_class = list(set(train_label))
        possible_class.sort(reverse=False)
        stimulus_num = len(possible_class)
        # 每个类别的所有trial的增强信号的平均
        template_sig = [np.zeros((filterbank_num, channel_num * n_delay, signal_len + test_ref_sig_P[0].shape[1])) for _ in range(stimulus_num)]

        for filterbank_idx in range(filterbank_num):
            # list[list]: (class_num, ), list: (trial_num, ), array: (channel, time)
            X_train = [[train_data_copy[i][filterbank_idx,:,:] for i in np.where(np.array(train_label) == class_val)[0]] for class_val in possible_class]
            trial_num = len(X_train[0])

            P_combine_X_train, P_combine_X_train_mean = _augmentData(X_train, test_ref_sig_P, n_delay, self.n_jobs)

            for stim_idx, P_combine_X_train_mean_single_class in enumerate(P_combine_X_train_mean):
                template_sig[stim_idx][filterbank_idx,:,:] = P_combine_X_train_mean_single_class

        U = self.model['U']
        spatial_filtered_template_sig = [np.zeros((filterbank_num, self.n_component, signal_len + test_ref_sig_P[0].shape[1])) for _ in range(stimulus_num)]

        for k in range(filterbank_num):
            for i in range(stimulus_num):
                tmp = template_sig[i][k, :, :]  # (n_delay*channel, 2*time)
                A_r = U[k, i, :, :]  # (n_delay*channel, n_component)
                a = A_r.T @ tmp  # (n_component, 2*time)，待分类trial的空域滤波结果
                spatial_filtered_template_sig[i][k, :, :] = a


        return template_sig, spatial_filtered_template_sig



    def predict(self,
            X: List[ndarray], test_ref_sig) -> List[int]:
        weights_filterbank = self.model['weights_filterbank']
        if weights_filterbank is None:
            weights_filterbank = [1 for _ in range(X[0].shape[0])]
        if type(weights_filterbank) is list:
            weights_filterbank = np.expand_dims(np.array(weights_filterbank),1).T
        else:
            if len(weights_filterbank.shape) != 2:
                raise ValueError("'weights_filterbank' has wrong shape")
            if weights_filterbank.shape[0] != 1:
                weights_filterbank = weights_filterbank.T
        if weights_filterbank.shape[0] != 1:
            raise ValueError("'weights_filterbank' has wrong shape")
        n_delay = self.n_delay

        X_delay = _gen_delay_X(X, n_delay)  # list: (epoch, ), array: (filterbank, n_delay*channel, time)
        U = self.model['U']

        test_ref_sig_P = _cal_ref_sig_P(test_ref_sig)

        template_sig, spatial_filtered_template_sig = self.get_template_sig(self.train_data, self.train_label, test_ref_sig_P)

        # 保存test set中每个trial经过空域滤波的结果
        # list: (trials, ), ndarray: (filterbank, stimulus_num, n_component, 2*time)
        spatial_filtered_test_trials = []

        if self.n_jobs is not None:
            r = Parallel(n_jobs=self.n_jobs)(delayed(partial(_r_canoncorr_withUV, Y=template_sig, P=test_ref_sig_P, U=U, V=U))(X=a) for a in X_delay)
        else:
            r = []
            for a in X_delay:
                trial_r, trial_spatial_filtered_data = _r_canoncorr_withUV(X=a, Y=template_sig, P=test_ref_sig_P, U=U, V=U)
                r.append(trial_r)
                spatial_filtered_test_trials.append(trial_spatial_filtered_data)
            # r: list: (epoch, ), array: (filterbank, class_num)

        self.test_trials_correlations = [weights_filterbank @ r_tmp for r_tmp in r]
        Y_pred = [int(np.argmax(trial_correlations)) for trial_correlations in self.test_trials_correlations]
        # Y_pred: list: (epoch, ), int
        
        return Y_pred, spatial_filtered_test_trials, template_sig, spatial_filtered_template_sig

    def score(self, X, y, test_ref_sig, returnTemplate=False):
        y_pred, spatial_filtered_test_trials, template_sig, spatial_filtered_template_sig = self.predict(X, test_ref_sig)
        if returnTemplate == False:
            # print('acc: ', accuracy_score(y, y_pred))
            return accuracy_score(y, y_pred), spatial_filtered_test_trials
        else:
            return accuracy_score(y, y_pred), spatial_filtered_test_trials, template_sig, spatial_filtered_template_sig




class SSVEP_Egraph(SSVEP_TDCA):
    def __init__(self,
                 n_component: int = 1,
                 n_jobs: Optional[int] = None,
                 weights_filterbank: Optional[List[float]] = None,
                 n_delay: int = 0,
                 electrodes_adjacent_matrix: ndarray = None,
                 electrodes_names: list = None,
                 interpolation_matrices: ndarray = None):
        super(SSVEP_Egraph, self).__init__(n_component, n_jobs, weights_filterbank, n_delay)
        self.electrodes_adjacent_matrix = electrodes_adjacent_matrix
        self.mode = 'egraph'
        self.electrodes_names = electrodes_names
        self.interpolation_matrices = interpolation_matrices  # 仅在MNE模式下只有一个通道需要插值时使用

        if electrodes_adjacent_matrix is None and electrodes_names is None:
            raise ValueError('One of electrodes_adjacent_matrix and electrodes_names must not be None!')

        if self.electrodes_names is not None:
            self.mne_info = create_eeg_info(self.electrodes_names, srate=250) #这里采样率随便赋值，不影响结果


    def predict(self,
            X_test: List[ndarray], test_ref_sig, bad_channel_indices: list = None) -> List[int]:
        X = X_test[:]  # 复制，不改变原值
        # 先对坏的通道进行补全操作
        if self.electrodes_adjacent_matrix is not None:
            for trial_idx, test_trial in enumerate(X):
                X[trial_idx] = _replace_with_weighted_average(test_trial, self.electrodes_adjacent_matrix, bad_channel_indices)
        else:
            # for trial_idx, test_trial in enumerate(X):
            #     X[trial_idx] = interpolate_bad(test_trial, self.mne_info, bad_channel_indices)

            # 只有一个通道插值的情况下可以使用下面这串代码，用提前保存的interpolation matrix进行插值提高效率
            if len(bad_channel_indices) > 0:
                good_channel_indices = list(np.delete(np.arange(0, X[0].shape[1]), bad_channel_indices))
                for trial_idx, test_trial in enumerate(X):
                    X[trial_idx][:, bad_channel_indices, :] = np.einsum('pmn,km->pkn', X[trial_idx][:, good_channel_indices, :], self.interpolation_matrices[bad_channel_indices])


        weights_filterbank = self.model['weights_filterbank']
        if weights_filterbank is None:
            weights_filterbank = [1 for _ in range(X[0].shape[0])]
        if type(weights_filterbank) is list:
            weights_filterbank = np.expand_dims(np.array(weights_filterbank), 1).T
        else:
            if len(weights_filterbank.shape) != 2:
                raise ValueError("'weights_filterbank' has wrong shape")
            if weights_filterbank.shape[0] != 1:
                weights_filterbank = weights_filterbank.T
        if weights_filterbank.shape[0] != 1:
            raise ValueError("'weights_filterbank' has wrong shape")
        n_delay = self.n_delay

        X_delay = _gen_delay_X(X, n_delay)  # list: (epoch, ), array: (filterbank, n_delay*channel, time)
        U = self.model['U']

        test_ref_sig_P = _cal_ref_sig_P(test_ref_sig)

        template_sig, spatial_filtered_template_sig = self.get_template_sig(self.train_data, self.train_label,
                                                                            test_ref_sig_P)

        # 保存test set中每个trial经过空域滤波的结果
        # list: (trials, ), ndarray: (filterbank, stimulus_num, n_component, 2*time)
        spatial_filtered_test_trials = []

        if self.n_jobs is not None:
            r = Parallel(n_jobs=self.n_jobs)(
                delayed(partial(_r_canoncorr_withUV, Y=template_sig, P=test_ref_sig_P, U=U, V=U))(X=a) for a in
                X_delay)
        else:
            r = []
            for a in X_delay:
                trial_r, trial_spatial_filtered_data = _r_canoncorr_withUV(X=a, Y=template_sig, P=test_ref_sig_P,
                                                                                U=U, V=U)
                r.append(trial_r)
                spatial_filtered_test_trials.append(trial_spatial_filtered_data)
            # r: list: (epoch, ), array: (filterbank, class_num)

        self.test_trials_correlations = [weights_filterbank @ r_tmp for r_tmp in r]
        Y_pred = [int(np.argmax(trial_correlations)) for trial_correlations in self.test_trials_correlations]
        # Y_pred: list: (epoch, ), int

        return Y_pred, spatial_filtered_test_trials, template_sig, spatial_filtered_template_sig, X

    def score(self, X, y, test_ref_sig, bad_channel_indices: list=None, returnTemplate=False, returnRecon=False):
        y_pred, spatial_filtered_test_trials, template_sig, spatial_filtered_template_sig, recon_X = self.predict(X, test_ref_sig, bad_channel_indices)
        if returnTemplate == False and returnRecon == False:
            # print('acc: ', accuracy_score(y, y_pred))
            return accuracy_score(y, y_pred), spatial_filtered_test_trials
        elif returnTemplate == True and returnRecon == False:
            return accuracy_score(y, y_pred), spatial_filtered_test_trials, template_sig, spatial_filtered_template_sig
        elif returnTemplate == False and returnRecon == True:
            return accuracy_score(y, y_pred), spatial_filtered_test_trials, recon_X
        elif returnTemplate == True and returnRecon == True:
            return accuracy_score(y, y_pred), spatial_filtered_test_trials, template_sig, spatial_filtered_template_sig, recon_X

class SSVEP_Sgraph(SSVEP_TDCA):
    def __init__(self,
                 n_component: int = 1,
                 n_jobs: Optional[int] = None,
                 weights_filterbank: Optional[List[float]] = None,
                 n_delay: int = 0,
                 ):
        super(SSVEP_Sgraph, self).__init__(n_component, n_jobs, weights_filterbank, n_delay)
        self.mode = 'sgraph'
        self.electrodes_adjacent_matrices = None



    def _CAM_r_canoncorr_withUV(self,
                                 X: ndarray,
                                 Y: List[ndarray],
                                 P: List[ndarray],
                                 U: ndarray,
                                 V: ndarray,
                                 bad_channel_indices: list = None) -> ndarray:
        """
        Calculate correlation of CCA based on canoncorr for single trial data using existing U and V

        Parameters
        ----------
        X : ndarray
            Single trial EEG data
            EEG shape: (filterbank_num, channel_num*n_delay, signal_len)
        Y : List[ndarray]
            List of reference signals
        P : List[ndarray]
            List of P
        U : ndarray
            Spatial filter
            shape: (filterbank_num * stimulus_num * channel_num * n_component)
        V : ndarray
            Weights of harmonics
            shape: (filterbank_num * stimulus_num * harmonic_num * n_component)

        Returns
        -------
        R : ndarray
            Correlation
            shape: (filterbank_num * stimulus_num)

        trial_spatial_filtered_data : ndarray

        """
        # X: (filterbank, n_delay*channel, time)
        # Y: List: (class_num, ), array: (filterbank_num, channel_num * n_delay, signal_len + ref_sig_Q[0].shape[0])
        # P: List: (class_num, ), array: (time, time)
        X_copy = X.copy()
        filterbank_num, _, signal_len = X_copy.shape
        n_component = U.shape[-1]

        if len(Y[0].shape) == 2:
            harmonic_num = Y[0].shape[0]
        elif len(Y[0].shape) == 3:
            harmonic_num = Y[0].shape[1]
        else:
            raise ValueError('Unknown data type')
        stimulus_num = len(Y)

        R = np.zeros((filterbank_num, stimulus_num))

        # 一个trial中多个filterbank频带数据经过n_component个空域滤波器的结果
        # ndarray: (filterbank, stimulus_num, n_component, 2*time)
        trial_spatial_filtered_data = np.zeros((filterbank_num, stimulus_num, n_component, signal_len + P[0].shape[-1]))
        recon_X_trial = np.zeros(X_copy.shape)
        for k in range(filterbank_num):
            tmp_X = X_copy[k, :, :]
            for i in range(stimulus_num):

                for delay_idx in range(self.n_delay):
                    delay_tmp_X = tmp_X[delay_idx*self.ch_num:(delay_idx+1)*self.ch_num, :]
                    tmp_X[delay_idx*self.ch_num:(delay_idx+1)*self.ch_num, :] = _replace_with_weighted_average(delay_tmp_X, self.electrodes_adjacent_matrices[i][k, :, :], bad_channel_indices)

                recon_X_trial[k] = tmp_X.copy()

                tmp = np.concatenate([tmp_X, tmp_X @ P[i]], axis=-1)  # (n_delay*channel, 2*time)

                if len(Y[i].shape) == 2:
                    Y_tmp = Y[i]
                elif len(Y[i].shape) == 3:
                    Y_tmp = Y[i][k, :, :]
                else:
                    raise ValueError('Unknown data type')

                A_r = U[k, i, :, :]  # (n_delay*channel, n_component)
                B_r = V[k, i, :, :]

                a = A_r.T @ tmp  # (n_component, 2*time)，待分类trial的空域滤波结果
                b = B_r.T @ Y_tmp  # 第i类的template的空域滤波结果

                trial_spatial_filtered_data[k, i, :, :] = a

                a = np.reshape(a, (-1))
                b = np.reshape(b, (-1))

                # r2 = stats.pearsonr(a, b)[0]
                # r = stats.pearsonr(a, b)[0]
                r = np.corrcoef(a, b)[0, 1]
                R[k, i] = r

        return R, trial_spatial_filtered_data, recon_X_trial



    def predict(self,
                X_test: List[ndarray], test_ref_sig, bad_channel_indices: list = None) -> List[int]:

        X = X_test[:]  # 复制，不改变原值
        recon_X = []
        weights_filterbank = self.model['weights_filterbank']
        if weights_filterbank is None:
            weights_filterbank = [1 for _ in range(X[0].shape[0])]
        if type(weights_filterbank) is list:
            weights_filterbank = np.expand_dims(np.array(weights_filterbank), 1).T
        else:
            if len(weights_filterbank.shape) != 2:
                raise ValueError("'weights_filterbank' has wrong shape")
            if weights_filterbank.shape[0] != 1:
                weights_filterbank = weights_filterbank.T
        if weights_filterbank.shape[0] != 1:
            raise ValueError("'weights_filterbank' has wrong shape")
        n_delay = self.n_delay
        filterbank_num = X[0].shape[0]
        test_class_num = len(test_ref_sig)
        sig_len = X[0].shape[2]


        X_delay = _gen_delay_X(X, n_delay)  # list: (epoch, ), array: (filterbank, n_delay*channel, time)
        U = self.model['U']

        test_ref_sig_P = _cal_ref_sig_P(test_ref_sig)

        template_sig, spatial_filtered_template_sig = self.get_template_sig(self.train_data, self.train_label,
                                                                            test_ref_sig_P)

        # 每个filterbank下的每个stimulus形状为(n_delay*channel, 2*time)
        # 用未delay的通道计算一个邻接矩阵
        correlation_matrices = [np.zeros((filterbank_num, self.ch_num, self.ch_num)) for _ in range(test_class_num)]
        for fb_idx in range(filterbank_num):
            for sti_idx in range(test_class_num):
                # 一个刺激类别下的其中一个fb, 取出未delay的通道
                stimulus_fb_template = template_sig[sti_idx][fb_idx, :self.ch_num, :sig_len]
                stimulus_fb_correlation_matrix = _cal_ch_correlation_matrix(stimulus_fb_template)
                correlation_matrices[sti_idx][fb_idx, :, :] = stimulus_fb_correlation_matrix


        self.electrodes_adjacent_matrices = correlation_matrices

        # 保存test set中每个trial经过空域滤波的结果
        # list: (trials, ), ndarray: (filterbank, stimulus_num, n_component, 2*time)
        spatial_filtered_test_trials = []

        if self.n_jobs is not None:
            r = Parallel(n_jobs=self.n_jobs)(
                delayed(partial(self._CAM_r_canoncorr_withUV, Y=template_sig, P=test_ref_sig_P, U=U, V=U, bad_channel_indices=bad_channel_indices))(X=a) for a in
                X_delay)
        else:
            r = []
            for a in X_delay:
                trial_r, trial_spatial_filtered_data, recon_X_trial = self._CAM_r_canoncorr_withUV(X=a, Y=template_sig, P=test_ref_sig_P,
                                                                                U=U, V=U, bad_channel_indices=bad_channel_indices)
                recon_X.append(recon_X_trial)
                r.append(trial_r)
                spatial_filtered_test_trials.append(trial_spatial_filtered_data)
            # r: list: (epoch, ), array: (filterbank, class_num)

        # list: (trial_num, ), ndarray: (1, stimulus_num)
        self.test_trials_correlations = [weights_filterbank @ r_tmp for r_tmp in r]
        Y_pred = [int(np.argmax(trial_correlations)) for trial_correlations in self.test_trials_correlations]
        # Y_pred: list: (epoch, ), int

        return Y_pred, spatial_filtered_test_trials, template_sig, spatial_filtered_template_sig, recon_X

    def score(self, X, y, test_ref_sig, bad_channel_indices: list=None, returnTemplate=False, returnRecon=False):
        y_pred, spatial_filtered_test_trials, template_sig, spatial_filtered_template_sig, recon_X = self.predict(X,
                                                                                                         test_ref_sig, bad_channel_indices)

        if returnTemplate == False and returnRecon == False:
            # print('acc: ', accuracy_score(y, y_pred))
            return accuracy_score(y, y_pred), spatial_filtered_test_trials
        elif returnTemplate == True and returnRecon == False:
            return accuracy_score(y, y_pred), spatial_filtered_test_trials, template_sig, spatial_filtered_template_sig
        elif returnTemplate == False and returnRecon == True:
            return accuracy_score(y, y_pred), spatial_filtered_test_trials, recon_X
        elif returnTemplate == True and returnRecon == True:
            return accuracy_score(y, y_pred), spatial_filtered_test_trials, template_sig, spatial_filtered_template_sig, recon_X



class weighted_sum(nn.Module):
    def __init__(self):
        super(weighted_sum, self).__init__()
        self.weights = nn.Parameter(torch.zeros((1, 2), requires_grad=True))


    def forward(self, x):
        eam = x[:, :40]
        cam = x[:, 40:]
        x = self.weights[0, 0] * eam + self.weights[0, 1] * cam
        return x


class SSVEP_MGIF():
    def __init__(self,
                 n_component: int = 1,
                 n_jobs: Optional[int] = None,
                 weights_filterbank: Optional[List[float]] = None,
                 n_delay: int = 0,
                 ssvep_selected_channels: list = None,
                 interpolation_matrices: ndarray = None,
                 method: str = None,  # max, sum, weights, all
                 ):

        if method is None:
            raise ValueError('Method must not be None! Must be one of max, sum or weights')
        self.method = method
        self.normal = SSVEP_TDCA(n_component, n_jobs, weights_filterbank, n_delay)
        self.egraph = SSVEP_Egraph(n_component, n_jobs, weights_filterbank, n_delay,
                               electrodes_names=ssvep_selected_channels, interpolation_matrices=interpolation_matrices)
        self.sgraph = SSVEP_Sgraph(n_component, n_jobs, weights_filterbank, n_delay)
        self.mode = 'mgif'
        if self.method == 'weights' or self.method == 'all':
            self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


    def fit(self,
            X: Optional[List[ndarray]] = None,
            Y: Optional[List[int]] = None,
            train_ref_sig: Optional[List[ndarray]] = None,):

        self.normal.fit(X, Y, train_ref_sig)
        self.egraph.fit(X, Y, train_ref_sig)
        self.sgraph.fit(X, Y, train_ref_sig)

        if self.method == 'weights' or self.method == 'all':
            self.moe = weighted_sum().to(self.device)

            train_correlations, train_labels = self.create_channel_attacked_correlations(X, Y, train_ref_sig)

            # 只使用EAM和CAM的结果
            train_correlations = train_correlations[:, 40:]

            train_correlations = torch.from_numpy(train_correlations).type(torch.float32)
            train_labels = torch.from_numpy(train_labels).type(torch.LongTensor)
            train_dataset = TensorDataset(train_correlations, train_labels)
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=2000)

            optimizer = torch.optim.Adam(self.moe.parameters(), lr=1e-3)
            loss_fn = nn.CrossEntropyLoss()

            print('********* train model **********')
            for epoch in range(100):
                epoch_correct_num = 0
                epoch_total_num = 0
                epoch_step = 0
                epoch_loss = 0

                self.moe.train()
                for batch_idx, (train_input, train_label) in enumerate(train_loader
                                                                       ):
                    epoch_step += 1
                    train_input, train_label = train_input.to(self.device), train_label.to(self.device)
                    optimizer.zero_grad()
                    predict_label = self.moe(train_input)

                    loss = loss_fn(predict_label, train_label)
                    loss.backward()
                    optimizer.step()

                    epoch_loss += loss.item()
                    epoch_total_num += len(train_input)
                    _, max_idx = torch.max(predict_label, 1)
                    epoch_correct_num += (max_idx == train_label).sum().item()

                epoch_acc = epoch_correct_num / epoch_total_num
                epoch_loss /= epoch_step

                # print('epoch ', str(epoch + 1), ' train loss: ', epoch_loss)
            print('train acc: ', epoch_acc)
            print('learned weights for MoE: ', self.moe.weights)




    def create_channel_attacked_correlations(self, X, Y, ref_sig):
        # 调用完fit后调用此方法
        # 对每种通道缺失类型的训练数据进行测试，得到训练数据的相关系数
        # 前9个循环分别对每个通道进行attack，最后一个循环加入certain数据

        correlations = []
        labels = []
        for ch_idx in range(X[0].shape[1] + 1):
            X_attack = []
            for trial_idx in range(len(X)):
                attacked_trial = X[trial_idx].copy()  # 不能改变原值
                if ch_idx < X[0].shape[1]:
                    attacked_trial[:, ch_idx, :] = 0
                    # 一个trial的所有filterbank替换成一样的高斯噪声
                    noise = add_gaussian_white_noise(attacked_trial[0, ch_idx, :], target_noise_db=0,
                                                     mode='noisePower')
                    attacked_trial[:, ch_idx, :] = np.stack([noise for _ in range(attacked_trial.shape[0])], axis=0)
                X_attack.append(attacked_trial)

            self.normal.predict(X_attack, ref_sig)


            if ch_idx < X[0].shape[1]:
                self.egraph.predict(X_attack, ref_sig, [ch_idx])
                self.sgraph.predict(X_attack, ref_sig, [ch_idx])
            else:
                self.egraph.predict(X_attack, ref_sig, [])
                self.sgraph.predict(X_attack, ref_sig, [])

            normal_correlations = self.normal.test_trials_correlations
            eam_correlations = self.egraph.test_trials_correlations
            cam_correlations = self.sgraph.test_trials_correlations

            normal_correlations = np.concatenate(normal_correlations, axis=0)
            eam_correlations = np.concatenate(eam_correlations, axis=0)
            cam_correlations = np.concatenate(cam_correlations, axis=0)

            concat_correlations = np.hstack(
                (normal_correlations, eam_correlations, cam_correlations))

            correlations.append(concat_correlations)
            labels.append(Y)

        correlations = np.concatenate(correlations, axis=0)
        labels = np.concatenate(labels, axis=0)

        return correlations, labels


    def score(self, X, y, test_ref_sig, bad_channel_indices: list=None):
        '''

        Parameters
        ----------
        - method : str
            'max': 三种方法在每一类别的相关系数取最大值后再从整合后的相关系数中选择最大值所在的类别
            'sum': 三种方法的相关系数先在方法内部进行归一化，然后加权

        Returns
        -------

        '''

        def _weights(normal_correlations, eam_correlations, cam_correlations):
            normal_correlations = np.concatenate(normal_correlations, axis=0)
            eam_correlations = np.concatenate(eam_correlations, axis=0)
            cam_correlations = np.concatenate(cam_correlations, axis=0)
            concat_correlations = np.hstack((normal_correlations, eam_correlations, cam_correlations))

            # 只用eam，cam的结果
            concat_correlations = concat_correlations[:, 40:]

            concat_correlations = torch.from_numpy(concat_correlations).type(torch.float32).to(self.device)

            prediction = self.moe(concat_correlations)
            _, robust_Y_pred = torch.max(prediction, 1)
            robust_Y_pred = robust_Y_pred.cpu().detach()
            robust_acc = accuracy_score(y, robust_Y_pred)
            return robust_acc

        def _max(normal_correlations, eam_correlations, cam_correlations):
            robust_correlations = []
            for normal_correlation, eam_correlation, cam_correlation in zip(normal_correlations, eam_correlations,
                                                                            cam_correlations):
                robust_correlations.append(
                    np.max(np.concatenate((normal_correlation, eam_correlation, cam_correlation), axis=0), axis=0))

            robust_Y_pred = [int(np.argmax(trial_correlations)) for trial_correlations in robust_correlations]
            robust_acc = accuracy_score(y, robust_Y_pred)
            return robust_acc

        def _sum(normal_correlations, eam_correlations, cam_correlations):
            robust_correlations = []
            for idx, (normal_correlation, eam_correlation, cam_correlation) in enumerate(
                    zip(normal_correlations, eam_correlations,
                        cam_correlations)):

                normal_correlation = scs.softmax(normal_correlation)
                eam_correlation = scs.softmax(eam_correlation)
                cam_correlation = scs.softmax(cam_correlation)

                weighted_sum = normal_correlation + eam_correlation + cam_correlation

                robust_correlations.append(weighted_sum)

            robust_Y_pred = [int(np.argmax(trial_correlations)) for trial_correlations in robust_correlations]
            robust_acc = accuracy_score(y, robust_Y_pred)
            return robust_acc


        normal_acc, _ = self.normal.score(X, y, test_ref_sig)
        eam_acc, _ = self.egraph.score(X, y, test_ref_sig, bad_channel_indices)
        cam_acc, _ = self.sgraph.score(X, y, test_ref_sig, bad_channel_indices)

        normal_correlations = self.normal.test_trials_correlations
        eam_correlations = self.egraph.test_trials_correlations
        cam_correlations = self.sgraph.test_trials_correlations


        if self.method == 'weights':
            robust_acc = _weights(normal_correlations, eam_correlations, cam_correlations)
        elif self.method == 'max':
            robust_acc = _max(normal_correlations, eam_correlations, cam_correlations)
        elif self.method == 'sum':
            robust_acc = _sum(normal_correlations, eam_correlations, cam_correlations)
        elif self.method == 'all':
            #before_n, before_e, before_c = normal_correlations.copy(), eam_correlations.copy(), cam_correlations.copy()
            robust_acc_weights = _weights(normal_correlations, eam_correlations, cam_correlations)
            robust_acc_max = _max(normal_correlations, eam_correlations, cam_correlations)
            robust_acc_sum = _sum(normal_correlations, eam_correlations, cam_correlations)
            # print(are_lists_equal(before_n, normal_correlations))
            # print(are_lists_equal(before_e, eam_correlations))
            # print(are_lists_equal(before_c, cam_correlations))

        else:
            raise ValueError('method must be one of sum, max, weights or all!')

        if self.method == 'all':
            return robust_acc_weights, robust_acc_max, robust_acc_sum, normal_acc, eam_acc, cam_acc
        else:
            return robust_acc, normal_acc, eam_acc, cam_acc







