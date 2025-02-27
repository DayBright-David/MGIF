import numpy as np
import pickle
import scipy
from utils import get_ref_sig, filterbank, loadmat, notch_filter, filter
import matplotlib.pyplot as plt
from sklearn import preprocessing

def readSubjectData(dataset_name, subject_name,
                    data_divide_method='class', test_block_idx=0, filterbank_num=2,
                    harmonic_number=5,
                    latency=0.14,
                    train_winLEN=3,
                    test_winLEN=3,
                    root_dir='./',
                    chnNames=None,
                    attack_method=None,
                    attack_channel='OZ',
                    mergeBlock=False,
                    test_uncertainty_frame=0):
    '''
    - data_divide_method: bool
        'class': the stimulus frequencies in the training set and test set are not overlapped.
        'block': divide the training set and test set according to blocks.
    - filterbank_num: int
         number of filter band. if filterband_num=0, no filter band is used.
    - latency: float
         This is the time latency used to remove segment where VEP has not appeared (in seconds).
         default=0.14 s.
    - train_winLEN: float
         Signal length in train set.
    - test_winLEN: float
         Signal length in test set.
    - chnNames : None or list
        if None, return the full channel data
        if list, select the channels in the list.

         SSVEP: ['PZ', 'PO5', 'PO3', 'POZ', 'PO4', 'PO6', 'O1', 'OZ', 'O2'] 枕区九导
         cVEP: ['PZ', 'PO5', 'POZ', 'PO4', 'PO6', 'O1', 'OZ', 'PO3', 'O2', 'P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7',
                    'P8', 'PO7', 'PO8', 'CB1', 'CB2']
    - attack_method : str
        Only used when dataset_name == 'Exp_pretrain_channel_attacker_9B'
        if None, no channel will be attacked.
        'all_block_diff': For the i-th block, the i-th channel is attacked.
        'all_block_same': For all blocks, the channel specified by parameter attack_channel is attacked.

    - attack_channel : str
        Only used when dataset_name == 'Exp_pretrain_channel_attacker_9B' and when attack_method == 'all_block_same'.
        Specify the attacked channel for all blocks.

    - mergeBlock : bool
        Only used when dataset_name == 'Exp_pretrain_channel_attacker_9B'
        If true, data in all blocks will be merged into one list.
        If false, data in different blocks will be placed in a single dict. Then a list
            consisting all block list will be returned.

    - test_uncertainty_frame : int
        Only used when dataset_name == 'Exp_stimulus_to_task'.
        if 0, return the certain 6B stimulus and test data.
        if >0, return the uncertain stimulus and test data under which the frame specified
            by the parameter is lost.

    :return:
    - train_dict: dict
       if stimulus type is cVEP:
        {'data': (epoch, channel, time), 'label': (epoch, ), 'stimulus': (class_num, time)}
       if stimulus type is SSVEP:
        {'data': (epoch, channel, time), 'label': (epoch, ), 'stimulus': (class_num, )}
    - test_dict: dict
       same form as train_data

    '''

    if dataset_name == 'data':
        srate = 250

        restrain = open('data/datasets/pre.pickle', 'rb')
        eegtrain = pickle.load(restrain)
        restest = open('data/datasets/formal.pickle', 'rb')
        eegtest = pickle.load(restest)

        subindex = -1
        for possible_subindex in range(len(eegtrain)):
         if eegtrain[possible_subindex]['name'] == subject_name:
             subindex = possible_subindex

        if subindex == -1:
            print('No matching subject !')

        chnINX = [eegtrain[subindex]['channel'].index(i) for i in chnNames]

        # data for train
        train_data = eegtrain[subindex]['stimulus']['X'][:, chnINX, :]  
        train_label = eegtrain[subindex]['stimulus']['y'] - 1 
        train_stimulus = eegtrain[subindex]['stimulus']['STI']  
        train_dict = {'data': train_data, 'label': train_label, 'stimulus': train_stimulus}
        # data for test
        test_data = eegtest[subindex]['wn']['X'][:, chnINX, :]  
        test_label = eegtest[subindex]['wn']['y'] - 1  
        test_stimulus = eegtest[subindex]['wn']['STI']  
        test_dict = {'data': test_data, 'label': test_label, 'stimulus': test_stimulus}


    elif dataset_name == 'Benchmark':

        srate = 250

        _CHANNELS = [
            'FP1', 'FPZ', 'FP2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1', 'FZ', 'F2',
            'F4', 'F6', 'F8', 'FT7', 'FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6',
            'FT8', 'T7', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'T8', 'M1', 'TP7',
            'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8', 'M2', 'P7', 'P5',
            'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO5', 'PO3', 'POZ', 'PO4',
            'PO6', 'PO8', 'CB1', 'O1', 'OZ', 'O2', 'CB2'
        ]

        _FREQS = [
            8, 9, 10, 11, 12, 13, 14, 15,
            8.2, 9.2, 10.2, 11.2, 12.2, 13.2, 14.2, 15.2,
            8.4, 9.4, 10.4, 11.4, 12.4, 13.4, 14.4, 15.4,
            8.6, 9.6, 10.6, 11.6, 12.6, 13.6, 14.6, 15.6,
            8.8, 9.8, 10.8, 11.8, 12.8, 13.8, 14.8, 15.8
        ]



        _PHASES = [
            0, 0.5, 1, 1.5, 0, 0.5, 1, 1.5,
            0.5, 1, 1.5, 0, 0.5, 1, 1.5, 0,
            1, 1.5, 0, 0.5, 1, 1.5, 0, 0.5,
            1.5, 0, 0.5, 1, 1.5, 0, 0.5, 1,
            0, 0.5, 1, 1.5, 0, 0.5, 1, 1.5
        ]

        subject_file = root_dir + '/Benchmark/' + subject_name + '.mat'
        subject_data = scipy.io.loadmat(subject_file)
        subject_data = subject_data['data']   # (64, 1500, 40, 6)

        if chnNames is None:
            subject_data = subject_data[:, round(srate * (0.5 + latency)):round(srate * 5.5), :,
                           :]  
        else:
            chnINX = [_CHANNELS.index(i) for i in chnNames]
            subject_data = subject_data[chnINX, round(srate*(0.5+latency)):round(srate*5.5), :, :]  # (9, 1215, 40, 6)

        subject_label = np.repeat(np.arange(0, 40)[:, np.newaxis], repeats=6, axis=1)  

        if data_divide_method == 'block':
            
            subject_data = subject_data.transpose(3, 2, 0, 1) 
            subject_label = subject_label.T

            for block_idx in range(subject_data.shape[0]):
                for trial_idx in range(subject_data.shape[1]):
                    for ch_idx in range(subject_data.shape[2]):
                        subject_data[block_idx, trial_idx, ch_idx, :] = preprocessing.scale(subject_data[block_idx, trial_idx, ch_idx, :])

            train_data = np.delete(subject_data, test_block_idx, axis=0)[:, :, :, :round(train_winLEN*srate)]
            train_label = np.delete(subject_label, test_block_idx, axis=0)
            train_data = np.concatenate([block for block in train_data], axis=0)
            train_label = train_label.reshape(-1)

            train_data = [trial for trial in train_data]
            train_label = list(train_label)


            test_data = subject_data[test_block_idx:test_block_idx+1, :, :, :round(test_winLEN*srate)]
            test_label = subject_label[test_block_idx:test_block_idx+1, :]
            test_data = np.concatenate([block for block in test_data], axis=0)
            test_label = test_label.reshape(-1)

            test_data = [trial for trial in test_data]
            test_label = list(test_label)

            train_ref_sig = get_ref_sig(freqs=_FREQS, phases=_PHASES, srate=srate,
                sig_len=train_data[0].shape[-1]/srate,
                N=harmonic_number,   # harmonic number
                ignore_stim_phase=False)  # # list: (class_num, ), array: (2*N, train_data_time)
            test_ref_sig = get_ref_sig(freqs=_FREQS, phases=_PHASES, srate=srate,
                                        sig_len=test_data[0].shape[-1] / srate,
                                        N=harmonic_number,  # harmonic number
                                        ignore_stim_phase=False)  # # list: (class_num, ), array: (2*N, test_data_time)



            if filterbank_num != 0:
                train_data = [filterbank(trial, srate=srate, num_subbands=filterbank_num) for trial in train_data]
                test_data = [filterbank(trial, srate=srate, num_subbands=filterbank_num) for trial in test_data]

            train_dict = {'data': train_data, 'label': train_label, 'ref_sig': train_ref_sig, 'freqs': _FREQS,
                          'phases': _PHASES, 'ch_names': _CHANNELS}
            test_dict = {'data': test_data, 'label': test_label, 'ref_sig': test_ref_sig, 'freqs': _FREQS, 'phases': _PHASES,
                         'ch_names': _CHANNELS}

    elif dataset_name == 'Exp_pretrain_channel_attacker_6B':

        srate = 250

        _FREQS = [8, 8.2, 8.4, 8.6, 8.8,
                  9, 9.2, 9.4, 9.6, 9.8,
                  10, 10.2, 10.4, 10.6, 10.8,
                  11, 11.2, 11.4, 11.6, 11.8,
                  12, 12.2, 12.4, 12.6, 12.8,
                  13, 13.2, 13.4, 13.6, 13.8,
                  14, 14.2, 14.4, 14.6, 14.8,
                  15, 15.2, 15.4, 15.6, 15.8]

        _PHASES = [
            0, 0.5, 1, 1.5, 0, 0.5, 1, 1.5,
            0.5, 1, 1.5, 0, 0.5, 1, 1.5, 0,
            1, 1.5, 0, 0.5, 1, 1.5, 0, 0.5,
            1.5, 0, 0.5, 1, 1.5, 0, 0.5, 1,
            0, 0.5, 1, 1.5, 0, 0.5, 1, 1.5
        ]

        subject_file = open(root_dir +
            '/Exp_pretrain_channel_attacker/Pickle_data/' + subject_name + '_cnt/Reliable_6B_SSVEP_3s.pickle',
            'rb') 
        subject_data_dict = pickle.load(subject_file)  
        subject_data, subject_label, channels = subject_data_dict['SSVEP_freqs'].values()
        ch_names = channels[:, 0]

        if chnNames is not None:
            chnINX = [list(ch_names).index(i) for i in chnNames]
        else:
            chnINX = [i for i in range(len(ch_names))]



        subject_data = subject_data[chnINX, round(srate * latency):, :, :]  # (21, 715, 40, 6)
        subject_label = subject_label - 41

        if data_divide_method == 'block':
            subject_data = subject_data.transpose(3, 2, 0, 1)  # (6, 40, 21, 715)
            subject_label = subject_label.T


            train_data = np.delete(subject_data, test_block_idx, axis=0)[:, :, :, :round(train_winLEN*srate)]
            train_label = np.delete(subject_label, test_block_idx, axis=0)
            train_data = np.concatenate([block for block in train_data], axis=0)
            train_label = train_label.reshape(-1)

            train_data = [trial for trial in train_data]
            train_label = list(train_label)


            test_data = subject_data[test_block_idx:test_block_idx+1, :, :, :round(test_winLEN*srate)]
            test_label = subject_label[test_block_idx:test_block_idx+1, :]
            test_data = np.concatenate([block for block in test_data], axis=0)
            test_label = test_label.reshape(-1)

            test_data = [trial for trial in test_data]
            test_label = list(test_label)

            train_ref_sig = get_ref_sig(freqs=_FREQS, phases=_PHASES, srate=srate,
                sig_len=train_data[0].shape[-1]/srate,
                N=harmonic_number,   # harmonic number
                ignore_stim_phase=False)  # # list: (class_num, ), array: (2*N, train_data_time)
            test_ref_sig = get_ref_sig(freqs=_FREQS, phases=_PHASES, srate=srate,
                                        sig_len=test_data[0].shape[-1] / srate,
                                        N=harmonic_number,  # harmonic number
                                        ignore_stim_phase=False)  # # list: (class_num, ), array: (2*N, test_data_time)



            if filterbank_num != 0:
                train_data = [filterbank(trial, srate=srate, num_subbands=filterbank_num) for trial in train_data]
                test_data = [filterbank(trial, srate=srate, num_subbands=filterbank_num) for trial in test_data]

            train_dict = {'data': train_data, 'label': train_label, 'ref_sig': train_ref_sig, 'freqs': _FREQS,
                          'phases': _PHASES, 'ch_names': ch_names}
            test_dict = {'data': test_data, 'label': test_label, 'ref_sig': test_ref_sig, 'freqs': _FREQS, 'phases': _PHASES, 'ch_names': ch_names}

    elif dataset_name == 'Exp_stimulus_to_task':

        srate = 250

        train_filename = root_dir + '/Exp_stimulus_to_task/Pickle_data/' + subject_name + '_cnt/20TARGET_cVEP_singletarget_6B_3s.pickle'
        restrain = open(train_filename, 'rb')
        eegtrain_dict = pickle.load(restrain)

        if test_uncertainty_frame == 0:
            test_filename = root_dir + '/Exp_stimulus_attack/Pickle_data/' + subject_name + '_cnt/certainty_6B_randomcVEP.pickle'
            restest = open(test_filename, 'rb')
            eegtest_dict = pickle.load(restest)
        else:
            test_filename = root_dir + '/Exp_stimulus_attack/Pickle_data/' + subject_name + '_cnt/uncertainty_1B_randomcVEP_stim' + str(test_uncertainty_frame) + '.pickle'
            restest = open(test_filename, 'rb')
            eegtest_dict = pickle.load(restest)

        train_data, train_label, channels = eegtrain_dict['SSVEP_freqs'].values()
        test_data, test_label, channels = eegtest_dict['cVEP_freqs'].values()
        ch_names = channels[:, 0]



        chnINX = [list(ch_names).index(i) for i in chnNames]

        train_data = train_data[chnINX, :, :, :].transpose(3, 2, 0, 1)
        test_data = test_data[chnINX, :, :, :].transpose(3, 2, 0, 1)

        train_label = train_label.T - 1
        test_label = test_label.T - 41


        train_data = np.concatenate([block for block in train_data], axis=0)
        test_data = np.concatenate([block for block in test_data], axis=0)

        train_label = train_label.reshape(-1)
        test_label = test_label.reshape(-1)

        train_stimulus = np.load(root_dir + '/Exp_stimulus_to_task/20_target_train_stimulus.npy')

        if test_uncertainty_frame == 0:
            test_stimulus = np.load(root_dir + '/Exp_stimulus_attack/40_target_certain_test_stimulus.npy')

        else:
            test_stimulus = np.load(
                root_dir + '/Exp_stimulus_attack/uncertain_stimulus/srate_250_3s/resampled_stim_code_uncertainty_' + str(
                    test_uncertainty_frame) + '.npy')



        train_dict = {'data': train_data, 'label': train_label, 'stimulus': train_stimulus}
        test_dict = {'data': test_data, 'label': test_label, 'stimulus': test_stimulus}

    elif dataset_name == 'Beta':

        srate = 250

        _CHANNELS = [
            'FP1', 'FPZ', 'FP2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6',
            'F8', 'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8', 'T7', 'C5',
            'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8', 'M1', 'TP7', 'CP5', 'CP3', 'CP1', 'CPZ',
            'CP2', 'CP4', 'CP6', 'TP8', 'M2', 'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6',
            'P8', 'PO7', 'PO5', 'PO3', 'POZ', 'PO4', 'PO6', 'PO8', 'CB1', 'O1', 'OZ', 'O2', 'CB2'
        ]

        _FREQS = [
            8.6, 8.8,
            9, 9.2, 9.4, 9.6, 9.8,
            10, 10.2, 10.4, 10.6, 10.8,
            11, 11.2, 11.4, 11.6, 11.8,
            12, 12.2, 12.4, 12.6, 12.8,
            13, 13.2, 13.4, 13.6, 13.8,
            14, 14.2, 14.4, 14.6, 14.8,
            15, 15.2, 15.4, 15.6, 15.8,
            8, 8.2, 8.4
        ]

        _PHASES = [
            1.5, 0,
            0.5, 1, 1.5, 0, 0.5,
            1, 1.5, 0, 0.5, 1,
            1.5, 0, 0.5, 1, 1.5,
            0, 0.5, 1, 1.5, 0,
            0.5, 1, 1.5, 0, 0.5,
            1, 1.5, 0, 0.5, 1,
            1.5, 0, 0.5, 1, 1.5,
            0, 0.5, 1
        ]

        subject_file = root_dir + '/Beta/' + subject_name + '.mat'
        subject_data = loadmat(subject_file)
        subject_data = subject_data['data']['EEG']  # (64, 750, 4, 40)

        if chnNames is None:
            subject_data = subject_data[:, :, :, :]
        else:
            chnINX = [_CHANNELS.index(i) for i in chnNames]
            subject_data = subject_data[chnINX, :, :, :]

        # 判断数据是3s还是4s
        if subject_data.shape[1] > srate * 3.5:
            subject_data = subject_data[:, round(srate * (0.5 + latency)):round(srate * 3.5), :,
                       :]  # (64, 715, 4, 40)

        else:
            subject_data = subject_data[:, round(srate * (0.5 + latency)):round(srate * 2.5), :,
                           :]  # (64, 465, 4, 40)

        subject_label = np.repeat(np.arange(0, 40)[:, np.newaxis], repeats=4, axis=1)  # (40, 4)


        if data_divide_method == 'block':
            # 用于SSVEP方法:
            subject_data = subject_data.transpose(2, 3, 0, 1)  # (4, 40, 64, time)
            subject_label = subject_label.T  # (4, 40)

            for block_idx in range(subject_data.shape[0]):
                for trial_idx in range(subject_data.shape[1]):
                    for ch_idx in range(subject_data.shape[2]):
                        subject_data[block_idx, trial_idx, ch_idx, :] = preprocessing.scale(
                            subject_data[block_idx, trial_idx, ch_idx, :])

            train_data = np.delete(subject_data, test_block_idx, axis=0)[:, :, :, :round(train_winLEN * srate)]
            train_label = np.delete(subject_label, test_block_idx, axis=0)
            train_data = np.concatenate([block for block in train_data], axis=0)
            train_label = train_label.reshape(-1)

            train_data = [trial for trial in train_data]
            train_label = list(train_label)

            test_data = subject_data[test_block_idx:test_block_idx + 1, :, :, :round(test_winLEN * srate)]
            test_label = subject_label[test_block_idx:test_block_idx + 1, :]
            test_data = np.concatenate([block for block in test_data], axis=0)
            test_label = test_label.reshape(-1)

            test_data = [trial for trial in test_data]
            test_label = list(test_label)

            train_ref_sig = get_ref_sig(freqs=_FREQS, phases=_PHASES, srate=srate,
                                        sig_len=train_data[0].shape[-1] / srate,
                                        N=harmonic_number,  # harmonic number
                                        ignore_stim_phase=False)  # # list: (class_num, ), array: (2*N, train_data_time)
            test_ref_sig = get_ref_sig(freqs=_FREQS, phases=_PHASES, srate=srate,
                                       sig_len=test_data[0].shape[-1] / srate,
                                       N=harmonic_number,  # harmonic number
                                       ignore_stim_phase=False)  # # list: (class_num, ), array: (2*N, test_data_time)

            if filterbank_num != 0:
                train_data = [filterbank(trial, srate=srate, num_subbands=filterbank_num) for trial in train_data]
                test_data = [filterbank(trial, srate=srate, num_subbands=filterbank_num) for trial in test_data]


            train_dict = {'data': train_data, 'label': train_label, 'ref_sig': train_ref_sig, 'freqs': _FREQS,
                          'phases': _PHASES, 'ch_names': _CHANNELS}
            test_dict = {'data': test_data, 'label': test_label, 'ref_sig': test_ref_sig, 'freqs': _FREQS,
                         'phases': _PHASES, 'ch_names': _CHANNELS}

    elif dataset_name == 'Exp_pretrain_channel_attacker_9B':
      
        srate = 250

        _FREQS = [
            8, 9, 10, 11, 12, 13, 14, 15,
            8.2, 9.2, 10.2, 11.2, 12.2, 13.2, 14.2, 15.2,
            8.4, 9.4, 10.4, 11.4, 12.4, 13.4, 14.4, 15.4,
            8.6, 9.6, 10.6, 11.6, 12.6, 13.6, 14.6, 15.6,
            8.8, 9.8, 10.8, 11.8, 12.8, 13.8, 14.8, 15.8
        ]

        _PHASES = [
            0, 0.5, 1, 1.5, 0, 0.5, 1, 1.5,
            0.5, 1, 1.5, 0, 0.5, 1, 1.5, 0,
            1, 1.5, 0, 0.5, 1, 1.5, 0, 0.5,
            1.5, 0, 0.5, 1, 1.5, 0, 0.5, 1,
            0, 0.5, 1, 1.5, 0, 0.5, 1, 1.5
        ]

        ########### load attacked blocks #############
        subject_data_file = open(root_dir +
            '/Exp_pretrain_channel_attacker/Pickle_data/' + subject_name + '_cnt/Unreliable_9B_SSVEP_3s.pickle',
            'rb')  
        subject_data_dict = pickle.load(subject_data_file) 

        subject_data, subject_label, channels = subject_data_dict['SSVEP_freqs'].values()

        ########### load attacker file #############
        subject_attacker_file = open(root_dir +
                                     '/Exp_pretrain_channel_attacker/Pickle_data/' + subject_name + '_cnt/Uncertainty_1B_SSVEP_3s.pickle',
                                     'rb') 
        subject_attacker_data_dict = pickle.load(subject_attacker_file)
        attacker_data, _, _ = subject_attacker_data_dict['SSVEP_freqs'].values()


        nblock = subject_data.shape[-1]
        ch_names = channels[:, 0]

        if attack_method is not None:
            subject_data = np.concatenate((subject_data, attacker_data), axis=-1)  # (64, 750, 40, 10)

        if chnNames is not None:
            chnINX = [list(ch_names).index(i) for i in chnNames]
            subject_data = subject_data[chnINX, :, :, :]  # (9, 715, 40, 10)
            ch_names = chnNames

        subject_data = subject_data[:, round(srate * latency):, :, :]  # (64, 715, 40, 9)
        subject_data = subject_data[:, :round(test_winLEN * srate), :, :]

        subject_label = subject_label - 41

        subject_data = subject_data.transpose(3, 2, 0, 1)  # (10, 40, 9, time)
        subject_label = subject_label.T  # (9, 40)


        test_ref_sig = get_ref_sig(freqs=_FREQS, phases=_PHASES, srate=srate,
                                   sig_len=subject_data.shape[-1] / srate,
                                   N=harmonic_number,  # harmonic number
                                   ignore_stim_phase=False)  # # list: (class_num, ), array: (2*N, test_data_time)


        if attack_method == 'all_block_diff':
            for block_idx in range(nblock):
                subject_data[block_idx, :, block_idx, :] = subject_data[-1, :, block_idx, :]

            subject_data = subject_data[:nblock, :, :, :]  # (9, 40, 9, 715)
        elif attack_method == 'all_block_same':
            attack_ch_idx = []
            if attack_channel in ch_names:
                attack_ch_idx = ch_names.index(attack_channel)

            for block_idx in range(nblock):
                subject_data[block_idx, :, attack_ch_idx, :] = subject_data[-1, :, attack_ch_idx, :]

            subject_data = subject_data[:nblock, :, :, :]  # (9, 40, 9, 715)

        if mergeBlock == False:

            test_block_list = []
            for block_idx in range(nblock):
                block_data = subject_data[block_idx, :, :, :]  # (40, 9, 715)
                block_label = subject_label[block_idx, :]  # (40, )

                block_data = [trial for trial in block_data]
                block_label = list(block_label)

                if filterbank_num != 0:
                    block_data = [filterbank(trial, srate=srate, num_subbands=filterbank_num) for trial in block_data]

                block_dict = {'data': block_data, 'label': block_label, 'ref_sig': test_ref_sig, 'freqs': _FREQS, 'phases': _PHASES, 'ch_names': ch_names}
                test_block_list.append(block_dict)

            return test_block_list
        else:

            test_data = np.concatenate([block for block in subject_data], axis=0)
            test_label = list(subject_label.reshape(-1))



            if filterbank_num != 0:
                test_data = [filterbank(trial, srate=srate, num_subbands=filterbank_num) for trial in test_data]

            test_dict = {'data': test_data, 'label': test_label, 'ref_sig': test_ref_sig, 'freqs': _FREQS,
                          'phases': _PHASES, 'ch_names': ch_names}


            return test_dict






    return train_dict, test_dict

def build_Benchmark_dataset(root_dir, winLEN=1):
    ########### Benchmark ############
    dataset_name = 'Benchmark'
    sub_name_list = ['S' + str(idx) for idx in range(1, 36)]
    n_fold = 6
    ssvep_selected_channels = ['PZ', 'PO5', 'PO3', 'POZ', 'PO4', 'PO6', 'O1', 'OZ', 'O2']
    fs = 250
    num_subbands = 1
    harmonic_num = 5
    delay_num = 0  
    electrodes_adjacent_matrix = np.loadtxt(
        root_dir + '/' + dataset_name + '/Benchmark_9ch_normalized_adjacency_matrix.csv', delimiter=',')
    
    train_data = []
    test_data = []
    for sub_name in sub_name_list:
        train_dict, test_dict = readSubjectData(dataset_name, sub_name, train_winLEN=winLEN,
                                                       test_winLEN=winLEN,
                                                       data_divide_method='block',
                                                       filterbank_num=num_subbands, harmonic_number=harmonic_num,
                                                       root_dir=root_dir, chnNames=ssvep_selected_channels)
        sub_train = train_dict['data']
        sub_test = test_dict['data']

        sub_train = np.stack(sub_train, axis=0)
        sub_test = np.stack(sub_test, axis=0)

        train_data.append(sub_train)
        test_data.append(sub_test)

    train_data =np.concatenate(train_data, axis=0).squeeze()
    test_data = np.concatenate(test_data, axis=0).squeeze()

    train_data = torch.from_numpy(train_data)
    test_data = torch.from_numpy(test_data)


    train_dataset = TensorDataset(train_data)
    test_dataset = TensorDataset(test_data)

    return train_dataset, test_dataset
