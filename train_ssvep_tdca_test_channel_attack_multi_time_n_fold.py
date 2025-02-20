#!/usr/bin/env python3
# -*- coding:utf-8 -*-


import os
import numpy as np
from utils import suggested_weights_filterbank, ITR
from processing_toolkit import add_gaussian_white_noise
from ssvep_tdca import SSVEP_TDCA, SSVEP_EAM_TDCA, SSVEP_CAM_TDCA, SSVEP_ROBUST_TDCA
from dataset import readSubjectData
import time
import argparse

import warnings
warnings.filterwarnings('ignore')


def get_args():
    parser = argparse.ArgumentParser('TDCA parameters', add_help=False)
    parser.add_argument('--target_noise_db', default=0, type=int) # 对应噪声方差为1

    parser.add_argument('--tdca_mode', default='EAM', type=str,
                        help='normal, EAM, CAM, or ROBUST')
    parser.add_argument('--robust_method', default='weights', type=str,
                        help='sum, max, or weights')
    parser.add_argument('--root_dir', default='dataset', type=str,
                        help='root directory of the dataset')
    parser.add_argument('--dataset_name', default='Benchmark', type=str,
                        help='Benchmark or Beta')


    return parser.parse_args()



ssvep_selected_channels = ['PZ', 'PO5', 'PO3', 'POZ', 'PO4', 'PO6', 'O1', 'OZ', 'O2']
interpolation_matrices = np.load('interpolation_matrix/9ch_interpolation_matrix.npy')





def n_fold_evaluation(dataset_name, subject_name, clf_1, n_fold, train_winLEN, test_winLEN, num_subbands=5, harmonic_num=5):
    n_fold_accs = []
    n_fold_itrs = []

    for fold_idx in range(n_fold):
        fold_accs, fold_itrs = evaluation_on_attacked_data(dataset_name, subject_name, clf_1, fold_idx, train_winLEN, test_winLEN, num_subbands, harmonic_num)
        n_fold_accs.append(fold_accs)
        n_fold_itrs.append(fold_itrs)
    n_fold_accs = np.stack(n_fold_accs)  # (n_fold, ch_num+1)
    n_fold_itrs = np.stack(n_fold_itrs)

    return n_fold_accs, n_fold_itrs


def evaluation_on_attacked_data(dataset_name, subject_name, clf_1, fold_idx, train_winLEN, test_winLEN, num_subbands=5, harmonic_num=5):




    train_dict, common_test_dict = readSubjectData(dataset_name, subject_name, train_winLEN=train_winLEN,
                                            test_winLEN=test_winLEN,
                                            data_divide_method='block', test_block_idx=fold_idx,
                                            filterbank_num=num_subbands, harmonic_number=harmonic_num,
                                            root_dir=args.root_dir, chnNames=ssvep_selected_channels)

    # train the model
    X_train = train_dict['data']
    Y_train = train_dict['label']
    train_ref_sig = train_dict['ref_sig']


    clf_1.fit(X=X_train, Y=Y_train, train_ref_sig=train_ref_sig)

    ch_num = X_train[0].shape[1]
    # evaluation on test data that is not attacked
    accs = np.zeros((ch_num + 1,))
    itrs = np.zeros((ch_num + 1,))

    X_test_common = common_test_dict['data']
    Y_test_common = common_test_dict['label']
    test_ref_sig = common_test_dict['ref_sig']


    test_class_num = len(test_ref_sig)

    if clf_1.tdca_mode == 'normal':
        acc, _ = clf_1.score(X_test_common, Y_test_common, test_ref_sig)
    elif clf_1.tdca_mode == 'EAM' or clf_1.tdca_mode == 'CAM':
        acc, _ = clf_1.score(X_test_common, Y_test_common, test_ref_sig, [])
    elif clf_1.tdca_mode == 'ROBUST':
        acc, _, _, _ = clf_1.score(X_test_common, Y_test_common, test_ref_sig, [])

    itr = ITR(test_class_num, acc, test_winLEN)

    accs[0] += acc
    itrs[0] += itr


    for ch_idx in range(ch_num):
        X_test_attack = []
        for trial_idx in range(len(X_test_common)):
            attacked_trial = X_test_common[trial_idx].copy()  # 不能改变原值
            attacked_trial[:, ch_idx, :] = 0

            # 一个trial的所有filterbank替换成一样的高斯噪声
            noise = add_gaussian_white_noise(attacked_trial[0, ch_idx, :], target_noise_db=args.target_noise_db, mode='noisePower')

            attacked_trial[:, ch_idx, :] = np.stack([noise for _ in range(attacked_trial.shape[0])], axis=0)

            X_test_attack.append(attacked_trial)


        test_class_num = len(test_ref_sig)


        if clf_1.tdca_mode == 'normal':
            acc, _ = clf_1.score(X_test_attack, Y_test_common, test_ref_sig)
        elif clf_1.tdca_mode == 'EAM' or clf_1.tdca_mode == 'CAM':
            acc, _ = clf_1.score(X_test_attack, Y_test_common, test_ref_sig, [ch_idx])
        elif clf_1.tdca_mode == 'ROBUST':
            acc, _, _, _ = clf_1.score(X_test_attack, Y_test_common, test_ref_sig, [ch_idx])

        itr = ITR(test_class_num, acc, test_winLEN)

        accs[ch_idx+1] += acc
        itrs[ch_idx+1] += itr


    return accs, itrs   # (10, ), (10, )

def main(args):
    if args.dataset_name == 'Benchmark':
        sub_name_list = ['S' + str(idx) for idx in range(1, 36)]
        n_fold = 6
        fs = 250
        num_subbands = 5
        harmonic_num = 5
        delay_num = 0  # 数据增强时延迟的个数

    elif args.dataset_name == 'Beta':
        sub_name_list = ['S' + str(idx) for idx in range(1, 71)]
        n_fold = 4
        fs = 250
        num_subbands = 5
        harmonic_num = 5
        delay_num = 0  # 数据增强时延迟的个数

    acc_tdca_all = np.zeros((len(sub_name_list), 5, 10, n_fold))  # 1 common test + 9 attack test
    itr_tdca_all = np.zeros((len(sub_name_list), 5, 10, n_fold))

    print('########### Mode: ', args.tdca_mode, ' #############')

    for sub_idx, subject_name in enumerate(sub_name_list):
        print('======================== subject: ', subject_name, ' ========================')
        for t_idx in range(5):
            winLEN = 0.2 + t_idx * 0.2
            # winLEN = 1
            print('******** window length: ', str(winLEN), 's **********')
            time_start = time.time()
            # TDCA
            weights_filterbank = suggested_weights_filterbank()
            if args.tdca_mode == 'normal':
                model = SSVEP_TDCA(weights_filterbank=weights_filterbank, n_delay=delay_num)
            elif args.tdca_mode == 'EAM':
                model = SSVEP_EAM_TDCA(weights_filterbank=weights_filterbank, n_delay=delay_num,
                                           electrodes_names=ssvep_selected_channels, interpolation_matrices=interpolation_matrices)
            elif args.tdca_mode == 'CAM':
                model = SSVEP_CAM_TDCA(weights_filterbank=weights_filterbank, n_delay=delay_num)
            elif args.tdca_mode == 'ROBUST':
                model = SSVEP_ROBUST_TDCA(weights_filterbank=weights_filterbank, n_delay=delay_num,
                                       ssvep_selected_channels=ssvep_selected_channels, interpolation_matrices=interpolation_matrices, method=args.robust_method)

            n_fold_accs, n_fold_itrs = n_fold_evaluation(args.dataset_name, subject_name, model, n_fold, winLEN, winLEN, num_subbands, harmonic_num)
            acc_tdca_all[sub_idx, t_idx] = n_fold_accs.T
            itr_tdca_all[sub_idx, t_idx] = n_fold_itrs.T

            time_end = time.time()
            print('Accuracies=', n_fold_accs)
            print('Average of n_fold: ', np.mean(n_fold_accs, axis=0))
            print('time: ', time_end - time_start)


            np.save('acc_tdca_' + args.dataset_name + '_' + args.tdca_mode + '_all_sub_ch_attack_multi_time_n_fold.npy', acc_tdca_all)
            np.save('itr_tdca_' + args.dataset_name + '_' + args.tdca_mode + '_all_sub_ch_attack_multi_time_n_fold.npy', itr_tdca_all)



if __name__ == "__main__":
    args = get_args()
    main(args)
