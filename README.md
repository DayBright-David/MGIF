# A Reliability-Enhanced Brain-Computer Interface via Mixture-of-Graphs-driven Information Fusion

This repo contains the implementation of the paper [A Reliability-Enhanced Brain-Computer Interface via Mixture-of-Graphs-driven Information Fusion](https://www.sciencedirect.com/science/article/pii/S1566253525001423#:~:text=This%20study%20presents%20a%20novel%20Mixture-of-Graphs-driven%20Information%20Fusion,integration%20of%20multi-graph%20knowledge%20for%20stable%20EEG%20representat)

![](./figs/Why_MGIF.png)

Reliable Brain-Computer Interface (BCI) systems are essential for practical applications. Current BCIs often suffer from performance degradation due to environmental noise and external interference. These environmental factors significantly compromise the quality of EEG data acquisition. This study presents a novel Mixture-of-Graphs-driven Information Fusion (MGIF) framework to enhance BCI system robustness through the integration of multi-graph knowledge for stable EEG representations.
Initially, the framework constructs complementary graph architectures: electrode-based structures for capturing spatial relationships and signal-based structures for modeling inter-channel dependencies. Subsequently, the framework employs filter bank-driven multi-graph constructions to encode spectral information and incorporates a self-play-driven fusion strategy to optimize graph embedding combinations. Finally, an adaptive gating mechanism is implemented to monitor electrode states and enable selective information fusion, thereby minimizing the impact of unreliable electrodes and environmental disturbances. Extensive evaluations through offline datasets and online experiments validate the framework’s effectiveness. Results demonstrate that MGIF achieves significant improvements in BCI reliability across challenging real-world environments.  

![](./figs/MGIF.png)

## Installation

1. Create a conda environment with python version==3.12

   ```
   conda create -n [ENV_NAME] python==3.12
   ```

2. Install dependencies

   ```
   pip install -r requirements.txt
   ```

## Data preparation

Before executing the code, please download the dataset from https://bci.med.tsinghua.edu.cn/download.html and place it in the `dataset` folder at the same directory level as the scripts. For instance, store all Benchmark subject data in the `dataset/Benchmark` directory.

We also conducted additional validation experiments, and the data has been made publicly available at: [https://doi.org/10.5281/zenodo.14044363](https://doi.org/10.5281/zenodo.14044363), which we hope will serve as a valuable reference for future researchers.

## Get started

The command to compute accuracy and ITR for different recognition algorithms on the Benchmark and Beta datasets is:

```
python train_ssvep_recognition_algorithms_test_channel_attack_multi_time_n_fold.py --dataset_name [DATASET_NAME] --mode [MODE] --mgif_method [MGIF_METHOD] --target_noise_db=0
```

The detailed explanation of the parameters:

|    Parameter    | Type |                         Description                          |
| :-------------: | :--: | :----------------------------------------------------------: |
|  dataset_name   | str  |                    `Benchmark` or `Beta`                     |
|    root_dir     | str  |     Root directory of the dataset. Default is `dataset`.     |
|      mode       | str  | Type of the recognition method, can be specified as <br>`normal` -- Standard method <br/>`egraph` -- E-graph <br/>`sgraph` -- S-graph <br/>`mgif` -- MGIF |
|    mgif_mode    | str  | Only used when `mode==mgif`. Can be specified as `sum`, `max`, or `weights`. |
| target_noise_db | int  | Power of the noise in dB. Default is `0`, corresponding to a noise variance of `1`. |

## Citation

If you find this repository useful for your publications, please consider citing our paper.

```
@article{dai2025reliability,
  title={A reliability-enhanced Brain-Computer Interface via Mixture-of-Graphs-driven Information Fusion},
  author={Dai, Bo and Wang, Yijun and Mou, Xinyu and Gao, Xiaorong},
  journal={Information Fusion},
  pages={103069},
  year={2025},
  publisher={Elsevier}
}
```

