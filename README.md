# Deep Recurrent Survival Analysis (DRSA)
A `tensorflow` implementation of DRSA model. This is the experiment code for the working paper (https://arxiv.org/abs/1809.02403).
If you have any problems, please feel free to contact the authors [Kan Ren](http://saying.ren), [Jiarui Qin](mailto:qinjr@icloud.com) and [Lei Zheng](mailto:zhenglei2016@sjtu.edu.cn).

### Model Description
Our model is `DRSA` model. The baseline models are `Kaplan-Meier`, `Lasso-Cox`, `Gamma`, `MTLSA`, `STM`, `DeepSurv`, `DeepHit`, `DRN`, and `DRSA`.
Among the baseline implementations, we forked the code of [STM](https://github.com/zeromike/bid-lands) and [MTLSA](https://github.com/MLSurvival/MTLSA).
We made some minor modifications on the two projects to fit in our experiments. To get the modified code, you may click MTLSA @ ba353f8 and STM @ df57e70. Many thanks to the authors of `STM` and `MTLSA`.
Other baselines' implementations are in `python` directory.

### Data Preparation
We have uploaded a tiny data sample for training and evaluation.

The **full dataset** for this project can be directly downloaded from this link: https://goo.gl/nUFND4.
This dataset contains three large-scale datasets in three real-world tasks, which is the first dataset with such scale for experiment reproduction in survival analysis.

After download please replace the sample data in `data/` folder with the full data files.

| Dataset  | MD5 Code  | Size |
| ------------ | ------------ | --- |
| drsa.**zip** | b63c53559f58e6afa62c121b0dd1997d  | 2.6 GB |

#### Data specification
We have three datasets and each of them contains `.yzbx.txt`, `featureindex.txt` and `.log.txt`.
We created the first data file `.log.txt` from the raw data of the original data source (please refer to our paper).
Then we made feature engineering according to the created feature dictionary `featindex.txt`.
The corresponding feature engineered data are in `.yzbx.txt`.

If you need to reproduce the experiemtns, you may run over `.yzbx.txt`.
If you want to dive deep and explain the observations of experiments, you would need to look into the the other files like `.log.txt` and `featindex.txt`. 

In `yzbx.txt` file, each line is a sample containing the "`yztx`" data (here we use `t` and `b` exchangably), the information is splitted by `SPACE`.
Here `z` is the true event time, `t` is the observation time and `x` is the list of features (multi-hot encoded as `feat_id:1`).
In the experiment, we only use `ztx` data.
Note that, for the uncensored data, `z <= t`, while for the censored data, `z > t`.

We conduct a simulation of observation experiments which ranges from the whole timeline of each dataset. Then the end of each observation (in right-censored situation) is tracked as `t` in the final data `yztx` along with the true event time `z`.
The true event time `z` is originally logged in the raw data file.
The raw data file (without any feature engineering) is from the other related works as described in the exp. part of our paper. We put the download links as below:
* clinic: http://biostat.mc.vanderbilt.edu/wiki/Main/DataSets  (support2csv.zip)
* music: https://www.dtic.upf.edu/~ocelma/MusicRecommendationDataset/lastfm-1K.html
* bidding: https://github.com/wnzhang/make-ipinyou-data

### Installation and Reproduction
[TensorFlow](https://www.tensorflow.org/)(>=1.3) and the other dependant packages (e.g., `numpy`, `sklearn` and `matplotlib`) should be pre-installed before running the code.

After package installation, you can simply run the code in `python` directory with the demo tiny dataset(sampled from BIDDING dataset). The outputs of the code are in `python/output` directory.

The running command are listed as below.
```
python km.py             # for Kaplan-Meier
python gamma_model.py    # for Gamma
python cox.py            # for Lasso-Cox and DeepSurv
python deephit.py        # for DeepHit
python DRN.py 0.0001      # for DRN
python DRSA.py 0.0001     # for DRSA
```
We have set default hyperparameters in the model implementation. So the parameter arguments are optional for running the code.

The results will be printed on the screen with the format:
Subset, Train/Test,  Step,  Cross Entropy, AUC, ANLP, Total Loss, batch size, hidden state size, learing rate, anlp learning rate, alpha, beta.

### Citation
You are more than welcome to cite our paper:
```
@article{ren2019deep,
  title={Deep Recurrent Survival Analysis},
  author={Ren, Kan and Qin, Jiarui and Zheng, Lei and Yang, Zhengyu and Zhang, Weinan and Qiu, Lin and Yu, Yong},
  year={2019},
  organization={AAAI}
}
```
