# Deep Recurrent Survival Analysis (DRSA)
A `tensorflow` implementation of DRSA model. This is the experiment code for the working paper (https://arxiv.org/abs/1809.02403).
If you have any problems, please feel free to contact the authors [Kan Ren](http://saying.ren), [Jiarui Qin](mailto:qinjr@icloud.com) and [Lei Zheng](mailto:zhenglei2016@sjtu.edu.cn).

### Model Description
Our model is `DRSA` model. The baseline models are `Kaplan-Meier`, `Lasso-Cox`, `Gamma`, `MTLSA`, `STM`, `DeepSurv`, `DeepHit`, `DRN`, and `DRSA`.
Among the baseline implementations, we forked the code of [STM](https://github.com/zeromike/bid-lands) and [MTLSA](https://github.com/MLSurvival/MTLSA).
We made some minor modifications on the two projects to fit them for our experiments, **To get the modified code, you can click MTLSA @ ba353f8 and STM @ df57e70.** Many thanks to the authors of `STM` and `MTLSA`.
Other baselines' implementations are in `python` directory.

### Data Preparation
We have uploaded a tiny data sample for training and evaluation.

The **full dataset** for this project can be downloaded at this link: https://goo.gl/nUFND4.
This dataset contains three large-scale datasets in three real-world tasks, which is the first dataset with such scale for experiment reproduction in survival analysis.

After download please replace the sample data in `data/` folder with the full data files.

#### Data specification
Each line is a sample containing the "`yztx`" data, the information is splitted by `SPACE`.
Here `z` is the true event time, `t` is the observation time and `x` is the list of features (multi-hot encoded as `feat_id:1`).
In the experiment, we only use `ztx` data.
Note that, for the uncensored data, `z <= t`, while for the censored data, `z > t`.

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
@article{ren2018deep,
  title={Deep Recurrent Survival Analysis},
  author={Ren, Kan and Qin, Jiarui and Zheng, Lei and Yang, Zhengyu and Zhang, Weinan and Qiu, Lin and Yu, Yong},
  year={2019},
  organization={AAAI}
}
```
