# Deep Recurrent Survival Analysis (DRSA)
A `tensorflow` implementation of DRSA model. This is the experiment code for the working paper.

Our model is `DRSA` model. The baseline models are `Kaplan-Miere`, `Lasso-Cox`, `Gamma`, `MTLSA`, `STM`, `DeepSurv`, `DeepHit`, `DRN`, and `DRSA`.
Among the baseline implementations, we forked the code of [STM](https://github.com/zeromike/bid-lands) and [MTLSA](https://github.com/MLSurvival/MTLSA).
Of course, we also made some modifications on the two projects to fit them for our experiments, **To get the modified code, you can click MTLSA @ ba353f8 and STM @ df57e70 to get them.** Many thanks to the authors of `STM` and `MTLSA`.
Other baselines' implementations are in `python` directory.

### Data Preparation
We have uploaded a tiny data sample for training and evaluation.
The full dataset for this project will be published soon.
After download please replace the sample data in `data/` folder with the full data files.

### Installation and Running
[TensorFlow](https://www.tensorflow.org/)(>=1.3) and dependant packages (e.g., `numpy`, `sklearn` and `matplotlib`) should be pre-installed before running the code.

After package installation, you can simple run the code in `python` directory with the demo tiny dataset(sampled from BIDDING dataset's campaign 2259). The outputs of the code are in `python/output' directory.

```
python3 km.py             # for Kaplan-Miere
python3 gamma_model.py    # for Gamma
python3 cox.py            # for Lasso-Cox and DeepSurv
python3 deephit.py        # for DeepHit
python RNN.py 0.0001      # for RNN
python DNN.py 0.0001      # for DNN
python DESA.py 0.0001     # for DESA
```
We have set default hyperparameter in the model implementation. So the parameter arguments are optional for running the code.

The result will be printed on the screen with the format:
Camp, Train/Test,  Step,  Cross Entropy, AUC, ANLP, Total Loss, batch size, hidden state size, learing rate, anlp learning rate, alpha, beta.
