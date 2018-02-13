# DEep Survival Analysis(DESA)
A `tensorflow` implementation of DESA model. This is the experiment code for the KDD'18 submitted paper "Deep Survival Analysis for Fine-grained Bid Landscape Forecasting in Real-time Bidding Advertising". 
If you have any questions, please feel free to contact [Kan Ren](http://apex.sjtu.edu.cn/members/kren) (kren@apex.sjtu.edu.cn), [Jiarui Qin](http://apex.sjtu.edu.cn/members/qinjr) (qinjr@apex.sjtu.edu.cn) and [Lei Zheng](http://apex.sjtu.edu.cn/members/zhenglei)(zhenglei@apex.sjtu.edu.cn).

Baseline models include `STM`, `MM`, `MTLSA`, `COX-NN`, `RNN` and `DNN`, amoung all the baseline implementations, we use the code of [STM](https://github.com/zeromike/bid-lands)(it also has the implementation of `MM`) and [MTLSA](https://github.com/MLSurvival/MTLSA).
Of course, we made some modifies on those codes to fit them for our experiments, **To get the modified code, you can click MTLSA @ 712e3bc and STM_MM @ 2d57f03 to get them.** Many thanks to the authors of `STM/MM` and `MTLSA`.
Other baselines' implementations are in `python` directory.

### Data Preparation
We have upload a tiny data sample for training and evaluation.
The full dataset for this project can be download from this [link](http://apex.sjtu.edu.cn/datasets/14).
After download please replace the sample data in `data/` folder with the full data files.

### Installation and Running
[TensorFlow](https://www.tensorflow.org/)(>=1.3) and dependant packages (e.g., `numpy`, `sklearn` and `matplotlib`) should be pre-installed before running the code. 

After package installation, you can simple run the code in `python` directory with the demo tiny dataset(sampled from campaign 2259). The outputs of the code are in `python/output' directory.

```
python3 coxnn.py [campaign] [learn_rate] [batch_size] [hidden_layer_size] [threshold] [w_k] [w_lambda] #for COX-NN
python DESA.py 0.0001     # for DESA
python RNN.py 0.0001      # for RNN
python DNN.py 0.0001      # for DNN
```
We have set default hyperparameter in the model implementation. So the parameter arguments are optional for running the code.

Result will be printed on the screen with the format: 
Camp, Train/Test,  Step,  Cross Entropy, AUC, ANLP, Total Loss, batch size, hidden state size, learing rate, anlp learning rate, alpha, beta.
