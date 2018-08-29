import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os
import random
import time
from util import *
from sklearn.metrics import *
import sys

class Model:
    def __init__(self, lr_1, lr_2, l2_loss_weight, batch_size, dimension, theta0, util_train, util_test, campaign):
        self.lr_1 = lr_1
        self.lr_2 = lr_2
        self.util_train = util_train
        self.util_test = util_test
        self.train_data_amt = util_train.get_data_amt()
        self.test_data_amt = util_test.get_data_amt()
        self.batch_size = batch_size
        self.batch_num = int(self.train_data_amt / self.batch_size)
        self.l2_loss_weight = l2_loss_weight
        self.campaign = campaign

        # output directory
        model_name = "{0}_{1}_{2}_{3}".format(self.lr_1, self.lr_2, self.l2_loss_weight, self.batch_size)
        self.output_dir = 'output/gamma/{}/{}/'.format(campaign, model_name)
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # reset graph
        tf.reset_default_graph()

        # placeholders
        self.X = tf.sparse_placeholder(tf.float64)
        self.z = tf.placeholder(tf.float64)
        self.b = tf.placeholder(tf.float64)
        self.y = tf.placeholder(tf.float64)

        # trainable variables
        self.theta = tf.Variable([theta0], name = 'theta', dtype=tf.float64)
        # tf.reshape(self.theta, [1, 1])

        all_train_data = self.util_train.get_all_data_origin()
        self.init_ks_value = all_train_data[3] * all_train_data[2] / theta0 + (1 - all_train_data[3]) * all_train_data[1] / theta0
        self.ks = tf.Variable(self.init_ks_value, name='ks', dtype=tf.float64)
        self.w = tf.Variable(initial_value=tf.truncated_normal(shape=[dimension, 1], dtype=tf.float64), name='w')
        # computation graph phase1
        self.ps = tf.pow(self.z, (self.ks - 1.)) * tf.exp(-self.z / self.theta) \
             / tf.exp(tf.lgamma(self.ks)) / tf.pow(self.theta, self.ks)
        self.cs = tf.igamma(self.ks, self.b / self.theta) / tf.exp(tf.lgamma(self.ks))

        self.loss_win = tf.log(self.ps)
        self.loss_lose = tf.log(1 - self.cs)
        self.loss_phase1 = -tf.reduce_mean(self.y * self.loss_win + (1 - self.y) * self.loss_lose)
        self.optimizer1 = tf.train.GradientDescentOptimizer(self.lr_1)
        self.train_step1 = self.optimizer1.minimize(self.loss_phase1)

        # phase 2
        self.label_phase2 = tf.placeholder(tf.float64)
        self.log_label_phase2 = tf.log(self.label_phase2)
        self.loss_phase2 = tf.reduce_mean(tf.square(tf.sparse_tensor_dense_matmul(self.X, self.w) - self.log_label_phase2)) \
                           + self.l2_loss_weight * tf.nn.l2_loss(self.w)
        self.optimizer2 = tf.train.MomentumOptimizer(self.lr_2, 0.9)
        self.train_step2 = self.optimizer2.minimize(self.loss_phase2)

        # session initialization
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        tf.global_variables_initializer().run(session=self.sess)

    def train_phase1(self, train_round = 50):
        # get all batches data
        x, b, z, y = self.util_train.get_all_data_origin()
        feed_dict = {}
        feed_dict[self.X] = tf.SparseTensorValue(x, [1] * len(x), [b.shape[0], dimension])
        feed_dict[self.b] = b
        feed_dict[self.z] = z
        feed_dict[self.y] = y

        print("begin training phase 1")
        for i in range(train_round):
            self.sess.run(self.train_step1, feed_dict)
            loss = self.sess.run(self.loss_phase1, feed_dict)
            print("train loss of phase-1, iteration-{0} is {1}".format(i, loss))


    def train_phase2(self):
        self.ks_const = self.ks.eval(session=self.sess) #np array
        self.theta_const = self.theta.eval(session=self.sess) #np array

        step = 0
        epoch = 0
        loss_list = []
        batch_loss = []

        print("begin training phase 2")
        while True:
            x_batch, b_batch, z_batch, y_batch, ks_batch = self.util_train.get_batch_data_origin_with_ks(step, self.ks_const)
            feed_dict = {}
            feed_dict[self.X] = tf.SparseTensorValue(x_batch, [1] * len(x_batch), [self.batch_size, dimension])
            feed_dict[self.b] = b_batch
            feed_dict[self.z] = z_batch
            feed_dict[self.y] = y_batch
            feed_dict[self.label_phase2] = self.theta_const * ks_batch

            self.sess.run(self.train_step2, feed_dict)
            batch_loss.append(self.sess.run(self.loss_phase2, feed_dict))
            step += 1

            if step * self.batch_size - epoch * int(0.02 * self.train_data_amt) >= int(0.02 * self.train_data_amt):
                loss = np.mean(batch_loss[step - int(int(0.02 * self.train_data_amt) / self.batch_size) - 1:])
                loss_list.append(loss)
                print("train loss of phase2 epoch-{0} is {1}".format(epoch, loss))
                epoch += 1

            # stop condition
            if epoch * 0.02 * self.train_data_amt <= 5 * self.train_data_amt:
                continue
            if (loss_list[-1] - loss_list[-2] > 0 and loss_list[-2] - loss_list[-3] > 0):
                break
            if epoch * 0.02 * self.train_data_amt >= 20 * self.train_data_amt:
                break

        # draw SGD training process
        x = [i for i in range(len(loss_list))]
        plt.plot(x, loss_list)
        plt.savefig(self.output_dir + 'train_phase2.png')
        plt.gcf().clear()


    def test(self):
        print('Test begin')
        self.pred_mp = tf.exp(tf.sparse_tensor_dense_matmul(self.X, self.w))
        self.MSE = tf.reduce_mean(tf.square(self.z - self.pred_mp))

        x, b, z, y = self.util_test.get_all_data_origin()
        feed_dict = {}

        feed_dict[self.X] = tf.SparseTensorValue(x, [1] * len(x), [self.test_data_amt, dimension])
        feed_dict[self.z] = z
        feed_dict[self.y] = y
        feed_dict[self.b] = b

        # calculate MSE
        mse = self.sess.run(self.MSE, feed_dict)
        print("MSE: {}".format(mse))

        ks = self.pred_mp / self.theta
        ps = tf.pow(self.z, (ks - 1.)) * tf.exp(-self.z / self.theta) / tf.pow(self.theta, ks) / tf.exp(tf.lgamma(ks))
        cs = tf.igamma(ks, self.b / self.theta) / tf.exp(tf.lgamma(ks))
        # calculate AUC and LogLoss
        win_rate = self.sess.run(cs, feed_dict)
        auc = roc_auc_score(y, win_rate)
        print("AUC: {}".format(auc))
        logloss = log_loss(y, win_rate)
        print("Log Loss: {}".format(logloss))

        # calculate ANLP
        logp = -tf.log(ps)
        logp_arr = self.sess.run(logp, feed_dict)
        logp_arr[np.isnan(logp_arr)] = 1e-20 #for overflow values, minor
        logp_arr[logp_arr == 0] = 1e-20

        anlp = np.mean(logp_arr)
        print("ANLP: {}".format(anlp))

        # save result and params
        fin = open(self.output_dir + 'result.txt', 'w')
        fin.writelines(["MSE: {0}   AUC: {1}    Log Loss: {2}   ANLP: {3}\n".format(mse, auc, logloss, anlp)])
        fin.close()

        np.save(self.output_dir + 'w', self.sess.run(self.w))
        np.save(self.output_dir + 'k', self.sess.run(ks, feed_dict))
        np.save(self.output_dir + 'theta', self.sess.run(self.theta))


if __name__ == '__main__':
    if len(sys.argv) == 2:
        os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]
    campaign_list = ['2259']#['3386', '3427', '3476', '1458']#['2997', '2259', '2261', '2821']

    for campaign in campaign_list:
        train_file = '../data/' + campaign + '/train.yzbx.txt'
        test_file = '../data/' + campaign + '/test.yzbx.txt'
        feat_index = '../data/' + campaign + '/featindex.txt'

        # hyper parameters
        lr_1s = [1e-3]
        lr_2s = [1e-3]
        l2_loss_weights = [0.0001]
        batch_sizes = [128]
        dimension = int(open(feat_index).readlines()[-1].split('\t')[1][:-1]) + 1

        params = []

        for lr_1 in lr_1s:
            for lr_2 in lr_2s:
                for l2_loss_weight in l2_loss_weights:
                    for batch_size in batch_sizes:
                        util_train = Util(train_file, feat_index, batch_size, 'train')
                        util_test = Util(test_file, feat_index, batch_size, 'test')
                        params.append([lr_1, lr_2, l2_loss_weight, batch_size, util_train, util_test])

        # search hyper parameters
        random.shuffle(params)
        for para in params:
            model = Model(lr_1=para[0], lr_2=para[1], l2_loss_weight=para[2], batch_size=para[3],
                          dimension=dimension, theta0=para[4].get_max_z(), util_train=para[4], util_test=para[5], campaign=campaign)
            model.train_phase1()
            model.train_phase2()
            try:
                model.test()
            except:
                continue
