import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os
import time
from sklearn.metrics import *
from util import *
import sys

class COX:
    def __init__(self, lr, batch_size, dimension, util_train, util_test, campaign, reg_lambda, nn=False):
        # hyperparameters
        self.lr = lr
        self.batch_size = batch_size
        self.util_train = util_train
        self.util_test = util_test
        self.reg_lambda = reg_lambda

        self.train_data_amt = util_train.get_data_amt()
        self.test_data_amt = util_test.get_data_amt()
        
        # output dir
        model_name = "{}_{}_{}".format(self.lr, self.reg_lambda, self.batch_size)
        if nn:
            self.output_dir = "output/coxnn/{}/{}/".format(campaign, model_name)
        else:
            self.output_dir = "output/cox/{}/{}/".format(campaign, model_name)
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # reset graph
        tf.reset_default_graph()

        # placeholders, sorted value
        self.X = tf.sparse_placeholder(tf.float64)
        self.z = tf.placeholder(tf.float64)
        self.b = tf.placeholder(tf.float64)
        self.y = tf.placeholder(tf.float64)

        # computation graph, linear estimator or neural network
        if nn:
            hidden_size = 20
            self.w1 = tf.Variable(initial_value=tf.truncated_normal(shape=[dimension, hidden_size], dtype=tf.float64), name='w1')
            self.w2 = tf.Variable(initial_value=tf.truncated_normal(shape=[hidden_size, 1], dtype=tf.float64), name='w2')
            self.hidden_values = tf.nn.relu(tf.sparse_tensor_dense_matmul(self.X, self.w1))
            self.index = tf.matmul(self.hidden_values, self.w2)
            self.reg = tf.nn.l2_loss(self.w1[1:,]) + tf.nn.l2_loss(self.w2[1:,])
        else:
            self.w = tf.Variable(initial_value=tf.truncated_normal(shape=[dimension, 1], dtype=tf.float64), name='w')
            self.index = tf.sparse_tensor_dense_matmul(self.X, self.w)
            self.reg = tf.reduce_sum(tf.abs(self.w[1:,]))
        
        self.multiple_times = tf.exp(self.index)
        self.loss = -tf.reduce_sum((self.index - tf.log(tf.cumsum(self.multiple_times, reverse=True))) * self.y) + \
                    self.reg
        self.optimizer = tf.train.GradientDescentOptimizer(self.lr)
        self.train_step = self.optimizer.minimize(self.loss)

        # for test h0
        self.base = self.z * self.y + self.b * (1 - self.y)
        self.candidate = (1 / tf.cumsum(tf.exp(self.index), reverse=True)) * self.y

        # session initialization
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        tf.global_variables_initializer().run(session=self.sess)

    def train(self):
        step = 0
        epoch = 0
        loss_list = []
        batch_loss = []

        while True:
            x_batch, b_batch, z_batch, y_batch = self.util_train.get_batch_data_origin_sorted(step)
            feed_dict = {}
            feed_dict[self.X] = tf.SparseTensorValue(x_batch, [1] * len(x_batch), [self.batch_size, dimension])
            feed_dict[self.b] = b_batch
            feed_dict[self.z] = z_batch
            feed_dict[self.y] = y_batch

            self.sess.run(self.train_step, feed_dict)
            batch_loss.append(self.sess.run(self.loss, feed_dict))
            step += 1

            if step * self.batch_size - epoch * int(0.02 * self.train_data_amt) >= int(0.02 * self.train_data_amt):
                loss = np.mean(batch_loss[step - int(int(0.02 * self.train_data_amt) / self.batch_size) - 1:])
                loss_list.append(loss)
                print("train loss of epoch-{0} is {1}".format(epoch, loss))
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
        plt.savefig(self.output_dir + 'train.png')
        plt.gcf().clear()

    def test(self):
        batch_num = int(self.test_data_amt / self.batch_size)
        anlp_batch = []
        auc_batch = []
        logloss_batch = []

        for b in range(batch_num):
            x, b, z, y = self.util_test.get_batch_data_origin(b)
            feed_dict = {}

            feed_dict[self.X] = tf.SparseTensorValue(x, [1] * len(x), [self.batch_size, dimension])
            feed_dict[self.z] = z
            feed_dict[self.y] = y
            feed_dict[self.b] = b

            base = self.sess.run(self.base, feed_dict)
            candidate = self.sess.run(self.candidate, feed_dict)
            multiple_times = self.sess.run(self.multiple_times, feed_dict)

            #get survival rate of b and b+1
            H0_b = np.zeros([self.batch_size, 1])
            H0_z = np.zeros([self.batch_size, 1])
            H0_z1 = np.zeros([self.batch_size, 1])
            for i in range(self.batch_size):
                bid = b[i][0]
                mp = z[i][0]
                H0_b[i][0] = np.sum(candidate[base <= bid])
                H0_z[i][0] = np.sum(candidate[base <= mp])
                H0_z1[i][0] = np.sum(candidate[base <= mp + 1])
            S0_b = np.exp(-H0_b)
            S0_z = np.exp(-H0_z)
            S0_z1 = np.exp(-H0_z1)

            S_b = np.power(S0_b, multiple_times)
            S_z = np.power(S0_z, multiple_times)
            S_z1 = np.power(S0_z1, multiple_times)

            p = S_z - S_z1
            p[p <= 0] = 1e-20
            # print(p[p == 0].size)
            # print(p[p < 0].size)
            anlp = np.average(-np.log(p))

            W_b = 1 - S_b
            auc = roc_auc_score(y, W_b)
            logloss = log_loss(y, W_b)

            anlp_batch.append(anlp)
            auc_batch.append(auc)
            logloss_batch.append(logloss)

        ANLP = np.mean(anlp_batch)
        AUC = np.mean(auc_batch)
        LOGLOSS = np.mean(logloss_batch)

        print("AUC: {}".format(AUC))
        print("Log-Loss: {}".format(LOGLOSS))
        print("ANLP: {}".format(ANLP))

        with open(self.output_dir + 'result.txt', 'w') as f:
            f.writelines(["AUC:{}\tANLP:{}\tLog-Loss:{}".format(AUC, ANLP, LOGLOSS)])


if __name__ == '__main__':
    if len(sys.argv) == 2:
        os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]

    campaign_list = ['2259']#['2997', '2259', '3476', '1458', '3386', '3427', '2261', '2821', '3358']

    for campaign in campaign_list:
        train_file = '../data/' + campaign + '/train.yzbx.txt'
        test_file = '../data/' + campaign + '/test.yzbx.txt'
        feat_index = '../data/' + campaign + '/featindex.txt'

        # hyper parameters
        lrs = [1e-3]
        batch_sizes = [256]
        reg_lambdas = [0.01]
        nns = [False, True]
        dimension = int(open(feat_index).readlines()[-1].split('\t')[1][:-1]) + 1

        params = []

        for lr in lrs:
            for batch_size in batch_sizes:
                for reg_lambda in reg_lambdas:
                    for nn in nns:
                        util_train = Util(train_file, feat_index, batch_size, 'train')
                        util_test = Util(test_file, feat_index, batch_size, 'test')
                        params.append([lr, batch_size, util_train, util_test, reg_lambda, nn])

        # search hyper parameters
        random.shuffle(params)
        for para in params:
            cox = COX(lr=para[0], batch_size=para[1], dimension=dimension, util_train=para[2], util_test=para[3], campaign=campaign, reg_lambda=para[4], nn=para[5])
            cox.train()
            cox.test()