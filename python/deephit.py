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

HIDDEN_SIZE1 = 500
OUT_SIZE1 = 500
HIDDEN_SIZE2 = 300
OUT_SIZE2 = 300

class DeepHit:
    def __init__(self, lr, batch_size, dimension, util_train, util_test, campaign, reg_lambda, sigma):
        # hyperparameters
        self.lr = lr
        self.batch_size = batch_size
        self.util_train = util_train
        self.util_test = util_test
        self.reg_lambda = reg_lambda
        self.sigma = sigma
        self.emb_size = 20

        self.train_data_amt = util_train.get_data_amt()
        self.test_data_amt = util_test.get_data_amt()

        # output dir
        model_name = "{}_{}_{}_{}".format(self.lr, self.reg_lambda, self.batch_size, self.sigma)
        self.output_dir = "output/deephit/{}/{}/".format(campaign, model_name)
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        # reset graph
        tf.reset_default_graph()

        # field params
        self.field_sizes = self.util_train.feat_sizes
        self.field_num = len(self.field_sizes)

        # placeholders
        self.X = [tf.sparse_placeholder(tf.float64) for i in range(0, self.field_num)]
        self.z = tf.placeholder(tf.float64)
        self.b = tf.placeholder(tf.float64)
        self.y = tf.placeholder(tf.float64)

        # embedding layer
        self.var_map = {}
        # for truncated
        self.var_map['embed_0'] = tf.Variable(
                tf.truncated_normal([self.field_sizes[0], 1], dtype=tf.float64))
        for i in range(1, self.field_num):
            self.var_map['embed_%d' % i] = tf.Variable(
                tf.truncated_normal([self.field_sizes[i], self.emb_size], dtype=tf.float64))
        
        # after embedding
        w0 = [self.var_map['embed_%d' % i] for i in range(self.field_num)]
        self.dense_input = tf.concat([tf.sparse_tensor_dense_matmul(self.X[i], w0[i]) for i in range(self.field_num)], 1)

        # shared network
        self.hidden1 = tf.Variable(initial_value=tf.truncated_normal(shape=[(self.field_num - 1) * self.emb_size + 1, HIDDEN_SIZE1], dtype=tf.float64), name='h1')
        self.out1 = tf.Variable(initial_value=tf.truncated_normal(shape=[HIDDEN_SIZE1, OUT_SIZE1], dtype=tf.float64), name='o1')
        self.hidden2 = tf.Variable(initial_value=tf.truncated_normal(shape=[OUT_SIZE1, HIDDEN_SIZE2], dtype=tf.float64), name='h2')
        self.out2 = tf.Variable(initial_value=tf.truncated_normal(shape=[HIDDEN_SIZE2, OUT_SIZE2], dtype=tf.float64), name='o2')

        # cause-specific network
        self.hidden1_val = tf.nn.relu(tf.matmul(self.dense_input, self.hidden1))
        self.out1_val = tf.sigmoid(tf.matmul(self.hidden1_val, self.out1))
        self.hidden2_val = tf.nn.relu(tf.matmul(self.out1_val, self.hidden2))
        self.out2_val = tf.sigmoid(tf.matmul(self.hidden2_val, self.out2))

        # p_z and w_b
        self.p = tf.nn.softmax(self.out2_val)
        self.w = tf.cumsum(self.p, exclusive=True, axis = 1)

        idx_z = tf.stack([tf.reshape(tf.range(tf.shape(self.z)[0]), (-1,1)), tf.cast(self.z - 1, tf.int32)], axis=-1)
        idx_b = tf.stack([tf.reshape(tf.range(tf.shape(self.b)[0]), (-1,1)), tf.cast(self.b - 1, tf.int32)], axis=-1)

        self.pz = tf.gather_nd(self.p, idx_z)
        self.wb = tf.gather_nd(self.w, idx_b)
        self.wz = tf.gather_nd(self.w, idx_z)

        # loss and train step
        self.loss1 = -tf.reduce_sum(tf.log(self.pz) * self.y)
        self.loss2 = -tf.reduce_sum(tf.log(1 - self.wb) * (1 - self.y))
        self.reg_loss = tf.nn.l2_loss(self.hidden1[1:,]) + tf.nn.l2_loss(self.hidden2[1:,]) + \
                        tf.nn.l2_loss(self.out1[1:,]) + tf.nn.l2_loss(self.out2[1:,])

        # get ranking loss
        self.w_of_pair = tf.transpose(tf.nn.embedding_lookup(tf.transpose(self.w), tf.cast(self.z[:,0] - 1, tf.int32)))
        self.w_of_self = tf.reshape(tf.tile(tf.reshape(self.wz, (self.batch_size, )), [self.batch_size]), (self.batch_size, self.batch_size))
        self.win_label = tf.reshape(tf.tile(tf.reshape(self.y, (self.batch_size, )), [self.batch_size]), (self.batch_size, self.batch_size))
        self.delta = self.w_of_self - self.w_of_pair
        self.candidate = tf.exp(-self.delta / self.sigma)
        self.rank_loss = tf.reduce_sum(tf.matrix_band_part(self.candidate, -1, 0) * self.win_label)

        self.loss = self.loss1 + self.loss2 + self.reg_lambda * self.reg_loss + self.rank_loss

        self.optimizer = tf.train.GradientDescentOptimizer(self.lr)
        self.train_step = self.optimizer.minimize(self.loss)

        # session initialization
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        tf.global_variables_initializer().run(session=self.sess)

    def train(self):
        step = 0
        epoch = 0
        batch_loss = []
        loss_list = []

        while True:
            x_batch_field, b_batch, z_batch, y_batch = self.util_train.get_batch_data_sorted(step)
            feed_dict = {}
            for j in range(len(self.X)):
                feed_dict[self.X[j]] = tf.SparseTensorValue(x_batch_field[j], [1] * len(x_batch_field[j]),
                                                                  [self.batch_size, self.field_sizes[j]])
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
        for i in range(batch_num):
            x_batch_field, b_batch, z_batch, y_batch = self.util_test.get_batch_data_sorted(i)
            feed_dict = {}
            for j in range(len(self.X)):
                feed_dict[self.X[j]] = tf.SparseTensorValue(x_batch_field[j], [1] * len(x_batch_field[j]),
                                                                  [self.batch_size, self.field_sizes[j]])
            feed_dict[self.b] = b_batch
            feed_dict[self.z] = z_batch
            feed_dict[self.y] = y_batch

            pz = self.sess.run(self.pz, feed_dict)
            wb = self.sess.run(self.wb, feed_dict)
            
            pz[pz == 0] = 1e-20
            anlp = np.average(-np.log(pz))
            auc = roc_auc_score(y_batch, wb)
            logloss = log_loss(y_batch, wb)

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
    
    def output_s(self):
        batch_num = int(self.test_data_amt / self.batch_size)
        output = np.ones([self.batch_size, OUT_SIZE2])
        for i in range(batch_num):
            x_batch_field, b_batch, z_batch, y_batch = self.util_test.get_batch_data(i)
            feed_dict = {}
            for j in range(len(self.X)):
                feed_dict[self.X[j]] = tf.SparseTensorValue(x_batch_field[j], [1] * len(x_batch_field[j]),
                                                                  [self.batch_size, self.field_sizes[j]])
            feed_dict[self.b] = b_batch
            feed_dict[self.z] = z_batch
            feed_dict[self.y] = y_batch
            output = np.vstack([output, self.sess.run(self.w, feed_dict)])
        print(output.shape)
        np.savetxt(self.output_dir + 's.txt', 1 - output[self.batch_size:,], delimiter='\t', fmt='%.4f')




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
        batch_sizes = [128]
        reg_lambdas = [1e-1]
        sigmas = [1]
        dimension = int(open(feat_index).readlines()[-1].split('\t')[1][:-1]) + 1

        params = []

        for lr in lrs:
            for batch_size in batch_sizes:
                for reg_lambda in reg_lambdas:
                    for sigma in sigmas:
                        util_train = Util(train_file, feat_index, batch_size, 'train')
                        util_test = Util(test_file, feat_index, batch_size, 'test')
                        params.append([lr, batch_size, util_train, util_test, reg_lambda, sigma])

        # search hyper parameters
        random.shuffle(params)
        for para in params:
            deephit = DeepHit(lr=para[0], batch_size=para[1], dimension=dimension, util_train=para[2], util_test=para[3], campaign=campaign, 
                              reg_lambda=para[4], sigma=para[5])
            deephit.train()
            deephit.test()
            # deephit.output_s()
