import tensorflow as tf
import numpy as np
import os
import time
import sys
from sklearn.metrics import *

def generate_indices(src_list):
    indices = []
    for i in range(len(src_list)):
        line = src_list[i]
        for pos in line:
            indices.append([i, pos])
    return indices

class COXNN:
    def __init__(self, campaign, train_file, test_file, max_dimen, learn_rate, batch_size, hidden_layer_size, threshold, w_k, w_lambda):
        self.train_file = train_file
        self.test_file = test_file
        self.max_dimen = max_dimen
        self.learn_rate = learn_rate
        self.batch_size = batch_size
        self.hidden_layer_size = hidden_layer_size
        self.threshold = threshold
        self.w_k = w_k
        self.w_lambda = w_lambda
        self.campaign = campaign
        self.output_dir = 'output' + os.path.sep + "cox_nn" + os.path.sep + str(campaign) + os.path.sep + str(learn_rate) + '|' + str(hidden_layer_size) + os.path.sep
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        """
        prepare training data
        """
        self.censored_list = []
        self.features_list = []

        fi = open(self.train_file)
        lines = fi.readlines()
        for line in lines:
            line = line[:-1].replace(':1', '')
            items = line.split(' ')
            if int(items[1]) >= int(items[2]):
                self.censored_list.append(1)
            else:
                self.censored_list.append(0)
            self.features_list.append([int(x) for x in items[3:]])

        self.features_list_batches = []
        self.censored_list_batches = []
        batch_num = int(len(self.features_list) / self.batch_size)
        for i in range(batch_num):
            start = i * self.batch_size
            end = start + self.batch_size - 1
            indices = generate_indices(self.features_list[start:end + 1])
            self.features_list_batches.append(indices)
            self.censored_list_batches.append(self.censored_list[start:end + 1])
        print("input training dataset complete")

        """
        prepare test data for overfitting analysis
        """
        self.test_censored_list = []
        self.test_features_list = []

        fi = open(self.test_file)
        lines = fi.readlines()
        for line in lines:
            line = line[:-1].replace(':1', '')
            items = line.split(' ')
            if int(items[1]) >= int(items[2]):
                self.test_censored_list.append(1)
            else:
                self.test_censored_list.append(0)
            self.test_features_list.append([int(x) for x in items[3:]])
        self.test_censored_list = np.array(self.test_censored_list)

    # NOTE:no activation function, has regularization
    # NOTE:RK's email, tf's sparse placeholder and dense matmul
    def train(self):
        print("begin nn")
        x = tf.sparse_placeholder(tf.float64, shape=[None, self.max_dimen])
        # x = tf.placeholder(tf.float64, shape = [None, MAX_DIMENSION])
        e = tf.placeholder(tf.float64, shape=[None, 1])
        in_size = self.max_dimen
        hidden = tf.Variable(tf.truncated_normal([in_size, self.hidden_layer_size], dtype=tf.float64))
        bias = tf.Variable(tf.zeros([self.hidden_layer_size], dtype=tf.float64))
        # out = tf.matmul(input_vec, hidden) + bias
        out = tf.nn.softplus(tf.sparse_tensor_dense_matmul(x, hidden) + bias)
        in_size = self.hidden_layer_size
        # final hidden layer to single output
        final_weights = tf.Variable(tf.truncated_normal([in_size, 1], dtype=tf.float64))
        final_bias = tf.Variable(tf.zeros([1], dtype=tf.float64))
        hazard = tf.nn.softplus(tf.matmul(out, final_weights) + final_bias)
        # loss = -tf.reduce_mean(hazard / tf.cumsum(hazard, reverse=True))
        loss = -tf.reduce_mean((hazard - tf.log(tf.cumsum(tf.exp(hazard), reverse=True))) * (1 - e))
        # regularization
        hidden_size = tf.cast(tf.size(hidden), dtype=tf.float64)
        final_size = tf.cast(tf.size(final_weights), dtype=tf.float64)
        loss += tf.reduce_mean(tf.square(hidden)) * hidden_size / (hidden_size + final_size)
        loss += tf.reduce_mean(tf.square(final_weights)) * final_size / (hidden_size + final_size)

        optimizer = tf.train.GradientDescentOptimizer(self.learn_rate)
        train_step = optimizer.minimize(loss)

        # training loop, handy generate batch
        print("begin training process")
        sess = tf.Session()
        init = tf.global_variables_initializer()
        sess.run(init)
        # plot learning curve
        learn_cur_y = []
        learn_cur_y_test = []

        # indices of the whole dataset
        start_time = time.time()
        indices_f = generate_indices(self.features_list)
        indices_test = generate_indices(self.test_features_list)
        data_cnt = 0
        while True:
            cnt = 0
            data_cnt += 1
            for i in range(len(self.features_list_batches)):
                cnt += 1
                #t = time.time()
                #print("training step: ", len(learn_cur_y))
                x_batch = tf.SparseTensorValue(self.features_list_batches[i], [1] * len(self.features_list_batches[i]),
                                               [self.batch_size, self.max_dimen])
                e_batch = np.array(self.censored_list_batches[i]).reshape(-1, 1)
                sess.run(train_step, {x: x_batch, e: e_batch})
                #print("step time: ", time.time() - t)

                if cnt * self.batch_size >= len(self.features_list) * 0.05:
                    print(i - cnt)
                    # calculate the curr_loss
                    curr_loss = sess.run(loss, {
                        x: tf.SparseTensorValue(indices_f, [1] * len(indices_f), [len(self.features_list), self.max_dimen]),
                        e: np.array(self.censored_list).reshape(-1, 1)})
                    curr_test_loss = sess.run(loss, {
                        x: tf.SparseTensorValue(indices_test, [1] * len(indices_test), [len(self.test_features_list), self.max_dimen]),
                        e: np.array(self.test_censored_list).reshape(-1, 1)})
                    print("current loss: %s" % (curr_loss))
                    print("current test loss: %s" % (curr_test_loss))
                    learn_cur_y.append(curr_loss)
                    learn_cur_y_test.append(curr_test_loss)
                    cnt = 0
            if data_cnt >= 12:
                break
            if learn_cur_y[-2] - learn_cur_y[-1] <= self.threshold and learn_cur_y[-3] - learn_cur_y[-2]\
                    <= self.threshold and learn_cur_y[-2] - learn_cur_y[-1] > 0 and learn_cur_y[-3] - learn_cur_y[-2] > 0:
                break
            else:
                continue
        # evaluate training accuracy
        curr_loss = sess.run(loss, {
            x: tf.SparseTensorValue(indices_f, [1] * len(indices_f), [len(self.features_list), self.max_dimen]),
            e: np.array(self.censored_list).reshape(-1, 1)})
        print("final loss: %s" % (curr_loss))
        print("nn model training time: ", time.time() - start_time)

        np.save(self.output_dir + 'nnlayer_hidden', sess.run(hidden))
        np.save(self.output_dir + 'bias_hidden', sess.run(bias))
        np.save(self.output_dir + 'nnlayer_final', sess.run(final_weights))
        np.save(self.output_dir + 'bias_final', sess.run(final_bias))

        fy = open(self.output_dir + 'learn_curve_nn_y', 'w')
        fy.writelines([str(len(learn_cur_y)) + '\n'])
        fy.writelines([str(y) + '\n' for y in learn_cur_y])
        fy.close()

        fy = open(self.output_dir + 'learn_curve_nn_y_test', 'w')
        fy.writelines([str(len(learn_cur_y_test)) + '\n'])
        fy.writelines([str(y) + '\n' for y in learn_cur_y_test])
        fy.close()

    def eval(self):
        features_list = []
        price_list = []
        label = []
        bid_list = []

        fi = open(self.test_file)
        lines = fi.readlines()
        for line in lines:
            line = line[:-1].replace(':1', '')
            items = line.split(' ')
            price_list.append(int(items[1]))
            bid_list.append(int(items[2]))
            features_list.append([int(x) for x in items[3:]])
            if (int(items[2]) > int(items[1])):
                label.append(1)
            else:
                label.append(0)
        label = np.array(label).reshape(-1, 1)
        pricevec = np.array(price_list).reshape(-1, 1) / (np.max(price_list))
        featvecs = tf.sparse_placeholder(tf.float64, shape=[None, self.max_dimen])
        bids = np.array(bid_list).reshape(-1, 1) / (np.mean(bid_list))

        hidden = np.load(self.output_dir + 'nnlayer_hidden.npy')
        bias = np.load(self.output_dir + 'bias_hidden.npy')
        output1 = tf.sparse_tensor_dense_matmul(featvecs, hidden) + bias

        nnlayer_final = np.load(self.output_dir + 'nnlayer_final.npy')
        bias_final = np.load(self.output_dir + 'bias_final.npy')

        indices = generate_indices(features_list)
        sess = tf.Session()
        init = tf.global_variables_initializer()
        sess.run(init)
        output = tf.matmul(output1, nnlayer_final) + bias_final
        out = sess.run(output, {featvecs: tf.SparseTensorValue(indices, [1] * len(indices), [len(features_list), self.max_dimen])})
        
        hazards = np.exp(out)
        lambda_z = hazards * (self.w_k / self.w_lambda) * np.power((pricevec / self.w_lambda), self.w_k - 1) * np.exp(-np.power(pricevec / self.w_lambda, self.w_k))
        s_b = np.exp(-hazards * (1 - np.exp(-(np.power(bids / self.w_lambda, self.w_k)))))
        s_z = np.exp(-hazards * (1 - np.exp(-(np.power(pricevec / self.w_lambda, self.w_k)))))
        win_rate = 1 - s_b
        p = lambda_z * s_z
        p[p == 0] = 1e-20
        ANLP = np.mean(-np.log(p))
        auc = roc_auc_score(label, win_rate)
        cross = log_loss(label, win_rate)

        #save
        np.save(self.output_dir + 'win_rate', win_rate)
        np.save(self.output_dir + 'pz',p)

        print('ANLP for cox nn:', ANLP)
        print('AUC for cox nn:', auc)
        print('Log loss for cox nn: ', cross)

        f = open(self.output_dir + 'metrics', 'a')
        f.write(str(self.campaign) + '\t' + str(self.w_k) + '\t' + str(self.w_lambda) + '\t' + str(auc) + '\t' + str(ANLP) + '\n')
        f.close()

def fine_tune():
    campaign_list = ['2259']#['2997', '2821', '3358']#['3386', '3427', '1458']#['2997', '2261', '2259', '2821', '3476', '3427', '3386', '3358', '1458']
    for campaign in campaign_list:
        print(campaign)
        train_file = '../data/' + str(campaign) + '/train.yzbx.sort.txt'
        test_file = '../data/' + str(campaign) + '/test.yzbx.txt'
        featindex = open('../data/' + str(campaign) + '/featindex.txt')
        max_dimen = int(featindex.readlines()[-1].split('\t')[1][:-1]) + 1

        batch_size = 256
        threshold = 0.0001
        # hyper parameters
        learn_rates = [0.03]#[0.003, 0.01, 0.03, 0.1]
        hidden_layer_sizes = [20]
        w_lambdas = [1]#, 1e1, 1e2]
        w_ks = [3e5]#[0.1988e6]

        for learn_rate in learn_rates:
            for hidden_layer_size in hidden_layer_sizes:
                for w_k in w_ks:
                    for w_lambda in w_lambdas:
                        coxnn = COXNN(campaign, train_file, test_file, max_dimen, learn_rate, batch_size, hidden_layer_size, threshold, w_k, w_lambda)
                        coxnn.train()
                        coxnn.eval()
                        print("cox nn model completed")

'''
learn_rate: learning rate
hidden_layer_size: the size of the only hidden layer. (COX-NN only implements one hidden layer)
threshold: to calculate whether training should stop
w_k: weibull distribution's k
w_lambda: weibull distribution's lambda

NOTE:w_k and w_lambda need to fine tune to find good result
'''
#python3 coxnn.py 2259 learn_rate batch_size hidden_layer_size threshold w_k w_lambda
if __name__ == '__main__':
    if len(sys.argv) != 8 and len(sys.argv) != 1:
        print('Usage: python3 coxnn.py campaign(2259) learn_rate batch_size hidden_layer_size threshold w_k w_lambda')
        exit(-1)

    if len(sys.argv) == 1:
        campaign = '2259'
        batch_size = 256
        threshold = 0.0001
        learn_rate = 0.03
        hidden_layer_size = 20
        w_lambda = 1
        w_k = 100
    else:
        campaign = sys.argv[1]
        batch_size = int(sys.argv[2])
        threshold = float(sys.argv[3])
        learn_rate = float(sys.argv[4])
        hidden_layer_size = int(sys.argv[5])
        w_lambda = float(sys.argv[6])
        w_k = float(sys.argv[7])

    train_file = '../data/' + str(campaign) + '/train.yzbx.sort.txt'
    test_file = '../data/' + str(campaign) + '/test.yzbx.txt'
    featindex = open('../data/' + str(campaign) + '/featindex.txt')
    max_dimen = int(featindex.readlines()[-1].split('\t')[1][:-1]) + 1

    coxnn = COXNN(campaign, train_file, test_file, max_dimen, learn_rate, batch_size, hidden_layer_size, threshold, w_k, w_lambda)
    coxnn.train()
    coxnn.eval()
    print("cox nn model completed")



    
