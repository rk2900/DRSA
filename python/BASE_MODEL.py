import tensorflow as tf
import numpy as np
import sys
from sklearn.metrics import roc_auc_score
import random
from tensorflow.python.ops import tensor_array_ops, control_flow_ops
import os
import time
import datetime
import signal
import math
TRAING_TIME = 15
SHUFFLE = True
LOAD_LITTLE_DATA = False

class SparseData():

    def shuffle(self):
        if SHUFFLE:
            np.random.shuffle(self.index)
        return self.data[self.index], self.seqlen[self.index], self.labels[self.index], self.market_price[self.index]

    def __init__(self, INPUT_FILE, win, all, discount):
        self.data = []
        self.labels = []
        self.seqlen = []
        self.market_price = []
        fi = open(INPUT_FILE, 'r')
        COUNT = 1
        max_d = -1
        self.finish_epoch = False
        for line in fi:
            if COUNT > 10000 and LOAD_LITTLE_DATA:
                break
            COUNT += 1
            s = line.split(' ')
            slen = len(s)
            t_indices = []
            for i in range(3, slen):
                w = s[i].split(':')
                td = int(w[0])
                t_indices.append(td)
                max_d = max(td, max_d)
            market_price = int(s[1])
            bid_price = int(s[2])
            if all:
                if bid_price <= market_price:
                    self.data.append(t_indices)  # data only conclude indices use this to get embedding
                    self.seqlen.append(bid_price / discount)
                    self.market_price.append(market_price / discount)
                    self.labels.append([1., 0.])  # so far we always lose, it means we still survial
                else:
                    self.data.append(t_indices)  # data only conclude indices use this to get embedding
                    self.seqlen.append(bid_price / discount)
                    self.market_price.append(market_price / discount)
                    self.labels.append([0., 1.])  # we win means we dead

            else:
                if bid_price <= market_price:
                    if not win:
                        self.data.append(t_indices)  # data only conclude indices use this to get embedding
                        #bid_price= 3
                        #market_price = 2
                        self.seqlen.append(bid_price / discount)
                        self.market_price.append(market_price / discount)
                        self.labels.append([1., 0.])  # so far we always lose, it means we still survial
                else:
                    if win:
                        #bid_price = 3
                        #market_price = 2
                        self.data.append(t_indices)  # data only conclude indices use this to get embedding
                        self.seqlen.append(bid_price / discount)
                        self.market_price.append(market_price / discount)
                        self.labels.append([0., 1.])  # we win means we dead

        self.max_d = max_d
        fi.close()
        self.size = len(self.data)
        self.data = np.array(self.data)
        self.labels = np.array(self.labels)
        self.seqlen = np.array(self.seqlen)
        self.market_price = np.array(self.market_price)
        print("data size ", self.size, "\n")
        self.index = range(0, self.size)
        self.data, self.seqlen, self.labels, self.market_price = self.shuffle()
        self.batch_id = 0


    def next(self, batch_size):
        if self.batch_id + batch_size > len(self.data):
            self.data, self.seqlen, self.labels, self.market_price = self.shuffle()
            self.batch_id = 0
            self.finish_epoch = True
        batch_data = self.data[self.batch_id:self.batch_id + batch_size]
        batch_labels = self.labels[self.batch_id:self.batch_id + batch_size]
        batch_seqlen = self.seqlen[self.batch_id:self.batch_id + batch_size]
        batch_market_price = self.market_price[self.batch_id:self.batch_id + batch_size]
        self.batch_id = self.batch_id + batch_size
        return np.array(batch_data), np.array(batch_labels), np.array(batch_seqlen), np.array(batch_market_price)

class biSparseData():
    def __init__(self, INPUT_FILE, discount):
        random.seed(time.time())
        self.winData = SparseData(INPUT_FILE, True, False, discount)
        self.loseData = SparseData(INPUT_FILE, False, True, discount)#todo lose data get all data
        self.size = self.winData.size + self.loseData.size
    def next(self, batch):
        #win = int(random.random() * 100) % 11 == 1# todoe 1/10 get windata
	win = int(random.random() * 100) % 11 <= 5
        if win:
            a, b, c, d = self.winData.next(batch)
            return a, b, c, d, True
        else:
            a, b, c, d = self.loseData.next(batch)
            return a, b, c, d, False


class BASE_RNN():

    train_data = None
    def init_matrix(self, shape):
        return tf.random_normal(shape, stddev=0.1)

    def __init__(self,  EMB_DIM = 32,
                        FEATURE_SIZE = 13,
                        BATCH_SIZE = 128,
                        MAX_DEN = 1580000,
                        MAX_SEQ_LEN = 350,
                        TRAING_STEPS = 100000,
                        STATE_SIZE = 64,
                        LR = 0.001,
                        GRAD_CLIP = 5.0,
                        L2_NORM = 0.001,
                        INPUT_FILE = "2997",
                        ALPHA = 1.0,
                        BETA = 0.2,
                        ADD_TIME_FEATURE=False,
                        MIDDLE_FEATURE_SIZE = 30,
                        LOG_FILE_NAME=None,
                        FIND_PARAMETER = False,
                        SAVE_LOG=True,
                        OPEN_TEST=True,
                        ONLY_TRAIN_ANLP=False,
                        LOG_PREFIX="",
                        TEST_FREQUENT=False,
                        ANLP_LR = 0.001,
                        DNN_MODEL = False,
                        QRNN_MODEL = False,
                        GLOAL_STEP = 0,
                        COV_SIZE = 1,
                        DOUBLE_QRNN = False,
                        ANLP_ROUND_ROBIN_RATE = 0.2,
                        DISCOUNT = 1
):
        self.DISCOUNT = DISCOUNT
        self.DOUBLE_QRNN = DOUBLE_QRNN
        self.ANLP_ROUND_ROBIN_RATE = ANLP_ROUND_ROBIN_RATE
        self.QRNN_MODEL = QRNN_MODEL
        self.global_step = GLOAL_STEP
        self.DNN_MODEL = DNN_MODEL
        self.ANLP_LR = ANLP_LR
        self.TEST_FREQUENT = TEST_FREQUENT
        self.ONLY_TRAIN_ANLP = ONLY_TRAIN_ANLP
        self.FIND_PARAMETER = FIND_PARAMETER
        self.add_time_feature = ADD_TIME_FEATURE
        self.MIDDLE_FEATURE_SIZE = MIDDLE_FEATURE_SIZE
        tf.reset_default_graph()
        self.TRAING_STEPS = TRAING_STEPS
        self.BATCH_SIZE = BATCH_SIZE
        self.STATE_SIZE = STATE_SIZE
        self.EMB_DIM = EMB_DIM
        self.FEATURE_SIZE = FEATURE_SIZE
        self.MAX_DEN = MAX_DEN
        self.MAX_SEQ_LEN = MAX_SEQ_LEN / self.DISCOUNT + 10
        self.LR = LR
        self.GRAD_CLIP = GRAD_CLIP
        self.L2_NORM = L2_NORM
        self.ALPHA = ALPHA
        self.BETA = BETA
        self.INPUT_FILE = INPUT_FILE
        self.SAVE_LOG = SAVE_LOG
        self.TRAIN_FILE = "../data/" + INPUT_FILE + "/train.yzbx.txt"
        self.TEST_FILE = "../data/" + INPUT_FILE + "/test.yzbx.txt"
        self.OPEN_TEST = OPEN_TEST
        self.COV_SIZE = COV_SIZE

  

        para = None
        if LOG_FILE_NAME != None:
            para = LOG_FILE_NAME
        else:
            para = LOG_PREFIX + str(self.EMB_DIM) + "_" + \
                str(BATCH_SIZE) + "_" + \
                str(self.STATE_SIZE) + "_" + \
                "{:.6f}".format(self.LR) + "_" + "{:.6f}".format(self.ANLP_LR) + "_" + \
                "{:.6f}".format(self.L2_NORM) + "_" + \
                INPUT_FILE + "_" + \
                "{:.2f}".format(self.ALPHA) + "_" \
                "{:.2f}".format(self.BETA) + "_" + str(ADD_TIME_FEATURE) + \
                   "_" + str(self.QRNN_MODEL) + "_" + str(self.COV_SIZE) + "_" + str(DISCOUNT)
        print para, '\n'
        self.filename = para
        self.train_log_txt_filename = "./" + para + '.train.log.txt'
        if os.path.exists(self.train_log_txt_filename):
            self.exist = True
        else:
            if self.SAVE_LOG:
                self.exist = False
                self.train_log_txt = open(self.train_log_txt_filename, 'w')
                self.train_log_txt.close()

    def get_survival_data(self, model, sess):
        alltestdata = SparseData(self.TEST_FILE, True, True)
        ret = []
        while alltestdata.finish_epoch == False:
            test_batch_x, test_batch_y, test_batch_len, test_batch_market_price = alltestdata.next(self.BATCH_SIZE)
            bid_loss, bid_test_prob, anlp, preds = sess.run(
                [self.cost, self.predict, self.anlp_node, self.preds],
                feed_dict={self.tf_x: test_batch_x,
                           self.tf_y: test_batch_y,
                           self.tf_bid_len: test_batch_len,
                           self.tf_market_price: test_batch_market_price
                           })
            ret.append(preds)
        return ret


    def load_data(self):
        self.train_data = biSparseData(self.TRAIN_FILE, self.DISCOUNT)
        self.test_data_win = SparseData(self.TEST_FILE, True, False, self.DISCOUNT)
        self.test_data_lose = SparseData(self.TEST_FILE, False, False, self.DISCOUNT)

    def is_exist(self):
        if self.SAVE_LOG == False:
            return False
        return self.exist


    def create_graph(self):
        BATCH_SIZE = self.BATCH_SIZE
        self.tf_x = tf.placeholder(tf.int32, [BATCH_SIZE, self.FEATURE_SIZE], name="tf_x")
        self.tf_y = tf.placeholder(tf.float32, [BATCH_SIZE, 2], name="tf_y")
        self.tf_bid_len = tf.placeholder(tf.int32, [BATCH_SIZE], name="tf_len")
        self.tf_market_price = tf.placeholder(tf.int32, [BATCH_SIZE], name="tf_market_price")
        self.tf_control_parameter = tf.placeholder(tf.float32, [2], name="tf_control_parameter")
        alpha = self.tf_control_parameter[0]
        beta = self.tf_control_parameter[1]
        self.tf_rnn_len = tf.maximum(self.tf_bid_len, self.tf_market_price) + 2
        embeddings = tf.Variable(self.init_matrix([self.MAX_DEN, self.EMB_DIM]))
        x_emds = tf.nn.embedding_lookup(embeddings, self.tf_x)
        input = tf.reshape(x_emds, [BATCH_SIZE, self.FEATURE_SIZE * self.EMB_DIM])
        input_x = None
        if self.add_time_feature:
            middle_layer = tf.layers.dense(input, self.MIDDLE_FEATURE_SIZE, tf.nn.relu)  # hidden layer

            def add_time(x):
                y = tf.reshape(tf.tile(x, [self.MAX_SEQ_LEN]), [self.MAX_SEQ_LEN, self.MIDDLE_FEATURE_SIZE])
                t = tf.reshape(tf.range(self.MAX_SEQ_LEN), [self.MAX_SEQ_LEN, 1])
                z = tf.concat([y, tf.cast(t, dtype=tf.float32)], 1)
                return z

            input_x = tf.map_fn(add_time, middle_layer)

        preds = None

        if self.DNN_MODEL:
            outlist = []
            for i in range(0, self.BATCH_SIZE):
                sigleout = tf.layers.dense(input_x[i], 1, tf.nn.sigmoid)
                outlist.append(sigleout)
            preds = tf.reshape(tf.stack(outlist, axis=0), [self.BATCH_SIZE, self.MAX_SEQ_LEN], name="preds")
        else:
            # input_x = tf.reshape(tf.tile(input, [1, self.MAX_SEQ_LEN]), [BATCH_SIZE, self.MAX_SEQ_LEN, self.FEATURE_SIZE * self.EMB_DIM])
            rnn_cell = None
            rnn_cell = tf.contrib.rnn.BasicLSTMCell(num_units=self.STATE_SIZE)


            outputs, (h_c, h_n) = tf.nn.dynamic_rnn(
                rnn_cell,                   # cell you have chosen
                input_x,                    # input
                initial_state=None,         # the initial hidden state
                dtype=tf.float32,           # must given if set initial_state = None
                time_major=False,           # False: (batch, time step, input); True: (time step, batch, input)
                sequence_length=self.tf_rnn_len
            )

            new_output = tf.reshape(outputs, [self.MAX_SEQ_LEN * BATCH_SIZE, self.STATE_SIZE])

            with tf.variable_scope('softmax'):
                W = tf.get_variable('W', [self.STATE_SIZE, 1])
                b = tf.get_variable('b', [1], initializer=tf.constant_initializer(0))

            logits = tf.matmul(new_output, W) + b
            preds = tf.transpose(tf.nn.sigmoid(logits, name="preds"), name="preds")[0]


        self.preds = preds
        survival_rate = preds
        batch_rnn_survival_rate = tf.reshape(survival_rate, [BATCH_SIZE, self.MAX_SEQ_LEN])

        map_parameter = tf.concat([batch_rnn_survival_rate,
                                   tf.cast(tf.reshape(self.tf_bid_len, [BATCH_SIZE, 1]), tf.float32)],
                                  1)
        map_parameter = tf.concat([map_parameter,
                                   tf.cast(tf.reshape(self.tf_market_price, [BATCH_SIZE, 1]), tf.float32)],
                                  1)
        def reduce_mul(x):
            bid_len = tf.cast(x[self.MAX_SEQ_LEN], dtype=tf.int32)
            market_len = tf.cast(x[self.MAX_SEQ_LEN + 1], dtype=tf.int32)
            survival_rate_last_one = tf.reduce_prod(x[0:bid_len])
            anlp_rate_last_one = tf.reduce_prod(x[0:market_len + 1])
            anlp_rate_last_two = tf.reduce_prod(x[0:market_len])
            ret = tf.stack([survival_rate_last_one, anlp_rate_last_one, anlp_rate_last_two])
            return ret

        self.mp_para = map_parameter
        rate_result = tf.map_fn(reduce_mul, elems=map_parameter ,name="rate_result")
        self.rate_result = rate_result
        log_minus = tf.log(tf.add(tf.transpose(rate_result)[2] - tf.transpose(rate_result)[1], 1e-20))#todo debug


        self.anlp_node = -tf.reduce_sum(log_minus) / self.BATCH_SIZE #todo load name
        self.anlp_node = tf.add(self.anlp_node, 0, name="anlp_node")
        self.final_survival_rate = tf.transpose(rate_result)[0]
        final_dead_rate = tf.subtract(tf.constant(1.0, dtype=tf.float32), self.final_survival_rate)

        self.predict = tf.transpose(tf.stack([self.final_survival_rate, final_dead_rate]), name="predict")
        cross_entropy = -tf.reduce_sum(self.tf_y*tf.log(tf.clip_by_value(self.predict,1e-10,1.0)))

        tvars = tf.trainable_variables()
        lossL2 = tf.add_n([ tf.nn.l2_loss(v) for v in tvars ]) * self.L2_NORM
        cost = tf.add(cross_entropy, lossL2, name = "cost")  / self.BATCH_SIZE
        self.cost = tf.add(cost, 0, name="cost")
        optimizer = tf.train.AdamOptimizer(learning_rate=self.LR, beta2=0.99)#.minimize(cost)
        optimizer_anlp = tf.train.AdamOptimizer(learning_rate=self.ANLP_LR, beta2=0.99)#.minimize(cost)

        grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars),
                                          self.GRAD_CLIP,
                                          )
        self.train_op = optimizer.apply_gradients(zip(grads, tvars), name="train_op")
        tf.add_to_collection('train_op', self.train_op)

        anlp_grads, _ = tf.clip_by_global_norm(tf.gradients(self.anlp_node, tvars),
                                          self.GRAD_CLIP,
                                          )
        self.anlp_train_op = optimizer_anlp.apply_gradients(zip(anlp_grads, tvars), name="anlp_train_op")
        tf.add_to_collection('anlp_train_op', self.anlp_train_op)


        self.com_cost = tf.add(alpha * self.cost, beta * self.anlp_node)
        com_grads, _ = tf.clip_by_global_norm(tf.gradients(self.com_cost, tvars),
                                          self.GRAD_CLIP,
                                          )

        self.com_train_op = optimizer.apply_gradients(zip(com_grads, tvars), name="train_op")
        tf.add_to_collection('com_train_op', self.com_train_op)

        correct_pred = tf.equal(tf.argmax(self.predict, 1), tf.argmax(self.tf_y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name="accuracy")


    def train_test(self,sess):
        self.load_data()
        init = tf.global_variables_initializer()
        self.sess = sess
        sess.run(init)
        saver = tf.train.Saver(max_to_keep=100)
        self.saver = saver
        TRAIN_LOG_STEP = int((self.train_data.size * 0.1) / self.BATCH_SIZE)
        train_auc_arr = []
        train_anlp_arr = []
        train_loss_arr = []
        train_auc_label = []
        train_auc_prob = []
        total_train_duration = 0
        total_test_duration = 0
        TEST_COUNT = 0
        max_auc = -1
        min_anlp = 200
        enough_test = 0
        last_loss = [9999.0, 9999.0]
        start_time = time.time()
        for step in range(1, self.TRAING_STEPS + 1):
            self.global_step = step
            batch_x, batch_y, batch_len, batch_market_price, win = self.train_data.next(self.BATCH_SIZE)
            if self.ONLY_TRAIN_ANLP:
                if win: #if win
                    _, train_anlp, train_loss, train_outputs = sess.run([self.com_train_op, self.anlp_node, self.cost, self.predict],
                                                                    feed_dict={self.tf_x: batch_x,
                                                                             self.tf_y: batch_y,
                                                                             self.tf_bid_len: batch_len,
                                                                             self.tf_market_price:batch_market_price,
                                                                             self.tf_control_parameter:[self.ALPHA, self.BETA]
                                                       })
                    train_anlp_arr.append(train_anlp)
                    train_loss_arr.append(train_loss)
                    train_auc_label.append(batch_y.T[0])
                    train_auc_prob.append(np.array(train_outputs).T[0])
                else:
                    train_loss, train_outputs = sess.run([self.cost, self.predict], feed_dict={self.tf_x: batch_x,
                                                       self.tf_y: batch_y,
                                                       self.tf_bid_len: batch_len,
                                                       self.tf_market_price:batch_market_price,
                                                       self.tf_control_parameter:[self.ALPHA, self.BETA]
                                                       })
                    #print train_outputs
                    train_loss_arr.append(train_loss)
                    train_auc_label.append(batch_y.T[0])
                    train_auc_prob.append(np.array(train_outputs).T[0])
            else:
                if win: #if win
                    _, train_anlp, train_loss, train_outputs, preds = sess.run([self.com_train_op, self.anlp_node, self.cost, self.predict, self.preds],
                                                                    feed_dict={self.tf_x: batch_x,
                                                                             self.tf_y: batch_y,
                                                                             self.tf_bid_len: batch_len,
                                                                             self.tf_market_price:batch_market_price,
                                                                             self.tf_control_parameter:[self.ALPHA, self.BETA]
                                                       })
                    train_anlp_arr.append(train_anlp)
                    train_loss_arr.append(train_loss)
                    train_auc_label.append(batch_y.T[0])
                    train_auc_prob.append(np.array(train_outputs).T[0])
                else:
                    _, train_loss, train_outputs = sess.run([self.train_op, self.cost, self.predict], feed_dict={self.tf_x: batch_x,
                                                       self.tf_y: batch_y,
                                                       self.tf_bid_len: batch_len,
                                                       self.tf_market_price:batch_market_price,
                                                       self.tf_control_parameter:[self.ALPHA, self.BETA]
                                                       })
                    #print train_outputs
                    train_loss_arr.append(train_loss)
                    train_auc_label.append(batch_y.T[0])
                    train_auc_prob.append(np.array(train_outputs).T[0])

            if step % 100 == 0:
                mean_anlp = np.array(train_anlp_arr[-99:]).mean()
                mean_loss = np.array(train_loss_arr[-99:]).mean()
                mean_auc = 0.0001
                if not self.ONLY_TRAIN_ANLP:
                    try:
                        mean_auc = roc_auc_score(np.reshape(train_auc_label, [1, -1])[0], np.reshape(train_auc_prob, [1, -1])[0])
                    except Exception:
                        print "AUC ERROE"
                        continue

                log = self.getStatStr("TRAIN", self.global_step, mean_auc, mean_loss, mean_anlp)
                print log
                self.force_write(log)
                train_loss_arr = []
                train_anlp_arr = []
                train_auc_label = []
                train_auc_prob = []
                if self.TEST_FREQUENT:
                    self.run_test(sess)
                    #self.save_model()

            if self.global_step < 300:
                if self.global_step % 100 == 0:
                    self.run_test(sess)
                    #self.save_model()
            elif self.global_step < 2000:
                if self.global_step % 500 == 0:
                    self.run_test(sess)
                    #self.save_model()
            elif self.global_step < 10000:
                if self.global_step % 2000 == 0:
                    self.run_test(sess)
                    #self.save_model()
            elif self.global_step <= 21000:
                if self.global_step % 3000 == 0:
                    self.run_test(sess)
                    #self.save_model()
            else:
                break

    def run_model(self):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            self.train_test(sess)

    def save_model(self):
        print "model name: ", self.filename, " ", self.global_step, "\n"
        self.saver.save(self.sess, "./saved_model/model" + self.filename, global_step=self.global_step)

    def getStatStr(self, category ,step, mean_auc, mean_loss, mean_anlp):
        statistics_log = str(self.INPUT_FILE) + "\t" + category + "\t" + str(step) + "\t" \
                         "{:.6f}".format(mean_loss) + "\t" + \
                         "{:.4f}".format(mean_auc) + "\t" + \
                         "{:.4f}".format(mean_anlp) + "\t" + \
                         "{:.4f}".format(self.ALPHA * mean_loss + self.BETA * mean_anlp) + \
                         str(self.EMB_DIM) + "\t" + str(self.BATCH_SIZE) + "\t" + \
                         str(self.STATE_SIZE) + "\t" + \
                         "{:.6f}".format(self.LR) + "\t" + \
                         "{:.6f}".format(self.ANLP_LR) + "\t" + \
                         "{:.6}".format(self.L2_NORM) + "\t" +\
                         str(self.ALPHA) + '\t' + \
                         str(self.BETA) + "\n"
        return statistics_log

    def load(self, meta, ckpt, step):
        tf.reset_default_graph()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        saver = tf.train.import_meta_graph(meta)
        #self.load_data()
        self.global_step = step
        #with tf.Session(config=config) as sess:
        sess = tf.Session(config=config)
        saver.restore(sess, ckpt)
        graph = tf.get_default_graph()
        self.tf_x = graph.get_tensor_by_name("tf_x:0")
        self.tf_y = graph.get_tensor_by_name("tf_y:0")
        self.tf_bid_len = graph.get_tensor_by_name("tf_len:0")
        self.tf_market_price = graph.get_tensor_by_name("tf_market_price:0")
        self.accuracy = graph.get_tensor_by_name("accuracy:0")
        self.cost = graph.get_tensor_by_name("cost:0")
        self.predict = graph.get_tensor_by_name("predict:0")
        self.anlp_node = graph.get_tensor_by_name("anlp_node:0")
        self.train_op = tf.get_collection('train_op')[0]

        #self.anlp_train_op = graph.get_collection("anlp_train_op")[0]
        #self.train _op = graph.get_tensor_by_name("train_op:0")
        self.preds = graph.get_tensor_by_name("preds:0")
        #self.com_train_op = tf.get_collection("com_train_op")[0]
        #self.tf_control_parameter = graph.get_tensor_by_name("tf_control_parameter:0")
        # self.train_log_txt.write(statistics_log)
        return sess

    def run_test(self, sess):
        auc_arr = []
        loss_arr = []
        anlp_arr = []
        auc_prob = []
        auc_label = []
        #print self.test_data_win.size + self.test_data_lose.size, "total size"
        total_time = 0
        for i in range(0, int(self.test_data_win.size / self.BATCH_SIZE)):
            test_batch_x, test_batch_y, test_batch_len, test_batch_market_price = self.test_data_win.next(
                self.BATCH_SIZE)
            start_time = time.time()
            bid_loss, bid_test_prob, anlp = sess.run(
                [self.cost, self.predict, self.anlp_node],
                feed_dict={self.tf_x: test_batch_x,
                           self.tf_y: test_batch_y,
                           self.tf_bid_len: test_batch_len,
                           self.tf_market_price:test_batch_market_price
                           })
            total_time += time.time() - start_time
            auc_prob.append(np.array(bid_test_prob).T[0])
            auc_label.append(test_batch_y.T[0])
            anlp_arr.append(anlp)
            loss_arr.append(bid_loss)
        mean_loss = np.array(loss_arr).mean()
        mean_anlp = np.array(anlp_arr).mean()
        log = self.getStatStr("TEST_WIN_DATA", self.global_step, 0.000001, mean_loss, mean_anlp)
        print log
        for i in range(0, int(self.test_data_lose.size / self.BATCH_SIZE)):
            test_batch_x, test_batch_y, test_batch_len, test_batch_market_price = self.test_data_lose.next(
                self.BATCH_SIZE)
            bid_loss, bid_test_prob, anlp= sess.run(
                                   [self.cost, self.predict,self.anlp_node],
                                   feed_dict={self.tf_x: test_batch_x,
                                              self.tf_y: test_batch_y,
                                              self.tf_bid_len: test_batch_len,
                                              self.tf_market_price: test_batch_market_price
                                              })
            auc_prob.append(np.array(bid_test_prob).T[0])
            auc_label.append(test_batch_y.T[0])
            anlp_arr.append(anlp)
            loss_arr.append(bid_loss)
        if len(auc_prob) > 0:
            try:
                auc = roc_auc_score(np.reshape(np.array(auc_label), [1, -1])[0],
                                np.reshape(np.array(auc_prob), [1, -1])[0])
            except Exception:
                print "AUC ERROR"
                return

            auc_arr.append(auc)
        mean_loss = np.array(loss_arr).mean()
        mean_auc = np.array(auc_arr).mean()
        mean_anlp = np.array(anlp_arr).mean()
        log = self.getStatStr("TEST", self.global_step, mean_auc, mean_loss, mean_anlp)
        self.force_write(log)
        print log
        return mean_loss, mean_auc, mean_anlp

    def force_write(self, log):
        if not self.SAVE_LOG:
            return
        self.train_log_txt = open(self.train_log_txt_filename, 'a')
        self.train_log_txt.write(log)
        self.train_log_txt.close()