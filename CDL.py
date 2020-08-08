import numpy as np
import pickle
import tensorflow as tf
from time import time
from evaluate import evaluate_model, evaluate_model_recall_precision

class MF():
    def __init__(self, rating_matrix):
        #### 參數設定
        self.num_u = rating_matrix.shape[0]  # 5551
        self.num_v = rating_matrix.shape[1]  # 16980
        self.u_lambda = 100
        self.v_lambda = 0.1
        self.k = 50  # latent維度
        self.a = 1
        self.b = 0.01
        self.R = np.mat(rating_matrix)
        self.C = np.mat(np.ones(self.R.shape)) * self.b
        self.C[np.where(self.R > 0)] = self.a
        self.I_U = np.mat(np.eye(self.k) * self.u_lambda)
        self.I_V = np.mat(np.eye(self.k) * self.v_lambda)
        self.U = np.mat(np.random.normal(0, 1 / self.u_lambda, size=(self.k, self.num_u)))
        self.V = np.mat(np.random.normal(0, 1 / self.v_lambda, size=(self.k, self.num_v)))

    def test(self):
        print(((U_cut * self.R[np.ravel(np.where(self.R[:, j] > 0)[1]), j] + self.v_lambda * self.V_sdae[j])).shape)

    def ALS(self, V_sdae):
        self.V_sdae = np.mat(V_sdae)

        V_sq = self.V * self.V.T * self.b
        for i in range(self.num_u):
            idx_a = np.ravel(np.where(self.R[i, :] > 0)[1])
            V_cut = self.V[:, idx_a]
            self.U[:, i] = np.linalg.pinv(V_sq + V_cut * V_cut.T * (self.a - self.b) + self.I_U) * (
                        V_cut * self.R[i, idx_a].T)  # V_sq+V_cut*V_cut.T*a_m_b = VCV^T

        U_sq = self.U * self.U.T * self.b
        for j in range(self.num_v):
            idx_a = np.ravel(np.where(self.R[:, j] > 0)[1])
            U_cut = self.U[:, idx_a]
            self.V[:, j] = np.linalg.pinv(U_sq + U_cut * U_cut.T * (self.a - self.b) + self.I_V) * (
                        U_cut * self.R[idx_a, j] + self.v_lambda * np.resize(self.V_sdae[j], (self.k, 1)))

        return self.U, self.V

def mask(corruption_level ,size):
    print("#### masking noise ")
    mask = np.random.binomial(1, 1 - corruption_level, [size[0],size[1]])
    return mask

def add_noise(x , corruption_level ):
    x = x * mask(corruption_level , x.shape)
    return x

class CDL():
    def __init__(self, rating_matrix, item_infomation_matrix, topK=10, recallK=300, precisionK=500, use_recall_precision=False):
        # model參數設定
        self.use_recall_precision = use_recall_precision
        self.topK =topK
        self.recallK = recallK
        self.precisionK = precisionK

        self.n_input = item_infomation_matrix.shape[1]
        self.n_hidden1 = 200
        self.n_hidden2 = 50
        self.k = 50

        self.lambda_w = 1
        self.lambda_n = 1
        self.lambda_u = 1
        self.lambda_v = 1

        self.drop_ratio = 0.01
        self.learning_rate = 0.001
        self.epochs = 20
        self.batch_size = 32

        self.num_u = rating_matrix.shape[0]
        self.num_v = rating_matrix.shape[1]

        self.Weights = {
            'w1': tf.Variable(tf.random_normal([self.n_input, self.n_hidden1], mean=0.0, stddev=1 / self.lambda_w)),
            'w2': tf.Variable(tf.random_normal([self.n_hidden1, self.n_hidden2], mean=0.0, stddev=1 / self.lambda_w)),
            'w3': tf.Variable(tf.random_normal([self.n_hidden2, self.n_hidden1], mean=0.0, stddev=1 / self.lambda_w)),
            'w4': tf.Variable(tf.random_normal([self.n_hidden1, self.n_input], mean=0.0, stddev=1 / self.lambda_w))
        }
        self.Biases = {
            'b1': tf.Variable(tf.random_normal([self.n_hidden1], mean=0.0, stddev=1 / self.lambda_w)),
            'b2': tf.Variable(tf.random_normal([self.n_hidden2], mean=0.0, stddev=1 / self.lambda_w)),
            'b3': tf.Variable(tf.random_normal([self.n_hidden1], mean=0.0, stddev=1 / self.lambda_w)),
            'b4': tf.Variable(tf.random_normal([self.n_input], mean=0.0, stddev=1 / self.lambda_w))
        }

        self.item_infomation_matrix = item_infomation_matrix

        self.build_model()

    def encoder(self, x, drop_ratio):
        w1 = self.Weights['w1']
        b1 = self.Biases['b1']
        L1 = tf.nn.sigmoid(tf.matmul(x, w1) + b1)
        L1 = tf.nn.dropout(L1, keep_prob=1 - drop_ratio)

        w2 = self.Weights['w2']
        b2 = self.Biases['b2']
        L2 = tf.nn.sigmoid(tf.matmul(L1, w2) + b2)
        L2 = tf.nn.dropout(L2, keep_prob=1 - drop_ratio)

        return L2

    def decoder(self, x, drop_ratio):
        w3 = self.Weights['w3']
        b3 = self.Biases['b3']
        L3 = tf.nn.sigmoid(tf.matmul(x, w3) + b3)
        L3 = tf.nn.dropout(L3, keep_prob=1 - drop_ratio)

        w4 = self.Weights['w4']
        b4 = self.Biases['b4']
        L4 = tf.nn.sigmoid(tf.matmul(L3, w4) + b4)
        L4 = tf.nn.dropout(L4, keep_prob=1 - drop_ratio)

        return L4

    def build_model(self):
        self.model_X_0 = tf.placeholder(tf.float32, shape=(None, self.n_input))
        self.model_X_c = tf.placeholder(tf.float32, shape=(None, self.n_input))
        self.model_V = tf.placeholder(tf.float32, shape=(None, self.k))
        self.model_drop_ratio = tf.placeholder(tf.float32)

        self.V_sdae = self.encoder(self.model_X_0, self.model_drop_ratio)
        self.y_pred = self.decoder(self.V_sdae, self.model_drop_ratio)

        self.Regularization = tf.reduce_sum(
            [tf.nn.l2_loss(w) + tf.nn.l2_loss(b) for w, b in zip(self.Weights.values(), self.Biases.values())])
        loss_r = 1 / 2 * self.lambda_w * self.Regularization
        loss_a = 1 / 2 * self.lambda_n * tf.reduce_sum(tf.pow(self.model_X_c - self.y_pred, 2))
        loss_v = 1 / 2 * self.lambda_v * tf.reduce_sum(tf.pow(self.model_V - self.V_sdae, 2))
        self.Loss = loss_r + loss_a + loss_v

        self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.Loss)

    def training(self, rating_matrix, test_ratings, test_negatives):
        # np.random.shuffle(self.item_infomation_matrix) #random index of train data
        evaluation_threads = 1
        num_items = rating_matrix.shape[1]

        self.item_infomation_matrix_noise = add_noise(self.item_infomation_matrix, 0.3)
        #self.item_infomation_matrix_noise = add_noise(self.item_infomation_matrix, 0.05)

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        mf = MF(rating_matrix)

        V_sdae = sess.run(self.V_sdae,
                          feed_dict={self.model_X_0: self.item_infomation_matrix_noise, self.model_drop_ratio: self.drop_ratio})

        U, V = mf.ALS(V_sdae)
        U=U.T

        # Init performance
        t1 = time()
        if use_recall_precision:
            (recalls, precisions) = evaluate_model_recall_precision(U @ V, num_items, test_ratings, self.recallK,
                                                                    self.precisionK, evaluation_threads)
            recall, precision = np.array(recalls).mean(), np.array(precisions).mean()
            print('Init: Recall = %.4f, Precision = %.4f\t [%.1f s]' % (recall, precision, time() - t1))
        else:
            (hits, ndcgs) = evaluate_model(U @ V, test_ratings, test_negatives, self.topK, evaluation_threads)
            hr, ndcg = np.array(hits).mean(), np.array(ndcgs).mean()
            print('Init: HR = %.4f, NDCG = %.4f\t [%.1f s]' % (hr, ndcg, time() - t1))

        # Train model
        if use_recall_precision:
            best_recall, best_precision, best_iter = recall, precision, -1
        else:
            best_hr, best_ndcg, best_iter = hr, ndcg, -1

        for epoch in range(self.epochs):
            #print("%d / %d" % (epoch + 1, self.epochs))

            t1 = time()

            V = np.resize(V, (num_items, self.k))
            for i in range(0, self.item_infomation_matrix.shape[0], self.batch_size):
                X_train_batch = self.item_infomation_matrix_noise[i:i + self.batch_size]
                y_train_batch = self.item_infomation_matrix[i:i + self.batch_size]
                V_batch = V[i:i + self.batch_size]
                _, my_loss = sess.run([self.optimizer, self.Loss],
                                      feed_dict={self.model_X_0: X_train_batch, self.model_X_c: y_train_batch,
                                                 self.model_V: V_batch, self.model_drop_ratio: self.drop_ratio})

            V_sdae = sess.run(self.V_sdae,
                              feed_dict={self.model_X_0: self.item_infomation_matrix_noise, self.model_drop_ratio: self.drop_ratio})

            U, V = mf.ALS(V_sdae)
            U = U.T

            t2 = time()

            # Evaluation
            if use_recall_precision:
                (recalls, precisions) = evaluate_model_recall_precision(U @ V, num_items, test_ratings,
                                                                        self.recallK,
                                                                        self.precisionK, evaluation_threads)
                recall, precision = np.array(recalls).mean(), np.array(precisions).mean()
                print('Iteration %d [%.1f s]: Recall = %.4f, Precision = %.4f, loss = %.4f [%.1f s]'
                      % (epoch, t2 - t1, recall, precision, my_loss, time() - t2))
                if recall > best_recall:
                    best_recall, best_precision, best_iter = recall, precision, epoch
            else:
                (hits, ndcgs) = evaluate_model(U @ V, test_ratings, test_negatives, self.topK, evaluation_threads)
                hr, ndcg = np.array(hits).mean(), np.array(ndcgs).mean()
                print('Iteration %d [%.1f s]: HR = %.4f, NDCG = %.4f, loss = %.4f [%.1f s]'
                      % (epoch, t2 - t1, hr, ndcg, my_loss, time() - t2))
                if hr > best_hr:
                    best_hr, best_ndcg, best_iter = hr, ndcg, epoch

        if use_recall_precision:
            print(
                "End. Best Iteration %d:  Recall = %.4f, Precision = %.4f. " % (best_iter, best_recall, best_precision))
        else:
            print("End. Best Iteration %d:  HR = %.4f, NDCG = %.4f. " % (best_iter, best_hr, best_ndcg))

        return U, V

#init random seed
np.random.seed(5)
print(tf.__version__)

import argparse
parser = argparse.ArgumentParser(description="Run CDL.")
parser.add_argument('--db', nargs='?', default='ml-1m',
                    help='Choose a dataset.')
parser.add_argument('--recall_precision', action='store_true', default=False,
                    help='use recall_precision eval.')
args = parser.parse_args()

use_recall_precision = args.recall_precision
db = args.db

if use_recall_precision: p=-10
else: p=1


print("#### load matrix from pickle")
with open(r'{db}/item_infomation_matrix.pickle'.format(db=db), 'rb') as handle:
    item_infomation_matrix = pickle.load(handle)

with open(r'{db}/rating_matrix_p{p}.pickle'.format(db=db,p=p), 'rb') as handle2:
    rating_matrix = pickle.load(handle2)

print("#### build model")
print()
print("#### matrix factorization model")

cdl = CDL(rating_matrix , item_infomation_matrix, use_recall_precision=use_recall_precision)
cdl.build_model()
from Dataset import Dataset
if use_recall_precision: dataset = Dataset("{db}/{db}.precision-recall".format(db=db), use_recall_precision)
else: dataset = Dataset("{db}/{db}.hr-ndcg".format(db=db), use_recall_precision)
U, V = cdl.training(rating_matrix,dataset.testRatings, dataset.testNegatives)
'''
print(rating_matrix.shape)
print(U.shape)
print(V.shape)

np.save("ml-1m/U",U)
np.save("ml-1m/V",V)
np.save("ml-1m/R",rating_matrix)
np.save("ml-1m/R_",U@V)
'''


