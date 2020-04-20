# 4876010998: XU KANGYAN
# CSCI561 HW 3 MLP NN
# 2020.4.22
# python3 NeuralNetwork3.py train_image.csv train_label.csv test_image.csv test_label.csv

import numpy as np
import sys
import math
# import time

# ------------------ Data ------------------ #
# train_image (60000, 784)
# train_label (60000,)
# test_image  (10000, 784)
# ----------------------------------------------- #

# ------------------ Parameter ------------------ #
train_batch_size = 1000
epoch_times = 100
learning_rate = 0.0001
nodes_num_1 = 128
nodes_num_2 = 64
# ----------------------------------------------- #


class Network:
    def __init__(self, num_1, num_2, train, label, test, batch_size):
        self.input = train  # image_num * 784
        self.label = label
        self.test = test
        # self.label_test = label_test
        self.batch_size = batch_size
        self.iteration = int(self.input.shape[0]/train_batch_size)
        self.input_layer_nodes_num = 784
        self.output_layer_nodes_num = 10
        self.hidden_layer_1_nodes_num = num_1
        self.hidden_layer_2_nodes_num = num_2

        # ------------------ Weight & Bias ------------------ #
        # -------------- Xavier initialization -------------- #
        bound_1 = math.sqrt(6.0 / (self.input_layer_nodes_num + self.hidden_layer_1_nodes_num))
        bound_2 = math.sqrt(6.0 / (self.hidden_layer_1_nodes_num + self.hidden_layer_2_nodes_num))
        bound_3 = math.sqrt(6.0 / (self.hidden_layer_2_nodes_num + self.output_layer_nodes_num))
        # 784 * layer_1_nodes_num
        self.W1 = np.random.uniform(-bound_1, bound_1, (self.input_layer_nodes_num, self.hidden_layer_1_nodes_num))
        self.b1 = np.random.randn(1, self.hidden_layer_1_nodes_num)
        # layer_1_nodes_num * layer_2_nodes_num
        self.W2 = np.random.uniform(-bound_2, bound_2, (self.hidden_layer_1_nodes_num, self.hidden_layer_2_nodes_num))
        self.b2 = np.random.randn(1, self.hidden_layer_2_nodes_num)
        # layer_2_nodes_num * 10
        self.W3 = np.random.uniform(-bound_3, bound_3, (self.hidden_layer_2_nodes_num, self.output_layer_nodes_num))
        self.b3 = np.random.randn(1, self.output_layer_nodes_num)

    def forward_pass(self, batch_input):
        self.hidden_layer_1_output = sigmoid(np.dot(batch_input, self.W1) + self.b1)
        self.hidden_layer_2_output = sigmoid(np.dot(self.hidden_layer_1_output, self.W2) + self.b2)
        self.output_layer_output = softmax(np.dot(self.hidden_layer_2_output, self.W3) + self.b3)

    def backward_propagation(self, batch_input, batch_label):
        '''
        z1 = x1.W1 + b1
        a1 = sigmoid(z1)  # batch_size * layer_1_nodes_num
        z2 = a1.W2 + b2
        a2 = sigmoid(z2)  # batch_size * layer_2_nodes_num
        z3 = a2.W3 + b3
        a3 = softmax(z3)  # batch_size * 10
        '''
        # ------------------ label vector to mat ------------------ #
        label_mat = np.zeros((self.batch_size, 10))
        i = range(0, self.batch_size)
        label_mat[i, batch_label[i]] = 1
        # --------------------------------------------------------- #
        # predict minus label | batch_size * 10
        derivative_loss_z3 = self.output_layer_output - label_mat
        # layer_2_nodes_num * 10
        derivative_loss_w3 = np.dot(self.hidden_layer_2_output.T, derivative_loss_z3)

        # batch_size * layer_2_nodes_num
        derivative_loss_a2 = np.dot(derivative_loss_z3, self.W3.T)
        # sigmoid derivative | batch_size * layer_2_nodes_num
        derivative_a2_z2 = self.hidden_layer_2_output * (1 - self.hidden_layer_2_output)
        # batch_size * layer_2_nodes_num
        derivative_loss_z2 = derivative_loss_a2 * derivative_a2_z2
        # layer_1_nodes_num * layer_2_nodes_num
        derivative_loss_w2 = np.dot(self.hidden_layer_1_output.T, derivative_loss_z2)

        # batch_size * layer_1_nodes_num
        derivative_loss_a1 = np.dot(derivative_loss_z2, self.W2.T)
        # sigmoid derivative | batch_size * layer_1_nodes_num
        derivative_a1_z1 = self.hidden_layer_1_output * (1 - self.hidden_layer_1_output)
        # batch_size * layer_1_nodes_num
        derivative_loss_z1 = derivative_loss_a1 * derivative_a1_z1
        # 784 * layer_1_nodes_num
        derivative_loss_w1 = np.dot(batch_input.T, derivative_loss_z1)

        derivative_loss_b3 = np.sum(derivative_loss_z3, axis=0, keepdims=True)
        derivative_loss_b2 = np.sum(derivative_loss_z2, axis=0, keepdims=True)
        derivative_loss_b1 = np.sum(derivative_loss_z1, axis=0, keepdims=True)

        # ------------------ Weight & Bias Update ------------------ #
        self.W3 -= learning_rate * derivative_loss_w3
        self.W2 -= learning_rate * derivative_loss_w2
        self.W1 -= learning_rate * derivative_loss_w1
        self.b3 -= learning_rate * derivative_loss_b3
        self.b2 -= learning_rate * derivative_loss_b2
        self.b1 -= learning_rate * derivative_loss_b1

    def train_by_batch(self):
        for i in range(self.iteration):
            input_batch = self.input[i*self.batch_size:(i+1)*self.batch_size, :]
            label_batch = self.label[i*self.batch_size:(i+1)*self.batch_size]
            self.forward_pass(input_batch)
            self.backward_propagation(input_batch, label_batch)

    def predict_test(self):
        test_hidden_layer_1_output = sigmoid(np.dot(self.test, self.W1) + self.b1)
        test_hidden_layer_2_output = sigmoid(np.dot(test_hidden_layer_1_output, self.W2) + self.b2)
        test_output_layer_output = softmax(np.dot(test_hidden_layer_2_output, self.W3) + self.b3)
        '''
        score = 0
        predict = np.argmax(test_output_layer_output, axis=1)
        for i in range(10000):
            if predict[i] == self.label_test[i]:
                score += 1
        test_accuracy = score/10000*100
        print('Accuracy on 10000 test images: %d %%' % test_accuracy)
        '''
        predict = np.argmax(test_output_layer_output, axis=1)
        return predict.T


def load_data():
    program_name = sys.argv[0]
    if len(sys.argv) > 1:
        train_im = np.loadtxt(sys.argv[1], dtype=int, delimiter=',')
        train_lb = np.loadtxt(sys.argv[2], dtype=int, delimiter=',')
        test_im = np.loadtxt(sys.argv[3], dtype=int, delimiter=',')
        # test_lb = np.loadtxt(sys.argv[4], dtype=int, delimiter=',')
        return train_im, train_lb, test_im


def sigmoid(mat):
    return 1.0/(1+np.exp(-mat))


def softmax(mat):
    # avoid nan
    temp = np.exp(mat - np.max(mat, axis=1, keepdims=True))
    return temp/np.sum(temp, axis=1, keepdims=True)

'''
def cross_entropy_loss_function(mat, label):
    image_count = mat.shape[0]
    probability = softmax(mat)
    i = range(0, image_count)
    loss = -np.log(probability[i, label[i]])
    return np.sum(loss)/image_count
'''

if __name__ == '__main__':
    train_image, train_label, test_image = load_data()

    mlp = Network(nodes_num_1, nodes_num_2, train_image, train_label, test_image, train_batch_size)
    for epoch in range(epoch_times):
        # ------------------ Train ------------------ #
        # start_time = time.time()
        mlp.train_by_batch()
        # print('Epoch %d - %2f sec' % (epoch, time.time() - start_time))

    # ------------------ output ------------------ #
    output = mlp.predict_test()
    np.savetxt('test_predictions.csv', output, delimiter=',', fmt='%d')

    # print("Training Finished")
