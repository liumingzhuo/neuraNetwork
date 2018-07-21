# *_*coding:utf-8 *_*
# 神经网络
import numpy

#numpy.seterr(over='ignore')
import scipy.special
import os
import matplotlib.pyplot


class neuraNetwork:

    # 初始化输入层 隐藏层和输出层 以及学习率
    def __init__(self, inputNodes, hideNodes, outputNodes, learnRate):
        self.inode = inputNodes
        self.hnode = hideNodes
        self.onode = outputNodes
        self.lr = learnRate
        # 生成权重(使用正态分布的方式采样)
        self.wih = numpy.random.normal(0.0, pow(self.hnode, -0.5), (self.hnode, self.inode))
        self.woh = numpy.random.normal(0.0, pow(self.onode, -0.5), (self.onode, self.hnode))
        # 初始化S激活函数
        self.activation_function = lambda x: scipy.special.expit(x)

    # 训练
    def training(self, input_list, targets_list):
        inputs = numpy.array(input_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T

        hident_outputs = numpy.dot(self.wih, inputs)

        final_hident_outputs = self.activation_function(hident_outputs)

        outputs = numpy.dot(self.woh, final_hident_outputs)
        final_outputs = self.activation_function(outputs)

        output_errors = targets - final_outputs
        '''
        errors(hidden)=weights**T(hidden_output) * errors(output)
        '''
        hiddent_errors = numpy.dot(self.woh.T, output_errors)

        # ΛWj,k=α*Ek * sigmoid(Ok) *(1-simoid(Ok)) 点乘 Oj^T
        # 隐藏层与输出层之间的运算
        self.woh += self.lr * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)),
                                        numpy.transpose(hident_outputs))
        # 输入层与隐藏层
        self.wih += self.lr * numpy.dot((hiddent_errors * final_hident_outputs * (1.0 - final_hident_outputs)),
                                        numpy.transpose(inputs))

    # 查询
    def query(self, inputs_list):
        inputs = numpy.array(inputs_list, ndmin=2).T
        # 计算隐藏层节点 的输出
        '''
        隐藏层的输出 等于 输入矩阵点成 权重
        输出层同理
        '''
        hident_outputs = numpy.dot(self.wih, inputs)
        final_hident_outputs = self.activation_function(hident_outputs)

        outputs = numpy.dot(self.woh, final_hident_outputs)
        final_outputs = self.activation_function(outputs)
        return final_outputs

    # 训练数据
    def train_data(self):
        file_path = os.path.join(os.path.abspath('..'), ('mnist_data/mnist_train.csv'))
        data_file = open(file_path, 'r')
        data_list = data_file.readlines()
        data_file.close()

        epochs = 5
        for e in range(epochs):
            for data in data_list:
                all_value = data.split(',')
                inputs = (numpy.asfarray(all_value[1:]) / 255.0 * 0.99) + 0.01
                targets = numpy.zeros(output_node) + 0.01
                targets[int(all_value[0])] = 0.99
                neuraNetwork.training(inputs, targets)

    # 测试数据
    def train_test_data(self):
        test_file_path = os.path.join(os.path.abspath('..'), ('mnist_data/mnist_test.csv'))
        test_data_file = open(test_file_path, 'r')
        test_data_list = test_data_file.readlines()
        test_data_file.close()
        scorecard = []
        for record in test_data_list:
            all_values = record.split(',')
            correct_lable = int(all_values[0])
            test_inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
            test_outputs = neuraNetwork.query(test_inputs)
            lable = numpy.argmax(test_outputs)
            # print('lable', lable)
            # print('correnct', correct_lable)
            if (lable == correct_lable):
                scorecard.append(1)
            else:
                scorecard.append(0)
        return scorecard


if __name__ == '__main__':
    input_nodes = 784
    hiddent_nodes = 200
    output_node = 20
    learning_rate = 0.1
    neuraNetwork = neuraNetwork(input_nodes, hiddent_nodes, output_node, learning_rate)
    neuraNetwork.train_data()
    scorecard = neuraNetwork.train_test_data()
    scorecard_array = numpy.asarray(scorecard)
    print(scorecard_array.sum())
    print(scorecard_array.size)
    print('训练成功率', scorecard_array.sum() / scorecard_array.size)

    # print(scraled_input)

    # image_array = numpy.asfarray(all_values[1:]).reshape((28, 28))
    # matplotlib.pyplot.imshow(image_array, cmap='Greys', interpolation='None')
    # matplotlib.pyplot.show()
