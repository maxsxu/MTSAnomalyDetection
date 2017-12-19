#!/usr/bin/env python
# coding=utf-8

# Created by max on 17-10-8

import sys

import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
# sklean接口的包装器KerasClassifier，作为sklearn的分类器接口
from keras.wrappers.scikit_learn import KerasClassifier
# 穷搜所有特定的参数值选出最好的模型参数
from sklearn.grid_search import GridSearchCV

# 类别的数目
nb_classes = 10
# 输入图像的维度
img_rows, img_cols = 28, 28
# 读取数据
(X_train, y_train), (X_test, y_test) = mnist.load_data()
# 读取的数据不包含通道维，因此shape为(60000,28,28)
# 为了保持和后端tensorflow的数据格式一致，将数据补上通道维
X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
# 新的数据shape为 (60000,28,28,1)， 1代表通道是1，也就是灰阶图片
# 指明输入数据的大小，便于后面搭建网络的第一层传入该参数
input_shape = (img_rows, img_cols, 1)
# 数据类型改为float32，单精度浮点数
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
# 数据归一化（图像数据常用）
X_train /= 255
X_test /= 255
# 将类别标签转换为one-hot编码
y_train = np_utils.to_categorical(y_train, nb_classes)
y_test = np_utils.to_categorical(y_test, nb_classes)  # 定义配置卷积网络模型的函数


def make_model(dense_layer_sizes, nb_filters, nb_conv, nb_pool):
    model = Sequential()
    model.add(Convolution2D(nb_filters, nb_conv, nb_conv,
                            border_mode='valid',
                            input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(Convolution2D(nb_filters, nb_conv, nb_conv))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    for layer_size in dense_layer_sizes:
        model.add(Dense(layer_size))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adadelta',
                  metrics=['accuracy'])
    return model


# 全连接层的备选参数列表
dense_size_candidates = [[32], [64], [32, 32], [64, 64]]
# 实现为Keras准备的sklearn分类器接口，创建一个分类器/评估器对象
# 传入的参数为：
# build_fn: callable function or class instance
# **sk_params: model parameters & fitting parameters
# 具体分析如下：
# 传入的第一个参数(build_fn)为可回调的函数，该函数建立、配置并返回一个Keras model，
# 该model将被用来训练/预测，这里我们传入了刚刚定义好的make_model函数
# 传入的第二个参数(**sk_params)为关键字参数(关键字参数在函数内部自动组装为一个dict),
# 既可以是模型的参数，也可以是训练的参数，合法的模型参数就是build_fn的参数，
# 注意，像所有sklearn中其他的评估器(estimator)一样，build_fn应当为其参数提供默认值，
# 以便我们在建立estimator的时候不用向sk_params传入任何值。
# sk_params也可以接收用来调用fit/predict/predict_proba/score方法的参数，
# 例如'nb_epoch','batch_size'
# fit/predict/predict_proba/score方法的参数将会优先从传入fit/predict/predict_proba/score
# 的字典参数中选择，其次才从传入sk_params的参数中选，最后才选择keras的Sequential模型的默认参数中选择
# 这里我们传入了用于调用fit方法的batch_size参数
my_classifier = KerasClassifier(make_model, batch_size=32)
# 当调用sklearn的grid_search接口时，合法的可调参数就是传给sk_params的参数，包括训练参数
# 换句话说，就是可以用grid_search来选择最佳的batch_size/nb_epoch，或者其他的一些模型参数

# GridSearchCV类，穷搜(Exhaustive search)评估器中所有特定的参数，
# 其重要的两类方法为fit和predict
# 传入参数为评估器对象my_classifier，由每一个grid point实例化一个estimator
# 参数网格param_grid，类型为dict，需要尝试的参数名称以及对应的数值
# 评估方式scoring，这里采用对数损失来评估
validator = GridSearchCV(my_classifier,
                         param_grid={'dense_layer_sizes': dense_size_candidates,
                                     'nb_epoch': [3, 6],
                                     'nb_filters': [8],
                                     'nb_conv': [3],
                                     'nb_pool': [2]},
                         scoring='log_loss')
# 根据各个参数值的不同组合在(X_train, y_train)上训练模型
validator.fit(X_train, y_train)
# 打印出训练过程中找到的最佳参数
print('Yhe parameters of the best model are: ')
print(validator.best_params_)

# validator.best_estimator_返回sklearn-warpped版本的最佳模型
# validator.best_estimator_.model返回未包装的最佳模型
best_model = validator.best_estimator_.model
# 度量值的名称
metric_names = best_model.metrics_names
# metric_names = ['loss', 'acc']
# 度量值的数值
metric_values = best_model.evaluate(X_test, y_test)
# metric_values = [0.0550, 0.9826]
print()
for metric, value in zip(metric_names, metric_values):
    print(metric, ': ', value)


def main(args):
    pass


if __name__ == "__main__":
    main(sys.argv)
