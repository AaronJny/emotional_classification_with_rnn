# -*- coding: utf-8 -*-
# @Time    : 18-3-14 下午3:33
# @Author  : AaronJny
# @Email   : Aaron__7@163.com
import numpy as np
import settings


class Dataset(object):
    def __init__(self, data_kind=0):
        """
        生成一个数据集对象
        :param data_kind: 决定了使用哪种数据集 0-训练集 1-开发集 2-测试集
        """
        self.data, self.labels = self.read_data(data_kind)
        self.start = 0  # 记录当前batch位置
        self.data_size = len(self.data)  # 样例数

    def read_data(self, data_kind):
        """
        从文件中加载数据
        :param data_kind:数据集种类 0-训练集 1-开发集 2-测试集
        :return:
        """
        # 获取数据集路径
        data_path = [settings.TRAIN_DATA, settings.DEV_DATA, settings.TEST_DATA][data_kind]
        # 加载
        data = np.load(data_path + '_data.npy')
        labels = np.load(data_path + '_labels.npy')
        return data, labels

    def next_batch(self, batch_size):
        """
        获取一个大小为batch_size的batch
        :param batch_size: batch大小
        :return:
        """
        start = self.start
        end = min(start + batch_size, self.data_size)
        self.start = end
        # 当遍历完成后回到起点
        if self.start >= self.data_size:
            self.start = 0
        # 返回一个batch的数据和标签
        return self.data[start:end], self.labels[start:end]


if __name__ == '__main__':
    Dataset()
