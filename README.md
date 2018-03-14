# 基于循环神经网络(RNN)的评论情感分类

使用循环神经网络，完成对影评的情感（正面、负面）分类。

训练使用的数据集为[https://www.cs.cornell.edu/people/pabo/movie-review-data/](https://www.cs.cornell.edu/people/pabo/movie-review-data/)上的[sentence polarity dataset v1.0](https://www.cs.cornell.edu/people/pabo/movie-review-data/rt-polaritydata.tar.gz)，包含正负面评论各5331条。

由于数据集较小，模型的泛化能力不是很好。

当训练集、开发集、测试集的分布为[0.8,0.1,0.1]，训练2000个batch_size=64的mini_batch时，模型在各数据集上的acc表现大致如下：

- 训练集 0.95

- 开发集 0.79

- 测试集 0.80

-------------------

## 说明

**1.数据预处理**

数据下载下来之后需要进行解压，得到`rt-polarity.neg`和`rt-polarity.pos`文件，这两个文件是`Windows-1252`编码的，先将它转成`unicode`处理起来会更方便。

数据预处理过程包括：

- 转码

- 生成词汇表

- 借助词汇表将影评转化为词向量

- 填充词向量并转化为np数组

- 按比例划分数据集（训练、开发、测试）

- 打乱数据集，写入文件

```cmd
python process_data.py 
```


**2.模型编写**

使用RNN完成分类功能，建模过程大致如下：

- 使用embedding构建词嵌入矩阵

- 使用LSTM作为循环神经网络的基本单元

- 对embedding和LSTM进行随机失活(dropout)

- 建立深度为2的深度循环神经网络

- 对深度循环神经网络的最后的输出做逻辑回归，通过sigmod判定类别


**3.模型训练**

训练：

- 使用移动平均

- 使用学习率指数衰减

```cmd
python train.py
```


**4.模型验证**

`eval.py`中存在如下代码：

```python
data = dataset.Dataset(0)
```

`Dataset`的参数，0代表验证训练集数据，1代表验证开发集数据，2代表验证测试集数据。

```cmd
python eval.py
```

**5.模型配置**

可配置参数集中在`settings`中。