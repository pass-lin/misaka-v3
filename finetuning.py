# -*- coding: utf-8 -*-
"""
Created on Tue May 16 16:52:47 2023

@author: Administrator
"""

import os

os.environ["RECOMPUTE"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


epochs = 20  # 总的epoch
batch_size = 20  # 根据显存设置
config_path = "misaka-v3/misaka_v3.json"  # config路径
dirs = "data/"  # 数据集的路径，会读取路径下的全部csv
model_load_weight = "misaka_v3.h5"  # 待读取权重的路径
max_input_len = 128  # 最大输入长度
max_output_len = 512  # 最大输出长度
model_save_weight = "expand.h5"  # 保存路径


from sklearn.utils import shuffle
import numpy as np
import pandas as pd
from sznlp.my_bert4keras.tokenizers import Tokenizer

tokenizer = Tokenizer("misaka-v3/vocab.txt", do_lower_case=True)
from sznlp.my_bert4keras.backend import tf, keras, K
from sznlp.my_bert4keras.layers import Loss

import time

while tf.test.is_gpu_available() == False:
    print("fingding gpu")
    time.sleep(1)


from sznlp.my_bert4keras.snippets import sequence_padding


print(tf.__version__)

from tqdm import tqdm

from sznlp.my_bert4keras.optimizers import Adam, AdaFactor
from sznlp.my_bert4keras.optimizers import extend_with_weight_decay, Tiger
from sznlp.misaka_models import *
from sznlp.my_bert4keras.optimizers import extend_with_layer_adaptation
from sznlp.my_bert4keras.optimizers import extend_with_piecewise_linear_lr
from sznlp.my_bert4keras.optimizers import extend_with_gradient_accumulation
from sznlp.my_bert4keras.optimizers import extend_with_piecewise_linear_lr
from sznlp.my_bert4keras.models import build_transformer_model


class CrossEntropy(Loss):
    """交叉熵作为loss，并mask掉输入部分@"""

    def compute_loss(self, inputs, mask=None):
        y_true, y_pred = inputs
        y_pred = keras.layers.Activation("linear", dtype="float32")(y_pred)

        y_mask = K.cast(K.not_equal(y_true, 0), y_pred.dtype)
        y_true = y_true[:, 1:]  # 目标token_ids
        y_mask = y_mask[:, 1:]  # segment_ids，刚好指示了要预测的部分
        y_pred = y_pred[:, :-1]  # 预测序列，错开一位
        acc = keras.metrics.sparse_categorical_accuracy(y_true, y_pred)
        acc = K.cast(acc, y_pred.dtype)
        acc = K.sum(acc * y_mask) / K.sum(y_mask)
        self.add_metric(acc, name="accuracy")  # , aggregation='mean')
        loss = K.sparse_categorical_crossentropy(
            y_true,
            y_pred,  # from_logits=True
        )

        loss = K.sum(loss * y_mask) / K.sum(y_mask)
        return loss * 1000


# with strategy.scope():
if True:
    misaka = build_transformer_model(
        config_path=config_path,
        model=Misaka_V3,
        # with_lm='linear',
        return_keras_model=False,
    )

    # model.summary()
    model = misaka.model
    output = CrossEntropy(1)(model.inputs[1:] + model.outputs)
    train_model = keras.models.Model(model.inputs, output)

    encoder = misaka.encoder
    decoder = misaka.decoder


optimizer = AdaFactor(
    learning_rate=2e-5,
)
train_model.compile(optimizer=optimizer)
model.summary()

try:
    train_model.load_weights(model_load_weight, by_name=True)
    print("成功加载权重")
except:
    try:
        misaka.encoder.load_weights(model_load_weight, by_name=True)
        print("成功加载权重编码器")
    except:
        print("模型加载失败")

from tqdm import tqdm


def load_data(filename):
    f = pd.read_csv(filename).values[:, 1:]
    f = shuffle(f)
    encoders, decoders = [], []
    for t in tqdm(f):
        for a in t[:3]:
            if type(a) == float:
                continue
        try:
            inputs, outputs = t[:2]

            encoder = tokenizer.encode(inputs.replace("氼。", "氼"))[0][
                -1 * max_input_len :
            ]
            decoder = tokenizer.encode(
                outputs.replace("氼。", "氼"), maxlen=max_output_len
            )[0]
            if len(decoder) < 128:
                continue
            encoder[0] = tokenizer._token_start_id
            encoders.append(encoder)
            decoders.append(decoder)
        except:
            continue

    return [encoders, decoders]


files = os.listdir(dirs)
x, y = [], []
print("开启数据加载")
for i, filename in enumerate(files):
    print(i, "/", len(files))
    if ".csv" not in filename.lower():
        continue
    x_t, y_t = load_data(dirs + filename)
    x.extend(x_t)
    y.extend(y_t)
print("开启数据填充")
x = sequence_padding(x)
y = sequence_padding(y)
print(x.shape, y.shape)

a = []

num = 0


class Save(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        train_model.save_weights(model_save_weight)


evaluator = Save()

train_model.fit(
    [x, y],
    epochs=epochs,
    verbose=1,
    batch_size=batch_size,
    shuffle=True,
    callbacks=[evaluator],
)

train_model.save_weights(model_save_weight)
