import tensorflow as tf
from tensorflow import keras
import numpy as np #배열 계산 관련
import pandas as pd # 데이터를 불러오거나 내보냄. 데이터 프레임 다룸
import matplotlib.pyplot as plt # 데이터 시각화 lib
mnist = tf.keras.datasets.mnist

#keras는 보다 상위 차원의 딥러닝 개발을 가능하게 해주는 lib
print("mnist download complete")
(x_train, y_train), (x_test, y_test) = mnist.load_data()
#학습용, 테스트용 분리
x_train, x_test = x_train/255.0, x_test/255.0
#0~255 data를 0.0~1.0으로 축소한다 (data 전처리)
print("normalization done")

#linear classifier
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28)), #28 by 28 mnist input flatten
    tf.keras.layers.Dense(10,activation='softmax')
])

model.compile(optimizer='SGD',
             loss='sparse_categorical_crossentropy',
             metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)
model.evaluate(x_test, y_test, verbose=2)

