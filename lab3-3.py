import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

X =[1,2,3]
Y=[1,2,3]

#Set wrong model weights 처음 W의 값을 지정한다. 결국 1으로 가게 됨. float형태로 입력해야함
W=tf.Variable(-3.0)
hypothesis = W*X
cost=tf.reduce_mean(tf.square(hypothesis-Y))

#cost를 미분하지 않고도 자동으로 해주는 코드
optimizer= tf.train.GradientDescentOptimizer(learning_rate=0.1)
train = optimizer.minimize(cost)

sess=tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(100):
    print(step,sess.run(W))
    sess.run(train)

#We can see this optimizer runs well