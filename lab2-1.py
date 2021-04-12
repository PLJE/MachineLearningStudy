import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

x_train =[1,2,3]
y_train =[1,2,3]

W = tf.Variable(tf.random_normal([1]),name='weight')
b = tf.Variable(tf.random_normal([1]),name='bias')

hypothesis = x_train*W+b
cost=tf.reduce_mean(tf.square(hypothesis - y_train ))

#first way to minimize
#train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)
#second way to minimize
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)

#launch the graph in a session
sess=tf.Session()
#initializes global variables in the graph
sess.run(tf.global_variables_initializer())

#fit the line
for step in range(2001):
    sess.run(train)
    if step%20==0:
        print(step,sess.run(cost),sess.run(W),sess.run(b))
