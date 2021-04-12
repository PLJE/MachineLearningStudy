import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

x_data = [[1, 2, 1, 1],
          [2, 1, 3, 2],
          [3, 1, 3, 4],
          [4, 1, 5, 5],
          [1, 7, 5, 5],
          [1, 2, 5, 6],
          [1, 6, 6, 6],
          [1, 7, 7, 7]]
y_data = [[0, 0, 1],
          [0, 0, 1],
          [0, 0, 1],
          [0, 1, 0],
          [0, 1, 0],
          [0, 1, 0],
          [1, 0, 0],
          [1, 0, 0]]

X = tf.placeholder("float" , [None,4])
Y = tf.placeholder("float", [None,3])
nb_classes = 3

W=tf.Variable(tf.random_normal([4,nb_classes]) , name = 'weight')
b=tf.Variable(tf.random_normal([nb_classes]) , name='bias')

#tf.nn.softmax computes softmax activations
#softmax = exp(Logits) / reduce_sum(exp(Logits) , dim )
hypothesis = tf.nn.softmax(tf.matmul(X,W)+b)

#cross entropy cost/loss
cost= tf.reduce_mean(-tf.reduce_sum(Y*tf.log(hypothesis) , axis =1))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

with tf.Session() as sess :
    sess.run(tf.global_variables_initializer())

    a = sess.run(hypothesis, feed_dict={X:[[1,11,7,9] ,[1,3,4,3] , [1,1,0,1] ]})
    print(a , sess.run(tf.arg_max(a,1)))
