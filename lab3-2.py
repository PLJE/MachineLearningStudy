import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

x_data =[1,2,3]
y_data= [1,2,3]

W=tf.Variable(tf.random_normal([1]) , name ='weight')

X=tf.placeholder(tf.float32)
Y=tf.placeholder(tf.float32)

hypothesis = X*W

cost=tf.reduce_mean(tf.square(hypothesis-Y))

#minimize: gradient descent using derivative : W-= learning_rate * derivative
learning_rate = 0.1 #식에서 alpha
gradient = tf.reduce_mean((W*X-Y)*X)
descent = W-learning_rate*gradient
update = W.assign(descent)

#launch the graph in a session
sess = tf.Session()
#initializes global variables in the graph
sess.run(tf.global_variables_initializer())

for step in range(21):
    sess.run(update, feed_dict={X:x_data,Y:y_data})
    print(step,sess.run(cost,feed_dict={X:x_data, Y:y_data}),sess.run(W))