import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import matplotlib.pyplot as plt #shows the graph

X=[1,2,3]
Y=[1,2,3]
W=tf.placeholder(tf.float32)
#our hypothesis for linear model X*W
hypothesis = X*W

#cost/loss function
cost = tf.reduce_mean(tf.square(hypothesis-Y))
#launch the graph in a session
sess = tf.Session()
#initializes global variables in the graph
sess.run(tf.global_variables_initializer())
#variables for plotting cost function
W_val=[] #list format
cost_val=[]

for i in range(-30,50):
    feed_W = i*0.1 #-3~5
    curr_cost,curr_W = sess.run([cost,W] , feed_dict = {W:feed_W})
    W_val.append(curr_W)
    cost_val.append(curr_cost)

#show the cost function
plt.plot(W_val,cost_val)
plt.show()
