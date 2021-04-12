#full code with placeholders
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

W=tf.Variable(tf.random_normal([1]),name='weight')
b=tf.Variable(tf.random_normal([1]),name='bias')
X=tf.placeholder(tf.float32,shape=[None])
Y=tf.placeholder(tf.float32,shape=[None])

#our hypothesis WX+b
hypothesis = W*X + b
#cost/loss function
cost = tf.reduce_mean(tf.square(hypothesis-Y))
#minimize
optimizer= tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)
#launch the graph in a session
sess=tf.Session()
#initializes global variables in the graph
sess.run(tf.global_variables_initializer())

#fit in the line with new training data
for step in range(2001):
    cost_val,W_val,b_val,_ = sess.run([cost,W,b,train],
                                      feed_dict={ X:[1,2,3,4,5],Y:[2.1,3.1,4.1,5.1,6.1]})
    if step%20==0:
        print(step,cost_val,W_val,b_val)

#testing our model
#print(sess.run(hypothesis, feed_dict={X:[5]}))
print(sess.run(hypothesis,feed_dict={X:[3,4]}))