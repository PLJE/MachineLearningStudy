#neural network for XOR (more accurate)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

x_data= [[0,0],[0,1],[1,0],[1,1]]
y_data = [[0],[1],[1],[0]]

X = tf.placeholder(tf.float32)
Y=tf.placeholder(tf.float32)

#layer1
W1 = tf.Variable(tf.random_normal([2,2]) , name = 'weight1')
b1 = tf.Variable(tf.random_normal([2]) , name='bias1')
layer1= tf.sigmoid(tf.matmul(X,W1)+b1)
#layer2
W2=tf.Variable(tf.random_normal([2,1]) , name='weight2')
b2=tf.Variable(tf.random_normal([1]) , name='bias2')
hypothesis = tf.sigmoid(tf.matmul(layer1 , W2) +b2)

cost= -tf.reduce_mean(Y*tf.log(hypothesis)+(1-Y)*tf.log(1-hypothesis))
train= tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

predicted = tf.cast(hypothesis>0.5 , dtype = tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted,Y) , dtype = tf.float32 ))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(10001):
        sess.run(train, feed_dict={X:x_data, Y: y_data})
        if step%100==0:
            print(step, sess.run(cost,feed_dict={X:x_data , Y:y_data}) , sess.run([W1,W2]))

    h,c,a = sess.run([hypothesis,predicted , accuracy] , feed_dict ={X:x_data , Y:y_data})
    print("\nhypothesis : " , h, "Correct : " ,c, "Accuracy : ", a)