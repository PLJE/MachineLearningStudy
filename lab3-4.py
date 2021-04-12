import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

W=tf.Variable(5.)
X =[1,2,3]
Y=[1,2,3]

hypothesis = W*X

#직접 계산한 gradient(기울기)
gradient = tf.reduce_mean((W*X - Y)*X)*2

cost=tf.reduce_mean(tf.square(hypothesis-Y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
#tensorflow가 자동으로 계산한 값.
gvs = optimizer.compute_gradients(cost)
apply_gradients = optimizer.apply_gradients(gvs)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(100):
    print(step, sess.run([gradient,W,gvs]))
    sess.run(apply_gradients)

#직접 계산한 것과 아래의 값은 결국 같음