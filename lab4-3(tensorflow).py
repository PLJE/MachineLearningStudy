#파일이 매우 큰 경우 numpy를 쓰면 메모리가 부족할 수도 있다. 이런 경우 tensorflow의 queue runner system사용. 여러개의 파일 읽어올 수 있음
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

filename_queue = tf.train.string_input_producer(['lab4-1_data_set'] , shuffle = False , name = 'filename_queue')

reader = tf.TextLineReader()
key, value = reader.read(filename_queue)

#default values, in case of empty columns. also specified the type of the decoded result.
record_defaults = [[0.],[0.],[0.],[0.]]
xy = tf.decode_csv(value, record_defaults=record_defaults)

#collects batches of csv in
train_x_batch , train_y_batch = \
    tf.train.batch([xy[0:-1],xy[-1:]] , batch_size=10)

X=tf.placeholder(tf.float32 , shape=[None,3])
Y=tf.placeholder(tf.float32 , shape=[None,1])

W= tf.Variable(tf.random_normal([3,1]) , name='weight')
b= tf.Variable(tf.random_normal([1]) , name ='bias')

hypothesis = tf.matmul(X,W) + b

cost = tf.reduce_mean(tf.square(hypothesis - Y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

#different part
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord = coord)

for step in range(2001):
    x_batch , y_batch = sess.run([train_x_batch,train_y_batch])
    cost_val , hy_val , _ = sess.run([cost,hypothesis,train] , feed_dict ={X : x_batch , Y:y_batch})
    if step %10==0:
        print(step, "Cost : " , cost_val , "\nPrediction:\n" , hy_val)

#different part
coord.request_stop()
coord.join(threads)