import tensorflow as tf  
from numpy import *

def weight_variable(shape):  
    return tf.Variable(initial_value=tf.truncated_normal(shape=shape, stddev=0.1), name='weight')  

def conv_2d(x, w):  
    return tf.nn.conv2d(input=x, filter=w, strides=[1, 1, 1, 1], padding="VALID")  

def evaluate(y, y_):  
    y = tf.arg_max(input=y, dimension=1)  
    y_ = tf.arg_max(input=y_, dimension=1)  
    return tf.reduce_mean(input_tensor=tf.cast(tf.equal(y, y_), tf.float32)) 

def bias_variable(shape):  
    return tf.Variable(initial_value=tf.constant(value=0.1, shape=shape), name='bias')  

test_=open('test128.txt')
fx=open('xX600.txt')
fy=open('yX600.txt')

testr=list(map(int,test_.read().split()))
fxr=list(map(int,fx.read().split()))
fyr=list(map(int,fy.read().split()))


test=array(testr).reshape((1,128,128,1))
trainx=array(fxr).reshape((600,128,128,1))
trainy=array(fyr).reshape((600,128,128,1))
trainxt=trainx.transpose((0,2,1,3))
trainyt=trainy.transpose((0,2,1,3))
#trainx=dstack((trainx,circle))
trainx=vstack((trainx,trainxt))
trainy=vstack((trainy,trainyt))





tfX = tf.placeholder(tf.float32, [1, 128,128,1])
w_convX1 = weight_variable(shape=[12, 12, 1, 3])  
b_convX1 = bias_variable(shape=[3])  
convX1_out = tf.nn.relu(conv_2d(tfX,w_convX1)+b_convX1)

w_convX2 = weight_variable(shape=[12, 12, 3, 9])  
b_convX2 = bias_variable(shape=[9])  
convX2_out = tf.nn.relu(conv_2d(convX1_out,w_convX2)+b_convX2)

w_convX3 = weight_variable(shape=[10, 10, 9, 27])  
b_convX3 = bias_variable(shape=[27])  
convX3_out = tf.nn.relu(conv_2d(convX2_out,w_convX3)+b_convX3)

w_convX4 = weight_variable(shape=[10, 10, 27, 54])  
b_convX4 = bias_variable(shape=[54])  
convX4_out = tf.nn.relu(conv_2d(convX3_out,w_convX4)+b_convX4)

w_convX5 = weight_variable(shape=[10, 10, 54, 108])  
b_convX5 = bias_variable(shape=[108])  
convX5_out = tf.nn.relu(conv_2d(convX4_out,w_convX5)+b_convX5)

w_convX6 = weight_variable(shape=[10, 10, 108, 216])  
b_convX6 = bias_variable(shape=[216])  
convX6_out = tf.nn.relu(conv_2d(convX5_out,w_convX6)+b_convX6)







w_convT1 = weight_variable(shape=[10, 10, 108, 216])  
b_convT1= bias_variable(shape=[108])
convT1_out = tf.nn.relu(tf.nn.conv2d_transpose(convX6_out,w_convT1,output_shape=[1,79,79,108],strides=[1,1,1,1],padding="VALID")+b_convT1)

w_convT2 = weight_variable(shape=[10, 10, 54, 108])  
b_convT2= bias_variable(shape=[54])
convT2_out = tf.nn.relu(tf.nn.conv2d_transpose(convT1_out,w_convT2,output_shape=[1,88,88,54],strides=[1,1,1,1],padding="VALID")+b_convT2)

w_convT3 = weight_variable(shape=[10, 10, 18, 54])  
b_convT3= bias_variable(shape=[18])
convT3_out = tf.nn.relu(tf.nn.conv2d_transpose(convT2_out,w_convT3,output_shape=[1,97,97,18],strides=[1,1,1,1],padding="VALID")+b_convT3)

w_convT4 = weight_variable(shape=[10, 10, 9, 18])  
b_convT4= bias_variable(shape=[9])
convT4_out = tf.nn.relu(tf.nn.conv2d_transpose(convT3_out,w_convT4,output_shape=[1,106,106,9],strides=[1,1,1,1],padding="VALID")+b_convT4)

w_convT5 = weight_variable(shape=[12, 12, 3, 9])  
b_convT5= bias_variable(shape=[3])
convT5_out = tf.nn.relu(tf.nn.conv2d_transpose(convT4_out,w_convT5,output_shape=[1,117,117,3],strides=[1,1,1,1],padding="VALID")+b_convT5)

w_convT6 = weight_variable(shape=[12, 12, 1, 3])  
b_convT6= bias_variable(shape=[1])
y_pred = tf.nn.relu(tf.nn.conv2d_transpose(convT5_out,w_convT6,output_shape=[1,128,128,1],strides=[1,1,1,1],padding="VALID")+b_convT6)
#print('*************************')
#print(y_pred)


tfy = tf.placeholder(tf.float32, [1, 128,128,1])
Loss = tf.nn.l2_loss(y_pred - tfy)
Step_train = tf.train.AdamOptimizer(learning_rate=0.0003).minimize(loss=Loss)  

initialized_variables = tf.initialize_all_variables()  
sess = tf.Session()
sess.run(fetches=initialized_variables) 
saver=tf.train.Saver()
saver.restore(sess,"/20170501/1/X30.ckpt")
for times in range(5):
    for iter in range(1):
        #batch = dataset.train.next_batch(batch_size=batch_size)
        #sess.run(fetches=Step_train, feed_dict={x:batch[0], y:batch[1], dropout_prob:0.5})  
        #Accuracy = sess.run(fetches=accuracy, feed_dict={x:batch[0], y:batch[1], dropout_prob:1})  
        #print('Iter num %d ,the train accuracy is %.3f' % (iter+1, Accuracy))  
        a=0
        for i in range(1200):
            sess.run(fetches=Step_train, feed_dict={tfX:trainx[i,0:128,0:128,0:1].reshape((1,128,128,1)), tfy:trainy[i,0:128,0:128,0:1].reshape((1,128,128,1))})
			
        a=sess.run(fetches=Loss, feed_dict={tfX:trainx[i,0:128,0:128,0:1].reshape((1,128,128,1)), tfy:trainy[i,0:128,0:128,0:1].reshape((1,128,128,1))})
        print(iter+1, a)
		
    save_path=saver.save(sess,"/20170501/1/X"+str(times+31)+".ckpt")


a=sess.run(fetches=y_pred, feed_dict={tfX:trainx[0,0:128,0:128,0:1].reshape((1,128,128,1)), tfy:trainy[0,0:128,0:128,0:1].reshape((1,128,128,1))})
output=array(a).reshape((1,16384))

outputy=[]	  
for i in range(16384):
		outputy.append(output[0,i])
		
pre=str(outputy)
pre=pre.replace("[","")
pre=pre.replace("]","")+"\n"

f=open("a.txt","w")
f.write(pre)
f.close()
print(len(outputy))


















