import cv2
import sys
import tensorflow as tf
import numpy as np
from traffic_net import TrafficNet


# Load image
file_dir = sys.argv[1]
img = cv2.imread(file_dir)
img = img[:, :, ::-1]             # convert from BGR to RGB
img = np.expand_dims(img, axis=0)


# Declare TrafficNet logits
x = tf.placeholder(tf.float32, (None,32,32,3))
prob_keep = tf.placeholder(tf.float32)
logits = TrafficNet(x, prob_keep=prob_keep)  


# Restore model and obtain CNN results for `img`
sess = tf.Session()
saver = tf.train.Saver()
saver.restore(sess, './tf_model/model.ckpt')

logits_img = sess.run(logits, feed_dict={x: img, prob_keep: 1})
softmax_img = sess.run(tf.nn.softmax(logits_img))[0]
des_sort_idx = np.argsort(-softmax_img)

top_results = 3
print('Class \t Softmax')
for i in range(top_results):
  cur_class = des_sort_idx[i]
  print(cur_class, '\t {:.5f}'.format(softmax_img[cur_class]))