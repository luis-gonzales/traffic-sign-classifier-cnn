import pickle
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
plt.switch_backend('agg')           # for AWS
from sklearn.utils import shuffle
from traffic_net import TrafficNet
from plot_helper import plot_helper


# Set random seed
tf.set_random_seed(0)


# Load pickled data; X.shape = 32x32x3 w/ 34799 in train; 4410, val; 12630, test
training_file   = "traffic_signs_data/train.p"
validation_file = "traffic_signs_data/valid.p"
testing_file    = "traffic_signs_data/test.p"

pickle_in = open('data/train.p', mode='rb')
train = pickle.load(pickle_in)
pickle_in = open('data/valid.p', mode='rb')
val = pickle.load(pickle_in)
pickle_in = open('data/test.p', mode='rb')
test = pickle.load(pickle_in)

X_train, y_train = train['features'], train['labels']
X_val, y_val     = val['features'], val['labels']
X_test, y_test   = test['features'], test['labels']

n_classes = np.max(y_train) + 1
print('Size of training set:\t', np.shape(y_train)[0])
print('Size of validation set:\t', np.shape(y_val)[0])
print('Size of test set:\t', np.shape(y_test)[0])
print('Number of classes:\t', n_classes)


# Batch placeholders
x = tf.placeholder(tf.float32, (None,32,32,3))
y = tf.placeholder(tf.uint8, (None))
one_hot_y = tf.one_hot(y, n_classes)


# Training pipeline
rate = tf.placeholder(tf.float32)
prob_keep = tf.placeholder(tf.float32)
logits = TrafficNet(x, prob_keep=prob_keep)
x_entropies = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits)
loss = tf.reduce_mean(x_entropies)
optimizer = tf.train.AdamOptimizer(learning_rate=rate).minimize(loss)


# Model evaluation
correct_predicts = tf.equal( tf.argmax(logits,1), tf.cast(y, tf.int64) )
accuracy = tf.reduce_mean( tf.cast(correct_predicts, tf.float32) )

def evaluate(X_data, y_data, sess, batch_sz_eval):
  eval_acc = 0
  for j in range(0, len(X_data), batch_sz_eval):
    mini_x, mini_y = X_data[j:j+batch_sz_eval], y_data[j:j+batch_sz_eval]
    mini_acc = sess.run(accuracy, feed_dict={x: mini_x, y: mini_y, prob_keep: 1})
    eval_acc += mini_acc * len(mini_x)
  return eval_acc / len(X_data)


# Train and save network
f = plt.figure(figsize = (2.5,2.5))
ax = f.add_subplot(1,1,1)

saver = tf.train.Saver()
sess = tf.Session()

print('Training...')
train_accs, val_accs, alpha_list = [], [], []     # store stats
sess.run(tf.global_variables_initializer())
alpha, batch_sz, epochs = 0.0005, 64, 150
  
# Perform training routine `epoch` number of times
for curr_epoch in range(epochs):
  X_train, y_train = shuffle(X_train, y_train)
  learn_slow_down = curr_epoch > 80
  if learn_slow_down:
    alpha *= 0.95
  alpha_list.append(alpha)
  for j in range(0, len(X_train), batch_sz): # 0, batch_sz, 2*batch_sz
    batch_x, batch_y = X_train[j:j+batch_sz], y_train[j:j+batch_sz]
    sess.run(optimizer, feed_dict={x: batch_x, y: batch_y, rate: alpha, prob_keep: 0.5})

  print('Epoch', curr_epoch+1, 'complete')
  train_accs.append( evaluate(X_train, y_train, sess, 512) )
  val_accs.append( evaluate(X_val, y_val, sess, 512) )
  print('Evaluation complete')

print('Training complete!')
  
# Plot learning curves  
plot_helper(ax, epochs, train_accs, val_accs, alpha_list, batch_sz)

# Save model
saver.save(sess, './tf_model/model.ckpt')
print('Saved model.')
