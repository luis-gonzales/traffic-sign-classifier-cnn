import tensorflow as tf

# Define model
def TrafficNet(X, prob_keep=1):
  # X is assumed to be 32 x 32 x 3

  K = [16, 16, 16, 1000, 43]

  initializer = tf.contrib.layers.variance_scaling_initializer()
  
  W = [tf.Variable( initializer((3, 3, 1, K[0])) ),
       tf.Variable( initializer((3, 3, K[0], K[1])) ),
       tf.Variable( initializer((3, 3, K[1], K[2])) ),
       tf.Variable( initializer((16*16*K[2], K[3])) ),
       tf.Variable( initializer((K[3], K[4])) )]  
  
  b = [tf.Variable( tf.zeros(K[0]) ),
       tf.Variable( tf.zeros(K[1]) ),
       tf.Variable( tf.zeros(K[2]) ),
       tf.Variable( tf.zeros(K[3]) ),
       tf.Variable( tf.zeros(K[4]) )]

  # Preprocess: convert to grayscale and normalize
  X = tf.image.rgb_to_grayscale(X)
  X = tf.subtract(X, 128)
  X = tf.divide(X, 128)

  padding = 'SAME'
  strides = [1,1,1,1]
  k_pool  = [1,2,2,1]

  # Conv 1: 32 x 32 x 1 -> 32 x 32 x K[0]
  conv_1 = tf.nn.conv2d(X, W[0], strides, padding) + b[0]
  conv_1 = tf.nn.relu(conv_1)

  # Conv 2: 32 x 32 x K[0] -> 32 x 32 x K[1]
  conv_2 = tf.nn.conv2d(conv_1, W[1], strides, padding) + b[1]
  conv_2 = tf.nn.relu(conv_2)

  # Conv 3: 32 x 32 x K[1] -> 32 x 32 x K[2]
  conv_3 = tf.nn.conv2d(conv_2, W[2], strides, padding) + b[2]
  conv_3 = tf.nn.relu(conv_3 + conv_1)
  
  # Pool 1: 32 x 32 x K[2] -> 16 x 16 x K[2]
  pool_1 = tf.nn.max_pool(conv_3, k_pool, strides=[1,2,2,1], padding='VALID')  

  # FC 1: 16 x 16 x K[2] -> 1 x K[3]
  flat = tf.contrib.layers.flatten(pool_1)
  fc_1 = tf.nn.xw_plus_b(flat, W[3], b[3])
  fc_1 = tf.nn.relu(fc_1)
  fc_1 = tf.nn.dropout(fc_1, keep_prob=prob_keep)
      
  # Logits: 1 x K[3] -> 1 x K[4]
  logits = tf.nn.xw_plus_b(fc_1, W[4], b[4])

  return logits
