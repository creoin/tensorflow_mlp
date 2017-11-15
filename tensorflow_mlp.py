import tensorflow as tf
import data_tools as dt

import os
import csv

# Import Dataset
filepath = 'data/task/task1.csv'
data_manager = dt.TaskData(filepath, (0.8,0.10,0.10))
data_manager.init_dataset()

# Prepare training and validation examples
train_x, train_y = data_manager.prepare_train()
valid_x, valid_y = data_manager.prepare_valid()

# Input placeholders for feeding in the features, x, and ground truth labels, y_
x = tf.placeholder("float", shape=[None, 2])
y_ = tf.placeholder("float", shape=[None, 3])

# Dropout: probability of keeping a given weight where Dropout is applied
keep_prob = tf.placeholder("float")

# First Layer Weights
W1 = tf.Variable(tf.truncated_normal([2,100], stddev=0.1))
b1 = tf.Variable(tf.truncated_normal([100], stddev=0.1))

# Second Layer Weights
W2 = tf.Variable(tf.truncated_normal([100,3], stddev=0.1))
b2 = tf.Variable(tf.truncated_normal([3], stddev=0.1))

# Layer 1, FC Layer with ReLU activation
o1 = tf.nn.relu(tf.matmul(x,W1) + b1)
# Add Dropout to first layer
o1_dropout = tf.nn.dropout(o1, keep_prob)

# Layer 2
o2 = tf.matmul(o1_dropout,W2) + b2

# Predictions
y = tf.nn.softmax(o2)

# cross_entropy = -tf.reduce_sum(y_*tf.log(y))

# Regularisation
reg_lambda = 1e-3
regulariser = tf.nn.l2_loss(W1) + tf.nn.l2_loss(W2)

# Loss
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=o2, labels=y_)) + (reg_lambda * regulariser)

# Optimisation, minimise the above total loss with Gradient Descent, updating network weights (performs backward pass)
learning_rate = 0.01
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

# For Validation
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

# Begin TensorFlow session and initialise all the variables
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)


# Set up a logger for the training data we want to record
quantities = ['Iteration', 'Train_Accuracy', 'Valid_Accuracy', 'Loss']
train_logs = dt.Logger(*quantities)

# Training Loop
# Dropout to apply during training
train_keep_prob = 0.7

for i in range(3001):
    feed_dict = {x: train_x, y_: train_y, keep_prob: train_keep_prob}
    train_accuracy, avg_loss, _ = sess.run([accuracy, loss, train_step], feed_dict=feed_dict)
    if i % 100 == 0:
        valid_accuracy = sess.run(accuracy, feed_dict={x: valid_x, y_: valid_y, keep_prob: 1.0})

        train_logs.log(Iteration=i, Train_Accuracy=train_accuracy, Valid_Accuracy=valid_accuracy, Loss=avg_loss)

        print('Iteration {:10}: loss {:6.3f} train accuracy {:7.3f} valid accuracy {:7.3f}'.format(
               i, avg_loss, train_accuracy, valid_accuracy))

print('\n\nTraining done.\n\n')
# Write out logs to CSV file
train_logs.printlog()
experiment = 'tf_task_training'
train_logs.write_csv(os.path.join('experiment/',experiment+'.csv'))
