import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Read the CSV file with the mushroom dataset
mushrooms_data = pd.read_csv("mushrooms.csv")

# Splitting into features and labels
x = mushrooms_data.drop('class',axis=1)
y = mushrooms_data['class']

# Encoding from chars to int
Encoder_x = LabelEncoder() 
for col in x.columns:
    x[col] = Encoder_x.fit_transform(x[col])
Encoder_y=LabelEncoder()
y = Encoder_y.fit_transform(y)

# Dummy veriables for 0/1 values
x=pd.get_dummies(x,columns=x.columns,drop_first=True)

# Split into training and test data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

# Feature scaling
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

# Principal component analysis
#pca = PCA(n_components=2)
#x_train = pca.fit_transform(x_train)
#x_test = pca.transform(x_test)

print x_train.shape
# x_train shape: (5686, 2)
# y_train shape: (5686,)
# x_test shape: (2438, 2)
# y_test shape: (2438,)

# Parameters for the model
batch_size = 64
training_iterations = 100000
model_evaluation_frequency = 100
dropout_prob = 0.5
learning_rate = 0.001
layer_2_units = 400
layer_3_units = 800

# Returns a random batch of training data with size batch_size
def get_train_batch():
	indices = np.random.randint(low=0, high=x_train.shape[0], size=[batch_size])
	return x_train[indices], y_train[indices]

# Tensorflow graph
g = tf.Graph()
with g.as_default():

	# Defining placeholders for features x and their labes y
	x = tf.placeholder(dtype=tf.float32, shape=[None, 95])
	y = tf.placeholder(dtype=tf.int64, shape=[None,])

	# Defining a placeholder for the dropout probability
	p_dropout = tf.placeholder(dtype=tf.float32)

	# Defining the layers for the model
	layer_1 = tf.layers.dense(inputs=x, units=95, activation=tf.nn.relu)
	layer_2 = tf.layers.dense(inputs=layer_1, units=layer_2_units, activation=tf.nn.relu)
	layer_3 = tf.layers.dense(inputs=layer_2, units=layer_3_units, activation=tf.nn.relu)

	# Dropout before logits layer
	dropout = tf.nn.dropout(x=layer_3, keep_prob=p_dropout)

	# Logits layer
	logits = tf.layers.dense(inputs=dropout, units=2)

	# Defining the loss function (cross entropy) and the optimizer
	loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y))
	optimizer = tf.train.AdamOptimizer(learning_rate)
	minimize_op = optimizer.minimize(loss, var_list=tf.trainable_variables())

	# Running a session to feed the graph
	sess = tf.Session()
	sess.run(tf.global_variables_initializer())

	# Calculations for the evaluation of the model
	missclassified = (tf.to_float(tf.count_nonzero(y - tf.argmax(tf.nn.softmax(logits), axis=1))) / tf.to_float(tf.shape(y)[0])) * 100
	confusion_matrix = tf.confusion_matrix(y, tf.argmax(tf.nn.softmax(logits), axis=1))
	average_training_loss = 0

	# Training iteration loop
	for i in range(training_iterations):

		# Feed data to the graph in batches of batch_size
		train_features, train_labels = get_train_batch()
		sess.run(minimize_op, feed_dict={x: train_features, y: train_labels, p_dropout: dropout_prob})

		# Evaluate state of the model every model_evaluation_frequency iteration
		if i % model_evaluation_frequency == 0:
			print "Iteration number", i
			print "------------------------------------------------------------------------------------------"

			# Calculating loss and average loss over the training batch
			training_batch_loss = sess.run(loss, feed_dict={x: train_features, y: train_labels, p_dropout: 1.0})
			times_evaluated = i/model_evaluation_frequency + 1
			average_training_loss = (((times_evaluated - 1) * average_training_loss) + training_batch_loss) / times_evaluated
			print "Current training set loss:", training_batch_loss, "and average training set loss:", average_training_loss

			# Printing the confusion matrix for the test data
			print "Confusion matrix:"
			print sess.run(confusion_matrix, feed_dict={x: x_test, y: y_test, p_dropout: 1.0})

			# Printing the test data loss and the missclassification rate
			print "Test set loss:", sess.run(loss, feed_dict={x: x_test, y: y_test, p_dropout: 1.0})
			missclassification_rate = sess.run(missclassified, feed_dict={x: x_test, y: y_test, p_dropout: 1.0})
			print "Missclassification rate:",missclassification_rate , "%"
			print ""