from __future__ import print_function
import gym
import numpy as np
import tensorflow as tf
import random
import math
import os

# ==================================================

env = gym.make("Pong-v0")
obs = env.reset()
# print(len(obs[0][0]))

batchSize = 16
gamma = 0.99
lr = 0.001
startEpsilon = 1
endEpsilon = 0.1
path = "./dqn"
annelingSteps = 10000
maxNumEpisodes = 10000
preTrainSteps = 5000
size = 80
updateFrequency = 5
maxEpisodeLength = 2000
runningReward = None
hSize = 512
size = 80


def preProcess(obs):
	obs = obs[35:195]
	obs = obs[::2, ::2, 0]
	obs[obs == 144] = 0
	obs[obs == 109] = 0
	obs[obs!=0] = 1
	return np.reshape(obs.astype(np.float).ravel(), size*size)

class obsBuffer():
	def __init__(self, bufferSize = 4*size*size):
		self.buffer = []
		self.bufferSize = bufferSize

	def add(self, obs):
		if len(self.buffer) + len(obs) >= self.bufferSize:
			self.buffer[0:(len(obs) + len(self.buffer)) - self.bufferSize] = []
		self.buffer.extend(obs)
		
class expBuffer():
	def __init__(self, bufferSize = 50000):
		self.buffer = []
		self.bufferSize = bufferSize

	def add(self, experience):
		if len(self.buffer) + len(experience) >= self.bufferSize:
			self.buffer[0:(len(experience) + len(self.buffer)) - self.bufferSize] = []
		self.buffer.extend(experience)
	
	def sample(self, size):
		return np.reshape(np.array(random.sample(self.buffer,size)),[size,5])
		
class qNetwork():
	def __init__(self, hSize):
		self.a = 0
		self.observations = tf.placeholder(tf.float32, shape=[None, 4*size*size])
		self.obsIn = tf.reshape(self.observations, shape=[-1, size, size, 4])
		self.conv1 = tf.contrib.layers.convolution2d(
			inputs=self.obsIn, num_outputs=32, kernel_size=[7,7], stride=[1,1], padding='SAME',
			biases_initializer=None)
		# print(self.conv1.get_shape())
		self.pool1 = tf.nn.relu(tf.contrib.layers.max_pool2d(
			inputs=self.conv1, kernel_size=[4,4], stride=[4,4], padding='VALID'))
		self.conv2 = tf.contrib.layers.convolution2d(
			inputs=self.pool1, num_outputs=64, kernel_size=[5,5], stride=[1,1], padding='SAME',
			biases_initializer=None)
		# print(self.conv2.get_shape())
		self.pool2 = tf.nn.relu(tf.contrib.layers.max_pool2d(
			inputs=self.conv2, kernel_size=[2,2], stride=[2,2], padding='VALID'))
		self.conv3 = tf.contrib.layers.convolution2d(
			inputs=self.pool2, num_outputs=128, kernel_size=[5,5], stride=[1,1], padding='SAME',
			biases_initializer=None)
		# print(self.conv3.get_shape())
		self.pool3 = tf.nn.relu(tf.contrib.layers.max_pool2d(
			inputs=self.conv3, kernel_size=[2,2], stride=[2,2], padding='VALID'))
		# print(self.pool3.get_shape())
		self.conv4 = tf.contrib.layers.convolution2d(
			inputs=self.pool3, num_outputs=hSize, kernel_size=[5,5], stride=[5,5], padding='VALID',
			biases_initializer=None)
		# print(self.conv4.get_shape())
		self.fully_connected = tf.Variable(tf.truncated_normal(shape=[hSize, 2], mean=0.0, stddev=0.02))
		self.hidden = tf.nn.relu(tf.contrib.layers.flatten(self.conv4))
		self.qValues = tf.matmul(self.hidden, self.fully_connected)
		self.pickedAction = tf.argmax(self.qValues, 1)
		self.maxQ = tf.reduce_max(self.qValues, reduction_indices=[1])
		
		self.immediateRewards = tf.placeholder(tf.float32, shape=[None, 1], name="Rewards")
		self.y = tf.add(tf.mul(gamma, self.maxQ), self.immediateRewards)
		self.selectedQ = tf.placeholder(tf.float32, shape=[None, 1], name="Selected_Q")
		self.loss = tf.square(self.y - self.selectedQ)

		self.optimizer = tf.train.AdamOptimizer(lr).minimize(self.loss)



tf.reset_default_graph()
primaryQnet = qNetwork(hSize)
experienceBuffer = expBuffer()
stateBuffer = obsBuffer()
prevStateBuffer = obsBuffer()

epsilon = startEpsilon
stepEpsilon = (startEpsilon - endEpsilon) / annelingSteps

stepsPerEpisode = []
rewardList = []
totalSteps = 0
numEpisode = 0

saver = tf.train.Saver()

if not os.path.exists(path):
    os.makedirs(path)

sess = tf.Session()
sess.run(tf.initialize_all_variables())

for numEpisode in range(maxNumEpisodes):
	epBuffer = expBuffer()
	obs = env.reset()
	obs = preProcess(obs)
	stateBuffer.add(obs)
	epSteps = 0
	rewardEp = 0

	for step in range(maxEpisodeLength):
		totalSteps += 1
		epSteps += 1;
		#if totalSteps > preTrainSteps:
			#print("pretraining over.")
		if len(prevStateBuffer.buffer) < prevStateBuffer.bufferSize:
			# print("here I am.")
			action = np.random.randint(0,2)
		else :
			if np.random.rand(1) < epsilon or totalSteps < preTrainSteps:
				action = np.random.randint(0,2)
			else:
				inputFrames = []
				inputFrames.append(stateBuffer.buffer)
				feed_dict = {primaryQnet.observations: inputFrames}
				action = sess.run(primaryQnet.pickedAction, feed_dict = feed_dict)
		if action == 1:
			act = 2
		else:
			act = 3
		obs, reward, done, info = env.step(act)
		rewardEp += reward;
		if reward != 0: # Pong has either +1 or -1 reward exactly when game ends.
			print ('ep %d: game finished, reward: %f, number of steps: %d' % (numEpisode, reward, epSteps))
			epSteps = 0
			
		
		if done:
			runningReward = rewardEp if runningReward is None else runningReward * 0.99 + rewardEp * 0.01
			print('resetting env. episode reward total was %f. running mean: %f' % (rewardEp, runningReward))
			experienceBuffer.add(epBuffer.buffer)
			print("epsilon: %f", epsilon)
			if numEpisode % 1000 == 0:
				saver.save(sess, path + '/model-' + str(numEpisode) + '.ckpt')
				print("saved model")
			break;
		obs = preProcess(obs)
		stateBuffer.add(obs)
		if len(prevStateBuffer.buffer) == prevStateBuffer.bufferSize: 
			prevFrames = []
			curFrames = []
			prevFrames.append(prevStateBuffer.buffer)
			curFrames.append(stateBuffer.buffer)
			epBuffer.add(np.reshape(np.array(
				[prevFrames, action, reward, curFrames, done]), [1,5]))
			if totalSteps > preTrainSteps:
				if epsilon > endEpsilon:					
					epsilon -= stepEpsilon
			if totalSteps > preTrainSteps and epSteps % updateFrequency == 0:
				# print("training begins")
				batch = experienceBuffer.sample(batchSize)
				feed_dict = {primaryQnet.observations: np.vstack(batch[:, 0])}
				QV = sess.run(primaryQnet.qValues, feed_dict=feed_dict)
				pickedQ = []
				for expe in range(batchSize):
					pickedQ.append(QV[expe][batch[expe][1]])
					# print(batch[expe][1])
				rewardFeed = np.reshape(batch[:, 2], (batchSize, 1))
				pickedQFeed = np.reshape(pickedQ, (batchSize, 1))
				feed_dict = {primaryQnet.observations: np.vstack(batch[:,3]), 
							 primaryQnet.immediateRewards: rewardFeed, primaryQnet.selectedQ: pickedQFeed}
				sess.run(primaryQnet.optimizer, feed_dict=feed_dict)
				# print(sess.run(primaryQnet.fully_connected))	
		prevStateBuffer.add(obs)
		
	
