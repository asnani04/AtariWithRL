from __future__ import print_function
import gym
import os
import numpy as np
import tensorflow as tf
import pickle

# ========================================
# Gym Loads the game
# ========================================
size = 80
gamma = 0.99
decayRate = 0.99
global_step = tf.Variable(0, trainable=False)
start_learning_rate = 5e-3

def preProcess(obs):
	obs = obs[35:195]
	obs = obs[::2, ::2, 0]
	obs[obs == 144] = 0
	obs[obs == 109] = 0
	obs[obs!=0] = 1
	return np.reshape(obs.astype(np.float).ravel(), (size*size, 1))

def discountRewards(r):
  """ take 1D float array of rewards and compute discounted reward """
  discounted_r = np.zeros_like(r)
  running_add = 0
  for t in reversed(xrange(0, r.size)):
    if r[t] != 0: running_add = 0 # reset the sum, since this was a game boundary (pong specific!)
    running_add = running_add * gamma + r[t]
    discounted_r[t] = running_add
  return discounted_r

hiddenSize = 200
env = gym.make("Pong-v0")
obs = env.reset()
lastObs = np.zeros((size*size, 1))
prevObs = tf.placeholder(tf.float32, shape=(size**2, 1))
episodeCount = 0
curReward = 0
rewardSum = 0
batchSize = 10
render = False
runningReward = None
gradBuffer1 = np.zeros([hiddenSize, size**2]) / np.sqrt(size**2)
gradBuffer2 = np.zeros([hiddenSize, 1]) / np.sqrt(hiddenSize)
rmspropCache1 = np.zeros([hiddenSize, size**2]) / np.sqrt(size**2)
rmspropCache2 = np.zeros([hiddenSize, 1]) / np.sqrt(hiddenSize)

curObs = tf.placeholder(tf.float32, shape=(size**2, 1))
# modifiedObs = tf.placeholder(tf.zeros([size*size, 1]), dtype = float)
# curHidden = tf.zeros([hiddenSize, 1])

episodicObs, episodicHidden = [], []
errorHist, rewardHist = [], []

w1 = tf.Variable(tf.truncated_normal(shape=[size**2, hiddenSize], mean=0.0, stddev=0.02))
w2 = tf.Variable(tf.truncated_normal(shape=[hiddenSize, 1], mean=0.0, stddev=0.02))

Hiddens = tf.placeholder(tf.float32, shape=[None, hiddenSize])
Gradients = tf.placeholder(tf.float32, shape=[None, 1])
Observations = tf.placeholder(tf.float32, shape=[None, size**2])
grad1 = tf.placeholder(tf.float32, shape=[hiddenSize, size**2])
Cache1 = tf.placeholder(tf.float32, shape=[hiddenSize, size**2])
grad2 = tf.placeholder(tf.float32, shape=[hiddenSize, 1])
Cache2 = tf.placeholder(tf.float32, shape=[hiddenSize, 1])

learning_rate = tf.train.exponential_decay(start_learning_rate, global_step,
                                           100000, 0.96, staircase=True)
modifiedObs = tf.sub(curObs, prevObs)

curHidden = tf.transpose(tf.matmul(tf.transpose(modifiedObs), w1))
layer1 = tf.nn.relu(curHidden)
out = tf.sigmoid(tf.matmul(tf.transpose(layer1), w2))

deltaWeights2 = tf.matmul(tf.transpose(Hiddens), Gradients)
deltaH = tf.nn.relu(tf.matmul(Gradients, tf.transpose(w2)))
deltaWeights1 = tf.transpose(tf.matmul(tf.transpose(Observations), deltaH))

weights1Update = tf.assign_add(w1, tf.transpose(learning_rate * grad1 / (tf.sqrt(Cache1) + 1e-5)))
weights2Update = tf.assign_add(w2, learning_rate * grad2 / (tf.sqrt(Cache2) + 1e-5))

saver = tf.train.Saver({'w1':w1, 'w2':w2})

sess = tf.InteractiveSession()
sess.run(tf.initialize_all_variables())

if os.path.exists('./pongWeights.ckpt'):
	saver.restore(sess, './pongWeights.ckpt')


while True:
	if render:
		env.render()

	obs = preProcess(obs)
	# for i in range(size**2):
	# 	if obs[i] != 0:
	# 		print(i)
	feed_dict = {curObs: obs, prevObs: lastObs}
	outProb, hid, modObs = sess.run([out, layer1, modifiedObs], feed_dict = feed_dict)
	# for i in range(size**2):
	# 	if modObs[i]!=0:
	# 		print(i)

	# print(outProb)
	action = 2 if np.random.uniform() < outProb else 3
	# print(action)
	lastObs = obs

	episodicObs.append(modObs)
	episodicHidden.append(hid)
	y = 1 if action == 2 else 0
	errorHist.append(y - outProb)

	obs, reward, done, info = env.step(action)
	rewardSum += reward
	rewardHist.append(reward)

	# print(i)
	if done:
		episodicHidden = np.reshape(episodicHidden, (len(episodicHidden), hiddenSize))
		episodicObs = np.reshape(episodicObs, (len(episodicObs), size**2))
		episodeCount += 1
		print("episode over:", episodeCount)
		allObs = np.vstack(episodicObs)
		allHidden = np.vstack(episodicHidden)
		print(len(errorHist), len(errorHist[0]))
		allErrors = np.vstack(errorHist)
		allRewards = np.vstack(rewardHist)


		discountedRewards = discountRewards (allRewards);
		discountedRewards -= np.mean (discountedRewards);
		discountedRewards /= np.std (discountedRewards);

		allErrors *= discountedRewards;
		# print(allErrors)
		delWeights1, delWeights2 = sess.run([deltaWeights1, deltaWeights2],
											feed_dict = {Hiddens: episodicHidden, Gradients: allErrors,
														 Observations: episodicObs})
		# ct = 0
		# for i in range(hiddenSize):
		# 	if delWeights1[5][i] != 0:
		# 		ct += 1
		# print(delWeights1[5])
		# print(ct)
		gradBuffer1 += delWeights1
		gradBuffer2 += delWeights2

		episodicObs, episodicHidden, errorHist, rewardHist = [], [], [], []

		if episodeCount % batchSize == 0:
			rmspropCache1 = decayRate * rmspropCache1 + (1 - decayRate) * gradBuffer1 ** 2
			rmspropCache2 = decayRate * rmspropCache2 + (1 - decayRate) * gradBuffer2 ** 2
			w1_upd, w2_upd = sess.run([weights1Update, weights2Update],
									  feed_dict = {grad1: gradBuffer1, grad2: gradBuffer2,
												   Cache1: rmspropCache1, Cache2: rmspropCache2})
			# print(w1.eval()[0])
			gradBuffer1 = np.zeros_like(gradBuffer1)
			gradBuffer2 = np.zeros_like(gradBuffer2)
			saver.save(sess, './pongWeights.ckpt')

		runningReward = rewardSum if runningReward is None else runningReward * 0.99 + rewardSum * 0.01
		print('resetting env. episode reward total was %f. running mean: %f' % (rewardSum, runningReward))
		# if episodeCount % 10 == 0: pickle.dump(, open('save.p', 'wb'))
		rewardSum = 0
		obs = env.reset() # reset env
		lastObs = np.zeros([size**2, 1])

	if reward != 0: # Pong has either +1 or -1 reward exactly when game ends.
		print ('ep %d: game finished, reward: %f' % (episodeCount, reward))
		# obs, reward, done, info = env.step(action)
