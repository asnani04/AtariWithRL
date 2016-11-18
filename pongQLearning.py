import gym
import numpy as np
import random
import tensorflow as tf
import matplotlib.pyplot as plt
import scipy.misc
import os
%matplotlib inline
import functions #This is the file with all function defintions

#Import game environmenet (TO DO)




#parameters
batchSize = 32 #number of experiences to use for one training step
updateFreq = 4 # frequency of update
y = .99 # Discount factor
startE = 1 #Starting value of epsilon, i.e. probability of random action
endE = 0.1 #Final value of epsilon
eDecreaseSteps = 10000 #number of steps taken to go from startE to endE
numEpisodes = 10000 #number of episodes to train with 
preTrainSteps = 10000 #number of steps to take before training
FinalConvLayerSize=512 #final layer size of our convultion layer  
loadModel = False #Whether or not to laod our model
tau = 0.001 #Rate of update of target network towards primary network
path="./dqn" #The path in which model will be saved

def main():
    tf.reset_default_graph() #what does this do?
    mainQN = Qnetwork(FinalConvLayerSize)
    targetQN = Qnetwork(FinalConvLayerSize)

    init = tf.initialize_all_variable()

    saver = tf.train.Saver()

    trainables = tf.trainable_variables()

    targetOps = updateTargetGraph(trainables,tau)

    myBuffer = experience_buffer()

    #set the rate at which we will decrease epsilon

    e = startE
    stepDrop = (startE - endE)/eDecreaseSteps

    #Lists which contain the total rewards and steps per episode

    jList = []
    rList = []
    totalSteps = 0
    
    #Make a path if path doesn't exist
    if not os.path.exists(path):
        os.makedirs(path)

    with tf.Session() as sess:
        if loadModel ==True:
            print 'Loading previous model...'
            ckpt = tf.train.get_checkpoint_state(path)
            saver.restore(sess,ckpt.model_checkpoint_path)

        sess.run(init)

        updateTarget(targetOps,sess)

        for i in range(numEpisodes):
            episodeBuffer = experienceBuffer()

            #Reset environment and get new observation

            s = env.reset()
            s = processState(s)
            d = False
            rAll = 0
            j = 0
            # The Learning algorithm

            while j< maxEp
