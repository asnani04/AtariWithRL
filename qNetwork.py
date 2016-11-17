#inpired from https://medium.com/@awjuliani/simple-reinforcement-learning-with-tensorflow-part-4-deep-q-networks-and-beyond-8438a3e2b8df#.utgeghelt
#Tensorflow implementaion of a Q-leaning using a neural network as the function approximator

class QNetwork():

    def __init__(self,h_size):

        # We recieve a frame from game and resize it to process it throug convolutional layers.

        self.scalarInput = tf.placeholder(shape=[None,size**2],dype=tf.float32)
        self.imageIn = tf.reshape(self.scalarInput,shape=[-1,84,84,3])
        self.conv1 = tf.contrib.layers.convolution2d(\ inputs =self.imageIn, num_outputs=32,kernel_size=[8,8],stride=[4,4],padding='VALID',biases_initializer=None)
        self.conv2 = tf.contrib.layers.convolution2d( \ inputs=self.conv1,num_outputs=64,kernel_size=[4,4],stride=[2,2],padding='VALID',biases_initializer=None)
        self.conv3 = tf.contrib.layers.convolution2d(\ inputs=self.conv2,num_outputs=64,kernel_size=[3,3],stride=[1,1],padding='VALID',biases_initializer=None)
        self.conv4=tf.contrib.layers.convolution2d( \ inputs=self.conv3,num_outputs=512,kernel_size[7,7],stride=[1,1],padding='VALID',biases_initializer=None)

        # We split the output from the final conv layer into advantage and value streams, this allows use to get better estimates of the values of the states

        self.streamAC,self.streamVC = tf.split(3,2,self.conv4)
        self.streamA = tf.contrib.layers.flatten(self.streamAC)
        self.streamV = tf.contrib.layers.flatten(self.stremVC)
        self.AW = tf.Variable(tf.random_normal([h_size/2,env.actions]))
        self.VW = tf.Variable(tf.random_normal([h_size/2,1]))
        self.Advantage = tf.matmul(self.streamA,self.AW)
        self.Value = tf.matmul(self.streamV,self.VW)


        #We will combne the two streams to get the final Q-values

        self.Qout = self.Value + tf.sub(self.Advantage, tf.reduce_mean(self.Advantage,reduction_indices=1,keep_dims=True))
        self.predict=tf.argmax(self.Qout,1)

        #We now obtain the sqaured loss 

        self.targetQ = tf.placeholder(shape=[None],dtype=tf.float32) #where have we initiliazed this?
        self.actions = tf.placeholder(shape=[None],dtype=tf.int32)
        self.actions_onehot = tf.one_hot(self.actions,env.actions,dtype=tf.float32)

        self.Q = tf.reduce_sum(tf.mul(self.Qout,self.actions_onehot),reduction_indices=1) #explain this!

        self.td_error = tf.square(self.targetQ - self.Q)
        self.loss = tf.reduce_mean(self.td_error)
        self.trainer = tf.train.AdamOptimizer(learning_rate=0.0001)
        self.updateModel = self.trainer.minimize(self.loss) 


class experienceBuffer():

    def __init(self,buffer_size):
        self.buffer = []
        self.buffer_size = buffer_size

    def add(self,experience):
        #explain
        if len(self.buffer)+len(experience) >= self.buffer_size:

            self.buffer[0:(len(experience)+len(self.buffer))-self.buffer_size] = []
        
        self.buffer.extend(experience)

    def sample(self,size):

        return np.reshape(np.array(random.sample(self.buffer,size)),[size,5])


