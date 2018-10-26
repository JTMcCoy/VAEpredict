import random
import numpy as np
import tensorflow as tf

Normal = tf.contrib.distributions.Normal
np.random.seed(0)
tf.set_random_seed(0)

def xavier_init(fan_in, fan_out, constant=1): 
    """ Xavier initialization of network weights"""
    # https://stackoverflow.com/questions/33640581/how-to-do-xavier-initialization-on-tensorflow
    low = -constant*np.sqrt(6.0/(fan_in + fan_out)) 
    high = constant*np.sqrt(6.0/(fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out), 
                             minval=low, maxval=high, 
                             dtype=tf.float32)

class TFVariationalAutoencoder(object):
    """ Variation Autoencoder (VAE) with an sklearn-like interface implemented using TensorFlow.
    
    This implementation uses probabilistic encoders and decoders using Gaussian 
    distributions and realized by multi-layer perceptrons. The VAE can be learned
    end-to-end.
    
    See "Auto-Encoding Variational Bayes" by Kingma and Welling for more details.
    """
    def __init__(self, network_architecture, transfer_fct=tf.nn.relu, 
                 learning_rate=0.001, batch_size=100, lag_size = 1):
        self.network_architecture = network_architecture
        self.transfer_fct = transfer_fct
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.lag_size = lag_size
        
        # tf Graph input
        self.x = tf.placeholder(tf.float32, [None, network_architecture["n_input"]])
        
        # Create autoencoder network
        self._create_network()
        
        # Define loss function based variational upper-bound and 
        # corresponding optimizer
        self._create_loss_optimizer()
        
        # Initializing the tensor flow variables
        init = tf.global_variables_initializer()

        # Launch the session
        self.sess = tf.InteractiveSession()
        self.sess.run(init)
    
    def _create_network(self):
        # Initialize autoencoder network weights and biases
        network_weights = self._initialize_weights(**self.network_architecture)

        # Use recognition network to determine mean and 
        # (log) variance of Gaussian distribution in latent
        # space
        self.z_mean, self.z_log_sigma_sq = \
            self._recognition_network(network_weights["weights_recog"], 
                                      network_weights["biases_recog"])

        # Draw one sample z from Gaussian distribution
        eps = tf.random_normal(tf.shape(self.z_mean), 0, 1, 
                               dtype=tf.float32)
        # writing eps as above keeps self.z of the same size as the input, so
        # it is not tied to a specific batch size as in the original (below)
#        eps = tf.random_normal((self.batch_size, n_z), 0, 1, 
#                               dtype=tf.float32)
        # z = mu + sigma*epsilon
        self.z = tf.add(self.z_mean, 
                        tf.multiply(tf.sqrt(tf.exp(self.z_log_sigma_sq)), eps))

        # Use generator to determine mean and 
        # (log) variance of Gaussian distribution of reconstructed input
        self.x_hat_mean, self.x_hat_log_sigma_sq = \
            self._generator_network(network_weights["weights_gener"],
                                    network_weights["biases_gener"])
            
    def _initialize_weights(self, n_hidden_recog_1, n_hidden_recog_2, 
                            n_hidden_gener_1,  n_hidden_gener_2, 
                            n_input, n_z):
        all_weights = dict()
        all_weights['weights_recog'] = {
            'h1': tf.Variable(xavier_init(n_input, n_hidden_recog_1)),
            'h2': tf.Variable(xavier_init(n_hidden_recog_1, n_hidden_recog_2)),
            'out_mean': tf.Variable(xavier_init(n_hidden_recog_2, n_z)),
            'out_log_sigma': tf.Variable(xavier_init(n_hidden_recog_2, n_z))}
        all_weights['biases_recog'] = {
            'b1': tf.Variable(tf.zeros([n_hidden_recog_1], dtype=tf.float32)),
            'b2': tf.Variable(tf.zeros([n_hidden_recog_2], dtype=tf.float32)),
            'out_mean': tf.Variable(tf.zeros([n_z], dtype=tf.float32)),
            'out_log_sigma': tf.Variable(tf.zeros([n_z], dtype=tf.float32))}
        all_weights['weights_gener'] = {
            'h1': tf.Variable(xavier_init(n_z, n_hidden_gener_1)),
            'h2': tf.Variable(xavier_init(n_hidden_gener_1, n_hidden_gener_2)),
            'out_mean': tf.Variable(xavier_init(n_hidden_gener_2, n_input)),
            'out_log_sigma': tf.Variable(xavier_init(n_hidden_gener_2, n_input))}
        all_weights['biases_gener'] = {
            'b1': tf.Variable(tf.zeros([n_hidden_gener_1], dtype=tf.float32)),
            'b2': tf.Variable(tf.zeros([n_hidden_gener_2], dtype=tf.float32)),
            'out_mean': tf.Variable(tf.zeros([n_input], dtype=tf.float32)),
            'out_log_sigma': tf.Variable(tf.zeros([n_input], dtype=tf.float32))}
        return all_weights
            
    def _recognition_network(self, weights, biases):
        # Generate probabilistic encoder (recognition network), which
        # maps inputs onto a normal distribution in latent space.
        # The transformation is parametrized and can be learned.
        layer_1 = self.transfer_fct(tf.add(tf.matmul(self.x, weights['h1']), 
                                           biases['b1'])) 
        layer_2 = self.transfer_fct(tf.add(tf.matmul(layer_1, weights['h2']), 
                                           biases['b2'])) 
        z_mean = tf.add(tf.matmul(layer_2, weights['out_mean']),
                        biases['out_mean'])
        z_log_sigma_sq = \
            tf.add(tf.matmul(layer_2, weights['out_log_sigma']), 
                   biases['out_log_sigma'])
        return (z_mean, z_log_sigma_sq)
    
    def _generator_network(self, weights, biases):
        # Generate probabilistic decoder (decoder network), which
        # maps points in latent space onto a normal distribution in data space.
        # The transformation is parametrized and can be learned.
        layer_1 = self.transfer_fct(tf.add(tf.matmul(self.z, weights['h1']), 
                                           biases['b1'])) 
        layer_2 = self.transfer_fct(tf.add(tf.matmul(layer_1, weights['h2']), 
                                           biases['b2'])) 
        x_hat_mean = tf.add(tf.matmul(layer_2, weights['out_mean']),
                        biases['out_mean'])
        x_hat_log_sigma_sq = \
            tf.add(tf.matmul(layer_2, weights['out_log_sigma']), 
                   biases['out_log_sigma'])
        return (x_hat_mean, x_hat_log_sigma_sq)
            
    def _create_loss_optimizer(self):
        # The loss is composed of two terms:
        # 1.) The reconstruction loss (the negative log probability
        #     of the input under the reconstructed Gaussian distribution 
        #     induced by the decoder in the data space).
        #     This can be interpreted as the number of "nats" required
        #     for reconstructing the input when the activation in latent
        #     is given.
        
        X_hat_distribution = Normal(loc=self.x_hat_mean,
                                    scale=tf.exp(self.x_hat_log_sigma_sq))
        reconstr_loss = \
            -tf.reduce_sum(X_hat_distribution.log_prob(self.x), 1)
            
        # 2.) The latent loss, which is defined as the Kullback Leibler divergence 
        ##    between the distribution in latent space induced by the encoder on 
        #     the data and some prior. This acts as a kind of regularizer.
        #     This can be interpreted as the number of "nats" required
        #     for transmitting the latent space distribution given
        #     the prior.
        latent_loss = -0.5 * tf.reduce_sum(1 + self.z_log_sigma_sq 
                                           - tf.square(self.z_mean) 
                                           - tf.exp(self.z_log_sigma_sq), 1)
        self.cost = tf.reduce_mean(reconstr_loss + latent_loss)   # average over batch
        
#        # Use ADAM optimizer
#        self.optimizer = \
#            tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)
        
#        # Use RMSProp optimizer
#        self.optimizer = \
#            tf.train.RMSPropOptimizer(learning_rate=self.learning_rate).minimize(self.cost)
            
#        # Use AdaGrad optimizer
#        self.optimizer = \
#            tf.train.AdagradOptimizer(learning_rate=self.learning_rate).minimize(self.cost)
            
        # Use Adadelta optimizer
        self.optimizer = \
            tf.train.AdadeltaOptimizer(learning_rate=self.learning_rate).minimize(self.cost)
        
    def partial_fit(self, X):
        """Train model based on mini-batch of input data.
        
        Return cost of mini-batch.
        """
        opt, cost = self.sess.run((self.optimizer, self.cost), 
                                  feed_dict={self.x: X})
        return cost
    
    def transform(self, X):
        """Transform data by mapping it into the latent space."""
        # Note: This maps to mean of distribution, we could alternatively
        # sample from Gaussian distribution
        return self.sess.run(self.z_mean, feed_dict={self.x: X})
    
    def generate(self, z_mu=None, n_samples = 100):
        """ Generate data by sampling from latent space.
        
        If z_mu is not None, data for this point in latent space is
        generated. Otherwise, z_mu is drawn from prior in latent 
        space.
        """
        if z_mu is None:
            z_mu = np.random.normal(size=[n_samples,self.network_architecture["n_z"]])
        
        x_hat_mu, x_hat_logsigsq = self.sess.run((self.x_hat_mean, self.x_hat_log_sigma_sq), 
                             feed_dict={self.z: z_mu})
        
        eps = tf.random_normal(tf.shape(x_hat_mu), 0, 1, 
                               dtype=tf.float32)
        
        # x_hat_gen = mu + sigma*epsilon
        x_hat_gen = tf.add(x_hat_mu, 
                        tf.multiply(tf.sqrt(tf.exp(x_hat_logsigsq)), eps))
        
        return x_hat_gen
    
    def reconstruct(self, X, sample = 'mean'):
        """ Use VAE to reconstruct given data, using the mean of the 
            Gaussian distribution of the reconstructed variables by default, 
            as this gives better imputation results.
            Data can also be reconstructed by sampling from the Gaussian
            distribution of the reconstructed variables, by specifying the
            input variable "sample" to value 'sample'.
        """
        if sample == 'sample':
            x_hat_mu, x_hat_logsigsq = self.sess.run((self.x_hat_mean, self.x_hat_log_sigma_sq), 
                             feed_dict={self.x: X})
        
            eps = tf.random_normal(tf.shape(X), 0, 1, 
                               dtype=tf.float32)
            # x_hat = mu + sigma*epsilon
            x_hat = tf.add(x_hat_mu, 
                        tf.multiply(tf.sqrt(tf.exp(x_hat_logsigsq)), eps))
            # evaluate the tensor, as indexing into tensors seems to be a
            # a missing function in tf:
            x_hat = x_hat.eval()
        else:
            x_hat_mu = self.sess.run(self.x_hat_mean, 
                             feed_dict={self.x: X})
            x_hat = x_hat_mu
        
        return x_hat
    
    def predict(self, X, max_iter = 10):
        '''One step ahead prediction, given an observation
        '''
        X_input = X.copy()
        X_nan = np.zeros([X_input.shape[0],X_input.shape[1]*2])
        X_nan[:,0:X_input.shape[1]] = X_input
        
        for i in range(max_iter):
            # reconstruct the inputs, using the mean:
            x_reconstruct = self.reconstruct(X_nan)
            X_nan[:,X_input.shape[1]:] = x_reconstruct[:,X_input.shape[1]:]
        
        X_pred = X_nan.copy()
        X_pred = X_pred[:,X_input.shape[1]:]
        
        return X_pred
    
    def impute(self, X_corrupt, max_iter = 10):
        """ Use VAE to impute missing values in X_corrupt. Missing values
            are indicated by a NaN.
        """
        n_lag = self.lag_size
        if n_lag == 1:
            # Select the rows of the datset which have one or more missing values:
            NanRowIndex = np.where(np.isnan(np.sum(X_corrupt,axis=1)))
            x_miss_val = X_corrupt[NanRowIndex[0],:]
            
            # initialise missing values with arbitrary value
            NanIndex = np.where(np.isnan(x_miss_val))
            x_miss_val[NanIndex] = 0
            
            MissVal = np.zeros([max_iter,len(NanIndex[0])], dtype=np.float32)
            
            for i in range(max_iter):
                MissVal[i,:] = x_miss_val[NanIndex]
                
                # reconstruct the inputs, using the mean:
                x_reconstruct = self.reconstruct(x_miss_val)
                x_miss_val[NanIndex] = x_reconstruct[NanIndex]
            
            X_corrupt[NanRowIndex,:] = x_miss_val
            X_imputed = X_corrupt
            self.MissVal = MissVal
        else:
            # Imputation must be with lagged variables.
            
            # First, replace with the mean any missing values in the first n_lag
            # rows, as missing values here cannot be imputed using a model:
            NanIndex = np.where(np.isnan(X_corrupt[0:n_lag-1,:]))
            X_corrupt[NanIndex] = 0
            
            # Select the rows of the datset which have one or more missing values:
            NanRowIndex = np.where(np.isnan(np.sum(X_corrupt,axis=1)))
            
            NanIndex = np.where(np.isnan(X_corrupt))
            MissVal = np.zeros([max_iter,len(NanIndex[0])], dtype=np.float32)
            
            # Sliding window through the data to impute values:
            for j in range(len(NanRowIndex[0])):
#                print(j/len(NanRowIndex[0]))
                ij = NanRowIndex[0][j]
                lag_j = np.arange(ij-n_lag+1, ij+1)
                
                x_miss_val = np.copy(X_corrupt[lag_j,:])
                # index of missing values in x_miss_val:
                NanIndex = np.where(np.isnan(x_miss_val))
                
                lag_var = x_miss_val[:,:-1].flatten()                
                lag_var = np.append(lag_var, X_corrupt[ij,-1])
                # index of missing values in lag_var:
                NanIndexLag = np.where(np.isnan(lag_var))
                # Initialise with arbitrary value:
                lag_var[NanIndexLag] = 0
                
                for i in range(max_iter):
#                    MissVal[i,:] = x_miss_val[NanIndex]
                    
                    # reconstruct the inputs, using the mean:
                    lag_feed = lag_var.reshape(1,len(lag_var))
                    x_reconstruct = self.reconstruct(lag_feed)
                    lag_var[NanIndexLag] = x_reconstruct[0][NanIndexLag]
                
                x_miss_val[NanIndex] = lag_var[NanIndexLag]
                X_corrupt[lag_j,:] = x_miss_val
            
            X_imputed = X_corrupt
        
        return X_imputed
    
    def train(self, XData, training_epochs=10, display_step=10):
        """ Train VAE in a loop, using numerical data"""
        
        def next_batch(Xdata, batch_size, lag_size, MissingVals = False):
            """ Sample lagged sets of observations from data in Xdata.
                Xdata is an [NxM] matrix, N observations of M variables.
                
                Returns Xdata_sample, a [batch_size x (lag_size x M)] matrix.
            """
            if MissingVals:
                # This returns records with any missing values replaced by 0:
                Xdata_length = Xdata.shape[0]
                M = Xdata.shape[1]
                X_indices = random.sample(range(lag_size-1,Xdata_length),batch_size)
                
                Xdata_sample = np.zeros((batch_size,lag_size*M))
                for i in range(batch_size):
                    Xi = X_indices[i]
                    lag_i = np.arange(Xi-lag_size+1, Xi+1)
                    lag_var = np.copy(Xdata[lag_i,:]).flatten()
                    
                    Xdata_sample[i,:] = lag_var
                
                NanIndex = np.where(np.isnan(Xdata_sample))
                Xdata_sample[NanIndex] = 0
            else:
                # This returns complete records only:
                ObsIndex = np.where(np.isfinite(np.sum(Xdata[lag_size-1:,:],axis=1)))
                M = Xdata.shape[1]
                
                # sample from the observed indices:
                X_indices = random.sample(list(ObsIndex[0]),batch_size)
                
                Xdata_sample = np.zeros((batch_size,lag_size*M))
                for i in range(batch_size):
                    Xi = X_indices[i]
                    lag_i = np.arange(Xi-lag_size+1, Xi+1)
                    lag_var = np.copy(Xdata[lag_i,:]).flatten()
                    
                    Xdata_sample[i,:] = lag_var
            
            return Xdata_sample
        
        # number of rows with complete entries in XData
        NanRowIndex = np.where(np.isnan(np.sum(XData,axis=1)))
        n_samples = np.size(XData, 0) - NanRowIndex[0].shape[0]
        
        losshistory = []
        losshistory_epoch = []
        for epoch in range(training_epochs):
            avg_cost = 0
            total_batch = int(n_samples / self.batch_size)
            # Loop over all batches
            for i in range(total_batch):
                batch_xs = next_batch(XData,self.batch_size, self.lag_size, MissingVals = False)
                # Fit training using batch data
                cost = self.partial_fit(batch_xs)
                # Compute average loss
                avg_cost += cost / n_samples * self.batch_size
            # Display logs per epoch step
            if epoch % display_step == 0:
                losshistory_epoch.append(epoch)
                losshistory.append(-avg_cost)
                print(f'Epoch: {epoch+1:.4f} Cost= {avg_cost:.9f}')
        self.losshistory = losshistory
        self.losshistory_epoch = losshistory_epoch
        return self