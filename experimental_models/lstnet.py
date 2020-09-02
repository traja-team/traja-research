from numpy.lib.financial import rate
import tensorflow as tf
""" Tensorflow 2.0 implementation of LSTNet Pytorch https://github.com/laiguokun/LSTNet/blob/master/models/LSTNet.py
TODO:Pass args
TODO: Other models   
Encoder-Decoder models with Attention
Adversarial Auto-Encoders: https://arxiv.org/abs/2008.11426?fbclid=IwAR0mC_Fnci2N8At7t_WUUDGKaCEU6mPu8_GM1JmDqyS9nZFK6LjebE6DMLs
Attend and Diagnose: https://towardsdatascience.com/attention-for-time-series-classification-and-forecasting-261723e0006d """

class LSTNet(tf.keras.Model):
    
    def __init__(self,args):
        super(LSTNet,self).__init__()
        
        self.use_cuda = args.cuda
        self.P= args.window 
        self.m = args.m
        self.hidR = args.hidRNN
        self.hidC = args.hidCNN
        self.hidS = args.hidSkip
        self.Ck = args.CNN_kernel
        self.skip = args.skip;
        self.pt = (self.P - self.Ck)/self.skip
        self.hw = args.highway_window
        
        self.conv1 = tf.keras.layers.Conv2D(filters = self.hidC,kernel_size=(self.Ck, self.m))
        self.GRU1 = tf.keras.layers.GRU(self.hidC,self.hidR)
        
        self.dropout = tf.keras.layers.Dropout(rate = args.dropout)
        if (self.skip > 0):
            self.GRUskip = tf.keras.layers.GRU(self.hidC, self.hidS)
            self.linear1 = tf.keras.layers.Dense(self.hidR + self.skip * self.hidS, self.m)
        else:
            self.linear1 = tf.keras.layers.Dense(self.hidR, self.m)
        if (self.hw > 0):
            self.highway = tf.keras.layers.Dense(self.hw, 1)
        self.output = None
        if (args.output_fun == 'sigmoid'):
            self.output = tf.keras.activations.sigmoid
        if (args.output_fun == 'tanh'):
            self.output = tf.keras.activations.tanh
            
    def call(self, x):
        batch_size = x.shape[0]
        
        # CNN
        c = x.reshape(-1, 1, self.P, self.m)
        c = tf.keras.activations.relu(self.conv1(c))
        c = self.dropout(c)
        c = tf.squeeze(c, 3)
        
        # RNN 
        r = tf.transpose(c,perm=[2,0,1]) # Pytorch equivalent torch.permute
        _, r = self.GRU1(r)
        r = self.dropout(tf.squeeze(r,0))

        
        #Skip layer - RNN
        if (self.skip > 0):
            s = c[:,:, int(-self.pt * self.skip):]
            s = s.reshape(batch_size, self.hidC, self.pt, self.skip)
            s = tf.transpose(s, perm = [2,0,3,1])
            s = s.reshape(self.pt, batch_size * self.skip, self.hidC)
            _, s = self.GRUskip(s)
            s = s.reshape(batch_size, self.skip * self.hidS)
            s = self.dropout(s)
            r = tf.concat((r,s),1)
        res = self.linear1(r)
        
        # Highway 
        if (self.hw > 0):
            z = x[:, -self.hw:, :]
            z = tf.transpose(z, perm =[0,2,1]).reshape(-1, self.hw)
            z = self.highway(z)
            z = z.reshape(-1,self.m)
            res = res + z
        if (self.output):
            res = self.output(res)
        return res
    