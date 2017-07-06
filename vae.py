# -*- coding: utf-8 -*-
"""
Created on Sat Jul  1 14:57:43 2017

@author: xiaos
"""
from keras.layers import Input,Dense,Activation,Lambda,RepeatVector,merge,Reshape,Layer,Dropout
import keras.backend as K
from keras.models import Model
from helpers import measure,clustering,print_2D,print_heatmap,cart2polar,outliers_detection
from keras.utils.vis_utils import plot_model
from keras.utils.layer_utils import print_summary
import numpy as np
from keras.optimizers import RMSprop
from keras import metrics
from config import config

def sampling(args):
    epsilon_std = 1.0
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=K.shape(z_mean), 
                              mean=0.,
                              stddev=epsilon_std)
    return z_mean + K.exp(z_log_var / 2) * epsilon

def sampling_gumbel(shape,eps=1e-20):
    u = K.random_uniform( shape )
    return -K.log( -K.log(u+eps)+eps )

def compute_softmax(logits,temp):
    z = logits + sampling_gumbel( K.shape(logits) )
    return K.softmax( z / temp )

def gumbel_softmax(logits,temp=0.1):
    return compute_softmax(logits,temp)
 
class VAE:
    
    def __init__(self,in_dim,loss):
        self.in_dim =in_dim
        self.vae = None
        self.ae = None
        self.loss = loss
    
    def vaeBuild( self ):
        in_dim = self.in_dim
        expr_in = Input( shape=(self.in_dim,) )
        h0 = Dropout(0.2)(expr_in) 
        ## Encoder layers
        h1 = Dense( units=512,name='encoder_1' )(h0)
        #h1_relu = Activation('relu')(h1)
        
        h2 = Dense( units=128,name='encoder_2' )(h1)
        h2_relu = Activation('relu')(h2)
        
        h3 = Dense( units=32,name='encoder_3' )(h2_relu)
        h3_relu = Activation('relu')(h3)
       
 
        z_mean = Dense( units=2,name='z_mean' )(h3_relu)
        z_log_var = Dense( units=2,name='z_log_var' )(h3_relu)
        #z_log_var = Lambda( lambda x:K.log(x) )(z_log_var)
        
        drop_ratio = Dense(units=1,name='drop_ratio',activation='sigmoid')(h3_relu)
        drop_ratio = RepeatVector( self.in_dim )(drop_ratio)
        drop_ratio = Reshape( target_shape=(self.in_dim,) )(drop_ratio)
        
        ## sampling new samples
        z = Lambda(sampling, output_shape=(2,))([z_mean, z_log_var])
        
        ## Decoder layers
        decoder_h1 = Dense( units=128,name='decoder_1' )(z)
        decoder_h1_relu = Activation('relu')(decoder_h1)
        decoder_h2 = Dense( units=512,name='decoder_2' )(decoder_h1_relu)
        decoder_h2_relu = Activation('relu')(decoder_h2)
        
        expr_x = Dense(units=self.in_dim,activation='tanh')(decoder_h2_relu)
        expr_x = Activation('relu')(expr_x)
        
        expr_x_drop = Lambda( lambda x:-x ** 2 )(expr_x)
        expr_x_drop_log = merge( [drop_ratio,expr_x_drop],mode='mul' )  ###  log p_drop =  log(exp(-\lambda x^2))
        expr_x_drop_p = Lambda( lambda x:K.exp(x) )(expr_x_drop_log)
        expr_x_nondrop_p = Lambda( lambda x:1-x )( expr_x_drop_p )
        expr_x_nondrop_log = Lambda( lambda x:K.log(x+1e-20) )(expr_x_nondrop_p)
        
        expr_x_drop_log = Reshape( target_shape=(self.in_dim,1) )(expr_x_drop_log)
        expr_x_nondrop_log = Reshape( target_shape=(self.in_dim,1) )(expr_x_nondrop_log)
        
        logits = merge( [expr_x_drop_log,expr_x_nondrop_log],mode='concat',concat_axis=-1 )
        samples = Lambda( gumbel_softmax,output_shape=(self.in_dim,2,) )( logits )
             
        samples = Lambda( lambda x:x[:,:,1] )(samples)
        samples = Reshape( target_shape=(self.in_dim,) )(samples)
        
        #print(samples.shape)
        expr_x = merge( [expr_x,samples],mode='mul' )

        class VariationalLayer(Layer):
            def __init__(self, **kwargs):
                self.is_placeholder = True
                super(VariationalLayer, self).__init__(**kwargs)
        
            def vae_loss(self, x, x_decoded_mean):
                xent_loss = in_dim * metrics.binary_crossentropy(x, x_decoded_mean)
                kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
                return K.mean(xent_loss + kl_loss)
        
            def call(self, inputs):
                x = inputs[0]
                x_decoded_mean = inputs[1]
                loss = self.vae_loss(x, x_decoded_mean)
                self.add_loss(loss, inputs=inputs)
                # We won't actually use the output.
                return x
        
        y = VariationalLayer()([expr_in, expr_x])
        vae = Model( inputs= expr_in,outputs=y )
        
        opt = RMSprop( lr=0.0001 )
        vae.compile( optimizer=opt,loss=None )
        
        ae = Model( inputs=expr_in,outputs=[h1,h2,h3,z_mean])
        
        self.vae = vae
        self.ae = ae

def SCVAE(expr,patience=30,metric='Silhouette',outliers=True,prefix=None,k=None,label=None,id_map=None,log=True,scale=True,rep=0):
    expr[expr<0] = 0.0

    if log:
        expr = np.log2( expr + 1 )
    if scale:
        for i in range(expr.shape[0]):
            expr[i,:] = expr[i,:] / np.max(expr[i,:])
   
    if outliers:
        o = outliers_detection(expr)
        expr = expr[o==1,:]
        if label is not None:
            label = label[o==1]

    if rep > 0:
        expr_train = np.matlib.repmat( expr,rep,1 )
    else:
        expr_train = np.copy( expr )
 
    vae_ = VAE( in_dim=expr.shape[1],loss = config['loss'] )
    vae_.vaeBuild()
    print_summary( vae_.vae )
    
    if prefix is not None:
        plot_model( vae_.vae,to_file=prefix+'_model.eps',show_shapes=True )
    
    wait = 0
    best_metric = -np.inf
    
    epoch = config['epoch']
    batch_size = config['batch_size']
    
    for e in range(epoch):
        print( "Epoch %d/%d"%(e+1,epoch) )
        
        loss = vae_.vae.fit( expr_train,expr_train,epochs=1,batch_size=batch_size,shuffle=True )
        train_loss = -loss.history['loss'][0]
        print( "Loss:"+str(train_loss) )
        res = vae_.ae.predict(expr)
        #print(h1)
        if k is None and label is not None:
            k=len(np.unique(label))
        
        for r in res:
            print("======"+str(r.shape[1])+"========")
            #if r.shape[1] == 2:
            #    r = cart2polar(r)
            pred,si = clustering( r,k=k )
            if label is not None:
                metrics_ = measure( pred,label )
            metrics_['Silhouette'] = si
               
        cur_metric = train_loss#metrics[metric]
        
        if best_metric < cur_metric:
            best_metric = cur_metric
            wait = 0
            if prefix is not None:
                model_file = prefix+'_model_weights_'+str(e)+'.h5'
                vae_.vae.save_weights(model_file)
        else:
            wait += 1
        
        if  e> 100 and wait > patience:
            break
        
    ## visualization
    if prefix is not None:
        for r in res:
            _,dim = r.shape
            pic_name = prefix + '_dim'+str(dim)+'.eps'
            if dim == 2:
                #r = cart2polar(r)
                if outliers:
                     o = outliers_detection(r)
                     r_p = r[o==1]
                     label_p = label[o==1]
                else:
                     r_p = np.copy(r)
                     label_p = np.copy(label)
                fig = print_2D( r_p,label_p,id_map=id_map)
                fig.savefig( pic_name  )
            else:
                fig32 = print_heatmap( r,label=label,id_map=id_map )
                fig32.savefig( pic_name )
    
    return res
    
    
    
    
