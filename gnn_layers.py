from turtle import down
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Layer
import tensorflow as tf
import utils 
from scipy.special import comb
from healpy_layers import *
from pygsp.graphs import SphereHealpix
import healpy as hp

class Chebyshev(Layer):
    """
    A graph convolutional layer using the Chebyshev approximation
    """

    def __init__(self, L, K,M, Fout=None, initializer=None, activation=None, use_bias=False,
                 use_bn=False,use_se = False,use_residual =False,**kwargs):
        """
        Initializes the graph convolutional layer, assuming the input has dimension (B, M, F)
        :param L: The graph Laplacian (MxM), as numpy array
        :param K: Order of the polynomial to use
        :param Fout: Number of features (channels) of the output, default to number of input channels
        :param initializer: initializer to use for weight initialisation
        :param activation: the activation function to use after the layer, defaults to linear
        :param use_bias: Use learnable bias weights
        :param use_bn: Apply batch norm before adding the bias
        :param kwargs: additional keyword arguments passed on to add_weight
        """

        # This is necessary for every Layer
        super(Chebyshev, self).__init__()

        # save necessary params
        self.L = L
        self.K = K
        self.Fout = Fout
        self.M =M
        self.use_bias = use_bias
        self.use_bn = use_bn
        self.use_se = use_se
        #self.residual =residual
        #self.alpha =alpha
        #self.use_residual = use_residual
        if self.use_bn:
            self.bn = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.9, epsilon=1e-5, center=False, scale=False)
        self.initializer = initializer
        if activation is None or callable(activation):
            self.activation = activation
        elif hasattr(tf.keras.activations, activation):
            self.activation = getattr(tf.keras.activations, activation)
        else:
            raise ValueError(f"Could not find activation <{activation}> in tf.keras.activations...")
        self.kwargs = kwargs
        self.act =  tf.keras.activations.relu
        self.down=tf.keras.layers.Dense(Fout/8,activation='swish')
        self.up = tf.keras.layers.Dense(Fout,activation='sigmoid')
        self.pool = tf.keras.layers.AveragePooling1D(pool_size = M)
        # Rescale Laplacian and store as a TF sparse tensor. Copy to not modify the shared L.
        L = sparse.csr_matrix(L)
        lmax = 1.02 * eigsh(L, k=1, which='LM', return_eigenvectors=False)[0]
        L = utils.rescale_L(L, lmax=lmax, scale=0.75)
        L = L.tocoo()
        indices = np.column_stack((L.row, L.col))
        L = tf.SparseTensor(indices, L.data, L.shape)
        self.sparse_L = tf.sparse.reorder(L)

    def build(self, input_shape):
        """
        Build the weights of the layer
        :param input_shape: shape of the input, batch dim has to be defined
        :return: the kernel variable to train
        """

        # get the input shape
        Fin = int(input_shape[-1])
        M = int(input_shape[1])

        # get Fout if necessary
        if self.Fout is None:
            Fout = Fin
        else:
            Fout = self.Fout

        if self.initializer is None:
            # Filter: Fin*Fout filters of order K, i.e. one filterbank per output feature.
            stddev = 1 / np.sqrt(Fin * (self.K + 0.5) / 2)
            initializer = tf.initializers.TruncatedNormal(stddev=stddev)
            self.kernel = self.add_weight("kernel", shape=[self.K * Fin, Fout],
                                          initializer=initializer, **self.kwargs)
        else:
            self.kernel = self.add_weight("kernel", shape=[self.K * Fin, Fout],
                                          initializer=self.initializer, **self.kwargs)

        if self.use_bias:
            self.bias = self.add_weight("bias", shape=[1, 1, Fout])

        # we cast the sparse L to the current backend type
        if tf.keras.backend.floatx() == 'float32':
            self.sparse_L = tf.cast(self.sparse_L, tf.float32)
        if tf.keras.backend.floatx() == 'float64':
            self.sparse_L = tf.cast(self.sparse_L, tf.float64)
        

        

    def call(self, input_tensor, training=False, *args, **kwargs):
        """
        Calls the layer on a input tensor
        :param input_tensor: input of the layer shape (batch, nodes, channels)
        :param args: further arguments
        :param training: wheter we are training or not
        :param kwargs: further keyword arguments
        :return: the output of the layer
        """

        # shapes, this fun is necessary since sparse_matmul_dense in TF only supports
        # the multiplication of 2d matrices, therefore one has to do some weird reshaping
        # this is not strictly necessary but leads to a huge performance gain...
        # See: https://arxiv.org/pdf/1903.11409.pdf
        N, M, Fin = input_tensor.get_shape()
        M, Fin = int(M), int(Fin)

        # get Fout if necessary
        if self.Fout is None:
            Fout = Fin
        else:
            Fout = self.Fout

        # Transform to Chebyshev basis
        x0 = tf.transpose(input_tensor, perm=[1, 2, 0])  # M x Fin x N
        x0 = tf.reshape(x0, [M, -1])  # M x Fin*N

        # list for stacking
        stack = [x0]

        if self.K > 1:
            x1 = tf.sparse.sparse_dense_matmul(self.sparse_L, x0)
            stack.append(x1)
        for k in range(2, self.K):
            x2 = 2 * tf.sparse.sparse_dense_matmul(self.sparse_L, x1) - x0  # M x Fin*N
            stack.append(x2)
            x0, x1 = x1, x2
        x = tf.stack(stack, axis=0)
        x = tf.reshape(x, [self.K, M, Fin, -1])  # K x M x Fin x N
        x = tf.transpose(x, perm=[3, 1, 2, 0])  # N x M x Fin x K
        x = tf.reshape(x, [-1, Fin * self.K])  # N*M x Fin*K
        # Filter: Fin*Fout filters of order K, i.e. one filterbank per output feature.
        #if self.use_residual:
            #x = (1-self.alpha)*x+self.alpha*self.residual
        #else:
           # x = x
        x = tf.matmul(x, self.kernel)  # N*M x Fout
        x = tf.reshape(x, [-1, M, Fout])  # N x M x Fout
        # x = tf.keras.activations.swish(self.bn(x, training=training))
        # pool = tf.keras.layers.AveragePooling1D(pool_size = M)(x)
        # down=tf.keras.layers.Dense(Fout/4,activation='swish')(pool)
        # up = tf.keras.layers.Dense(Fout,activation='sigmoid')(down)
        if self.use_bn:
            x = self.bn(x, training=training)

        if self.use_bias:
            x = tf.add(x, self.bias)

        if self.activation is not None:
            x = self.activation(x)

        if self.use_se:
            out = self.up(self.down(self.pool(x)))
            output =x*out #hadmard product
        #out = up(down(pool(x)))
        else:
            output =x 

        

        return output


class Monomial(Layer):
    """
    A graph convolutional layer using Monomials
    """
    def __init__(self, L, K, Fout=None, initializer=None, activation=None, use_bias=False,
                 use_bn=False, **kwargs):
        """
        Initializes the graph convolutional layer, assuming the input has dimension (B, M, F)
        :param L: The graph Laplacian (MxM), as numpy array
        :param K: Order of the polynomial to use
        :param Fout: Number of features (channels) of the output, default to number of input channels
        :param initializer: initializer to use for weight initialisation
        :param activation: the activation function to use after the layer, defaults to linear
        :param use_bias: Use learnable bias weights
        :param use_bn: Apply batch norm before adding the bias
        :param kwargs: additional keyword arguments passed on to add_weight
        """

        # This is necessary for every Layer
        super(Monomial, self).__init__()

        # save necessary params
        self.L = L
        self.K = K
        self.Fout = Fout
        self.use_bias = use_bias
        self.use_bn = use_bn
        if self.use_bn:
            self.bn = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.9, epsilon=1e-5, center=False, scale=False)
        self.initializer = initializer
        if activation is None or callable(activation):
            self.activation = activation
        elif hasattr(tf.keras.activations, activation):
            self.activation = getattr(tf.keras.activations, activation)
        else:
            raise ValueError(f"Could not find activation <{activation}> in tf.keras.activations...")
        self.kwargs = kwargs

        # Rescale Laplacian and store as a TF sparse tensor. Copy to not modify the shared L.
        L = sparse.csr_matrix(L)
        lmax = 1.02 * eigsh(L, k=1, which='LM', return_eigenvectors=False)[0]
        L = utils.rescale_L(L, lmax=lmax)
        L = L.tocoo()
        indices = np.column_stack((L.row, L.col))
        L = tf.SparseTensor(indices, L.data, L.shape)
        self.sparse_L = tf.sparse.reorder(L)

    def build(self, input_shape):
        """
        Build the weights of the layer
        :param input_shape: shape of the input, batch dim has to be defined
        """

        # get the input shape
        Fin = int(input_shape[-1])
        M = int(input_shape[1])

        # get Fout if necessary
        if self.Fout is None:
            Fout = Fin
        else:
            Fout = self.Fout

        if self.initializer is None:
            # Filter: Fin*Fout filters of order K, i.e. one filterbank per output feature.
            initializer = tf.initializers.TruncatedNormal(stddev=np.sqrt(6 /(Fin+Fout)))
            self.kernel = self.add_weight("kernel", shape=[self.K * Fin, Fout],
                                          initializer=initializer, **self.kwargs)
        else:
            self.kernel = self.add_weight("kernel", shape=[self.K * Fin, Fout],
                                          initializer=self.initializer, **self.kwargs)

        if self.use_bias:
            self.bias = self.add_weight("bias", shape=[1, 1, Fout])

        # we cast the sparse L to the current backend type
        if tf.keras.backend.floatx() == 'float32':
            self.sparse_L = tf.cast(self.sparse_L, tf.float32)
        if tf.keras.backend.floatx() == 'float64':
            self.sparse_L = tf.cast(self.sparse_L, tf.float64)
        self.down=tf.keras.layers.Dense(Fout/4,activation='swish')
        self.up = tf.keras.layers.Dense(Fout,activation='sigmoid')
        self.pool = tf.keras.layers.AveragePooling1D(pool_size = M)
    def call(self, input_tensor, training=False, *args, **kwargs):
        """
        Calls the layer on a input tensor
        :param input_tensor: input of the layer shape (batch, nodes, channels)
        :param training: wheter we are training or not
        :param args: further arguments
        :param kwargs: further keyword arguments
        :return: the output of the layer
        """

        # shapes, this fun is necessary since sparse_matmul_dense in TF only supports
        # the multiplication of 2d matrices, therefore one has to do some weird reshaping
        # this is not strictly necessary but leads to a huge performance gain...
        # See: https://arxiv.org/pdf/1903.11409.pdf
        N, M, Fin = input_tensor.get_shape()
        M, Fin = int(M), int(Fin)

        # get Fout if necessary
        if self.Fout is None:
            Fout = Fin
        else:
            Fout = self.Fout

        # Transform to monomial basis.
        x0 = tf.transpose(input_tensor, perm=[1, 2, 0])  # M x Fin x N
        x0 = tf.reshape(x0, [M, -1])  # M x Fin*N

        # list for stacking
        stack = [x0]

        for k in range(1, self.K):
            x1 = tf.sparse.sparse_dense_matmul(self.sparse_L, x0)  # M x Fin*N
            stack.append(x1)
            x0 = x1

        x = tf.stack(stack, axis=0)
        x = tf.reshape(x, [self.K, M, Fin, -1])  # K x M x Fin x N
        x = tf.transpose(x, perm=[3, 1, 2, 0])  # N x M x Fin x K
        x = tf.reshape(x, [-1, Fin * self.K])  # N*M x Fin*K
        # Filter: Fin*Fout filters of order K, i.e. one filterbank per output feature.
        x = tf.matmul(x, self.kernel)  # N*M x Fout
        x = tf.reshape(x, [-1, M, Fout])  # N x M x Fout
        out = self.up(self.down(self.pool(x)))
        x = x*out

        if self.use_bn:
            x = self.bn(x, training=training)

        if self.use_bias:
            x = tf.add(x, self.bias)

        if self.activation is not None:
            x = self.activation(x)

        return x


class GCNN_ResidualLayer(Model):
    """
    A generic residual layer of the form
    in -> layer -> layer -> out + alpha*in
    with optional batchnorm in the end
    """

    def __init__(self, layer_type, layer_kwargs, activation=None, act_before=False, use_bn=False,
                 norm_type="batch_norm", bn_kwargs=None, alpha=1.0):
        """
        Initializes the residual layer with the given argument
        :param layer_type: The layer type, either "CHEBY" or "MONO" for chebychev or monomials
        :param layer_kwargs: A dictionary with the inputs for the layer
        :param activation: activation function to use for the res layer
        :param act_before: use activation before skip connection
        :param use_bn: use batchnorm inbetween the layers
        :param norm_type: type of batch norm, either batch_norm for normal batch norm or layer_norm for
                          tf.keras.layers.LayerNormalization
        :param bn_kwargs: An optional dictionary containing further keyword arguments for the normalization layer
        :param alpha: Coupling strength of the input -> layer(input) + alpha*input
        """
        # This is necessary for every Layer
        super(GCNN_ResidualLayer, self).__init__()

        # save variables
        self.layer_type = layer_type
        self.layer_kwargs = layer_kwargs
        if activation is None or callable(activation):
            self.activation = activation
        elif hasattr(tf.keras.activations, activation):
            self.activation = getattr(tf.keras.activations, activation)
        else:
            raise ValueError(f"Could not find activation <{activation}> in tf.keras.activations...")
        self.act_before = act_before
        self.use_bn = use_bn
        self.norm_type = norm_type
        # set the default axis if necessary
        if bn_kwargs is None:
                self.bn_kwargs = {"axis": -1}
        else:
            self.bn_kwargs = bn_kwargs
            if "axis" not in self.bn_kwargs and norm_type != "moving_norm":
                self.bn_kwargs.update({"axis": -1})
        if self.layer_type == "BERN":
            self.layer1 = Bernstein(**self.layer_kwargs)
            self.layer2 = Bernstein(**self.layer_kwargs)
        if self.layer_type == "CHEBY":
            self.layer1 = Chebyshev(**self.layer_kwargs)
            self.layer2 = Chebyshev(**self.layer_kwargs)
        elif self.layer_type == "MONO":
            self.layer1 = Monomial(**self.layer_kwargs)
            self.layer2 = Monomial(**self.layer_kwargs)
        else:
            raise IOError(f"Layertype not understood: {self.layer_type}")

        if use_bn:
            if norm_type == "layer_norm":
                self.bn1 = tf.keras.layers.LayerNormalization(**self.bn_kwargs)
                self.bn2 = tf.keras.layers.LayerNormalization(**self.bn_kwargs)
            elif norm_type == "batch_norm":
                self.bn1 = tf.keras.layers.BatchNormalization(**self.bn_kwargs)
                self.bn2 = tf.keras.layers.BatchNormalization(**self.bn_kwargs)
            else:
                raise ValueError(f"norm_type <{norm_type}> not understood!")

        self.alpha = alpha

    def call(self, input_tensor, training=False, *args, **kwargs):
        """
        Calls the layer on a input tensorf
        :param input_tensor: The input of the layer
        :param training: wheter we are training or not
        :param args: further arguments (ignored)
        :param kwargs: further keyword arguments (ignored)
        :return: the output of the layer
        """
        x = self.layer1(input_tensor)

        # bn
        if self.use_bn:
            x = self.bn1(x, training=training)

        # 2nd layer
        x = self.layer2(x)

        # bn
        if self.use_bn:
            x = self.bn2(x, training=training)

        # deal with the activation
        if self.activation is None:
            return x + input_tensor

        if self.act_before:
            return self.activation(x) + self.alpha*input_tensor
        else:
            return self.activation(x + self.alpha*input_tensor)

class Bernstein(Layer):
    """
    A graph convolutional layer using the Chebyshev approximation
    """

    def __init__(self, L, K, Fout=None, initializer=None, activation=None, use_bias=False,
                 use_bn=False, **kwargs):
        """
        Initializes the graph convolutional layer, assuming the input has dimension (B, M, F)
        :param L: The graph Laplacian (MxM), as numpy array
        :param K: Order of the polynomial to use
        :param Fout: Number of features (channels) of the output, default to number of input channels
        :param initializer: initializer to use for weight initialisation
        :param activation: the activation function to use after the layer, defaults to linear
        :param use_bias: Use learnable bias weights
        :param use_bn: Apply batch norm before adding the bias
        :param kwargs: additional keyword arguments passed on to add_weight
        """

        # This is necessary for every Layer
        super(Bernstein, self).__init__()

        # save necessary params
        self.L = L
        self.K = K-1
        self.Fout = Fout
        self.use_bias = use_bias
        self.use_bn = use_bn
        if self.use_bn:
            self.bn = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.9, epsilon=1e-5, center=False, scale=False)
        self.initializer = initializer
        if activation is None or callable(activation):
            self.activation = activation
        elif hasattr(tf.keras.activations, activation):
            self.activation = getattr(tf.keras.activations, activation)
        else:
            raise ValueError(f"Could not find activation <{activation}> in tf.keras.activations...")
        self.kwargs = kwargs

        # Rescale Laplacian and store as a TF sparse tensor. Copy to not modify the shared L.
        L = sparse.csr_matrix(L)
        lmax = 1.02 * eigsh(L, k=1, which='LM', return_eigenvectors=False)[0]
        L = utils.rescale_L(L, lmax=lmax, scale=0.75)
        L = L.tocoo()
        indices = np.column_stack((L.row, L.col))
        L = tf.SparseTensor(indices, L.data, L.shape)
        self.sparse_L = tf.sparse.reorder(L)

    def build(self, input_shape):
        """
        Build the weights of the layer
        :param input_shape: shape of the input, batch dim has to be defined
        :return: the kernel variable to train
        """

        # get the input shape
        Fin = int(input_shape[-1])

        # get Fout if necessary
        if self.Fout is None:
            Fout = Fin
        else:
            Fout = self.Fout

        if self.initializer is None:
            # Filter: Fin*Fout filters of order K, i.e. one filterbank per output feature.
            stddev = np.sqrt(6 /(Fin+Fout)) # 1 / np.sqrt(Fin * (self.K + 0.5) / 2)
            initializer = tf.initializers.TruncatedNormal(stddev=stddev)
            self.kernel = self.add_weight("kernel", shape=[(self.K+1) * Fin, Fout],
                                          initializer=initializer, **self.kwargs)
        else:
            self.kernel = self.add_weight("kernel", shape=[(self.K+1) * Fin, Fout],
                                          initializer=self.initializer, **self.kwargs)

        if self.use_bias:
            self.bias = self.add_weight("bias", shape=[1, 1, Fout])

        # we cast the sparse L to the current backend type
        if tf.keras.backend.floatx() == 'float32':
            self.sparse_L = tf.cast(self.sparse_L, tf.float32)
        if tf.keras.backend.floatx() == 'float64':
            self.sparse_L = tf.cast(self.sparse_L, tf.float64)

    def call(self, input_tensor, training=False, *args, **kwargs):
        """
        Calls the layer on a input tensor
        :param input_tensor: input of the layer shape (batch, nodes, channels)
        :param args: further arguments
        :param training: wheter we are training or not
        :param kwargs: further keyword arguments
        :return: the output of the layer
        """

        # shapes, this fun is necessary since sparse_matmul_dense in TF only supports
        # the multiplication of 2d matrices, therefore one has to do some weird reshaping
        # this is not strictly necessary but leads to a huge performance gain...
        # See: https://arxiv.org/pdf/1903.11409.pdf
        N, M, Fin = input_tensor.get_shape()
        M, Fin = int(M), int(Fin)

        # get Fout if necessary
        if self.Fout is None:
            Fout = Fin
        else:
            Fout = self.Fout

        # Transform to Chebyshev basis
        x0 = tf.transpose(input_tensor, perm=[1, 2, 0])  # M x Fin x N
        x0 = tf.reshape(x0, [M, -1])  # M x Fin*N
        
        # list for stacking
        stack = []
        for i in range(0,self.K+1):
            x1 = x0
            theta = comb(self.K,i)/(2**self.K)
            for j in range(i):
                x2= tf.sparse.sparse_dense_matmul(self.sparse_L, x1)
                x1 =x2
            x2=x1
            for k in range(self.K-i):
                x3 = 2*x2-tf.sparse.sparse_dense_matmul(self.sparse_L, x2)
                x2 =x3
            x3 = theta*x3
            stack.append(x3)
        x = tf.stack(stack, axis=0)
        x = tf.reshape(x, [(self.K+1), M, Fin, -1])  # K+1 x M x Fin x N
        x = tf.transpose(x, perm=[3, 1, 2, 0])  # N x M x Fin x K+1
        x = tf.reshape(x, [-1, Fin * (self.K+1)])  # N*M x Fin*K+1
        # Filter: Fin*Fout filters of order K, i.e. one filterbank per output feature.
        x = tf.matmul(x, self.kernel)  # N*M x Fout
        x = tf.reshape(x, [-1, M, Fout])  # N x M x Fout

        if self.use_bn:
            x = self.bn(x, training=training)

        if self.use_bias:
            x = tf.add(x, self.bias)

        if self.activation is not None:
            x = self.activation(x)

        return x
class GCNN_ResBern(Model):
    """
    A generic residual layer of the form
    in -> layer -> layer -> out + alpha*in
    with optional batchnorm in the end
    """

    def __init__(self, layer_type, layer_kwargs, activation=None, act_before=False, use_bn=False,
                 norm_type="batch_norm", bn_kwargs=None, alpha=1.0):
        """
        Initializes the residual layer with the given argument
        :param layer_type: The layer type, either "CHEBY" or "MONO" for chebychev or monomials
        :param layer_kwargs: A dictionary with the inputs for the layer
        :param activation: activation function to use for the res layer
        :param act_before: use activation before skip connection
        :param use_bn: use batchnorm inbetween the layers
        :param norm_type: type of batch norm, either batch_norm for normal batch norm or layer_norm for
                          tf.keras.layers.LayerNormalization
        :param bn_kwargs: An optional dictionary containing further keyword arguments for the normalization layer
        :param alpha: Coupling strength of the input -> layer(input) + alpha*input
        """
        # This is necessary for every Layer
        super(GCNN_ResBern, self).__init__()

        # save variables
        self.layer_type = layer_type
        self.layer_kwargs = layer_kwargs
        if activation is None or callable(activation):
            self.activation = activation
        elif hasattr(tf.keras.activations, activation):
            self.activation = getattr(tf.keras.activations, activation)
        else:
            raise ValueError(f"Could not find activation <{activation}> in tf.keras.activations...")
        self.act_before = act_before
        self.use_bn = use_bn
        self.norm_type = norm_type
        # set the default axis if necessary
        if bn_kwargs is None:
                self.bn_kwargs = {"axis": -1}
        else:
            self.bn_kwargs = bn_kwargs
            if "axis" not in self.bn_kwargs and norm_type != "moving_norm":
                self.bn_kwargs.update({"axis": -1})
        self.layer1 = Bernstein(**self.layer_kwargs)
        self.layer2 = Bernstein(**self.layer_kwargs)

        if use_bn:
            if norm_type == "layer_norm":
                self.bn1 = tf.keras.layers.LayerNormalization(**self.bn_kwargs)
                self.bn2 = tf.keras.layers.LayerNormalization(**self.bn_kwargs)
            elif norm_type == "batch_norm":
                self.bn1 = tf.keras.layers.BatchNormalization(**self.bn_kwargs)
                self.bn2 = tf.keras.layers.BatchNormalization(**self.bn_kwargs)
            else:
                raise ValueError(f"norm_type <{norm_type}> not understood!")

        self.alpha = alpha

    def call(self, input_tensor, training=False, *args, **kwargs):
        """
        Calls the layer on a input tensorf
        :param input_tensor: The input of the layer
        :param training: wheter we are training or not
        :param args: further arguments (ignored)
        :param kwargs: further keyword arguments (ignored)
        :return: the output of the layer
        """
        x = self.layer1(input_tensor)

        # bn
        if self.use_bn:
            x = self.bn1(x, training=training)

        # 2nd layer
        x = self.layer2(x)

        # bn
        if self.use_bn:
            x = self.bn2(x, training=training)

        # deal with the activation
        if self.activation is None:
            return x + input_tensor

        if self.act_before:
            return self.activation(x) + self.alpha*input_tensor
        else:
            return self.activation(x + self.alpha*input_tensor)


class GCNNII(Model):
    """
    a four 8 layers of re-implemented GCNN with Initial residual.
    """

    def __init__(self,layer_kwargs, alpha=0.2):
        """
        Initializes the residual layer with the given argument
        :param layer_type: The layer type, either "CHEBY" or "MONO" for chebychev or monomials
        :param layer_kwargs: A dictionary with the inputs for the layer
        :param activation: activation function to use for the res layer
        :param act_before: use activation before skip connection
        :param use_bn: use batchnorm inbetween the layers
        :param norm_type: type of batch norm, either batch_norm for normal batch norm or layer_norm for
                          tf.keras.layers.LayerNormalization
        :param bn_kwargs: An optional dictionary containing further keyword arguments for the normalization layer
        :param alpha: Coupling strength of the input -> layer(input) + alpha*input
        """
        # This is necessary for every Layer
        super(GCNNII, self).__init__()

        sphere32 = SphereHealpix(subdivisions=32, indexes=np.arange(hp.nside2npix(32)), nest=True,
                                       k=40, lap_type='normalized')
        L32 = sphere32.L
        sphere16 = SphereHealpix(subdivisions=16, indexes=np.arange(hp.nside2npix(16)), nest=True,
                                       k=40, lap_type='normalized')
        L16 = sphere16.L
        sphere8 = SphereHealpix(subdivisions=8, indexes=np.arange(hp.nside2npix(8)), nest=True,
                                       k=40, lap_type='normalized')
        L8 = sphere8.L
        sphere4 = SphereHealpix(subdivisions=4, indexes=np.arange(hp.nside2npix(4)), nest=True,
                                       k=40, lap_type='normalized')
        L4 = sphere4.L
        sphere2 = SphereHealpix(subdivisions=2, indexes=np.arange(hp.nside2npix(2)), nest=True,
                                       k=40, lap_type='normalized')
        L2 = sphere2.L


        # save variables
        self.layer_kwargs = layer_kwargs
        self.bn = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.9, epsilon=1e-5, center=False, scale=True)
        self.layer1 = Chebyshev(**self.layer_kwargs,L=L32,K=10, Fout=16, use_bias=False, use_bn=True, activation='relu')
        self.layer2 = Chebyshev(**self.layer_kwargs,L=L32,K=10, Fout=32, use_bias=True, use_bn=True, activation='relu')
        self.pool1 = tf.keras.layers.MaxPool1D(pool_size=4, strides=4,
                                                    padding='valid', data_format='channels_last')
        self.layer3 = Chebyshev(**self.layer_kwargs,L=L16,K=10, Fout=32, use_bias=True, use_bn=True, activation='relu')
        self.layer4 = Chebyshev(**self.layer_kwargs,L=L16,K=10, Fout=64, use_bias=True, use_bn=True, activation='relu')
        #768
        self.layer5 = Chebyshev(**self.layer_kwargs,L=L8,K=10, Fout=48, use_bias=True, use_bn=True, activation='relu')
        self.layer6 = Chebyshev(**self.layer_kwargs,L=L8,K=10, Fout=96, use_bias=True, use_bn=True, activation='relu')
        #192
        self.layer7 = Chebyshev(**self.layer_kwargs,L=L4,K=10, Fout=96, use_bias=True, use_bn=True, activation='relu')
        self.layer8 = Chebyshev(**self.layer_kwargs,L=L4,K=10, Fout=128, use_bias=True, use_bn=True, activation='relu')

        self.layer9 = Chebyshev(**self.layer_kwargs,L=L2,K=10, Fout=180, use_bias=True, use_bn=True, activation='relu')
        self.up1 = tf.keras.layers.Conv1D(16, 1, activation=None,data_format='channels_last')
        self.up2 = tf.keras.layers.Conv1D(32, 1, activation=None,data_format='channels_last')
        self.up3 = tf.keras.layers.Conv1D(64, 1, activation=None,data_format='channels_last')
        self.up4 = tf.keras.layers.Conv1D(96, 1, activation=None,data_format='channels_last')
        self.up5 = tf.keras.layers.Conv1D(128, 1, activation=None,data_format='channels_last')
        self.up6 = tf.keras.layers.Conv1D(180, 1, activation=None,data_format='channels_last')
        self.up7 = tf.keras.layers.Conv1D(48, 1, activation=None,data_format='channels_last')
        #self.up4 = tf.keras.layers.Conv1D(48, 1, activation=None,data_format='channels_last')
        self.alpha = alpha

    def call(self, input_tensor, training=False ,*args, **kwargs):
        """
        Calls the layer on a input tensorf
        :param input_tensor: The input of the layer
        :param training: wheter we are training or not
        :param args: further arguments (ignored)
        :param kwargs: further keyword arguments (ignored)
        :return: the output of the layer
        """
        input = self.bn(input_tensor,training = False)
        x = (1-self.alpha)*self.layer1(input)+self.alpha*self.up1(input) #first block
        x = (1-self.alpha)*self.layer2(x)+self.alpha*self.up2(input)
        x =  self.pool1(x)
        x = (1-self.alpha)*self.layer3(x)+self.alpha*self.up2(self.pool1(input)) #second block
        x = (1-self.alpha)*self.layer4(x)+self.alpha*self.up3(self.pool1(input))
        x =  self.pool1(x)
        x = (1-self.alpha)*self.layer5(x)+self.alpha*self.up7(self.pool1(self.pool1(input))) #second block
        x = (1-self.alpha)*self.layer6(x)+self.alpha*self.up4(self.pool1(self.pool1(input)))
        x =  self.pool1(x)
        x = (1-self.alpha)*self.layer7(x)+self.alpha*self.up4(self.pool1(self.pool1(self.pool1(input)))) #second block
        x = (1-self.alpha)*self.layer8(x)+self.alpha*self.up5(self.pool1(self.pool1(self.pool1(input))))
        x =  self.pool1(x)
        x = (1-self.alpha)*self.layer9(x)+self.alpha*self.up6(self.pool1(self.pool1(self.pool1(self.pool1(input))))) #second block
        x = self.pool1(x)

        # bn
        return x