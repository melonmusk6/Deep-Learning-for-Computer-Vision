"""
Implements convolutional networks in PyTorch.
WARNING: you SHOULD NOT use ".to()" or ".cuda()" in each implementation block.
"""
import torch
from a3_helper import softmax_loss
from fully_connected_networks import Linear_ReLU, Linear, Solver, adam, ReLU


def hello_convolutional_networks():
    """
    This is a sample function that we will try to import and run to ensure that
    our environment is correctly set up on Google Colab.
    """
    print('Hello from convolutional_networks.py!')


class Conv(object):

    @staticmethod
    def forward(x, w, b, conv_param):
        """
        A naive implementation of the forward pass for a convolutional layer.
        The input consists of N data points, each with C channels, height H and
        width W. We convolve each input with F different filters, where each
        filter spans all C channels and has height HH and width WW.

        Input:
        - x: Input data of shape (N, C, H, W)
        - w: Filter weights of shape (F, C, HH, WW)
        - b: Biases, of shape (F,)
        - conv_param: A dictionary with the following keys:
          - 'stride': The number of pixels between adjacent receptive fields
            in the horizontal and vertical directions.
          - 'pad': The number of pixels that is used to zero-pad the input.

        During padding, 'pad' zeros should be placed symmetrically (i.e equally
        on both sides) along the height and width axes of the input. Be careful
        not to modfiy the original input x directly.

        Returns a tuple of:
        - out: Output data of shape (N, F, H', W') where H' and W' are given by
          H' = 1 + (H + 2 * pad - HH) / stride
          W' = 1 + (W + 2 * pad - WW) / stride
        - cache: (x, w, b, conv_param)
        """
        out = None
        ####################################################################
        # TODO: Implement the convolutional forward pass.                  #
        # Hint: you can use function torch.nn.functional.pad for padding.  #
        # You are NOT allowed to use anything in torch.nn in other places. #
        ####################################################################
        # Replace "pass" statement with your code
        N, C, H_in, W_in = x.shape 
        F_filters, _, HH, WW = w.shape
        stride = conv_param['stride'] 
        pad = conv_param['pad'] 

        H_out = 1 + (H_in + 2 * pad - HH) // stride
        W_out = 1 + (W_in + 2 * pad - WW) // stride

        out = torch.zeros((N, F_filters, H_out, W_out), dtype=x.dtype, device=x.device)

        x_padded = torch.nn.functional.pad(x, (pad, pad, pad, pad), mode='constant', value=0)
        #torch.nn.functional.pad：获取输入张量 x，并在其高度和宽度维度的每一侧都添加 pad 个零。

        for n in range(N): 
            for f in range(F_filters):  
                for i in range(H_out): 
                    for j in range(W_out):
                        h_start = i * stride
                        h_end = h_start + HH
                        w_start = j * stride
                        w_end = w_start + WW
                        
                        receptive_field = x_padded[n, :, h_start:h_end, w_start:w_end]
                        
                        out[n, f, i, j] = torch.sum(receptive_field * w[f]) + b[f]
        #####################################################################
        #                          END OF YOUR CODE                         #
        #####################################################################
        cache = (x, w, b, conv_param)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        """
        A naive implementation of the backward pass for a convolutional layer.
          Inputs:
        - dout: Upstream derivatives.
        - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

        Returns a tuple of:
        - dx: Gradient with respect to x
        - dw: Gradient with respect to w
        - db: Gradient with respect to b
        """
        dx, dw, db = None, None, None
        ###############################################################
        # TODO: Implement the convolutional backward pass.            #
        ###############################################################
        # Replace "pass" statement with your code
        x_orig, w_orig, b_orig, conv_param = cache 
        N, C, H_in, W_in = x_orig.shape 
        F_filters, _, HH, WW = w_orig.shape 
        stride = conv_param['stride'] 
        pad = conv_param['pad'] 

        H_out = 1 + (H_in + 2 * pad - HH) // stride
        W_out = 1 + (W_in + 2 * pad - WW) // stride

        dx = torch.zeros_like(x_orig)
        dw = torch.zeros_like(w_orig)
        db = torch.zeros_like(b_orig)

        x_padded =  torch.nn.functional.pad(x_orig, (pad, pad, pad, pad), mode='constant', value=0)
        dx_padded = torch.zeros_like(x_padded)

        for n in range(N): 
            for f in range(F_filters):  
                for i in range(H_out): 
                    for j in range(W_out):
                        h_start = i * stride
                        h_end = h_start + HH
                        w_start = j * stride
                        w_end = w_start + WW
                        db[f] += dout[n,f,i,j]
                        
                        receptive_field = x_padded[n, :, h_start:h_end, w_start:w_end]
                        
                        dw[f] += receptive_field * dout[n, f, i, j]

                        dx_padded[n, :, h_start:h_end, w_start:w_end] += w_orig[f] * dout[n, f, i, j]
        
        if pad > 0:
            dx = dx_padded[:, :, pad:-pad, pad:-pad]
        else:
            dx = dx_padded
        ###############################################################
        #                       END OF YOUR CODE                      #
        ###############################################################
        return dx, dw, db


class MaxPool(object):

    @staticmethod
    def forward(x, pool_param):
        """
        A naive implementation of the forward pass for a max-pooling layer.

        Inputs:
        - x: Input data, of shape (N, C, H, W)
        - pool_param: dictionary with the following keys:
          - 'pool_height': The height of each pooling region
          - 'pool_width': The width of each pooling region
          - 'stride': The distance between adjacent pooling regions
        No padding is necessary here.

        Returns a tuple of:
        - out: Output of shape (N, C, H', W') where H' and W' are given by
          H' = 1 + (H - pool_height) / stride
          W' = 1 + (W - pool_width) / stride
        - cache: (x, pool_param)
        """
        out = None
        ####################################################################
        # TODO: Implement the max-pooling forward pass                     #
        ####################################################################
        # Replace "pass" statement with your code
        N, C, H_in, W_in = x.shape 
        pool_height = pool_param['pool_height'] 
        pool_width = pool_param['pool_width'] 
        stride = pool_param['stride'] 

        H_out = 1 + (H_in - pool_height) // stride
        W_out = 1 + (W_in - pool_width) // stride

        out = torch.zeros((N, C, H_out, W_out), dtype=x.dtype, device=x.device)

        for n in range(N):
            for c in range(C):
                for i in range(H_out):
                    for j in range(W_out):
                        h_start = i * stride
                        h_end = h_start + pool_height
                        w_start = j * stride
                        w_end = w_start + pool_width
                        
                        pool_region = x[n, c, h_start:h_end, w_start:w_end]

                        out[n, c, i, j] = torch.max(pool_region)
        ####################################################################
        #                         END OF YOUR CODE                         #
        ####################################################################
        cache = (x, pool_param)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        """
        A naive implementation of the backward pass for a max-pooling layer.
        Inputs:
        - dout: Upstream derivatives
        - cache: A tuple of (x, pool_param) as in the forward pass.
        Returns:
        - dx: Gradient with respect to x
        """
        dx = None
        #####################################################################
        # TODO: Implement the max-pooling backward pass                     #
        #####################################################################
        # Replace "pass" statement with your code
        x_orig, pool_param = cache 
        N, C, H_in, W_in = x_orig.shape 
        pool_height = pool_param['pool_height']
        pool_width = pool_param['pool_width'] 
        stride = pool_param['stride']

        H_out = 1 + (H_in - pool_height) // stride
        W_out = 1 + (W_in - pool_width) // stride

        dx = torch.zeros_like(x_orig)

        for n in range(N):
            for c in range(C): 
                for i in range(H_out):
                    for j in range(W_out): 
                        h_start = i * stride
                        h_end = h_start + pool_height
                        w_start = j * stride
                        w_end = w_start + pool_width
                        
                        pool_region = x_orig[n, c, h_start:h_end, w_start:w_end]
                        max_val = torch.max(pool_region)
                        
                        mask = (pool_region == max_val)
                        dx[n, c, h_start:h_end, w_start:w_end] += mask * dout[n, c, i, j]
        ####################################################################
        #                          END OF YOUR CODE                        #
        ####################################################################
        return dx


class ThreeLayerConvNet(object):
    """
    A three-layer convolutional network with the following architecture:
    conv - relu - 2x2 max pool - linear - relu - linear - softmax
    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(self,
                 input_dims=(3, 32, 32),
                 num_filters=32,
                 filter_size=7,
                 hidden_dim=100,
                 num_classes=10,
                 weight_scale=1e-3,
                 reg=0.0,
                 dtype=torch.float,
                 device='cpu'):
        """
        Initialize a new network.
        Inputs:
        - input_dims: Tuple (C, H, W) giving size of input data
        - num_filters: Number of filters to use in the convolutional layer
        - filter_size: Width/height of filters to use in convolutional layer
        - hidden_dim: Number of units to use in fully-connected hidden layer
        - num_classes: Number of scores to produce from the final linear layer.
        - weight_scale: Scalar giving standard deviation for random
          initialization of weights.
        - reg: Scalar giving L2 regularization strength
        - dtype: A torch data type object; all computations will be performed
          using this datatype. float is faster but less accurate, so you
          should use double for numeric gradient checking.
        - device: device to use for computation. 'cpu' or 'cuda'
        """
        self.params = {}
        self.reg = reg
        self.dtype = dtype

        ######################################################################
        # TODO: Initialize weights，biases for the three-layer convolutional #
        # network. Weights should be initialized from a Gaussian             #
        # centered at 0.0 with standard deviation equal to weight_scale;     #
        # biases should be initialized to zero. All weights and biases       #
        # should be stored in thedictionary self.params. Store weights and   #
        # biases for the convolutional layer using the keys 'W1' and 'b1';   #
        # use keys 'W2' and 'b2' for the weights and biases of the hidden    #
        # linear layer, and key 'W3' and 'b3' for the weights and biases of  #
        # the output linear layer                                            #
        #                                                                    #
        # IMPORTANT: For this assignment, you can assume that the padding    #
        # and stride of the first convolutional layer are chosen so that     #
        # **the width and height of the input are preserved**. Take a        #
        # look at the start of the loss() function to see how that happens.  #
        ######################################################################
        # Replace "pass" statement with your code
        self.device = device 
        C_in, H_in, W_in = input_dims

        self.params['W1'] = torch.randn(num_filters, C_in, filter_size, filter_size, device=self.device, dtype=self.dtype) * weight_scale
        self.params['b1'] = torch.zeros(num_filters, device=self.device, dtype=self.dtype)

        # 卷积层输出维度
        H_conv = H_in
        W_conv = W_in
        
        # 2x2 最大池化层
        pool_height, pool_width, pool_stride = 2, 2, 2
        H_pool = 1 + (H_conv - pool_height) // pool_stride
        W_pool = 1 + (W_conv - pool_width) // pool_stride
        
        # 第一个全连接层
        fc1_input_dim = num_filters * H_pool * W_pool

        # 隐藏线性层
        self.params['W2'] = torch.randn(fc1_input_dim, hidden_dim, device=self.device, dtype=self.dtype) * weight_scale
        self.params['b2'] = torch.zeros(hidden_dim, device=self.device, dtype=self.dtype)
        
        # 输出线性层
        self.params['W3'] = torch.randn(hidden_dim, num_classes, device=self.device, dtype=self.dtype) * weight_scale
        self.params['b3'] = torch.zeros(num_classes, device=self.device, dtype=self.dtype)
        ######################################################################
        #                            END OF YOUR CODE                        #
        ######################################################################

    def save(self, path):
        checkpoint = {
          'reg': self.reg,
          'dtype': self.dtype,
          'params': self.params,
        }
        torch.save(checkpoint, path)
        print("Saved in {}".format(path))

    def load(self, path):
        checkpoint = torch.load(path, map_location='cpu')
        self.params = checkpoint['params']
        self.dtype = checkpoint['dtype']
        self.reg = checkpoint['reg']
        print("load checkpoint file: {}".format(path))

    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.
        Input / output: Same API as TwoLayerNet.
        """
        X = X.to(self.dtype)
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']

        # pass conv_param to the forward pass for the convolutional layer
        # Padding and stride chosen to preserve the input spatial size
        filter_size = W1.shape[2]
        conv_param = {'stride': 1, 'pad': (filter_size - 1) // 2}

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        scores = None
        ######################################################################
        # TODO: Implement the forward pass for three-layer convolutional     #
        # net, computing the class scores for X and storing them in the      #
        # scores variable.                                                   #
        #                                                                    #
        # Remember you can use functions defined in your implementation      #
        # above                                                              #
        ######################################################################
        # Replace "pass" statement with your code
        # 第 1 层: conv - relu - 2x2 max pool
        out1, cache1 = Conv_ReLU_Pool.forward(X, W1, b1, conv_param, pool_param)
        N_batch, F_filters, H_pool_out, W_pool_out = out1.shape 
        fc_input = out1.reshape(N_batch, -1) 
        
        # 第 2 层: linear - relu
        out2, cache2 = Linear_ReLU.forward(fc_input, W2, b2)
        
        # 第 3 层: linear 
        scores, cache3 = Linear.forward(out2, W3, b3)
        
        self.cache_for_backward = (cache1, cache2, cache3, out1.shape)
        ######################################################################
        #                             END OF YOUR CODE                       #
        ######################################################################

        if y is None:
            return scores

        loss, grads = 0.0, {}
        ####################################################################
        # TODO: Implement backward pass for three-layer convolutional net, #
        # storing the loss and gradients in the loss and grads variables.  #
        # Compute data loss using softmax, and make sure that grads[k]     #
        # holds the gradients for self.params[k]. Don't forget to add      #
        # L2 regularization!                                               #
        #                                                                  #
        # NOTE: To ensure that your implementation matches ours and you    #
        # pass the automated tests, make sure that your L2 regularization  #
        # does not include a factor of 0.5                                 #
        ####################################################################
        # Replace "pass" statement with your code
        cache1, cache2, cache3, out1_shape = self.cache_for_backward 

        loss, dscores = softmax_loss(scores, y)
        
        loss += self.reg * (torch.sum(W1 * W1) + torch.sum(W2 * W2) + torch.sum(W3 * W3))

        # 第 3 层 (linear) 的反向传播
        dout2, dW3, db3 = Linear.backward(dscores, cache3)
        grads['W3'] = dW3 + 2 * self.reg * W3 
        grads['b3'] = db3
        
        # 第 2 层 (linear - relu) 的反向传播
        dfc_input, dW2, db2 = Linear_ReLU.backward(dout2, cache2)
        grads['W2'] = dW2 + 2 * self.reg * W2
        grads['b2'] = db2
        
        
        dfc_input_reshaped = dfc_input.reshape(out1_shape)
        
        # 第 1 层 (conv - relu - pool) 的反向传播
        dX, dW1, db1 = Conv_ReLU_Pool.backward(dfc_input_reshaped, cache1)
        grads['W1'] = dW1 + 2 * self.reg * W1 
        grads['b1'] = db1
        ###################################################################
        #                             END OF YOUR CODE                    #
        ###################################################################

        return loss, grads


class DeepConvNet(object):
    """
    A convolutional neural network with an arbitrary number of convolutional
    layers in VGG-Net style. All convolution layers will use kernel size 3 and
    padding 1 to preserve the feature map size, and all pooling layers will be
    max pooling layers with 2x2 receptive fields and a stride of 2 to halve the
    size of the feature map.

    The network will have the following architecture:

    {conv - [batchnorm?] - relu - [pool?]} x (L - 1) - linear

    Each {...} structure is a "macro layer" consisting of a convolution layer,
    an optional batch normalization layer, a ReLU nonlinearity, and an optional
    pooling layer. After L-1 such macro layers, a single fully-connected layer
    is used to predict the class scores.

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """
    def __init__(self,
                 input_dims=(3, 32, 32),
                 num_filters=[8, 8, 8, 8, 8],
                 max_pools=[0, 1, 2, 3, 4],
                 batchnorm=False,
                 num_classes=10,
                 weight_scale=1e-3,
                 reg=0.0,
                 weight_initializer=None,
                 dtype=torch.float,
                 device='cpu'):
        """
        Initialize a new network.

        Inputs:
        - input_dims: Tuple (C, H, W) giving size of input data
        - num_filters: List of length (L - 1) giving the number of
          convolutional filters to use in each macro layer.
        - max_pools: List of integers giving the indices of the macro
          layers that should have max pooling (zero-indexed).
        - batchnorm: Whether to include batch normalization in each macro layer
        - num_classes: Number of scores to produce from the final linear layer.
        - weight_scale: Scalar giving standard deviation for random
          initialization of weights, or the string "kaiming" to use Kaiming
          initialization instead
        - reg: Scalar giving L2 regularization strength. L2 regularization
          should only be applied to convolutional and fully-connected weight
          matrices; it should not be applied to biases or to batchnorm scale
          and shifts.
        - dtype: A torch data type object; all computations will be performed
          using this datatype. float is faster but less accurate, so you should
          use double for numeric gradient checking.
        - device: device to use for computation. 'cpu' or 'cuda'
        """
        self.params = {}
        self.num_layers = len(num_filters)+1
        self.max_pools = max_pools
        self.batchnorm = batchnorm
        self.reg = reg
        self.dtype = dtype

        if device == 'cuda':
            device = 'cuda:0'

        #####################################################################
        # TODO: Initialize the parameters for the DeepConvNet. All weights, #
        # biases, and batchnorm scale and shift parameters should be        #
        # stored in the dictionary self.params.                             #
        #                                                                   #
        # Weights for conv and fully-connected layers should be initialized #
        # according to weight_scale. Biases should be initialized to zero.  #
        # Batchnorm scale (gamma) and shift (beta) parameters should be     #
        # initilized to ones and zeros respectively.                        #
        #####################################################################
        # Replace "pass" statement with your code
        C_curr, H_curr, W_curr = input_dims 
        conv_kernel_size = 3 

        self.device = device
        for i in range(len(num_filters)): 
            F_out = num_filters[i] 
            
            if weight_scale == 'kaiming': 
                self.params[f'W{i+1}'] = kaiming_initializer(C_curr, F_out, K=conv_kernel_size, relu=True, device=self.device, dtype=self.dtype)
            else: 
                self.params[f'W{i+1}'] = torch.randn(F_out, C_curr, conv_kernel_size, conv_kernel_size, device=self.device, dtype=self.dtype) * weight_scale
            self.params[f'b{i+1}'] = torch.zeros(F_out, device=self.device, dtype=self.dtype)

            if self.batchnorm:
                self.params[f'gamma{i+1}'] = torch.ones(F_out, device=self.device, dtype=self.dtype) 
                self.params[f'beta{i+1}'] = torch.zeros(F_out, device=self.device, dtype=self.dtype) 
            
            C_curr = F_out 
            
            if i in self.max_pools:
                H_curr = H_curr // 2 
                W_curr = W_curr // 2
        
        fc_input_dim = C_curr * H_curr * W_curr
        if weight_scale == 'kaiming':
            self.params[f'W{self.num_layers}'] = kaiming_initializer(fc_input_dim, num_classes, K=None, relu=False, device=self.device, dtype=self.dtype)
        else:
            self.params[f'W{self.num_layers}'] = torch.randn(fc_input_dim, num_classes, device=self.device, dtype=self.dtype) * weight_scale
        self.params[f'b{self.num_layers}'] = torch.zeros(num_classes, device=self.device, dtype=self.dtype)
        ################################################################
        #                      END OF YOUR CODE                        #
        ################################################################

        # With batch normalization we need to keep track of running
        # means and variances, so we need to pass a special bn_param
        # object to each batch normalization layer. You should pass
        # self.bn_params[0] to the forward pass of the first batch
        # normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.batchnorm:
            self.bn_params = [{'mode': 'train'}
                              for _ in range(len(num_filters))]

        # Check that we got the right number of parameters
        if not self.batchnorm:
            params_per_macro_layer = 2  # weight and bias
        else:
            params_per_macro_layer = 4  # weight, bias, scale, shift
        num_params = params_per_macro_layer * len(num_filters) + 2
        msg = 'self.params has the wrong number of ' \
              'elements. Got %d; expected %d'
        msg = msg % (len(self.params), num_params)
        assert len(self.params) == num_params, msg

        # Check that all parameters have the correct device and dtype:
        for k, param in self.params.items():
            msg = 'param "%s" has device %r; should be %r' \
                  % (k, param.device, device)
            assert param.device == torch.device(device), msg
            msg = 'param "%s" has dtype %r; should be %r' \
                  % (k, param.dtype, dtype)
            assert param.dtype == dtype, msg

    def save(self, path):
        checkpoint = {
          'reg': self.reg,
          'dtype': self.dtype,
          'params': self.params,
          'num_layers': self.num_layers,
          'max_pools': self.max_pools,
          'batchnorm': self.batchnorm,
          'bn_params': self.bn_params,
        }
        torch.save(checkpoint, path)
        print("Saved in {}".format(path))

    def load(self, path, dtype, device):
        checkpoint = torch.load(path, map_location='cpu')
        self.params = checkpoint['params']
        self.dtype = dtype
        self.reg = checkpoint['reg']
        self.num_layers = checkpoint['num_layers']
        self.max_pools = checkpoint['max_pools']
        self.batchnorm = checkpoint['batchnorm']
        self.bn_params = checkpoint['bn_params']

        for p in self.params:
            self.params[p] = \
                self.params[p].type(dtype).to(device)

        for i in range(len(self.bn_params)):
            for p in ["running_mean", "running_var"]:
                self.bn_params[i][p] = \
                    self.bn_params[i][p].type(dtype).to(device)

        print("load checkpoint file: {}".format(path))

    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the deep convolutional
        network.
        Input / output: Same API as ThreeLayerConvNet.
        """
        X = X.to(self.dtype)
        mode = 'test' if y is None else 'train'

        # Set train/test mode for batchnorm params since they
        # behave differently during training and testing.
        if self.batchnorm:
            for bn_param in self.bn_params:
                bn_param['mode'] = mode
        scores = None

        # pass conv_param to the forward pass for the
        # convolutional layer
        # Padding and stride chosen to preserve the input
        # spatial size
        filter_size = 3
        conv_param = {'stride': 1, 'pad': (filter_size - 1) // 2}

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        scores = None
        #########################################################
        # TODO: Implement the forward pass for the DeepConvNet, #
        # computing the class scores for X and storing them in  #
        # the scores variable.                                  #
        #                                                       #
        # You should use the fast versions of convolution and   #
        # max pooling layers, or the convolutional sandwich     #
        # layers, to simplify your implementation.              #
        #########################################################
        # Replace "pass" statement with your code
        current_out = X 
        caches_list = [] 

        for i in range(self.num_layers - 1):
            W_curr = self.params[f'W{i+1}']
            b_curr = self.params[f'b{i+1}']
            
            layer_cache = None 
            if self.batchnorm: 
                gamma_curr = self.params[f'gamma{i+1}']
                beta_curr = self.params[f'beta{i+1}']
                bn_param_curr = self.bn_params[i] 
                if i in self.max_pools: 
                    current_out, layer_cache = Conv_BatchNorm_ReLU_Pool.forward(current_out, W_curr, b_curr, gamma_curr, beta_curr, conv_param, bn_param_curr, pool_param)
                else: 
                    current_out, layer_cache = Conv_BatchNorm_ReLU.forward(current_out, W_curr, b_curr, gamma_curr, beta_curr, conv_param, bn_param_curr)
            else: 
                if i in self.max_pools: 
                    current_out, layer_cache = Conv_ReLU_Pool.forward(current_out, W_curr, b_curr, conv_param, pool_param)
                else:
                    current_out, layer_cache = Conv_ReLU.forward(current_out, W_curr, b_curr, conv_param)
            caches_list.append(layer_cache) 
            
        self.shape_before_fc = current_out.shape 

        N_batch_final = current_out.shape[0]
        fc_input = current_out.reshape(N_batch_final, -1)
        
        W_final_fc = self.params[f'W{self.num_layers}']
        b_final_fc = self.params[f'b{self.num_layers}']
        scores, fc_layer_cache = Linear.forward(fc_input, W_final_fc, b_final_fc)
        caches_list.append(fc_layer_cache) 
        
        self.caches_for_backward = caches_list 
        #####################################################
        #                 END OF YOUR CODE                  #
        #####################################################

        if y is None:
            return scores

        loss, grads = 0, {}
        ###################################################################
        # TODO: Implement the backward pass for the DeepConvNet,          #
        # storing the loss and gradients in the loss and grads variables. #
        # Compute data loss using softmax, and make sure that grads[k]    #
        # holds the gradients for self.params[k]. Don't forget to add     #
        # L2 regularization!                                              #
        #                                                                 #
        # NOTE: To ensure that your implementation matches ours and you   #
        # pass the automated tests, make sure that your L2 regularization #
        # does not include a factor of 0.5                                #
        ###################################################################
        # Replace "pass" statement with your code
        all_caches = self.caches_for_backward 
        
        loss, dscores = softmax_loss(scores, y)
        
        for i in range(self.num_layers):
            W_reg = self.params[f'W{i+1}']
            loss += self.reg * torch.sum(W_reg * W_reg)

        fc_layer_cache = all_caches.pop() 
        W_final_fc_reg = self.params[f'W{self.num_layers}'] 
        
        dfc_input, dW_fc, db_fc = Linear.backward(dscores, fc_layer_cache)
        grads[f'W{self.num_layers}'] = dW_fc + 2 * self.reg * W_final_fc_reg 
        grads[f'b{self.num_layers}'] = db_fc
        
        dout_current = dfc_input.reshape(self.shape_before_fc)

        for i in range(self.num_layers - 2, -1, -1): 
            current_layer_cache = all_caches.pop() 
            W_curr_reg = self.params[f'W{i+1}'] 

            if self.batchnorm:
                if i in self.max_pools: 
                    dout_current, dW, db, dgamma, dbeta = Conv_BatchNorm_ReLU_Pool.backward(dout_current, current_layer_cache)
                else:
                    dout_current, dW, db, dgamma, dbeta = Conv_BatchNorm_ReLU.backward(dout_current, current_layer_cache)
                grads[f'gamma{i+1}'] = dgamma 
                grads[f'beta{i+1}'] = dbeta
            else: 
                if i in self.max_pools: 
                    dout_current, dW, db = Conv_ReLU_Pool.backward(dout_current, current_layer_cache)
                else: 
                    dout_current, dW, db = Conv_ReLU.backward(dout_current, current_layer_cache)
            
            grads[f'W{i+1}'] = dW + 2 * self.reg * W_curr_reg 
            grads[f'b{i+1}'] = db 
        #############################################################
        #                       END OF YOUR CODE                    #
        #############################################################

        return loss, grads


def find_overfit_parameters():
    weight_scale = 2e-3   # Experiment with this!
    learning_rate = 1e-5  # Experiment with this!
    ###########################################################
    # TODO: Change weight_scale and learning_rate so your     #
    # model achieves 100% training accuracy within 30 epochs. #
    ###########################################################
    # Replace "pass" statement with your code
    weight_scale = 1e-1  
    learning_rate = 1e-3
    ###########################################################
    #                       END OF YOUR CODE                  #
    ###########################################################
    return weight_scale, learning_rate


def create_convolutional_solver_instance(data_dict, dtype, device):
    model = None
    solver = None
    #########################################################
    # TODO: Train the best DeepConvNet that you can on      #
    # CIFAR-10 within 60 seconds.                           #
    #########################################################
    # Replace "pass" statement with your code
    input_dims = data_dict['X_train'].shape[1:] 
    
    num_filters = [64, 128, 256] 
    max_pools = [0, 1]
    batchnorm = True 
    weight_scale = 'kaiming'
    reg = 1e-6 

    model = DeepConvNet(input_dims=input_dims,
                        num_filters=num_filters,
                        max_pools=max_pools,
                        batchnorm=batchnorm,
                        num_classes=10, 
                        weight_scale=weight_scale,
                        reg=reg,
                        dtype=dtype,
                        device=device)

    solver = Solver(model, data_dict,
                    update_rule=adam,       
                    optim_config={'learning_rate': 3e-4},
                    lr_decay=0.98,
                    num_epochs=7,
                    batch_size=128,
                    print_every=100,
                    verbose=True,
                    device=device)
    #########################################################
    #                  END OF YOUR CODE                     #
    #########################################################
    return solver


def kaiming_initializer(Din, Dout, K=None, relu=True, device='cpu',
                        dtype=torch.float32):
    """
    Implement Kaiming initialization for linear and convolution layers.

    Inputs:
    - Din, Dout: Integers giving the number of input and output dimensions
      for this layer
    - K: If K is None, then initialize weights for a linear layer with
      Din input dimensions and Dout output dimensions. Otherwise if K is
      a nonnegative integer then initialize the weights for a convolution
      layer with Din input channels, Dout output channels, and a kernel size
      of KxK.
    - relu: If ReLU=True, then initialize weights with a gain of 2 to
      account for a ReLU nonlinearity (Kaiming initializaiton); otherwise
      initialize weights with a gain of 1 (Xavier initialization).
    - device, dtype: The device and datatype for the output tensor.

    Returns:
    - weight: A torch Tensor giving initialized weights for this layer.
      For a linear layer it should have shape (Din, Dout); for a
      convolution layer it should have shape (Dout, Din, K, K).
    """
    gain = 2. if relu else 1.
    weight = None
    if K is None:
        ###################################################################
        # TODO: Implement Kaiming initialization for linear layer.        #
        # The weight scale is sqrt(gain / fan_in),                        #
        # where gain is 2 if ReLU is followed by the layer, or 1 if not,  #
        # and fan_in = num_in_channels (= Din).                           #
        # The output should be a tensor in the designated size, dtype,    #
        # and device.                                                     #
        ###################################################################
        # Replace "pass" statement with your code
        fan_in = Din 
        scale = torch.sqrt(torch.tensor(gain / fan_in, device=device, dtype=dtype)) 
        weight = torch.randn(Din, Dout, device=device, dtype=dtype) * scale
        ###################################################################
        #                            END OF YOUR CODE                     #
        ###################################################################
    else:
        ###################################################################
        # TODO: Implement Kaiming initialization for convolutional layer. #
        # The weight scale is sqrt(gain / fan_in),                        #
        # where gain is 2 if ReLU is followed by the layer, or 1 if not,  #
        # and fan_in = num_in_channels (= Din) * K * K                    #
        # The output should be a tensor in the designated size, dtype,    #
        # and device.                                                     #
        ###################################################################
        # Replace "pass" statement with your code
        fan_in = Din * K * K 
        scale = torch.sqrt(torch.tensor(gain / fan_in, device=device, dtype=dtype)) 
        weight = torch.randn(Dout, Din, K, K, device=device, dtype=dtype) * scale
        ###################################################################
        #                         END OF YOUR CODE                        #
        ###################################################################
    return weight


class BatchNorm(object):

    @staticmethod
    def forward(x, gamma, beta, bn_param):
        """
        Forward pass for batch normalization.

        During training the sample mean and (uncorrected) sample variance
        are computed from minibatch statistics and used to normalize the
        incoming data. During training we also keep an exponentially decaying
        running mean of the mean and variance of each feature, and these
        averages are used to normalize data at test-time.

        At each timestep we update the running averages for mean and
        variance using an exponential decay based on the momentum parameter:

        running_mean = momentum * running_mean + (1 - momentum) * sample_mean
        running_var = momentum * running_var + (1 - momentum) * sample_var

        Note that the batch normalization paper suggests a different
        test-time behavior: they compute sample mean and variance for
        each feature using a large number of training images rather than
        using a running average. For this implementation we have chosen to use
        running averages instead since they do not require an additional
        estimation step; the PyTorch implementation of batch normalization
        also uses running averages.

        Input:
        - x: Data of shape (N, D)
        - gamma: Scale parameter of shape (D,)
        - beta: Shift paremeter of shape (D,)
        - bn_param: Dictionary with the following keys:
          - mode: 'train' or 'test'; required
          - eps: Constant for numeric stability
          - momentum: Constant for running mean / variance.
          - running_mean: Array of shape (D,) giving running mean
            of features
          - running_var Array of shape (D,) giving running variance
            of features

        Returns a tuple of:
        - out: of shape (N, D)
        - cache: A tuple of values needed in the backward pass
        """
        mode = bn_param['mode']
        eps = bn_param.get('eps', 1e-5)
        momentum = bn_param.get('momentum', 0.9)

        N, D = x.shape
        running_mean = bn_param.get('running_mean',
                                    torch.zeros(D,
                                                dtype=x.dtype,
                                                device=x.device))
        running_var = bn_param.get('running_var',
                                   torch.zeros(D,
                                               dtype=x.dtype,
                                               device=x.device))

        out, cache = None, None
        if mode == 'train':
            ##################################################################
            # TODO: Implement the training-time forward pass for batch norm. #
            # Use minibatch statistics to compute the mean and variance, use #
            # these statistics to normalize the incoming data, and scale and #
            # shift the normalized data using gamma and beta.                #
            #                                                                #
            # You should store the output in the variable out.               #
            # Any intermediates that you need for the backward pass should   #
            # be stored in the cache variable.                               #
            #                                                                #
            # You should also use your computed sample mean and variance     #
            # together with the momentum variable to update the running mean #
            # and running variance, storing your result in the running_mean  #
            # and running_var variables.                                     #
            #                                                                #
            # Note that though you should be keeping track of the running    #
            # variance, you should normalize the data based on the standard  #
            # deviation (square root of variance) instead!                   #
            # Referencing the original paper                                 #
            # (https://arxiv.org/abs/1502.03167) might prove to be helpful.  #
            ##################################################################
            # Replace "pass" statement with your code
            sample_mean = x.mean(dim=0)
            sample_var = x.var(dim=0, unbiased=False) 
            
            x_normalized = (x - sample_mean) / torch.sqrt(sample_var + eps)
            out = gamma * x_normalized + beta
            
            running_mean = momentum * running_mean + (1 - momentum) * sample_mean
            running_var = momentum * running_var + (1 - momentum) * sample_var
            
            cache = (x, gamma, x_normalized, sample_mean, sample_var, eps, mode)
            ################################################################
            #                           END OF YOUR CODE                   #
            ################################################################
        elif mode == 'test':
            ################################################################
            # TODO: Implement the test-time forward pass for               #
            # batch normalization. Use the running mean and variance to    #
            # normalize the incoming data, then scale and shift the        #
            # normalized data using gamma and beta. Store the result       #
            # in the out variable.                                         #
            ################################################################
            # Replace "pass" statement with your code
            x_normalized = (x - running_mean) / torch.sqrt(running_var + eps)
            out = gamma * x_normalized + beta
            cache = (x, gamma, x_normalized, running_mean, running_var, eps, mode)
            ################################################################
            #                      END OF YOUR CODE                        #
            ################################################################
        else:
            raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

        # Store the updated running means back into bn_param
        bn_param['running_mean'] = running_mean.detach()
        bn_param['running_var'] = running_var.detach()

        return out, cache

    @staticmethod
    def backward(dout, cache):
        """
        Backward pass for batch normalization.

        For this implementation, you should write out a
        computation graph for batch normalization on paper and
        propagate gradients backward through intermediate nodes.

        Inputs:
        - dout: Upstream derivatives, of shape (N, D)
        - cache: Variable of intermediates from batchnorm_forward.

        Returns a tuple of:
        - dx: Gradient with respect to inputs x, of shape (N, D)
        - dgamma: Gradient with respect to scale parameter gamma,
          of shape (D,)
        - dbeta: Gradient with respect to shift parameter beta,
          of shape (D,)
        """
        dx, dgamma, dbeta = None, None, None
        #####################################################################
        # TODO: Implement the backward pass for batch normalization.        #
        # Store the results in the dx, dgamma, and dbeta variables.         #
        # Referencing the original paper (https://arxiv.org/abs/1502.03167) #
        # might prove to be helpful.                                        #
        # Don't forget to implement train and test mode separately.         #
        #####################################################################
        # Replace "pass" statement with your code
        x, gamma, x_normalized, stat_mean, stat_var, eps, mode = cache
        N, D = dout.shape

        dbeta = dout.sum(dim=0) 
        
        dgamma = (dout * x_normalized).sum(dim=0) 
        
        dx_normalized = dout * gamma
        
        if mode == 'train':
            sample_mean = stat_mean 
            sample_var = stat_var    
            
            inv_std = 1.0 / torch.sqrt(sample_var + eps) 

            # dx 的计算公式
            # dx = (1/N) * gamma * inv_std * [ N*dout - sum(dout) - x_normalized * sum(dout * x_normalized) ]
            # 等价于:
            # dx = (1/N) * inv_std * [ N*dx_normalized - sum(dx_normalized) - x_normalized * sum(dx_normalized * x_normalized) ]
            dx = (1.0 / N) * inv_std * (N * dx_normalized - dx_normalized.sum(dim=0) - x_normalized * (dx_normalized * x_normalized).sum(dim=0))

        elif mode == 'test':
            running_mean = stat_mean
            running_var = stat_var  
            # 在测试模式下，running_mean 和 running_var 对于 x 是常数
            # x_normalized = (x - running_mean) / sqrt(running_var + eps)
            # d(x_normalized)/dx = 1 / sqrt(running_var + eps)
            # dL/dx = dL/dx_normalized * d(x_normalized)/dx = dx_normalized / sqrt(running_var + eps)
            dx = dx_normalized / torch.sqrt(running_var + eps)
        #################################################################
        #                      END OF YOUR CODE                         #
        #################################################################

        return dx, dgamma, dbeta

    @staticmethod
    def backward_alt(dout, cache):
        """
        Alternative backward pass for batch normalization.
        For this implementation you should work out the derivatives
        for the batch normalizaton backward pass on paper and simplify
        as much as possible. You should be able to derive a simple expression
        for the backward pass. See the jupyter notebook for more hints.

        Note: This implementation should expect to receive the same
        cache variable as batchnorm_backward, but might not use all of
        the values in the cache.

        Inputs / outputs: Same as batchnorm_backward
        """
        dx, dgamma, dbeta = None, None, None
        ###################################################################
        # TODO: Implement the backward pass for batch normalization.      #
        # Store the results in the dx, dgamma, and dbeta variables.       #
        #                                                                 #
        # After computing the gradient with respect to the centered       #
        # inputs, you should be able to compute gradients with respect to #
        # the inputs in a single statement; our implementation fits on a  #
        # single 80-character line.                                       #
        ###################################################################
        # Replace "pass" statement with your code
        x, gamma, x_normalized, stat_mean, stat_var, eps, mode = cache
        N, D = dout.shape

        dbeta = dout.sum(dim=0)
        dgamma = (dout * x_normalized).sum(dim=0)
        
        dx_normalized = dout * gamma
        
        if mode == 'train':
            sample_var = stat_var 
            inv_std = 1.0 / torch.sqrt(sample_var + eps) 
            dx = (1.0/N) * inv_std * (N * dx_normalized - dx_normalized.sum(dim=0) - x_normalized * (dx_normalized * x_normalized).sum(dim=0))
        elif mode == 'test':
            running_var = stat_var 
            inv_std = 1.0 / torch.sqrt(running_var + eps) 
            dx = dx_normalized * inv_std
        #################################################################
        #                        END OF YOUR CODE                       #
        #################################################################

        return dx, dgamma, dbeta


class SpatialBatchNorm(object):

    @staticmethod
    def forward(x, gamma, beta, bn_param):
        """
        Computes the forward pass for spatial batch normalization.

        Inputs:
        - x: Input data of shape (N, C, H, W)
        - gamma: Scale parameter, of shape (C,)
        - beta: Shift parameter, of shape (C,)
        - bn_param: Dictionary with the following keys:
          - mode: 'train' or 'test'; required
          - eps: Constant for numeric stability
          - momentum: Constant for running mean / variance. momentum=0
            means that old information is discarded completely at every
            time step, while momentum=1 means that new information is never
            incorporated. The default of momentum=0.9 should work well
            in most situations.
          - running_mean: Array of shape (C,) giving running mean of
            features
          - running_var Array of shape (C,) giving running variance
            of features

        Returns a tuple of:
        - out: Output data, of shape (N, C, H, W)
        - cache: Values needed for the backward pass
        """
        out, cache = None, None

        ################################################################
        # TODO: Implement the forward pass for spatial batch           #
        # normalization.                                               #
        #                                                              #
        # HINT: You can implement spatial batch normalization by       #
        # calling the vanilla version of batch normalization you       #
        # implemented above. Your implementation should be very short; #
        # ours is less than five lines.                                #
        ################################################################
        # Replace "pass" statement with your code
        N, C, H, W_img = x.shape 
        
        x_permuted = x.permute(0, 2, 3, 1) 
        x_reshaped = x_permuted.reshape(N * H * W_img, C)
        
        out_reshaped, bn_cache_internal = BatchNorm.forward(x_reshaped, gamma, beta, bn_param)
        
        out_permuted_back = out_reshaped.reshape(N, H, W_img, C)
        out = out_permuted_back.permute(0, 3, 1, 2) 
        
        cache = (bn_cache_internal, x.shape) 
        ################################################################
        #                       END OF YOUR CODE                       #
        ################################################################

        return out, cache

    @staticmethod
    def backward(dout, cache):
        """
        Computes the backward pass for spatial batch normalization.
        Inputs:
        - dout: Upstream derivatives, of shape (N, C, H, W)
        - cache: Values from the forward pass
        Returns a tuple of:
        - dx: Gradient with respect to inputs, of shape (N, C, H, W)
        - dgamma: Gradient with respect to scale parameter, of shape (C,)
        - dbeta: Gradient with respect to shift parameter, of shape (C,)
        """
        dx, dgamma, dbeta = None, None, None

        #################################################################
        # TODO: Implement the backward pass for spatial batch           #
        # normalization.                                                #
        #                                                               #
        # HINT: You can implement spatial batch normalization by        #
        # calling the vanilla version of batch normalization you        #
        # implemented above. Your implementation should be very short;  #
        # ours is less than five lines.                                 #
        #################################################################
        # Replace "pass" statement with your code
        bn_cache_internal, original_x_shape = cache 
        N, C, H, W_img = original_x_shape 
        
        dout_permuted = dout.permute(0, 2, 3, 1)
        dout_reshaped = dout_permuted.reshape(N * H * W_img, C)
        
        dx_reshaped, dgamma, dbeta = BatchNorm.backward_alt(dout_reshaped, bn_cache_internal)
        
        dx_permuted_back = dx_reshaped.reshape(N, H, W_img, C)
        dx = dx_permuted_back.permute(0, 3, 1, 2)
        ##################################################################
        #                       END OF YOUR CODE                         #
        ##################################################################

        return dx, dgamma, dbeta

##################################################################
#           Fast Implementations and Sandwich Layers             #
##################################################################


class FastConv(object):

    @staticmethod
    def forward(x, w, b, conv_param):
        N, C, H, W = x.shape
        F, _, HH, WW = w.shape
        stride, pad = conv_param['stride'], conv_param['pad']
        layer = torch.nn.Conv2d(C, F, (HH, WW), stride=stride, padding=pad)
        layer.weight = torch.nn.Parameter(w)
        layer.bias = torch.nn.Parameter(b)
        tx = x.detach()
        tx.requires_grad = True
        out = layer(tx)
        cache = (x, w, b, conv_param, tx, out, layer)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        try:
            x, _, _, _, tx, out, layer = cache
            out.backward(dout)
            dx = tx.grad.detach()
            dw = layer.weight.grad.detach()
            db = layer.bias.grad.detach()
            layer.weight.grad = layer.bias.grad = None
        except RuntimeError:
            dx, dw, db = torch.zeros_like(tx), \
                         torch.zeros_like(layer.weight), \
                         torch.zeros_like(layer.bias)
        return dx, dw, db


class FastMaxPool(object):

    @staticmethod
    def forward(x, pool_param):
        N, C, H, W = x.shape
        pool_height, pool_width = \
            pool_param['pool_height'], pool_param['pool_width']
        stride = pool_param['stride']
        layer = torch.nn.MaxPool2d(kernel_size=(pool_height, pool_width),
                                   stride=stride)
        tx = x.detach()
        tx.requires_grad = True
        out = layer(tx)
        cache = (x, pool_param, tx, out, layer)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        try:
            x, _, tx, out, layer = cache
            out.backward(dout)
            dx = tx.grad.detach()
        except RuntimeError:
            dx = torch.zeros_like(tx)
        return dx


class Conv_ReLU(object):

    @staticmethod
    def forward(x, w, b, conv_param):
        """
        A convenience layer that performs a convolution
        followed by a ReLU.
        Inputs:
        - x: Input to the convolutional layer
        - w, b, conv_param: Weights and parameters for the
          convolutional layer
        Returns a tuple of:
        - out: Output from the ReLU
        - cache: Object to give to the backward pass
        """
        a, conv_cache = FastConv.forward(x, w, b, conv_param)
        out, relu_cache = ReLU.forward(a)
        cache = (conv_cache, relu_cache)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        """
        Backward pass for the conv-relu convenience layer.
        """
        conv_cache, relu_cache = cache
        da = ReLU.backward(dout, relu_cache)
        dx, dw, db = FastConv.backward(da, conv_cache)
        return dx, dw, db


class Conv_ReLU_Pool(object):

    @staticmethod
    def forward(x, w, b, conv_param, pool_param):
        """
        A convenience layer that performs a convolution,
        a ReLU, and a pool.
        Inputs:
        - x: Input to the convolutional layer
        - w, b, conv_param: Weights and parameters for
          the convolutional layer
        - pool_param: Parameters for the pooling layer
        Returns a tuple of:
        - out: Output from the pooling layer
        - cache: Object to give to the backward pass
        """
        a, conv_cache = FastConv.forward(x, w, b, conv_param)
        s, relu_cache = ReLU.forward(a)
        out, pool_cache = FastMaxPool.forward(s, pool_param)
        cache = (conv_cache, relu_cache, pool_cache)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        """
        Backward pass for the conv-relu-pool
        convenience layer
        """
        conv_cache, relu_cache, pool_cache = cache
        ds = FastMaxPool.backward(dout, pool_cache)
        da = ReLU.backward(ds, relu_cache)
        dx, dw, db = FastConv.backward(da, conv_cache)
        return dx, dw, db


class Linear_BatchNorm_ReLU(object):

    @staticmethod
    def forward(x, w, b, gamma, beta, bn_param):
        """
        Convenience layer that performs an linear transform,
        batch normalization, and ReLU.
        Inputs:
        - x: Array of shape (N, D1); input to the linear layer
        - w, b: Arrays of shape (D2, D2) and (D2,) giving the
          weight and bias for the linear transform.
        - gamma, beta: Arrays of shape (D2,) and (D2,) giving
          scale and shift parameters for batch normalization.
        - bn_param: Dictionary of parameters for batch
          normalization.
        Returns:
        - out: Output from ReLU, of shape (N, D2)
        - cache: Object to give to the backward pass.
        """
        a, fc_cache = Linear.forward(x, w, b)
        a_bn, bn_cache = BatchNorm.forward(a, gamma, beta, bn_param)
        out, relu_cache = ReLU.forward(a_bn)
        cache = (fc_cache, bn_cache, relu_cache)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        """
        Backward pass for the linear-batchnorm-relu
        convenience layer.
        """
        fc_cache, bn_cache, relu_cache = cache
        da_bn = ReLU.backward(dout, relu_cache)
        da, dgamma, dbeta = BatchNorm.backward(da_bn, bn_cache)
        dx, dw, db = Linear.backward(da, fc_cache)
        return dx, dw, db, dgamma, dbeta


class Conv_BatchNorm_ReLU(object):

    @staticmethod
    def forward(x, w, b, gamma, beta, conv_param, bn_param):
        a, conv_cache = FastConv.forward(x, w, b, conv_param)
        an, bn_cache = SpatialBatchNorm.forward(a, gamma,
                                                beta, bn_param)
        out, relu_cache = ReLU.forward(an)
        cache = (conv_cache, bn_cache, relu_cache)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        conv_cache, bn_cache, relu_cache = cache
        dan = ReLU.backward(dout, relu_cache)
        da, dgamma, dbeta = SpatialBatchNorm.backward(dan, bn_cache)
        dx, dw, db = FastConv.backward(da, conv_cache)
        return dx, dw, db, dgamma, dbeta


class Conv_BatchNorm_ReLU_Pool(object):

    @staticmethod
    def forward(x, w, b, gamma, beta, conv_param, bn_param, pool_param):
        a, conv_cache = FastConv.forward(x, w, b, conv_param)
        an, bn_cache = SpatialBatchNorm.forward(a, gamma, beta, bn_param)
        s, relu_cache = ReLU.forward(an)
        out, pool_cache = FastMaxPool.forward(s, pool_param)
        cache = (conv_cache, bn_cache, relu_cache, pool_cache)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        conv_cache, bn_cache, relu_cache, pool_cache = cache
        ds = FastMaxPool.backward(dout, pool_cache)
        dan = ReLU.backward(ds, relu_cache)
        da, dgamma, dbeta = SpatialBatchNorm.backward(dan, bn_cache)
        dx, dw, db = FastConv.backward(da, conv_cache)
        return dx, dw, db, dgamma, dbeta
