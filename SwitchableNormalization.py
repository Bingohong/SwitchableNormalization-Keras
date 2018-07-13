import os
os.environ["KERAS_BACKEND"] = "tensorflow"

from keras.engine import Layer, InputSpec
from keras import initializers
from keras import regularizers
from keras import constraints
from keras import backend as K

from keras.utils.generic_utils import get_custom_objects


class SwitchableNormalization(Layer):
    """Switchable normalization layer

    # Arguments
        axis: Integer, the axis that should be normalized
            (typically the features axis).
            For instance, after a `Conv2D` layer with
            `data_format="channels_first"`,
            set `axis=1` in `BatchNormalization`.
        momentum: Momentum for the moving mean and the moving variance.
        epsilon: Small float added to variance to avoid dividing by zero.
        center: If True, add offset of `beta` to normalized tensor.
            If False, `beta` is ignored.
        scale: If True, multiply by `gamma`.
            If False, `gamma` is not used.
            When the next layer is linear (also e.g. `nn.relu`),
            this can be disabled since the scaling
            will be done by the next layer.
        mean_weight_initializer: Initializer for mean weight
        variance_weight_initializer: Initializer for variance weight
        beta_initializer: Initializer for the beta weight.
        gamma_initializer: Initializer for the gamma weight.
        beta_regularizer: Optional regularizer for the beta weight.
        gamma_regularizer: Optional regularizer for the gamma weight.
        beta_constraint: Optional constraint for the beta weight.
        gamma_constraint: Optional constraint for the gamma weight.

    # Input shape
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.

    # Output shape
        Same shape as input.

    # References
        - [Differentiable Learning-to-Normalize via Switchable Normalization](https://arxiv.org/abs/1806.10779)
        - [SN by pytorch](https://github.com/switchablenorms/Switchable-Normalization/blob/master/models/switchable_norm.py)
    """

    def __init__(self,
                 axis=-1,
                 momentum=0.99,
                 epsilon=1e-5,
                 center=True,
                 scale=True,
                 mean_weight_initializer = 'ones',
                 variance_weight_initializer = 'ones',
                 beta_initializer='zeros',
                 gamma_initializer='ones',
                 moving_mean_initializer='zeros',
                 moving_variance_initializer='ones',
                 beta_regularizer=None,
                 gamma_regularizer=None,
                 beta_constraint=None,
                 gamma_constraint=None,
                 **kwargs):
        super(SwitchableNormalization, self).__init__(**kwargs)
        self.supports_masking = True
        self.axis = axis
        self.momentum = momentum
        self.epsilon = epsilon
        self.center = center
        self.scale = scale
        self.mean_weight_initializer = initializers.get(mean_weight_initializer)
        self.variance_weight_initializer = initializers.get(variance_weight_initializer)
        self.beta_initializer = initializers.get(beta_initializer)
        self.gamma_initializer = initializers.get(gamma_initializer)
        self.moving_mean_initializer = initializers.get(moving_mean_initializer)
        self.moving_variance_initializer = initializers.get(moving_variance_initializer)
        self.beta_regularizer = regularizers.get(beta_regularizer)
        self.gamma_regularizer = regularizers.get(gamma_regularizer)
        self.beta_constraint = constraints.get(beta_constraint)
        self.gamma_constraint = constraints.get(gamma_constraint)

    def build(self, input_shape):
        dim = input_shape[self.axis]

        if dim is None:
            raise ValueError('Axis ' + str(self.axis) + ' of '
                                                        'input tensor should have a defined dimension '
                                                        'but the layer received an input with shape ' +
                             str(input_shape) + '.')

        self.input_spec = InputSpec(ndim=len(input_shape),
                                    axes={self.axis: dim})
        
        # add mean/variance weight
        self.mean_weight = self.add_weight(shape=(3,),
                                            name="mean_weight",
                                            initializer=self.mean_weight_initializer)
        self.variance_weight = self.add_weight(shape=(3,),
                                                name="variance_weight",
                                                initializer=self.variance_weight_initializer)

        # add gamma/beta weight
        shape = (dim,)
        if self.scale:
            self.gamma = self.add_weight(shape=shape,
                                         name='gamma',
                                         initializer=self.gamma_initializer,
                                         regularizer=self.gamma_regularizer,
                                         constraint=self.gamma_constraint)
        else:
            self.gamma = None
        if self.center:
            self.beta = self.add_weight(shape=shape,
                                        name='beta',
                                        initializer=self.beta_initializer,
                                        regularizer=self.beta_regularizer,
                                        constraint=self.beta_constraint)
        else:
            self.beta = None

        self.moving_mean = self.add_weight(
            shape=shape,
            name="moving_mean",
            initializer=self.moving_mean_initializer,
            trainable=False)
        self.moving_variance = self.add_weight(
            shape=shape,
            name="moving_variance",
            initializer=self.moving_variance_initializer,
            trainable=False)

        self.built = True

    def call(self, inputs, training=None):
        input_shape = K.int_shape(inputs)
        # Prepare broadcasting shape.
        ndim = len(input_shape)
        broadcast_shape = [1] * len(input_shape)
        broadcast_shape[self.axis] = input_shape[self.axis]

        # mean/variance of instance_normalization
        reduction_axes_in = list(range(len(input_shape)))
        del reduction_axes_in[self.axis]
        del reduction_axes_in[0]
        mean_in = K.mean(inputs, axis=reduction_axes_in, keepdims=True)
        variance_in = K.var(inputs, axis=reduction_axes_in, keepdims=True)

        # mean/variance of layer_normalization
        reduction_axes_ln = list(range(len(input_shape)))
        del reduction_axes_ln[0]
        mean_ln = K.mean(inputs, axis=reduction_axes_ln, keepdims=True)
        variance_ln = K.var(inputs, axis=reduction_axes_ln, keepdims=True)

        # mean/variance of batch_normalization 
        reduction_axes_bn = list(range(len(input_shape)))
        del reduction_axes_bn[self.axis]

        def normed_training():
            mean_bn = K.mean(inputs, axis=reduction_axes_bn,keepdims=True)
            variance_bn = K.var(inputs, axis=reduction_axes_bn,keepdims=True)
            mean = [mean_in, mean_ln, mean_bn]
            variance = [variance_in, variance_ln, variance_bn]

            # If the learning is either dynamic, or set to training:
            self.add_update([K.moving_average_update(self.moving_mean,
                                                     K.reshape(mean_bn,(input_shape[self.axis],)),
                                                     self.momentum),
                             K.moving_average_update(self.moving_variance,
                                                     K.reshape(variance_bn,(input_shape[self.axis],)),
                                                     self.momentum)],
                            inputs)
            return norm(mean, variance)

        def normalize_inference():
            mean_bn = self.moving_mean
            variance_bn = self.moving_variance
            mean = [mean_in, mean_ln, mean_bn]
            variance = [variance_in, variance_ln, variance_bn]
            return norm(mean, variance)

        def norm(mean,variance):
            mean_weight = K.softmax(self.mean_weight)
            variance_weight = K.softmax(self.variance_weight)
            norm_mean = mean_weight[0]*mean[0] + mean_weight[1]*mean[1] + mean_weight[2]*mean[2]
            norm_variance = variance_weight[0]*variance[0] + variance_weight[1]*variance[1] + variance_weight[2]*variance[2]
            normd = (inputs - norm_mean) / (K.sqrt(norm_variance + self.epsilon))
            if self.scale:
                broadcast_gamma = K.reshape(self.gamma, broadcast_shape)
                normd = normd * broadcast_gamma
            if self.center:
                broadcast_beta = K.reshape(self.beta, broadcast_shape)
                normd = normd + broadcast_beta
            return normd

        if training in {0,False}:
            return normalize_inference

        # Pick the normalized form corresponding to the training phase.
        return K.in_train_phase(normed_training,
                                normalize_inference,
                                training=training)

    def get_config(self):
        config = {
            'axis': self.axis,
            'momentum': self.momentum,
            'epsilon': self.epsilon,
            'center': self.center,
            'scale': self.scale,
            'mean_weight_initializer': self.mean_weight_initializer,
            'variance_weight_initializer': self.variance_weight_initializer,
            'beta_initializer': initializers.serialize(self.beta_initializer),
            'gamma_initializer': initializers.serialize(self.gamma_initializer),
            'moving_mean_initializer': initializers.serialize(self.moving_mean_initializer),
            'moving_variance_initializer': initializers.serialize(self.moving_variance_initializer),
            'beta_regularizer': regularizers.serialize(self.beta_regularizer),
            'gamma_regularizer': regularizers.serialize(self.gamma_regularizer),
            'beta_constraint': constraints.serialize(self.beta_constraint),
            'gamma_constraint': constraints.serialize(self.gamma_constraint)
        }
        base_config = super(SwitchableNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape


get_custom_objects().update({'SwitchableNormalization': SwitchableNormalization})