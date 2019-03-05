# Basic NN layers                                                        TFNN #

from collections import defaultdict
import numpy as np
import tensorflow as tf
from . import util, ops


class Dense:
    def __init__(
        self, name,
        inp_size, out_size, activ=ops.nop,
        matrix=None, bias=None,
        matrix_initializer=None, bias_initializer=None
    ):

        """
        <name>/W
        <name>/b

        User can explicitly specify matrix to use instead of W (<name>/W is
        not created then), but this is not recommended to external users.
        """
        self.name = name
        self.activ = activ
        self.inp_size = inp_size
        self.out_size = out_size

        with tf.variable_scope(name):
            if matrix is None:
                self.W = ops.get_model_variable('W', shape=[inp_size, out_size], initializer=matrix_initializer)
            else:
                self.W = matrix

            if bias is None:
                self.b = ops.get_model_variable('b', shape=[out_size], initializer=bias_initializer)
            else:
                self.b = bias

    def __call__(self, inp):
        """
        inp: [..., inp_size]
        --------------------
        Ret: [..., out_size]
        """
        with tf.variable_scope(self.name):
            out = self.activ(ops.dot(inp, self.W) + self.b)
            out.set_shape([None]*(out.shape.ndims-1) + [self.out_size])
            return out

    @property
    def input_size(self):
        return self.inp_size

    @property
    def output_size(self):
        return self.out_size


class Embedding:
    def __init__(self, name, voc_size, emb_size, matrix=None, initializer=None, device=''):
        """
        Parameters:

          <name>/mat
        """
        self.name = name
        self.voc_size = voc_size
        self.emb_size = emb_size
        self.device = device

        if matrix is not None:
            self.mat = matrix
        else:
            with tf.variable_scope(name), (tf.device(device) if device is not None else util.nop_ctx()):
                self.mat = ops.get_model_variable('mat', shape=[voc_size, emb_size], initializer=initializer)

    def __call__(self, inp):
        """
        inp: [...]
        --------------------
        Ret: [..., emb_size]
        """
        with tf.name_scope(self.name), (tf.device(self.device) if self.device is not None else util.nop_ctx()):
            return tf.gather(self.mat, inp)


class TransformerEmbedding:
    """
    Embedding layer which applies output scaling, adds timing signal ans may have bias
    """
    def __init__(
        self, name,
        voc_size, emb_size,
        bias=False,
        rescale=False,
        **kwargs
    ):
        self.name = name
        self.emb_size = emb_size
        self.voc_size = voc_size
        self.bias = bias
        self.rescale = rescale

        self.emb = Embedding(
            name, voc_size, emb_size,
            initializer=tf.random_normal_initializer(0, emb_size**-.5),
            **kwargs
        )

        if bias:
            with tf.variable_scope(name):
                self.emb_bias = ops.get_model_variable('bias', shape=[1, 1, emb_size])

    def __call__(self, words, shift_right=False, offset=0):
        with tf.name_scope(self.name):
            emb = self.emb(words)  # [batch_size * ninp * emb_dim]

            if self.rescale:
                emb *= self.emb_size ** .5

            if self.bias:
                emb += self.emb_bias

            if shift_right:
                emb = tf.pad(emb, [[0, 0], [1, 0], [0, 0]])[:, :-1, :]

            return emb + ops.make_transformer_timing_signal(emb, offset=offset)


class LayerNorm:
    """
    Performs Layer Normalization
    """
    def __init__(self, name, inp_size, epsilon=1e-6):
        self.name = name
        self.epsilon = epsilon

        with tf.variable_scope(name):
            self.scale = ops.get_model_variable('scale', shape=[inp_size], initializer=tf.ones_initializer())
            self.bias = ops.get_model_variable('bias', shape=[inp_size], initializer=tf.zeros_initializer())

    def __call__(self, inp):
        with tf.variable_scope(self.name):
            mean, variance = tf.nn.moments(inp, axes=[-1], keep_dims=True)
            norm_x = (inp - mean) * tf.rsqrt(variance + self.epsilon)
            return norm_x * self.scale + self.bias


class BatchNorm:
    def __init__(self, name, inp_size, epsilon=1e-6, alpha=0.01,
                 maintain_stats=True, implicit_update_stats=True):
        """
        Performs Batch Normalization over last axis. https://arxiv.org/abs/1502.03167
        :param maintain_stats: if True, maintains moving averages of mean and inv std over batches
        :param alpha: moving average alpha to update statistics (only if maintain_stats)
        :param implicit_update_stats: if True, implicitly performs moving average update of mean and inv std
            stats every time layer is called in training mode
        """
        self.name = name
        self.epsilon = epsilon

        with tf.variable_scope(name):
            self.bias = ops.get_model_variable('bias', shape=[inp_size], initializer=tf.zeros_initializer())
            self.scale = ops.get_model_variable('scale', shape=[inp_size], initializer=tf.ones_initializer())

            if maintain_stats:
                self.alpha = alpha
                self.implicit_update_stats = implicit_update_stats
                self.mean_stats = ops.get_model_variable('mean_moving_average', shape=[inp_size],
                                                         initializer=tf.zeros_initializer(), trainable=False)
                self.inv_std_stats = ops.get_model_variable('mean_inverse_sqrt_variance', shape=[inp_size],
                                                            initializer=tf.ones_initializer(), trainable=False)

    def __call__(self, inp, **kwargs):

        with tf.variable_scope(self.name):
            if ops.is_dropout_enabled():
                normalize_over_axes = list(range(inp.shape.ndims - 1))
                mean, variance = tf.nn.moments(inp, axes=normalize_over_axes, keep_dims=True)
                inv_std = tf.rsqrt(variance + self.epsilon)
                context = util.nop_ctx()

                if hasattr(self, 'alpha'):
                    mean_ma = (1 - self.alpha) * self.mean_stats + self.alpha * tf.reshape(mean, [-1])
                    inv_std_ma = (1 - self.alpha) * self.inv_std_stats + self.alpha * tf.reshape(inv_std, [-1])
                    update_mean = tf.assign(self.mean_stats, mean_ma)
                    update_inv_std = tf.assign(self.inv_std_stats, inv_std_ma)
                    tf.add_to_collection(util.IMPLICIT_UPDATES, update_mean)
                    tf.add_to_collection(util.IMPLICIT_UPDATES, update_inv_std)
                    if self.implicit_update_stats:
                        context = tf.control_dependencies([update_mean, update_inv_std])

                with context:
                    norm_x = (inp - mean) * inv_std

            else:  # not training
                norm_x = (inp - self.mean_stats) * self.inv_std_stats
            return norm_x * self.scale + self.bias


class Conv2D:
    def __init__(self, name, inp_size, num_filers, filter_size=(3, 3),
                 padding='SAME', strides=(1, 1, 1, 1), kernel=None, bias=None,
                 activation=ops.nop):
        self.name, self.inp_size, self.num_filters = name, inp_size, num_filers
        self.filter_size, self.padding, self.strides = filter_size, padding, strides
        self.activation = activation

        with tf.variable_scope(self.name):
            if kernel is not None:
                self.kernel = kernel
            else:
                self.kernel = ops.get_model_variable(
                    'kernel', shape=[filter_size[0], filter_size[1], inp_size, num_filers]
                )
            if bias is not None:
                self.bias = bias
            else:
                self.bias = ops.get_model_variable('bias', shape=[num_filers])


    def __call__(self, inp, **kwargs):
        hid = tf.nn.conv2d(inp, self.kernel, strides=self.strides, padding=self.padding, **kwargs)
        hid += self.bias
        hid = self.activation(hid)
        return hid


class ImageEncoder:
    def __init__(self, name, inp_channels=1, filter_sizes=(64, 128, 256, 512, 512),
                 filter_size=(3, 3), padding='SAME', batchnorm=False, convs_per_pool=1,
                 activation=tf.nn.elu, add_timing_signal=False, **kwargs):
        """ A sequence of of conv-norm-activation-pool cycles """
        self.name = name
        self.layers = []
        self.add_timing_signal = add_timing_signal
        input_sizes = (inp_channels,) + filter_sizes

        with tf.variable_scope(name) as self.scope:
            for i, (inp_size, num_filters) in enumerate(zip(input_sizes, filter_sizes)):
                for j in range(convs_per_pool):
                    conv = Conv2D('layer%i_%i' % (i, j), inp_size, num_filters,
                                  filter_size=filter_size, padding=padding)
                    inp_size = num_filters
                    self.layers.append(conv)

                    if batchnorm:
                        bn = BatchNorm('batchnorm%i' % i, filter_sizes[i])
                        self.layers.append(bn)

                    self.layers.append(activation)

                pool = tf.keras.layers.MaxPooling2D((2, 2), padding='same')
                self.layers.append(pool)
                # WARNING! keras layers with parameters may cause problems with variable scopes
                # max pool is safe cuz it has no parameters

    def __call__(self, x):
        with tf.name_scope(self.name):
            for layer in self.layers:
                x = layer(x)

            if self.add_timing_signal:
                shape = tf.shape(x)
                ts = ops.make_2d_timing_signal(shape[1], shape[2], shape[3])
                ts = tf.tile(ts[None, :, :, :], [shape[0], 1, 1, 1])

                out = tf.concat([x, ts], axis=-1)
                out.set_shape([None, x.shape[1], x.shape[2], 2 * x.shape[3]])
                x = out

            return x


class ResidualLayerWrapper:
    def __init__(self, name, wrapped_layer, inp_size, out_size, steps='nlda', dropout=0.0, dropout_seed=None):
        """
        Applies any number of residual connection, dropout and/or layer normalization before or after wrapped layer
        :param steps: a sequence of operations to perform, containing any combination of:
            - 'l' - call wrapped [l]ayer, this operation should be used exactly once
            - 'd' - apply [d]ropout with p = dropout and seed = dropout_seed
            - 'a' - [a]dd inputs to output (residual connection)
            - 'n' - apply layer [n]ormalization here, can only be done once
        """
        assert steps.count('l') == 1, "residual wrapper must call wrapped layer exactly once"
        assert steps.count('n') <= 1, "in current implementaion, there can be at most one layer normalization step"
        assert inp_size == out_size or 'a' not in steps, "residual step only works if inp_size == out_size"
        self.name = name
        self.wrapped_layer = wrapped_layer

        if 'n' in steps:
            ln_size = inp_size if steps.index('n') < steps.index('l') else out_size
            with tf.variable_scope(name):
                self.norm_layer = LayerNorm("layer_norm", ln_size)

        self.preprocess_steps = steps[:steps.index('l')]
        self.postprocess_steps = steps[steps.index('l') + 1:]
        self.dropout = dropout
        self.dropout_seed = dropout_seed

    def __call__(self, inp, *args, **kwargs):
        out = self.preprocess(inp)
        out = self.wrapped_layer(out, *args, **kwargs)

        # Apply postprocessing only to first element of multi-output layer
        return util.map_first(lambda x: self.postprocess(x, inp), out)

    def __getattr__(self, attr):
        if attr in self.__dict__:
            return getattr(self, attr)
        return getattr(self.wrapped_layer, attr)

    def preprocess(self, inp):
        return self._perform(self.preprocess_steps, inp)

    def postprocess(self, out, inp=None):
        return self._perform(self.postprocess_steps, out, inp=inp)

    def _perform(self, steps, out, inp=None):
        if inp is None:
            inp = out
        for s in steps:
            if s == 'd':
                if ops.is_dropout_enabled():
                    out = ops.dropout(out, 1.0 - self.dropout, seed=self.dropout_seed)
            elif s == 'a':
                out += inp
            elif s == 'n':
                out = self.norm_layer(out)
            else:
                raise RuntimeError("Unknown process step: %s" % s)
        return out


class ResidualChain:
    """
    Block which consists of several layers with residual wrappers
    """
    def __init__(
        self, name, size,
        inp_size=None,
        out_norm=True,
        res_steps='nlda', res_dropout=0.0, inp_dropout=True
    ):
        self.name = name
        self.layers = []
        self.layer_arg_names = []
        self.layer_kwarg_names = []
        self.size = size
        self.inp_size = inp_size if inp_size is not None else size
        self.res_steps = res_steps
        self.res_dropout = res_dropout
        self.inp_dropout = res_dropout if inp_dropout is True else float(inp_dropout)
        with tf.variable_scope(name):
            self.out_norm = LayerNorm('out_norm', inp_size=size) if out_norm else None

    def __call__(self, inp, *args, **kwargs):
        aux = {}
        with tf.name_scope(self.name):
            if ops.is_dropout_enabled():
                inp = ops.dropout(inp, 1.0 - self.inp_dropout)

            for layer, layer_arg_names, layer_kwarg_names in zip(self.layers,
                                                                 self.layer_arg_names,
                                                                 self.layer_kwarg_names):
                layer_args = [kwargs[a] for a in layer_arg_names]
                layer_kwargs = dict((k, kwargs[a]) for k, a in layer_kwarg_names.items())
                layer_out = layer(inp, *layer_args, **layer_kwargs)
                if isinstance(layer_out, (tuple, list)):
                    inp = layer_out[0]
                    aux[layer.name] = layer_out[1] if len(layer_out) == 2 else layer_out[1:]
                else:
                    inp = layer_out
            if self.out_norm is not None:
                inp = self.out_norm(inp)

        return inp, aux

    def add_layer(self, layer, *args, **kwargs):
        """
        Adds a layer to block
        Layer will be wrapped around with residual connections
        Any additional arguments represent mapping of layer arguments to arguments of residual block
        """
        wrapped_layer = ResidualLayerWrapper(
            layer.name,
            layer,
            inp_size=self.inp_size if len(self.layers) == 0 else self.size,
            out_size=self.size,
            steps=self.res_steps,
            dropout=self.res_dropout)

        self.layers.append(wrapped_layer)
        self.layer_arg_names.append(args)
        self.layer_kwarg_names.append(kwargs)
        return wrapped_layer


class FFN:
    """
    Transformer feed-forward layer
    """
    def __init__(self, name,
                 inp_size, hid_size, out_size,
                 relu_dropout):
        assert isinstance(hid_size, int), "List of hidden sizes not is not supported"
        self.name = name
        self.relu_dropout = relu_dropout

        with tf.variable_scope(name):
            self.first_conv = Dense(
                'conv1',
                inp_size, hid_size,
                activ=tf.nn.relu,
                bias_initializer=tf.zeros_initializer())

            self.second_conv = Dense(
                'conv2',
                hid_size, out_size,
                activ=lambda x: x,
                bias_initializer=tf.zeros_initializer())

    def __call__(self, inputs, summarize_preactivations=False):
        """
        inp: [batch_size * ninp * inp_dim]
        ---------------------------------
        out: [batch_size * ninp * out_dim]
        """
        with tf.variable_scope(self.name):
            hidden = self.first_conv(inputs)
            if ops.is_dropout_enabled():
                hidden = ops.dropout(hidden, 1.0 - self.relu_dropout)
            outputs = self.second_conv(hidden)

        return outputs


class MultiHeadAttn:
    """
    Multihead scaled-dot-product attention with input/output transformations
    """
    ATTN_BIAS_VALUE = -1e9

    def __init__(
            self, name, inp_size,
            key_depth, value_depth, output_depth,
            num_heads, attn_dropout=0, attn_value_dropout=0,
            kv_inp_size=None, combine_weight_matrices=None,
    ):
        """
        Multi-head attention or self-attention layer
        :param inp_size: input size for attention queries
        :param kv_inp_size: input size for keys and values. default kv_inp_size = inp_size
        :param key_depth: total query/key vector size, sum over all heads. must be divisible by num_heads
        :param value_depth: total value vector size, sum over all heads. must be divisible by num_heads
        :param output_depth: output vector, obtained by linear projection over selected values over all heads
        :param attn_dropout: dropout rate on attention weights before averaging
        :param attn_value_dropout: dropout rate on attention vector before output projection
        :param combine_weight_matrices: if True, stores both query, key and value weight matrices together (faster self-attn)
            if False, stores query and key_and_value weight matrices separately (allows different input sizes)
            defaults to True if kv_inp_size is not specified(None), False otherwise.
        """
        assert key_depth % num_heads == 0, "Number of heads must divide key_depth"
        assert value_depth % num_heads == 0, "Number of heads must divide value_depth"

        self.name = name
        self.key_depth = key_depth
        self.value_depth = value_depth
        self.num_heads = num_heads
        self.attn_dropout = attn_dropout
        self.attn_value_dropout = attn_value_dropout
        if combine_weight_matrices is None:
            combine_weight_matrices = kv_inp_size is None

        with tf.variable_scope(name):
            self.scope = tf.get_variable_scope()
            if combine_weight_matrices:
                assert kv_inp_size is None or kv_inp_size == inp_size, "if combine_weight matrices, kv_inp_size must " \
                                                                       "be equal to inp_size."
                self.combined_conv = Dense(
                    'combined_conv',
                    inp_size, key_depth * 2 + value_depth,
                    activ=lambda x: x,
                    bias_initializer=tf.zeros_initializer())

                self.query_conv = Dense(
                    'query_conv',
                    inp_size, key_depth,
                    activ=lambda x: x,
                    matrix=self.combined_conv.W[:, :key_depth],
                    bias=self.combined_conv.b[:key_depth],
                )

                self.kv_conv = Dense(
                    'kv_conv',
                    inp_size, key_depth + value_depth,
                    activ=lambda x: x,
                    matrix=self.combined_conv.W[:, key_depth:],
                    bias=self.combined_conv.b[key_depth:],
                )
            else:
                kv_inp_size = kv_inp_size or inp_size
                self.query_conv = Dense(
                    'query_conv',
                    inp_size, key_depth,
                    activ=lambda x: x,
                    bias_initializer=tf.zeros_initializer(),
                )

                self.kv_conv = Dense(
                    'kv_conv',
                    kv_inp_size, key_depth + value_depth,
                    activ=lambda x: x,
                    bias_initializer=tf.zeros_initializer(),
                )
                if kv_inp_size == inp_size:
                    self.combined_conv = Dense(
                        'combined_conv',
                        inp_size, key_depth * 2 + value_depth,
                        activ=lambda x: x,
                        matrix=tf.concat([self.query_conv.W, self.kv_conv.W], axis=1),
                        bias=tf.concat([self.query_conv.b, self.kv_conv.b], axis=0),
                        )

            self.out_conv = Dense(
                'out_conv',
                value_depth, output_depth,
                activ=lambda x: x,
                bias_initializer=tf.zeros_initializer())

    @staticmethod
    def split_heads(x, num_heads):
        """
        Split channels (dimension 3) into multiple heads (dimension 1)
        input: (batch_size * ninp * inp_dim)
        output: (batch_size * n_heads * ninp * (inp_dim/n_heads))
        """
        old_shape = x.get_shape().dims
        dim_size = old_shape[-1]
        new_shape = old_shape[:-1] + [num_heads] + [dim_size // num_heads if dim_size else None]
        ret = tf.reshape(x, tf.concat([tf.shape(x)[:-1], [num_heads, tf.shape(x)[-1] // num_heads]], 0))
        ret.set_shape(new_shape)
        return tf.transpose(ret, [0, 2, 1, 3])  # [batch_size * n_heads * ninp * (hid_dim//n_heads)]

    @staticmethod
    def combine_heads(x):
        """
        Inverse of split heads
        input: (batch_size * n_heads * ninp * (inp_dim/n_heads))
        out: (batch_size * ninp * inp_dim)
        """
        x = tf.transpose(x, [0, 2, 1, 3])
        old_shape = x.get_shape().dims
        a, b = old_shape[-2:]
        new_shape = old_shape[:-2] + [a * b if a and b else None]
        ret = tf.reshape(x, tf.concat([tf.shape(x)[:-2], [tf.shape(x)[-2] * tf.shape(x)[-1]]], 0))
        ret.set_shape(new_shape)
        return ret

    def __call__(self, query_inp, attn_mask, kv_inp=None, kv=None):
        """
        query_inp: [batch_size * n_q * inp_dim]
        attn_mask: [batch_size * 1 * n_q * n_kv]
        kv_inp: [batch_size * n_kv * inp_dim]
        -----------------------------------------------
        results: [batch_size * n_q * output_depth]
        """
        assert kv is None or kv_inp is None, "please only feed one of kv or kv_inp"
        with tf.name_scope(self.name) as scope:
            if kv_inp is not None or kv is not None:
                q = self.query_conv(query_inp)
                if kv is None:
                    kv = self.kv_conv(kv_inp)
                k, v = tf.split(kv, [self.key_depth, self.value_depth], axis=2)
            else:
                assert hasattr(self, 'combined_conv'), "This attention layer was built for different input size" \
                                                       " for queries and kv. One must give it kv_inp kwarg (or kv)"
                combined = self.combined_conv(query_inp)
                q, k, v = tf.split(combined, [self.key_depth, self.key_depth, self.value_depth], axis=2)
            q = self.split_heads(q, self.num_heads)  # [batch_size * n_heads * n_q * (k_dim/n_heads)]
            k = self.split_heads(k, self.num_heads)  # [batch_size * n_heads * n_kv * (k_dim/n_heads)]
            v = self.split_heads(v, self.num_heads)  # [batch_size * n_heads * n_kv * (v_dim/n_heads)]

            key_depth_per_head = self.key_depth / self.num_heads
            q = q / np.sqrt(key_depth_per_head)

            # Dot-product attention
            # logits: (batch_size * n_heads * n_q * n_kv)
            attn_bias = MultiHeadAttn.ATTN_BIAS_VALUE * (1 - attn_mask)
            logits = tf.matmul(q, tf.transpose(k, perm=[0, 1, 3, 2])) + attn_bias
            weights = tf.nn.softmax(logits)

            if ops.is_dropout_enabled():
                weights = ops.dropout(weights, 1.0 - self.attn_dropout)
            x = tf.matmul(
                weights,                         # [batch_size * n_heads * n_q * n_kv]
                v                                # [batch_size * n_heads * n_kv * (v_deph/n_heads)]
            )
            combined_x = self.combine_heads(x)

            if ops.is_dropout_enabled():
                combined_x = ops.dropout(combined_x, 1.0 - self.attn_value_dropout)

            outputs = self.out_conv(combined_x)

            return outputs, {'attn': {'scope': scope, 'weights': weights, 'logits': logits, 'mask': attn_mask}}


class TransformerChain(ResidualChain):
    """
    Residual block with transformer-specific layers and hyperparams
    May serve as encoder or decoder, depending on mask used
    May include attentions to other blocks, specified by attn_inputs
    """
    def __init__(
            self,
            name,
            *_args,
            hid_size=512,
            emb_size=None,
            key_size=None, value_size=None,
            ff_size=None,
            num_heads=8, num_layers=6,
            attn_dropout=0.0, attn_value_dropout=0.0, relu_dropout=0.0, res_dropout=0.1,
            res_steps='nlda', normalize_out=False,
            attn_inputs=tuple(),
            attn_input_sizes=None,
            **_kwargs
    ):

        super().__init__(
            name, hid_size,
            inp_size=emb_size if emb_size else hid_size,
            out_norm=normalize_out,
            res_steps=res_steps, res_dropout=res_dropout
        )

        if _args:
            raise Exception("Unexpected positional arguments")

        if attn_input_sizes is None:
            attn_input_sizes = {}
        if not isinstance(attn_input_sizes, dict):
            attn_input_sizes = dict(zip(attn_inputs, attn_input_sizes))

        self.attn_inputs = attn_inputs
        self.num_layers = num_layers
        self.hid_size = hid_size
        self.emb_size = emb_size if emb_size else hid_size

        key_size = key_size if key_size else hid_size
        value_size = value_size if value_size else hid_size
        ff_size = ff_size if ff_size else hid_size

        def attn_layer(layer_name, first=False, kv_inp_size=None, **kwargs):
            return MultiHeadAttn(
                layer_name,
                inp_size=self.emb_size if first else self.hid_size,
                key_depth=key_size,
                value_depth=value_size,
                output_depth=self.hid_size,
                num_heads=num_heads,
                attn_dropout=attn_dropout,
                attn_value_dropout=attn_value_dropout,
                kv_inp_size=kv_inp_size,
                )

        def ffn_layer(layer_name):
            return FFN(
                layer_name,
                inp_size=self.hid_size,
                hid_size=ff_size,
                out_size=self.hid_size,
                relu_dropout=relu_dropout)

        self.self_attn_layers = []
        self.ffn_layers = []
        self.attn_layers = defaultdict(lambda: [])

        with tf.variable_scope(name):
            for i in range(num_layers):
                self.self_attn_layers.append(self.add_layer(
                    attn_layer('attn-%i' % i, first=(i == 0)),
                    'self_attn_mask'))

                for attn_inp in attn_inputs:
                    self.attn_layers[attn_inp].append(self.add_layer(
                        attn_layer('%s_attn-%i' % (attn_inp, i),
                                   kv_inp_size=attn_input_sizes.get(attn_inp, hid_size)),
                        '%s_attn_mask' % attn_inp,
                        '%s_out' % attn_inp))

                self.ffn_layers.append(self.add_layer(
                    ffn_layer('ffn-%i' % i)))
