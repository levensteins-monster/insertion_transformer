# Basic TF operations                                                    TFNN #

import tensorflow as tf
from tensorflow.python.client import device_lib

import math
import hashlib
import threading
import contextlib
from copy import copy


# Dropout scope
import lib.util

_tls = threading.local()


def is_dropout_enabled():
    if not hasattr(_tls, 'dropout_enabled'):
        _tls.dropout_enabled = True
    return _tls.dropout_enabled


@contextlib.contextmanager
def dropout_scope(enabled):
    was_enabled = is_dropout_enabled()
    _tls.dropout_enabled = enabled
    try:
        yield
    finally:
        _tls.dropout_enabled = was_enabled


def get_seed_from_name(name):
    full_name = '/'.join([tf.get_variable_scope().name, name])
    return int(hashlib.md5(full_name.encode()).hexdigest()[:8], 16)


def default_initializer(seed, dtype):
    scope_initializer = tf.get_variable_scope().initializer
    if scope_initializer is not None:
        return scope_initializer
    return tf.glorot_uniform_initializer(seed, dtype)


def get_model_variable(name, **kwargs):
    """ Get variable from MODEL_VARIABLES collection with initializer seeded from its name, not id """

    if kwargs.get('initializer') is None:
        kwargs['initializer'] = default_initializer(seed=get_seed_from_name(name), dtype=kwargs.get('dtype', tf.float32))
    elif hasattr(kwargs['initializer'], 'seed') and kwargs['initializer'].seed is None:
        kwargs['initializer'] = copy(kwargs['initializer'])
        kwargs['initializer'].seed = get_seed_from_name(name)

    return tf.contrib.framework.model_variable(name, **kwargs)


def dot(x, y):
    """
    x: [..., a]
    y: [a, ...]
    -------------
    Ret: [..., ...]
    """
    if x.shape[-1].value is not None and y.shape[0].value is not None:
        # check shapes at composition time
        assert x.shape[-1] == y.shape[0], "Last x dim ({}) must match first y dim ({})".format(
            x.shape[-1], y.shape[0]
        )
        context = lib.util.nop_ctx()
    else:
        # check shapes at inference time
        shapes_are_correct = tf.assert_equal(tf.shape(x)[-1], tf.shape(y)[0])
        context = tf.control_dependencies([shapes_are_correct])

    x_ndim = x.get_shape().ndims
    y_ndim = y.get_shape().ndims
    etc_x = tf.slice(tf.shape(x), [0], [x_ndim - 1])
    etc_y = tf.slice(tf.shape(y), [1], [-1])

    a = tf.shape(y)[0]

    # Reshape forth.
    if x_ndim != 2:
        x = tf.reshape(x, [-1, a])
    if y_ndim != 2:
        y = tf.reshape(y, [a, -1])

    # Compute
    with context:
        ret = tf.matmul(x, y)

    # Reshape back.
    if x_ndim != 2 or y_ndim != 2:
        ret = tf.reshape(ret, tf.concat([etc_x, etc_y], 0))

    return ret


def make_attn_mask(inp, inp_len, dtype=tf.float32):
    """
    Compute mask for encoder-like network - each sequence element in a batch has access to the full sequence
    :param inp: [batch_size * ninp]
    :param inp_len: [batch_size]
    :returns: mask, [batch_size * 1 * 1 * ninp]
    """
    with tf.name_scope("encoder_mask"):
        mask = tf.sequence_mask(inp_len, dtype=dtype, maxlen=tf.shape(inp)[1])
        return mask[:, None, None, :]


def infer_length(seq, eos=1):
    """
    compute length (including first eos) given output indices and eos code.
    :param seq: tf matrix [time,batch] if time_major else [batch,time]
    :param eos: integer index of end-of-sentence token
    :returns: lengths, int32 vector of [batch_size]
    """
    is_eos = tf.cast(tf.equal(seq, eos), 'int32')
    count_eos = tf.cumsum(is_eos, axis=0, exclusive=True)
    lengths = tf.reduce_sum(tf.cast(tf.equal(count_eos, 0), 'int32'), axis=1)
    return lengths


def infer_mask(seq, eos=1, dtype=tf.bool):
    """
    compute mask
    :param seq: tf matrix [time,batch] if time_major else [batch,time]
    :param eos: integer index of end-of-sentence token
    :returns: mask, matrix of same shape as seq and of given dtype (bool by default)
    """
    return tf.sequence_mask(infer_length(seq, eos), dtype=dtype, maxlen=tf.shape(seq)[1])


def dropout(x, keep_prob, *args, **kwargs):
    if keep_prob >= 1:
        return x
    return tf.nn.dropout(x, keep_prob, *args, **kwargs)


def clip_by_norm(t, clip_norm):
    """
    Sparse-aware tf.clip_by_norm().
    """
    if isinstance(t, tf.IndexedSlices):
        values = tf.convert_to_tensor(t.values)
    else:
        values = tf.convert_to_tensor(t)

    # Do the job.
    values = tf.clip_by_norm(values, clip_norm)

    if isinstance(t, tf.IndexedSlices):
        return tf.IndexedSlices(values, t.indices, t.dense_shape)
    return values


def clip_by_value(t, clip_value_min, clip_value_max):
    """
    Sparse-aware tf.clip_by_value().
    """
    if isinstance(t, tf.IndexedSlices):
        values = tf.convert_to_tensor(t.values)
    else:
        values = tf.convert_to_tensor(t)

    # Do the job.
    values = tf.clip_by_value(values, clip_value_min, clip_value_max)

    if isinstance(t, tf.IndexedSlices):
        return tf.IndexedSlices(values, t.indices, t.dense_shape)
    return values


def group(*ops):
    """
    Like tf.group(), but returns tf.constant(0) instead of tf.no_op(),
    which makes it suitable for use in tf.cond().
    """
    with tf.control_dependencies(ops):
        return tf.constant(0)


def select_values_over_last_axis(values, indices):
    """
    Auxiliary function to select logits corresponding to chosen tokens.
    :param values: logits for all actions: float32[batch,tick,action]
    :param indices: action ids int32[batch,tick]
    :returns: values selected for the given actions: float[batch,tick]
    """
    assert values.shape.ndims == 3 and indices.shape.ndims == 2
    batch_size, seq_len = tf.shape(indices)[0], tf.shape(indices)[1]

    time_i, batch_i = tf.meshgrid(tf.range(0, seq_len, dtype=indices.dtype),
                                  tf.range(0, batch_size, dtype=indices.dtype))

    indices_nd = tf.stack([batch_i, time_i, indices], axis=-1)

    return tf.gather_nd(values, indices_nd)


def nop(x):
    return x


def make_sinusoid_signal(position, num_channels, min_timescale=1.0, max_timescale=1e4):
    """
    Generate sinusoid timing signal
    :param position: tensor of position values
    :return: a tensor of the shape tf.shape(positon) + [num_channels]
    """
    num_timescales = num_channels // 2
    log_timescale_increment = (
        math.log(float(max_timescale) / float(min_timescale)) /
        (tf.to_float(num_timescales) - 1))
    inv_timescales = min_timescale * tf.exp(
        tf.to_float(tf.range(num_timescales)) * -log_timescale_increment)

    # scaled_time: tf.shape(positon) + [num_timescales]
    scaled_time = tf.expand_dims(position, -1) * tf.reshape(inv_timescales, [1] * position.shape.ndims + [-1])
    signal = tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=-1)
    signal = tf.pad(signal, [[0, 0]] * position.shape.ndims + [[0, tf.mod(num_channels, 2)]])
    return signal


def make_transformer_timing_signal(inp, min_timescale=1.0, max_timescale=1e4, offset=0, inp_reverse=None):
    """
    Generate timing signal like described in transformer article
    :param inp: (batch_size * ninp * hid_dim)
    :param offset: add this number to all character positions.
        if offset == 'random', picks this number uniformly from [-32000,32000] integers
    :type offset: number, tf.Tensor or 'random'
    """
    with tf.name_scope("timing_signal"):
        ninp = tf.shape(inp)[1]
        hid_size = tf.shape(inp)[2]

        position = tf.to_float(tf.range(ninp))[None, :]

        if offset == 'random':
            BIG_LEN = 32000
            offset = tf.random_uniform(tf.shape(position), minval=-BIG_LEN, maxval=BIG_LEN, dtype=tf.int32)

        # force broadcasting over batch axis
        if isinstance(offset * 1, tf.Tensor):  # multiply by 1 to also select variables, special generators, etc.
            assert offset.shape.ndims in (0, 1, 2)
            new_shape = [tf.shape(offset)[i] for i in range(offset.shape.ndims)]
            new_shape += [1] * (2 - len(new_shape))
            offset = tf.reshape(offset, new_shape)

        position += tf.to_float(offset)

        if inp_reverse is not None:
            position = tf.multiply(
                position,
                tf.where(
                    tf.equal(inp_reverse, 0),
                    tf.ones_like(inp_reverse, dtype=tf.float32),
                    -1.0 * tf.ones_like(inp_reverse, dtype=tf.float32)
                )[:, None, None]  # (batch_size * ninp * dim)
            )

        return make_sinusoid_signal(position, hid_size, min_timescale=min_timescale, max_timescale=max_timescale)


def make_2d_timing_signal(height, width, dim):
    position = tf.to_float(tf.range(width))[None, :]  # [1, width]
    signal = tf.tile(make_sinusoid_signal(position, dim // 2), [height, 1, 1])
    position = tf.to_float(tf.range(height))[:, None]  # [height, 1]
    signal = tf.concat([signal, tf.tile(make_sinusoid_signal(position, dim // 2), [1, width, 1])], axis=-1)
    return signal


def word_dropout(inp, inp_len, voc, dropout=0, method='unk', keep_first=False):
    """
    one function to rule all word dropout methods
    :param inp: tf tensor [batch, ninp]
    :param voc: tfnn vocabulary
    :param dropout: probability of dropping, float scalar in [0, 1]
    :param method: one of several supported methods
        'unk' - replace with voc.unk
        'random_word' - replace with random word except eos or unk
    :return: matrix of same shape as inp with dropout applied to it
    """
    if dropout == 0:
        return inp

    inp_shape = tf.shape(inp)
    border = tf.fill([inp_shape[0], 1], False)
    mask = tf.sequence_mask(inp_len - 1 - keep_first, inp_shape[1] - 1 - keep_first)
    if not keep_first:
        mask = tf.concat((mask, border), axis=1)
    else:
        mask = tf.concat((tf.tile(border, [int(keep_first), 1]), mask, border), axis=1)

    mask = tf.logical_and(mask, tf.random_uniform(inp_shape) < dropout)

    if method == 'unk':
        replacement = tf.fill(inp_shape, tf.cast(voc._unk, inp.dtype))
    elif method == 'random_word':
        replacement = tf.random_uniform(inp_shape, minval=max(voc.eos, voc.unk) + 1,
                                        maxval=len(voc), dtype=inp.dtype)
    else:
        raise ValueError("Unknown word dropout method: %r" % method)
    return tf.where(mask, replacement, inp)


def log_softmax_nd(logits, axes=(-1,)):
    """ log-softmax over several axes """
    logits -= tf.reduce_max(logits, axis=axes, keepdims=True)
    return logits - tf.reduce_logsumexp(logits, axis=axes, keepdims=True)


def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return tuple(x.name for x in local_device_protos if x.device_type == 'GPU')