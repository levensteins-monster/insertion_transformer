import numpy as np
import tensorflow as tf


def save(filename, vars):
    """
    Save specified variables to an NPZ archive.
    """
    p = {}
    for var in vars:
        p[var.name] = var
    keys = list(p.keys())
    values = tf.get_default_session().run([p[k] for k in keys])
    p = dict(zip(keys, values))
    np.savez(filename, **p)


def load(filename, vars, batch_size=10, sess=None):
    """
    Load NPZ archive into specified variables.

    If variable we want to load is not in NPZ, we ignore it.
    If NPZ has a value for a variable that is not in 'vars' list, we ignore it.
    """
    p = np.load(filename)
    ops = []
    feed_dict = {}

    with tf.variable_scope('load'):
        for var in vars:
            if var.name not in p:
                continue
            placeholder = tf.placeholder(var.dtype)
            feed_dict[placeholder] = p[var.name]
            ops.append(tf.assign(var, placeholder, validate_shape=False).op)

    if ops:
        for ofs in range(0, len(ops), batch_size):
            (sess or tf.get_default_session()).run(ops[ofs: ofs + batch_size], feed_dict)


def get_model_variables():
    """
    Return set of variables in tf.GraphKeys.MODEL_VARIABLES collection.
    """
    g = tf.get_default_graph()
    vars = set(g.get_collection(tf.GraphKeys.MODEL_VARIABLES))
    return vars


def get_state_variables():
    """
    Return set of all tensorflow variables except tf.GraphKeys.MODEL_VARIABLES collection
    """
    vars = set(tf.global_variables())
    vars -= get_model_variables()
    return vars
