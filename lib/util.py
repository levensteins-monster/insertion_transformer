# Utilities                                                              TFNN #
import numpy as np
import tensorflow as tf
import contextlib
import sys

IMPLICIT_UPDATES = 'IMPLICIT_UPDATES'


def nested_compare(t, u):
    """
    Return whether nested structure of t1 and t2 matches.
    """
    if isinstance(t, (list, tuple)):
        if not isinstance(u, type(t)):
            return False
        if len(t) != len(u):
            return False
        for a, b in zip(t, u):
            if not nested_compare(a, b):
                return False
        return True

    if isinstance(t, dict):
        if not isinstance(u, dict):
            return False
        if set(t.keys()) != set(u.keys()):
            return False
        for k in t:
            if not nested_compare(t[k], u[k]):
                return False
        return True

    else:
        return True


def nested_flatten(t):
    """
    Turn nested list/tuple/dict into a flat iterator.
    """
    if isinstance(t, (list, tuple)):
        for x in t:
            yield from nested_flatten(x)
    elif isinstance(t, dict):
        for k, v in sorted(t.items()):
            yield from nested_flatten(v)
    else:
        yield t


def nested_pack(flat, structure):
    return _nested_pack(iter(flat), structure)


def _nested_pack(flat_iter, structure):
    if is_namedtuple(structure):
        return type(structure)(*[
            _nested_pack(flat_iter, x)
            for x in structure]
                               )
    if isinstance(structure, (list, tuple)):
        return type(structure)(
            _nested_pack(flat_iter, x)
            for x in structure
        )
    elif isinstance(structure, dict):
        return {
            k: _nested_pack(flat_iter, v)
            for k, v in sorted(structure.items())
        }
    else:
        return next(flat_iter)


def is_namedtuple(x):
    """Checks if x is a namedtuple instance. Taken from https://stackoverflow.com/a/2166841 ."""
    t = type(x)
    b = t.__bases__
    if len(b) != 1 or b[0] != tuple: return False
    f = getattr(t, '_fields', None)
    if not isinstance(f, tuple): return False
    return all(type(n) == str for n in f)


def is_iterable(x):
    """Checks if x is iterable"""
    try:
        iter(x)
        return True
    except TypeError as te:
        return False


def nested_map(fn, *t):
    # Check arguments.
    if not t:
        raise ValueError('Expected 2+ arguments, got 1')
    for i in range(1, len(t)):
        if not nested_compare(t[0], t[i]):
            msg = 'Nested structure of %r and %r differs'
            raise ValueError(msg % (t[0], t[i]))

    # Map.
    flat = map(nested_flatten, t)
    return nested_pack(map(fn, *flat), t[0])


def merge_dicts(a, b):
    res = a.copy()
    res.update(b)
    return res


def map_first(fn, res):
    """
    Apply some transformation to first element of list or tuple.
    It's used to postprocess output of layer, where first element is main output
    and all other are some auxiliary outputs
    """
    if isinstance(res, (list, tuple)):
        assert len(res) > 0
        return (fn(res[0]),) + tuple(res[1:])
    else:
        return fn(res)


def is_scalar(var):
    """ checks if var is not scalar. Works for list, np.array, tf.tensor and many similar classes """
    return len(np.shape(var)) == 0


@contextlib.contextmanager
def nop_ctx():
    yield


def log_add_const(logp, logc=np.log(1e-3)):
    """ computes log(p + c) given log(p) and log(c)"""
    return logp + np.logaddexp(0, logc - logp)


def merge_summaries(inputs, collections=None, name=None):
    # Wrapper correctly working with inputs = []
    if len(inputs) == 0:
        # We should return simple tf operation that returns empty bytes
        return tf.identity(b'')
    return tf.summary.merge(inputs, collections, name)


def make_batch_placeholder(batch_data):
    batch_placeholder = {
        k: tf.placeholder(v.dtype, [None]*len(v.shape))
        for k, v in batch_data.items()}
    return batch_placeholder


def initialize_uninitialized_variables(sess=None, var_list=None):
    with tf.name_scope("initialize"):
        sess = sess or tf.get_default_session() or tf.InteractiveSession()
        uninitialized_names = set(sess.run(tf.report_uninitialized_variables(var_list)))
        uninitialized_vars = []
        for var in tf.global_variables():
            if var.name[:-2].encode() in uninitialized_names:
                uninitialized_vars.append(var)

        sess.run(tf.variables_initializer(uninitialized_vars))


def get_optimized_variables(model, verbose=True):
    """ returns a list of trainable variables that actually have nonzero gradients """

    batch_ph = make_batch_placeholder(model.make_feed_dict(model._get_batch_sample()))
    logp = model.compute_action_logprobs(batch_ph, is_train=True)
    out = sum(map(tf.reduce_sum, logp.values()))
    all_vars = tf.trainable_variables()
    vars_and_grads = list(zip(all_vars, tf.gradients(out, all_vars)))
    used_vars = [var for var, grad in vars_and_grads if grad is not None]

    if verbose and len(used_vars) != len(all_vars):
        print("Not all trainable variables will be optimized.")
        print("UNUSED_VARS:", [var.name for var, grad in vars_and_grads if grad is None], file=sys.stderr)
        print("OPTIMIZED_VARS:", [var.name for var, grad in vars_and_grads if grad is not None], file=sys.stderr)
    return used_vars


def make_symbolic_cache(values):
    """
    Generates symbolic cache for a dict of tensors
    
    returns cached state and an op to update the cache
    """
    cached_state = {}
    for key, value in values.items():
        cached_state[key] = tf.Variable(tf.zeros([], value.dtype), validate_shape=False,
                                                 trainable=False, name=value.name[:-2]+'_cached')
        cached_state[key].set_shape(value.shape)

    compute_state = list(nested_flatten(nested_map(
        lambda var, val: tf.assign(var, val, validate_shape=False),
        cached_state, values)))
    return cached_state, compute_state
