import sys
from collections import defaultdict
import numpy as np
import tensorflow as tf

import lib.ops
from lib.data import linelen, form_adaptive_batches
from lib.oracle import inserts_coo_to_tensor, batch_inserts_to_coo, GenerateReferenceInserts
from lib.inference import SampleReferenceInserts
from lib.util import nested_map, nested_flatten, get_optimized_variables, initialize_uninitialized_variables, make_symbolic_cache

class SampleBasedTrainer:
    def __init__(self, model, sess=None, optimized_variables=None,
                 name=None, verbose=False, is_train=True, initialize=True,
                 sampler_opts=None, optimizer_opts=None, grad_clip=0,
                 **kwargs
                 ):
        """
        An imperative trainer is an object that performs training on batches. Works out-of-graph (in python).
        It is hard-coded to do one thing - sample-based training - but it does that thing well.
        :type model: lib.models.Transformer
        :param sess: tf session to use. tf.get_default_session by default, create new if no default.
        """
        self.model = model
        self.name = name = name or 'trainer_' + model.name
        self.sess = sess = sess or tf.get_default_session() or tf.InteractiveSession()
        self.verbose = verbose

        with tf.name_scope(self.name), tf.variable_scope(self.name) as scope:
            optimized_variables = optimized_variables or get_optimized_variables(model, verbose)
            self.optimized_variables = optimized_variables
            self.step = tf.train.get_or_create_global_step(sess.graph)

            # gradient accumulators (for virtual batch training)
            self.accumulated_grads = [tf.Variable(tf.zeros_like(w), trainable=False, name=w.name[:-2] + '_acc')
                                      for w in optimized_variables]
            self.accumulated_num_batches = tf.Variable(tf.zeros(()), trainable=False, name='num_batches_since_update')
            
            ############
            # step 1: precompute encoder state for all unique input lines
            self.encoder_batch_ph = self.model.make_encoder_batch_ph()
            
            enc = model.encode(self.encoder_batch_ph, is_train)
            self.cached_enc_state, self.compute_enc_state = make_symbolic_cache(enc)
                        
            ############
            # step 2: path_sampler samples a batch of trajectories (sequences of inserts)
            # it also caches encoder state for efficiency
            self.path_sampler = SampleReferenceInserts(model, **(sampler_opts or {}), enc_state=self.cached_enc_state)
            self.cached_enc_state = nested_map(tf.stop_gradient, self.cached_enc_state)
            self.cached_grad_wrt_enc = nested_map(lambda v: tf.Variable(tf.zeros([]), validate_shape=False,
                                                                        trainable=False,
                                                                        name=v.name[:-2] + '_cached_grad'),
                                                  self.cached_enc_state)

            self.reset_cached_grad_wrt_enc = nested_map(lambda acc, tensor: tf.assign(acc, tf.zeros_like(tensor),
                                                                                      validate_shape=False),
                                                        self.cached_grad_wrt_enc, self.cached_enc_state)
            self.fetch_before_batch = tf.group([self.reset_cached_grad_wrt_enc])
            ############
            # step 3: a trajectory is split into slices (for memory efficiency),
            # for each slice we compute dL/d_w_dec and dL/d_enc_state
            self.slice_ph = {
                'out': tf.placeholder('int32', [None, None]),
                'out_len': tf.placeholder('int32', [None]),
                'out_to_inp_indices': tf.placeholder('int32', [None]),
                'ref_len': tf.placeholder('int32', [None]),
                'ref_inserts': tf.placeholder('int64', [None, 3]),
                'chosen_inserts': tf.placeholder('int64', [None, 3]),
            }
            loss_on_slice, counters_on_slice = self.get_loss_and_counters(
                self.slice_ph, self.cached_enc_state, is_train=is_train,
                **kwargs
            )

            flat_enc_keys = sorted(self.cached_enc_state.keys())
            flat_enc_cache = list(self.cached_enc_state[k] for k in flat_enc_keys)
            flat_accumulated_grad_wrt_enc = [self.cached_grad_wrt_enc[k] for k in flat_enc_keys]

            loss_grads_on_slice = tf.gradients(loss_on_slice, optimized_variables + flat_enc_cache)
            weight_and_enc_grad_accumulators = self.accumulated_grads + flat_accumulated_grad_wrt_enc
            self.update_grads_on_slice = [
                tf.assign_add(grad_acc, grad)
                for grad_acc, grad in zip(weight_and_enc_grad_accumulators, loss_grads_on_slice)
                if grad is not None
            ]
            # ^-- sess.run-ning this will update gradients w.r.t. decoder weights and encoder state

            # accumulators for metrics
            self.accumulated_counters = nested_map(lambda v: tf.Variable(tf.zeros(v.shape, v.dtype), trainable=False),
                                                   counters_on_slice)
            self.update_counters_on_slice = nested_map(tf.assign_add, self.accumulated_counters, counters_on_slice)
            self.fetch_on_slice = tf.group([self.update_grads_on_slice, self.update_counters_on_slice])

            ############
            # step 4: once we're finished with all slices in one batch, it's time we compute the remaining gradients
            # dL/d_w_enc = dL/d_enc_state * d_enc_state/d_w_enc
            
            encoder_state = model.encode(self.encoder_batch_ph, is_train=is_train)
            flat_encoder_state = [encoder_state[k] for k in flat_enc_keys]
            loss_grads_after_slice = tf.gradients(flat_encoder_state, optimized_variables,
                                                  grad_ys=flat_accumulated_grad_wrt_enc)
            self.update_grads_after_batch = [
                tf.assign_add(grad_acc, grad)
                for grad_acc, grad in zip(self.accumulated_grads, loss_grads_after_slice)
                if grad is not None
            ]

            self.fetch_after_batch = tf.group([
                self.update_grads_after_batch,
                tf.assign_add(self.accumulated_num_batches, 1)
            ])

            ############
            # step 5: after one or several batches, we use the accumulated gradients to perform optimization step,
            # compute metrics for summary and then reset all buffers

            with tf.control_dependencies([tf.assert_positive(self.accumulated_num_batches,
                                                             message='Accumulate gradients over at least one '
                                                                     'full batch before averaging them')]):
                loss_denominator = self.get_denominator(self.accumulated_counters)
                self.grads_avg = [grad_acc / loss_denominator for grad_acc in self.accumulated_grads]

            self.opt = self.get_optimizer(self.step, **(optimizer_opts or {}))

            if grad_clip:
                grads, self.grads_global_norm = tf.clip_by_global_norm(self.grads_avg, grad_clip)
            else:
                grads, self.grads_global_norm = self.grads_avg, tf.global_norm(self.grads_avg)

            self.apply_gradients = tf.group(self.opt.apply_gradients(zip(grads, optimized_variables),
                                                                     global_step=self.step))
            self.reset_gradients = tf.group(
                tf.variables_initializer(self.accumulated_grads + [self.accumulated_num_batches]))

            self.compute_metrics = self.aggregate_metrics_from_counters(self.accumulated_counters)
            self.reset_counters = tf.variables_initializer(list(nested_flatten(self.accumulated_counters)))

            if initialize:
                sess.run([self.reset_gradients, self.reset_counters, tf.assign(self.step, 1)])
                remaining_utility_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope.name)
                initialize_uninitialized_variables(sess=sess, var_list=remaining_utility_variables)


    def train_on_batch(self, batch, slice_max_len=None, optimizer_step=True, reset_counters=None):
        """
        Accumulates gradients and counters,
        :param batch: a list of pairs [(inp_line, out_line), ...]
        :param slice_max_len: maximum length of a single slice of hypotheses (in tokens)
        :return: total loss
        """
        assert optimizer_step or not reset_counters, "do not reset counters if you don't optimize." \
                                                     "Counters contain statistics used for apply_gradients"
        sess, model = self.sess, self.model

        batch = list(batch)
        reference_lengths = [linelen(ref) for _, ref in batch]

        # step 1 save encoder state to tf variables
        enc_feed = self.model.make_feed_dict(batch)
        sess.run(self.compute_enc_state, {self.encoder_batch_ph[k]: enc_feed[k] for k in self.encoder_batch_ph})

        # step 2: sample insert trajectories, cache encoder state
        batch_trajectories = list(self.path_sampler.generate_trajectories(batch, sess))

        sess.run(self.fetch_before_batch)

        # step 3: process hypos with decoder, accumulate gradients at encoder
        # 3.1 split data into slices

        if slice_max_len is None:
            slices = [batch_trajectories]
        else:
            slices = form_adaptive_batches(batch_trajectories,
                                           slice_max_len,
                                           cost_func=lambda row: linelen(row['hypo']))

        # 3.2 process hypos one slice at a time
        for slice in slices:

            slice_feed = {key: [row[key] for row in slice] for key in slice[0].keys()}

            slice_feed['out_len'] = [linelen(hypo) for hypo in slice_feed['hypo']]
            slice_feed['out'] = model.out_voc.to_matrix(slice_feed.pop('hypo'))
            slice_feed['ref_inserts'] = batch_inserts_to_coo(slice_feed['ref_inserts'], model.out_voc)
            slice_feed['chosen_inserts'] = batch_inserts_to_coo(slice_feed['chosen_inserts'], model.out_voc)
            slice_feed['ref_len'] = [reference_lengths[i] for i in slice_feed['out_to_inp_indices']]

            sess.run(self.fetch_on_slice, {self.slice_ph[k]: slice_feed[k] for k in self.slice_ph})

        # step 4. compute remaining gradients through encoder
        encoder_feed = self.model.make_feed_dict(batch)

        sess.run(self.fetch_after_batch,
                 {self.encoder_batch_ph[k]: encoder_feed[k] for k in self.encoder_batch_ph})

        metrics = sess.run(self.compute_metrics)

        if optimizer_step:
            sess.run(self.apply_gradients)
            sess.run(self.reset_gradients)
        if reset_counters is None:
            reset_counters = optimizer_step
        if reset_counters:
            sess.run(self.reset_counters)

        return metrics

    def get_loss_and_counters(self, batch_ph, cached_enc_state, is_train,
                              eos_coeff=None, entropy_reg=0.0, loss_use_logp_any_ref=True):
        # encode with cached enc state
        enc_batch_size = tf.shape(cached_enc_state['out'])[0]
        with tf.control_dependencies([tf.assert_equal(tf.shape(tensor)[0], enc_batch_size)
                                      for tensor in nested_flatten(cached_enc_state)]):
            enc_reordered = {k: tf.gather(v, batch_ph['out_to_inp_indices'])
                             for k, v in cached_enc_state.items()}

        logp = self.model.compute_action_logprobs(batch_ph, is_train=is_train, enc=enc_reordered)
        insert_logprobas = logp['insert']  # [batch, nout, voc_size]
        finish_logprobas = logp['finish']  # [batch]

        # get reference inserts
        is_ref_insert = inserts_coo_to_tensor(batch_ph['ref_inserts'],
                                              tf.shape(batch_ph['out']),
                                              len(self.model.out_voc))
        is_chosen_insert = inserts_coo_to_tensor(batch_ph['chosen_inserts'],
                                                 tf.shape(batch_ph['out']),
                                                 len(self.model.out_voc))

        # compute log-probability of any reference insert
        neg_inf_like_logp = tf.fill(tf.shape(insert_logprobas), -1e9)
        ref_logp = tf.where(is_ref_insert, insert_logprobas, neg_inf_like_logp)
        chosen_logp = tf.where(is_chosen_insert, insert_logprobas, neg_inf_like_logp)

        logp_ref_inserts = tf.reduce_logsumexp(ref_logp if loss_use_logp_any_ref else chosen_logp, axis=(1, 2))
        # ^-- [batch_size]

        should_finish = tf.reduce_any(is_ref_insert[:, :, self.model.out_voc.eos], axis=-1)

        xent_values = -tf.where(should_finish, finish_logprobas, logp_ref_inserts)
        # ^-- [batch_size]

        # reweighting
        if eos_coeff is None:
            xent_numerator = tf.reduce_sum(xent_values)
        else:
            samples_per_line = tf.to_float(batch_ph['ref_len'])
            weights = tf.where(should_finish,
                               eos_coeff * samples_per_line,
                               (1.0 - eos_coeff) * samples_per_line / (samples_per_line - 1.0))
            # ^-- [batch_size]
            xent_numerator = tf.reduce_sum(xent_values * weights)

        batch_size = tf.shape(insert_logprobas)[0]
        counters = dict(
            batch_size=tf.to_float(batch_size),
            xent_numerator=xent_numerator,
        )

        # assemble loss (crossentropy with some extra steps)
        loss_numerator = xent_numerator

        if entropy_reg != 0.0:
            insert_probas = tf.exp(insert_logprobas)  # [batch_size, nout, voc_size]
            insert_p_logp_sum = tf.reduce_sum(insert_probas * insert_logprobas, axis=2)  # [batch_size, nout]

            mask = lib.ops.infer_mask(batch_ph['out'], self.model.out_voc.eos, dtype=tf.float32)  # [batch_size, nout]
            insert_p_logp_sum = tf.reduce_sum(insert_p_logp_sum * mask, axis=1)  # [batch_size]

            finish_p_logp_sum = finish_logprobas * tf.exp(finish_logprobas)  # [batch_size]

            entropy_values = - finish_p_logp_sum - insert_p_logp_sum  # [batch_size]
            entropy_numerator = tf.reduce_sum(entropy_values)

            loss_numerator -= entropy_reg * entropy_numerator
            counters.update(entropy_numerator=entropy_numerator)

        # metrics
        p_correct_numerator = tf.reduce_sum(tf.exp(-xent_values))
        argmax_flat = tf.argmax(tf.reshape(insert_logprobas, [batch_size, -1]), axis=-1)
        is_argmax_correct = tf.gather_nd(tf.reshape(is_ref_insert, [batch_size, -1]),
                                         tf.stack([tf.range(batch_size), tf.to_int32(argmax_flat)], -1))

        is_argmax_correct = tf.where(should_finish, tf.exp(finish_logprobas) >= 0.5, is_argmax_correct)

        acc_numerator = tf.reduce_sum(tf.to_float(is_argmax_correct))
        counters.update(
            loss_numerator=loss_numerator,
            acc_numerator=acc_numerator,
            p_correct_numerator=p_correct_numerator,
        )

        return loss_numerator, counters

    def aggregate_metrics_from_counters(self, counters, numerator_suffix='_numerator'):
        """ Compute any utility metrics given accumulated counters from self.get_loss_and_counters(...)[-1]"""
        results = {
            key[:-len(numerator_suffix)]: counters[key] / self.get_denominator(counters)
            for key in counters if key.endswith(numerator_suffix)
        }
        results['grad_norm'] = self.grads_global_norm
        results['step'] = self.step
        if hasattr(self.opt, '_lr'):
            results['learning_rate'] = tf.identity(self.opt._lr)
        return results

    def get_denominator(self, accumulated_counters):
        """ return total batch size as loss denominator """
        return accumulated_counters['batch_size']

    def get_optimizer(self, step, base_lr=1e-4, warmup_time=None, **kwargs):
        if self.verbose:
            if len(kwargs):
                print("OPTIMIZER OPTS:", kwargs)
            print("base_lr={}, warmup_time={}".format(base_lr, warmup_time))
        step = tf.to_float(step)
        learning_rate = base_lr
        if warmup_time is not None:
            learning_rate *= tf.minimum(
                                tf.to_float(step + 1) ** -0.5 * warmup_time ** 0.5,
                                tf.to_float(step + 1) / warmup_time)

        return tf.contrib.opt.LazyAdamOptimizer(learning_rate, **kwargs)


class FixedOrderTrainer(SampleBasedTrainer):
    def __init__(self, *args, mode='random', sampler_opts=None, **kwargs):
        """
        An imperative trainer is an object that performs training on batches. Works out-of-graph (in python).
        It is hard-coded to do one thing - sample-based training - but it does that thing well.
        :type model: lib.models.Transformer
        :param sess: tf session to use. tf.get_default_session by default, create new if no default.
        """
        super().__init__(*args, **kwargs)
        # Don't pass sampler_opts as we change it later any way

        self.path_sampler = GenerateReferenceInserts(self.model.out_voc, mode=mode, **(sampler_opts or {}))

    def get_loss_and_counters(self, batch_ph, cached_enc_state, is_train, loss_use_logp_chosen=False, eos_coeff=None, **kwargs):

        # encode with cached enc state
        enc_batch_size = tf.shape(cached_enc_state['out'])[0]
        with tf.control_dependencies([tf.assert_equal(tf.shape(tensor)[0], enc_batch_size)
                                      for tensor in nested_flatten(cached_enc_state)]):
            enc_reordered = {k: tf.gather(v, batch_ph['out_to_inp_indices'])
                             for k, v in cached_enc_state.items()}

        logp = self.model.compute_action_logprobs(batch_ph, is_train=is_train, enc=enc_reordered)
        insert_logprobas = logp['insert']  # [batch]
        finish_logprobas = logp['finish']  # [batch, nout, voc_size]

        # get reference inserts
        is_ref_insert = inserts_coo_to_tensor(batch_ph['ref_inserts'],
                                              tf.shape(batch_ph['out']),
                                              len(self.model.out_voc))
        is_chosen_insert = inserts_coo_to_tensor(batch_ph['chosen_inserts'],
                                                 tf.shape(batch_ph['out']),
                                                 len(self.model.out_voc))

        mask_correct = is_chosen_insert if loss_use_logp_chosen else is_ref_insert

        # assumes that reference inserts for ended hypo are EOS tokens and after-reference are NULL
        should_finish = tf.reduce_any(is_ref_insert[:, :, self.model.out_voc.eos], axis=-1)

        logp_ref = tf.einsum("btl,btl->b", insert_logprobas, tf.to_float(mask_correct))
        # equivalent to tf.reduce_sum(insert_logprobas * mask_correct, (1, 2)), but without tmp tensor

        xent_values = logp_ref / (tf.reduce_sum(tf.to_float(mask_correct), (-2, -1)) + 1e-5)
        # logp_ref is divided by number of correct labels to properly compute xent

        xent_values = -tf.where(should_finish,
                                finish_logprobas,
                                xent_values)
        # ^-- [batch_size]

        if eos_coeff is None:
            xent_numerator = tf.reduce_sum(xent_values)
        else:
            samples_per_line = tf.to_float(batch_ph['ref_len'])
            weights = tf.where(should_finish,
                               eos_coeff * samples_per_line,
                               (1.0 - eos_coeff) * samples_per_line / (samples_per_line - 1.0))
            # ^-- [batch_size]
            xent_numerator = tf.reduce_sum(xent_values * weights)


        batch_size = tf.shape(insert_logprobas)[0]
        counters = dict(
            batch_size=tf.to_float(batch_size),
            xent_numerator=xent_numerator,
        )

        # assemble loss (crossentropy)
        loss_numerator = xent_numerator

        # metrics
        p_correct_numerator = tf.reduce_sum(tf.exp(logp_ref))
        argmax_flat = tf.argmax(tf.reshape(insert_logprobas, [batch_size, -1]), axis=-1)
        is_argmax_correct = tf.gather_nd(tf.reshape(is_ref_insert, [batch_size, -1]),
                                         tf.stack([tf.range(batch_size), tf.to_int32(argmax_flat)], -1))
        is_argmax_correct = tf.where(should_finish, tf.exp(finish_logprobas) >= 0.5, is_argmax_correct)


        acc_numerator = tf.reduce_sum(tf.to_float(is_argmax_correct))
        counters.update(
            loss_numerator=loss_numerator,
            acc_numerator=acc_numerator,
            p_correct_numerator=p_correct_numerator,
        )

        return loss_numerator, counters
