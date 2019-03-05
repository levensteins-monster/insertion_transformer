from itertools import count

import numpy as np
import tensorflow as tf

import lib.ops
from lib.oracle import get_optimal_inserts, inserts_coo_to_tensor, batch_inserts_to_coo
from lib.util import nested_flatten, nested_map, make_batch_placeholder


class BeamSearchInserts:
    def __init__(self, model, is_train=False):
        """
        An object that finds most likely sequence of inserts
        :type model: lib.models.Transformer
        """

        self.model = model
        self.batch_ph = make_batch_placeholder(model.make_feed_dict(model._get_batch_sample()))

        self.k_best = tf.placeholder('int32', [])
        self.hypo_base_logprobs = tf.placeholder('float32', [None])  # [batch_size]

        # a mask of allowed tokens
        not_special = np.array([(i not in model.out_voc._default_token_ix)
                                for i in range(len(model.out_voc))],
                               dtype=np.bool)
        self.allowed_tokens = tf.placeholder_with_default(not_special, shape=[len(model.out_voc)],)
        # ^-- [voc_size]

        # step 1: precompute encoder outputs for all unique input lines
        enc = model.encode(self.batch_ph, is_train)
        self.cached_enc_state = {}
        for key, value in enc.items():
            self.cached_enc_state[key] = tf.Variable(tf.zeros([], value.dtype), validate_shape=False,
                                                     trainable=False, name=value.name[:-2] + '_cached')
            self.cached_enc_state[key].set_shape(value.shape)

        self.compute_enc_state = list(nested_flatten(nested_map(
            lambda var, val: tf.assign(var, val, validate_shape=False),
            self.cached_enc_state, enc)))

        # step 2: assemble decoder outputs for each input
        # there may be several out hypos for the same inp line. Up to beam_size hypos to be exact.
        out_to_inp_ix = tf.zeros([tf.shape(self.batch_ph['out'])[0]], dtype=tf.int64)
        enc_reordered = {k: tf.gather(v, out_to_inp_ix) for k, v in self.cached_enc_state.items()}

        # step 3: compute logits and action log-probs for inserting tokens and finishing
        logp = model.compute_action_logprobs(self.batch_ph, is_train, enc=enc_reordered)

        ###################
        # insert operation
        hypo_logprobs_insert = self.hypo_base_logprobs[:, None, None] + logp['insert']
        # ^-- [batch, position, token]
        hypo_logprobs_insert -= 1e9 * tf.to_float(tf.logical_not(self.allowed_tokens))
        best_inserts_flat = tf.nn.top_k(tf.reshape(hypo_logprobs_insert, [-1]), k=self.k_best, sorted=True)

        batch_size, max_len = tf.shape(self.batch_ph['out'])[0], tf.shape(self.batch_ph['out'])[1]
        voc_size = len(model.out_voc)

        best_hypo_ix = best_inserts_flat.indices // (max_len * voc_size)
        best_insert_pos = (best_inserts_flat.indices // voc_size) % max_len
        best_token_ix = best_inserts_flat.indices % voc_size
        best_insert_logp = best_inserts_flat.values
        self.insert_kbest = [best_hypo_ix, best_insert_pos, best_token_ix, best_insert_logp]

        ##################
        # eos operation
        self.finished_hypo_logprobs = self.hypo_base_logprobs + logp['finish']

    def translate_line(self, src, beam_size=32, max_steps=None, beam_spread=float('inf'),
                       sess=None, verbose=False, eos_in_beam=True):
        """
        Translates a single line using beam search
        :param src: string, space-separated source tokens
        :param beam_size: maximum number of hypotheses considered at each step
        :param beam_spread: after a full hypo is found, drops any hypotheses
            whose scores are below score(best_full_hypo_so_far) - beam_spread
        :param max_steps: terminates after this many steps
        :param eos_in_beam: if True, a hypothesis can't be terminated if its termination is not in beam top-k
        :param sess: tf session to use. tf.get_default_session by default, create new if no default.
        :param verbose: if True, prints best-in-beam hypo at each step
        :return: a dictionary { (source, translation_hypothesis) -> score }
        """
        assert max_steps is not None or beam_spread < float('inf'), "please specify termination condition"
        sess = sess or tf.get_default_session() or tf.InteractiveSession()
        finished_translation_logp = {}    # {dst -> logp(reach this hypo)}, used for pruning with beam spread
        best_finished_logp = -float('inf')
        beam, beam_logp = [''], [0.0]

        # save encoder state to tf variables
        enc_feed = self.model.make_feed_dict([(src, '')])
        sess.run(self.compute_enc_state, {self.batch_ph[k]: enc_feed[k] for k in enc_feed})

        for t in count():
            feed = self.model.make_feed_dict([(src, hypo) for hypo in beam])
            feed = {self.batch_ph[k]: feed[k] for k in feed}
            feed[self.k_best] = beam_size
            feed[self.hypo_base_logprobs] = beam_logp

            (hypo_ix, insert_at, insert_token_ix, new_logp), termination_logp = \
                sess.run([self.insert_kbest, self.finished_hypo_logprobs], feed)

            ###
            # handle terminations (eos)
            for finished_hypo, finished_hypo_logp in zip(beam, termination_logp):
                if eos_in_beam and finished_hypo_logp < min(new_logp):
                    finished_hypo_logp = finished_hypo_logp - 1e9
                # trajectory termination
                previous_logp = finished_translation_logp.get(finished_hypo, -float('inf'))
                finished_translation_logp[finished_hypo] = np.logaddexp(finished_hypo_logp, previous_logp)
                best_finished_logp = max(best_finished_logp, finished_translation_logp[finished_hypo])
                # ^-- wtf happens here: if there are several trajectories tau1, tau2 that lead to the same Y,
                # the correct way to compute logp(Y) = log(p(tau1) + p(tau2)), which is done with logaddexp

            ###
            # handle inserts
            new_beam, new_beam_logp = [], []
            for hypo_ix, insert_i, token_i, hypo_logp in zip(hypo_ix, insert_at, insert_token_ix, new_logp):
                hypo = beam[hypo_ix]
                assert token_i != self.model.out_voc.eos
                hypo_tok = hypo.split()
                hypo_tok.insert(insert_i, self.model.out_voc.words(token_i))
                hypo = ' '.join(hypo_tok)

                if hypo in new_beam:
                    # merge hypo trajectory with another trajectory in beam that has higher score
                    # (older hypo's score is better since we've sorted them from highest score to lowest)
                    prev_ix = new_beam.index(hypo)
                    new_beam_logp[prev_ix] = np.logaddexp(new_beam_logp[prev_ix], hypo_logp)

                elif hypo_logp + beam_spread >= best_finished_logp:
                    new_beam.append(hypo)
                    new_beam_logp.append(hypo_logp)
                else:
                    pass  # pruned by beam spread

            beam, beam_logp = new_beam, new_beam_logp

            if not len(beam): break
            if max_steps is not None and t >= max_steps: break
            if verbose:
                print('%f %s' % (beam_logp[0], beam[0]))

        return finished_translation_logp

    @staticmethod
    def apply_length_penalty(hypo_scores, len_alpha=1.0, base=0.0, eps=1e-6):
        """ Boosts longer hypotheses score AFTER beam search by applying length penalty, see GNMT for details """
        return {
            hypo: hypo_scores[hypo] / ((max(eps, base + len(hypo.split()))) / (base + 1)) ** len_alpha
            for hypo in hypo_scores
        }


class SampleReferenceInserts:
    def __init__(self, model, enc_state, is_train=False, greedy=False, temperature=None):
        """
        A "supervised" inference that selects one sequence of inserts out of those that lead to ref
        :type model: lib.models.Transformer
        :param is_train: if True, model is used in training mode (with dropout, etc.)
        :param greedy: if True, returns most likely tokens. False - samples proportionally to probs
        :param temperature: if given, samples proportionally to probs ^ 1. / temperature
        """

        self.model = model
        self.batch_ph = make_batch_placeholder(model.make_feed_dict(model._get_batch_sample()))
        self.batch_ph['out_to_inp_indices'] = tf.placeholder('int64', [None])  # [batch_size]
        self.batch_ph['ref_inserts'] = tf.placeholder('int64', [None, 3])

        # step 1: precomputed encoder outputs for all unique input lines
        self.cached_enc_state = enc_state

        # step 2: decode, compute log-probs and use them to choose from correct inserts
        enc_reordered = {k: tf.gather(v, self.batch_ph['out_to_inp_indices']) for k, v in self.cached_enc_state.items()}
        action_logp = model.compute_action_logprobs(self.batch_ph, is_train, enc=enc_reordered, temperature=temperature)

        is_ref_insert = inserts_coo_to_tensor(self.batch_ph['ref_inserts'],
                                                  tf.shape(self.batch_ph['out']), len(model.out_voc))
        # ^-- [batch, time, voc_size]

        masked_logprobs = tf.where(is_ref_insert,
                                   action_logp['insert'],
                                   tf.fill(tf.shape(action_logp['insert']), -float('inf')))

        logprobs_flat = tf.reshape(masked_logprobs, [tf.shape(action_logp['insert'])[0], -1])  # [batch, time*voc_size]

        if greedy:
            chosen_index_flat = tf.argmax(logprobs_flat, axis=1)
        else:
            chosen_index_flat = tf.multinomial(logprobs_flat, num_samples=1)[:, 0]

        chosen_pos = chosen_index_flat // len(model.out_voc)
        chosen_token_ix = chosen_index_flat % len(model.out_voc)
        self.chosen_inserts = [chosen_pos, chosen_token_ix]
        chosen_insert_logprobs = tf.gather_nd(logprobs_flat, tf.stack(
            [tf.range(0, tf.to_int64(tf.shape(chosen_index_flat)[0]), dtype=chosen_index_flat.dtype),
             chosen_index_flat], axis=-1)
        )

        is_terminated = tf.equal(chosen_token_ix, model.out_voc.eos)
        self.chosen_insert_logprobs = tf.where(is_terminated, action_logp['finish'], chosen_insert_logprobs)

        logp_any_ref = tf.reduce_logsumexp(logprobs_flat, axis=-1)  # [batch_size]
        self.logp_any_ref = tf.where(is_terminated, action_logp['finish'], logp_any_ref)

    def generate_trajectories(self, batch, sess=None):
        """
        Samples trajectories that start at empty hypothesis and end on reference lines
        :param batch: a sequence of pairs[(inp_line, ref_out_line)]
        :param sess: tf session to use. tf.get_default_session by default, create new if no default.
        :return: a sequence of dicts {inp_line, hypo, out_to_inp_index, ref_inserts, chosen_inserts, ...}
        """
        sess = sess or tf.get_default_session() or tf.InteractiveSession()
        inp_lines, ref_lines = zip(*batch)
        hypos_tok = [list() for _ in ref_lines]
        out_voc = self.model.out_voc
        hypos_ref_tok = [out_voc.words(out_voc.ids(ref_line.split())) for ref_line in ref_lines]
        hypos_to_indices = list(range(len(inp_lines)))
        # ^-- inp_line index for each hypo in hypo_tok. hypos for shorter inp_lines will terminate earlier.

        while len(hypos_tok):
            # step 1, find all allowed inserts
            ref_inserts = []
            for hypo, ref in zip(hypos_tok, hypos_ref_tok):
                if hypo == ref:
                    ref_inserts.append([{self.model.out_voc.EOS} for _ in hypo])
                else:
                    ref_inserts.append(get_optimal_inserts(hypo, ref))

            # step 2, run model to choose inserts
            hypo_lines = [' '.join(hypo_tok) for hypo_tok in hypos_tok]
            hypo_inp_lines = [inp_lines[i] for i in hypos_to_indices]

            sample_inp = self.model._get_batch_sample()[0]
            feed = self.model.make_feed_dict([(*sample_inp[:1], hypo, *sample_inp[2:]) for hypo in hypo_lines])
            del feed['inp']
            if 'inp_len' in feed:  # we don't need inp cuz we use pre-computed enc state
                del feed['inp_len']
            feed['out_to_inp_indices'] = hypos_to_indices
            feed['ref_inserts'] = batch_inserts_to_coo(ref_inserts, self.model.out_voc)

            (chosen_pos, chosen_token_ix), chosen_insert_logp, logp_any_ref = \
                sess.run((self.chosen_inserts, self.chosen_insert_logprobs, self.logp_any_ref),
                         {self.batch_ph[k]: feed[k] for k in feed})

            # assemble chosen inserts
            chosen_inserts = []
            for hypo, pos, token_ix in zip(hypos_tok, chosen_pos, chosen_token_ix):
                oh_chosen = [set() for _ in range(len(hypo) + 1)]
                oh_chosen[pos].add(self.model.out_voc.words(token_ix))
                chosen_inserts.append(oh_chosen)

            # step 3, give away results after one step
            yield from [
                dict(step=len(hypos_tok[0]), inp_line=inp_line, hypo=hypo, ref_inserts=ref_ins,
                     chosen_inserts=chosen_ins, chosen_insert_logp=logp, logp_any_ref=any_logp,
                     out_to_inp_indices=inp_id)
                for inp_line, inp_id, hypo, ref_ins, chosen_ins, logp, any_logp in
                zip(hypo_inp_lines, hypos_to_indices, hypo_lines, ref_inserts, chosen_inserts,
                    chosen_insert_logp, logp_any_ref)
            ]

            # step 4, update hypos, form new batch
            new_hypos_tok, new_hypos_ref, new_hypos_indices = [], [], []
            for hypo, ref, line_index, pos, token_ix in zip(hypos_tok, hypos_ref_tok, hypos_to_indices,
                                                            chosen_pos, chosen_token_ix):

                if token_ix in (self.model.out_voc.eos, self.model.out_voc.null):
                    pass  # translation terminated. remove from batch.
                else:
                    hypo = list(hypo)
                    hypo.insert(pos, self.model.out_voc.words(token_ix))
                    new_hypos_tok.append(hypo)
                    new_hypos_ref.append(ref)
                    new_hypos_indices.append(line_index)

            hypos_tok, hypos_ref_tok, hypos_to_indices = new_hypos_tok, new_hypos_ref, new_hypos_indices

