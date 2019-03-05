"""
Transformer encoder / decoder layer chain
"""
import numpy as np
import tensorflow as tf

import lib.layers
from . import layers, ops
from .data import linelen


class Transformer:

    def __init__(
            self, name, inp_voc, out_voc,
            logits_bias=False, share_emb=False, dst_rand_offset=False,
            rescale_emb=True, inp_emb_bias=False, emb_inp_device='', emb_out_device='',
            **kwargs
    ):
        """
        Transformer-based model that predicts logp(insert(i, token) | x, y)
        :type inp_voc: lib.voc.Voc
        :type out_voc: lib.voc.Voc
        :param logits_bias: if True, final logits layer has bias term.
        :param share_emb: if True, input and output embeddings will use the same matrix.
            Useful for in case of shared vocabularies or when there is a
        :param dst_rand_offset: if True, adds a random offset to output embeddings, same for all positions
        :param kwargs: other hyperparameters - see TransformerChain and TransformerEmbedding
        """
        self.name = name
        self.inp_voc, self.out_voc = inp_voc, out_voc
        self.dst_rand_offset = dst_rand_offset
        self.hp = kwargs

        emb_size = kwargs.get('emb_size', kwargs.get('hid_size', 512))
        max_voc_size = max(len(inp_voc), len(out_voc))

        with tf.variable_scope(self.name) as self.scope:
            # Embeddings
            self.emb_inp = layers.TransformerEmbedding(
                'emb_inp', max_voc_size if share_emb else len(inp_voc), emb_size,
                bias=inp_emb_bias, rescale=rescale_emb, device=emb_inp_device)

            self.emb_out = layers.TransformerEmbedding(
                'emb_out', max_voc_size if share_emb else len(out_voc), emb_size,
                matrix=self.emb_inp.emb.mat if share_emb else None,
                rescale=rescale_emb, device=emb_out_device)

            # Model body
            self.encoder = layers.TransformerChain('enc', **kwargs)
            self.decoder = layers.TransformerChain('dec', attn_inputs=['enc'], **kwargs)

            # logits: token insertions plus one extra logit to predict position where to insert
            self.logits = layers.Dense(
                'logits', kwargs['hid_size'], len(out_voc) + 1,
                matrix=tf.transpose(self.emb_out.emb.mat) if kwargs.get('dwwt', False) else None,
                bias=None if logits_bias else 0
            )

    def _get_batch_sample(self):
        """ A minimal example of model input data """
        return [("i saw a cat", "i write the code")]

    def make_encoder_batch_ph(self):
        return {
            'inp': tf.placeholder('int32', [None, None]),
            'inp_len': tf.placeholder('int32', [None])
        }

    def make_feed_dict(self, batch, **kwargs):
        """ Take input data strings, return a dict { key: np.array(value) } """
        inp_lines, out_lines = zip(*batch)
        inp_len = [linelen(line) for line in inp_lines]
        out_len = [linelen(line) for line in out_lines]
        return {
            'inp': self.inp_voc.to_matrix(inp_lines),
            'inp_len': np.array(inp_len, 'int32'),
            'out': self.out_voc.to_matrix(out_lines),
            'out_len': np.array(out_len, 'int32')
        }

    def encode(self, batch, is_train):
        """ Take placeholders for data batch, return encoder state """
        with tf.name_scope(self.name), ops.dropout_scope(is_train):
            inp = batch['inp']  # [batch_size * ninp]
            inp_len = batch.get('inp_len', ops.infer_length(inp, self.inp_voc.eos))  # [batch]
            attn_mask = ops.make_attn_mask(inp, inp_len)  # [batch_size, 1, 1, ninp]
            out, _ = self.encoder(self.emb_inp(inp), self_attn_mask=attn_mask)
            # ^-- [batch_size, ninp, hid_size]
            return dict(out=out, attn_mask=attn_mask)

    def compute_action_logprobs(self, batch, is_train, enc=None, temperature=None):
        """
        Compute log-probabilities for all possible actions (aka agent policy)
        :param batch: a dict with
            - token matrix 'out'[batch_size, output_length]
            - optional length vector out_len[batch_size]
        :param is_train: whether or not to use training behavior (e.g. dropout)
        :returns: {'insert':logp(insert(i, c) | x, y), 'finish':logp(terminate| x, y)}
        """
        enc = self.encode(batch, is_train) if enc is None else enc
        with tf.name_scope(self.name), ops.dropout_scope(is_train):
            out = batch['out']  # partial translation, shape: [batch_size * nout]
            out_len = batch.get('out_len', ops.infer_length(out, self.out_voc.eos))  # [batch]

            # embedding. Note: at this point, a special "zero" vector is added
            # to the first position hence length is increased by 1

            out_padded = tf.concat([tf.zeros_like(out[:, :1]), out], axis=1)  # [batch_size, nout+1]
            dec_emb = self.emb_out(out_padded, offset='random' if self.dst_rand_offset else 0)
            # ^-- shape: [batch_size, nout + 1]

            # run decoder
            attn_mask = ops.make_attn_mask(out_padded, out_len + 1)  # [batch_size, 1, 1, nout + 1]
            dec_out, _ = self.decoder(dec_emb, self_attn_mask=attn_mask,
                                      enc_out=enc['out'], enc_attn_mask=enc['attn_mask'])
            # ^-- [batch_size, nout + 1, hid_size]

            logits = self.logits(dec_out)  # [batch_size, nout + 1, voc_size + 1]
            if temperature is not None:
                logits /= temperature

            # compute log-probabilities for actions

            # position log-probabilities, logP(insert(pos, *) | ...)
            # used to predict position of next insert and termination condition (EOS)
            position_logits = logits[:, :, -1]  # [batch_size, nout + 1]

            position_mask = tf.cast(attn_mask, tf.bool)[:, 0, 0, :]  # [batch_size, nout + 1]
            position_logits = tf.where(position_mask, position_logits,
                                       tf.fill(tf.shape(position_logits), -1e9))
            position_logp = tf.nn.log_softmax(position_logits, axis=-1)  # [batch_size, n_out]

            # two actions: insert - at any non-EOS position - or finish - defined as inserting at EOS
            finish_logp = tf.gather_nd(position_logp,
                                       tf.stack([tf.range(tf.shape(out_len)[0]), out_len], axis=1))
            # ^-- [batch_size]

            insert_position_logp = tf.where(position_mask[:, 1:], position_logp[:, :-1],
                                            tf.fill(tf.shape(position_logp[:, :-1]), -1e9))
            # ^-- [batch_size, nout]

            # insertion log-probabilities:
            # logP(insert(pos, tok) | ...) = logP(insert(pos, *) | ...) + logP(insert(pos, tok) | insert(pos, *), ...)

            token_logits = logits[:, :-1, :len(self.out_voc)]  # [batch_size, n_out, voc_size]
            token_logp_given_position = tf.nn.log_softmax(token_logits, axis=-1)
            # note: we do not need mask on token_logp_given_position cuz mask is already applied to insert_position_logp

            insert_logp = insert_position_logp[:, :, None] + token_logp_given_position

        return {
            # group 1 (exps sum to 1)
            'insert': insert_logp,  # [batch_size, nout, voc_size]
            'finish': finish_logp,  # [batch_size]
        }


class ImgToSeqTransformer(Transformer):
    def __init__(
            self, name, out_voc, inp_w, inp_h, inp_channels=3, make_encoder=lib.layers.ImageEncoder,
            logits_bias=False, share_emb=False, dst_rand_offset=False,
            rescale_emb=True, emb_out_device='',
            **kwargs
    ):
        """
        Transformer-based model that predicts logp(insert(i, token) | x, y)
        :type out_voc: lib.voc.Voc
        :param logits_bias: if True, final logits layer has bias term.
        :param dst_rand_offset: if True, adds a random offset to output embeddings, same for all positions
        :param kwargs: other hyperparameters - see TransformerChain and TransformerEmbedding
        """
        self.name = name
        self.inp_voc, self.out_voc = out_voc, out_voc  # inp voc is a stub, the same as out_voc
        self.dst_rand_offset = dst_rand_offset
        self.hp = kwargs
        self.w = inp_w
        self.h = inp_h
        self.inp_channels = inp_channels

        emb_size = kwargs.get('emb_size', kwargs.get('hid_size', 512))
        max_voc_size = len(out_voc)

        with tf.variable_scope(self.name) as self.scope:
            # Embeddings

            self.emb_out = layers.TransformerEmbedding(
                'emb_out', max_voc_size if share_emb else len(out_voc), emb_size,
                matrix=self.emb_inp.emb.mat if share_emb else None,
                rescale=rescale_emb, device=emb_out_device)

            # Model body
            self.encoder = make_encoder('enc', inp_h=inp_w, inp_w=inp_h, inp_channels=inp_channels, **kwargs)

            enc_out_shape = self.encode(self.make_encoder_batch_ph(), True)['out'].shape
            assert enc_out_shape.ndims == 3 and enc_out_shape[-1].value is not None, \
                "encoder output shape must be a 3d tensor with fixed num units, " \
                "got shape {}".format(enc_out_shape)

            self.decoder = layers.TransformerChain('dec', attn_inputs=['enc'],
                                                   attn_input_sizes={'enc': enc_out_shape[-1].value},
                                                   **kwargs)

            # logits: token insertions plus one extra logit to predict position where to insert
            self.logits = layers.Dense(
                'logits', kwargs['hid_size'], len(out_voc) + 1,
                bias=None if logits_bias else 0
            )


    def _get_batch_sample(self):
        """ A minimal example of model input data """
        return [(np.zeros((self.h, self.w, self.inp_channels)), 'A cat sat')]

    def make_feed_dict(self, batch, **kwargs):
        """ Take input data strings, return a dict { key: np.array(value) } """
        inp_imgs, out_lines = zip(*batch)

        out_len = [linelen(line) for line in out_lines]
        return {
            'inp': np.array(inp_imgs, 'float32'),
            'out': self.out_voc.to_matrix(out_lines),
            'out_len': np.array(out_len, 'int32')
        }

    def make_encoder_batch_ph(self):
        return {
            'inp': tf.placeholder('float32', [None, self.h, self.w, self.inp_channels]),
        }

    def encode(self, batch, is_train):
        """ Take placeholders for data batch, return encoder state """
        with tf.name_scope(self.name), ops.dropout_scope(is_train):
            inp = batch['inp']  # [batch_size * ninp]

            out = self.encoder(inp)
            assert out.shape[-1] is not None
            out_shape = tf.shape(out)

            out = tf.reshape(out, [out_shape[0], -1, out.shape[-1]])

            attn_mask = tf.ones((out_shape[0], 1, 1, out_shape[1] * out_shape[2]))  # [batch_size, 1, 1, ninp]

            return dict(out=out, attn_mask=attn_mask)
