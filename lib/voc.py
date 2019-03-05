import numpy as np
import lib.util


class Voc:
    """
    Vocab converts between strings of tokens and matrices of token indices.
    It should normally be treated as immutable
    """
    NULL, EOS, UNK = '_NULL_', '_EOS_', '_UNK_'
    _default_tokens = (NULL, EOS, UNK)

    def __init__(self, tokens):
        tokens = tuple(tokens)
        assert len(tokens) == len(set(tokens)), "tokens must be unique"
        for i, t in enumerate(self._default_tokens):
            assert t in tokens and tokens.index(t) == i, "token must have %s at index %i" % (t, i)

        self._tokens = tokens
        self._token2id = {token: i for i, token in enumerate(self._tokens)}
        self.null = self._tokens.index(self.NULL)
        self.eos = self._tokens.index(self.EOS)
        self.unk = self._tokens.index(self.UNK)
        self._default_token_ix = (self.null, self.eos, self.unk)

    def __len__(self):
        return len(self._tokens)

    def __contains__(self, item):
        return item in self._token2id

    def ids(self, inp):
        """ converts a token or list of tokens into integer token index or indices """
        if isinstance(inp, (str, bytes)):
            return self._token2id.get(inp, self.unk)
        else:
            return [self.ids(xi) for xi in inp]

    def words(self, inp):
        """ converts token indices into a list of tokens """
        if lib.util.is_iterable(inp):
            return [self.words(xi) for xi in inp]
        else:
            return self._tokens[inp]

    def to_matrix(self, lines, max_len=None):
        """
        converts a string or a list of strings into a fixed size matrix
        pads short sequences with self.EOS
        example usage:
        >>>sentences = ... # a list of strings
        >>>vocab = Voc.from_sequences(sentences)
        >>>print(vocab.tokenize_many(sentences[:3]))
        [[15 22 21 28 27 13  1  1  1  1  1  1]
         [30 21 15 15 21 14 28 27 13  1  1  1]
         [25 37 31 34 21 20 37 21 28 19 13  1]]
        """
        sequences = list(map(str.split, lines))
        max_len = max_len or max(map(len, sequences)) + 1  # 1 for eos
        matrix = np.full((len(sequences), max_len), fill_value=self.eos, dtype='int32')
        for i, seq in enumerate(sequences):
            tokens = self.ids(seq)[:max_len]
            matrix[i, :len(tokens)] = tokens
        return matrix

    def to_lines(self, indices, deprocess=True):
        """
        Convert tensor of token ids into strings
        :param matrix: matrix of tokens of int32, shape=[batch,time]
        :param deprocess: if True, removes all unknowns and EOS
        :return: string or strings
        """
        # if indices is a sequence of SEQUENCES of token_ids, not just token_ids
        if hasattr(next(iter(indices)), '__iter__'):
            return [self.to_lines(xi, deprocess=deprocess) for xi in indices]
        tokens = [self._tokens[token] for token in indices]
        if deprocess:
            tokens = [t for t in tokens if t not in self._default_tokens]
        return ' '.join(tokens)

    def save(self, file):
        with open(file, 'w', encoding='utf-8') as f:
            for token in self._tokens:
                f.write(token + '\n')

    @classmethod
    def load(cls, file):
        """ Parses vocab from a .voc file """
        with open(file, 'r', encoding='utf-8') as f:
            tokens = list(filter(len, f.read().split('\n')))
        return Voc(tokens)

    @classmethod
    def from_sequences(cls, sentences):
        """ Infers tokens from a corpora of sentences (tokens separated by spaces) """
        tokens = set()
        for s in sentences:
            tokens.update(s.split())
        return Voc(list(cls._default_tokens) + sorted(tokens))

    @classmethod
    def merge(cls, first, *others):
        """
        Constructs vocab out of several different vocabularies.
        Maintains existing token ids by first vocab
        """
        for vocab in (first,) + others:
            assert isinstance(vocab, Voc)

        # get all tokens from others that are not in first
        other_tokens = set()
        for vocab in others:
            other_tokens.update(set(vocab.tokens))

        # inplace substract first.tokens from other_tokens
        other_tokens.difference_update(set(first.tokens))

        return Voc(first.tokens + tuple(sorted(other_tokens)))
