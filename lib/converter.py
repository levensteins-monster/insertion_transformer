import numpy as np
from lib.voc import Voc
from warnings import warn

warn("DeprecationWarning: lib.converter is no longer supported and will be removed. "
     "Please train models using lib.training")


def convert_tfnn_checkpoint(npz_path):
    chkpt = np.load(npz_path)
    chkpt = {var: chkpt[var] for var in chkpt}
    logits_w = chkpt.pop('mod/loss_xent_lm/logits/W:0')
    chkpt['mod/logits/W:0'] = logits_w

    if 'mod/loss_xent_lm/logits/b:0' in chkpt:
        logits_b = chkpt.pop('mod/loss_xent_lm/logits/b:0')
        chkpt['mod/logits/b:0'] = logits_b

    return chkpt


def convert_voc(tfnn_voc):
    """ Converts tfnn voc into lib.voc.Voc preserves all tokens but swaps UNK/BOS """
    tokens = tfnn_voc.words(list(range(tfnn_voc.size())))
    tokens[0] = '_NULL_'
    return Voc(tokens)
