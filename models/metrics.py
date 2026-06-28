import torch
from utils.misc import *

def multi_label_focal_loss(pred, target, 
                           class_weights=None, sample_weights=None, gamma=0):
    '''
    Calculates focal loss given pred and target for multi-label input. 
    Weights are optional and are offered at sample- and class-level.
    If gamma = 0, this calculates the multi-label cross entropy.
    pred: Tensor (Batch, num_labels, num_classes)
    target: Tensor (Batch, num_labels, num_classes)
    class_weights: Tensor (num_labels, num_classes)
    sample_weights: Tensor (Batch, num_labels)
    gamma: float
    '''
    device = pred.device
    if pred.dim() != 3 or target.dim() != 3:
            raise ValueError('Both inputs must be of dim = 3.')
    class_weights = torch.ones(pred.shape[1:], device=device) if class_weights is None else class_weights
    class_weights = class_weights.unsqueeze(0)
    sample_weights = torch.ones(pred.shape[:2], device=device) if sample_weights is None else sample_weights
    sample_weights = sample_weights.unsqueeze(-1)

    # focal loss term
    log_probits = torch.nn.functional.log_softmax(pred, dim=-1)
    probits = torch.exp(log_probits)
    focal_term = (1 - probits) ** gamma

    # cross-entropy
    ce_loss = -target * log_probits
    loss = torch.sum(sample_weights * class_weights * focal_term * ce_loss, dim=[-1, -2])
    return torch.mean(loss)

def normalize_logits(tab):
    norm_tab = torch.zeros_like(tab, device=tab.device)
    idx1 = torch.arange(len(tab)).repeat_interleave(tab.shape[1])
    idx3 = torch.argmax(tab, dim=-1)
    idx2 = torch.tile(torch.arange(tab.shape[1], device=tab.device), (1, len(tab)))
    norm_tab[idx1, idx2, idx3.reshape(-1)] = 1
    return norm_tab

def tab_excl_silence(tab):
    tab = normalize_logits(tab)
    return tab[..., 1:]

def tab2son(tab):
    '''
    This function takes a batch of tablatures (N x 6, num_classes) and
    converts them to a sonority class a batch of numbers 0-2, where 0 means the current
    frame is a complete silence, 1 means only a single string was active (monophonic),
    and 2 means multiple strings were active (polyphonic).
    '''
    tab = normalize_logits(tab)
    tab = normalize_logits(tab)
    num_samples = tab.shape[0]
    son_classes = torch.zeros((num_samples, 3), device=tab.device)
    active_strings = tab[:, :, 1:].sum(dim=-1).sum(dim=-1)
    son_classes[active_strings == 0, 0] = 1  # Silence
    son_classes[active_strings == 1, 1] = 1  # Monophonic
    son_classes[active_strings > 1, 2] = 1   # Polyphonic
    return son_classes

def tab2pitch(tab):
    '''
    Takes in a torch tensor of shape (Batch x 6 x num_frets).
    It outputs a torch tensor of shape (Batch x num_pitches).
    '''
    tab = normalize_logits(tab)
    num_samples = tab.shape[0]
    num_frets = tab.shape[-1] - 1
    string_pitches = torch.tensor(MIDI_NOTE_NUMBERS, device=tab.device)
    num_pitches = max(string_pitches) + num_frets - min(string_pitches)
    pitches = torch.zeros(num_samples, num_pitches)
    # get rid of "closed" class, as we only want to count positives
    sample_nr, string, fret = torch.where(tab[..., 1:] == 1)
    active_pitches = string_pitches[string] + fret - min(string_pitches)
    pitches[sample_nr, active_pitches] = 1
    return pitches

def confusion_matrix(pred, target, num_classes=None,
                    micro_avg=False, flatten=False):
    '''
    This can return the entire matrix unless provided an index, in which case it returns
    the corresponding element from the confusion matrix.
    '''
    if target.dim() > 2:
        y_true = torch.argmax(target, dim=-1).view(pred.shape[0], -1)
        y_pred = torch.argmax(pred, dim=-1).view(target.shape[0], -1)
    else:
        y_true = target.view(pred.shape[0], -1).long()
        y_pred = pred.view(target.shape[0], -1).long()

    if micro_avg is True:
        tp = torch.sum((pred == 1) & (target == 1))
        fp = torch.sum((pred == 1) & (target == 0))
        fn = torch.sum((pred == 0) & (target == 1))
        tn = torch.sum((pred == 0) & (target == 0))
        conf_matrix = torch.tensor([[tp, fn], [fp, tn]], device=pred.device)
    else:
        K = pred.shape[1] if num_classes is None else num_classes
        conf_matrix = torch.zeros((K, K), device=pred.device)
        pred_idces, true_idces = torch.where(y_pred == 1)[1], torch.where(y_true == 1)[1]
        for i in range(len(y_true)):
            conf_matrix[pred_idces[i], true_idces[i]] += 1

    return conf_matrix if flatten is False else conf_matrix.reshape(-1)

def precision_fn(tp, fp):
    '''
    Calculates precision from aggregate TP and FP counts.
    '''
    denominator = tp + fp
    if denominator == 0:
        return torch.tensor(0)
    
    precision_val = tp / denominator
    return precision_val if not torch.isnan(precision_val) else torch.tensor(0)

def recall_fn(tp, fn):
    '''
    Calculates recall from aggregate TP and FN counts.
    '''
    denominator = tp + fn
    if denominator == 0:
        return torch.tensor(0)
        
    recall_val = tp / denominator
    return recall_val if not torch.isnan(recall_val) else torch.tensor(0)

def f_measure_fn(tp, fp, fn):
    '''
    Calculates F1-measure by reusing precision and recall.
    '''
    p = precision_fn(tp, fp)
    r = recall_fn(tp, fn)
    
    denominator = p + r
    if denominator == 0:
        return torch.tensor(0)
        
    f = (2 * p * r) / denominator
    return f if not torch.isnan(f) else torch.tensor(0)

def p_r_f_metrics(input, **kwargs):
    tp, fn, fp, tn = input
    prec = precision_fn(tp, fp)
    recall = recall_fn(tp, fn)
    f1 = f_measure_fn(tp, fp, fn)
    return prec, recall, f1

def tab_disamb_rate(tab_precision, pitch_precision, **kwargs):
    try:
        tab_disamb = float(tab_precision) / float(pitch_precision)
        if tab_disamb != tab_disamb: 
            return 0.0
        return tab_disamb
    except ZeroDivisionError:
        return 0.0