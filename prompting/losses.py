import torch.nn.functional as F

def transpose(x):
    return x.t() if x.dim() == 2 else x.permute(0, 2, 1)

def contrastive_loss(visual_features, class_prototypes, labels=None, t=0.07):
    logits = t.exp() * visual_features @ transpose(class_prototypes)
    if labels is not None:
        return F.cross_entropy(logits, labels), logits
    else:
        return None, logits