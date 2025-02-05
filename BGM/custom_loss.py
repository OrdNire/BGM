import torch
import torch.nn as nn
import torch.nn.functional as F

class LabelContrastiveLossHardway(nn.Module):
    def __init__(self):
        super(LabelContrastiveLossHardway, self).__init__()

    def forward(self, positive_features, negative_features, label_embeddings, beta=0.5):
        '''
        :param positive_features:   size [b, embedding_dim]
        :param negative_features:   size [b, embedding_dim]
        :param label_embeddings:    size [b, embedding_dim]
        :return:
        '''
        positive_sim = F.cosine_similarity(label_embeddings, positive_features, dim=-1) # [batch_size, num_label]
        negative_sim = F.cosine_similarity(label_embeddings, negative_features, dim=-1)
        loss = (1 - positive_sim) + torch.maximum((negative_sim - beta), torch.tensor(0.0))  # [batch_size, num_label]
        return loss.mean()

class LabelContrastiveLoss(nn.Module):
    def __init__(self, label_embedding=None):
        super(LabelContrastiveLoss, self).__init__()
        self.label_embeddings = label_embedding

    def _contrastive_loss(self, positive_features, negative_features, label_embeddings, beta):
        '''
        :param pred:                size [b, num_label]
        :param positive_features:   size [b, num_label, embedding_dim]
        :param negative_features:   size [b, num_label, embedding_dim]
        :param label_embeddings:    size [1, num_label, embedding_dim]
        :return:
        '''

        positive_sim = F.cosine_similarity(label_embeddings, positive_features, dim=-1) # [batch_size, num_label]
        negative_sim = F.cosine_similarity(label_embeddings, negative_features, dim=-1)
        loss = (1 - positive_sim) + torch.maximum((negative_sim - beta), torch.tensor(0.0))  # [batch_size, num_label]
        return loss


    def forward(self, pred, positive_features, negative_features, beta=0.5):
        '''
        :param pred:                size [b, num_label]
        :param positive_features:   size [b, num_label, embedding_dim]
        :param negative_features:   size [b, num_label, embedding_dim]
        :param label_embeddings:    size [num_label, embedding_dim]
        :return:
        '''
        if isinstance(beta, torch.Tensor):
            beta = beta.unsqueeze(1).repeat(1, pred.shape[-1]).cuda()

        # probabilities = torch.sigmoid(pred)
        probabilities = pred

        label_embeddings = self.label_embeddings.unsqueeze(0).expand(pred.size(0), -1, -1)  # [b, num_label, embedding_dim]

        loss = self._contrastive_loss(positive_features, negative_features, label_embeddings, beta)

        weighted_loss = probabilities * loss  # [batch_size, num_label, 1]

        total_loss = weighted_loss.mean()

        return total_loss

class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, pad_index=None):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.pad_index = pad_index

    def forward(self, x, target, smoothing=0.1):
        confidence = 1. - smoothing
        logprobs = F.log_softmax(x, dim=-1)

        if self.pad_index is not None:
            mask = target != self.pad_index
        else:
            mask = torch.ones_like(target, dtype=torch.bool)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1) * mask
        smooth_loss = -logprobs.mean(dim=-1) * mask
        loss = confidence * nll_loss + smoothing * smooth_loss
        return loss.mean()

def linear_combination(x, y, epsilon):
    return epsilon * x + (1 - epsilon) * y

def reduce_loss(loss, reduction='mean'):
    return loss.mean() if reduction == 'mean' else loss.sum() if reduction == 'sum' else loss


def cls_weights_conf_loss(logits, labels, attention_scores):
    # logits: (B, M, C) -> B: batch size, M: number of labels, C: number of classes per label
    # labels: (B, M) -> integer labels for each of the M categories
    # attention_scores: (B, M) -> attention scores (weights for each label)

    # 计算每个标签的多分类交叉熵损失
    # 使用 F.cross_entropy，它会自动对 logits 应用 softmax
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), reduction='none')  # (B * M,)

    # Reshape to (B, M) so each sample's loss per label is preserved
    loss = loss.view(logits.size(0), logits.size(1))  # (B, M)

    # 根据 attention_scores 对损失进行加权
    weighted_loss = loss * attention_scores  # (B, M)

    # 对所有样本和标签的加权损失求平均
    avg_loss = weighted_loss.mean()  # (scalar)

    return avg_loss

class LabelSmoothingCrossEntropyV2(nn.Module):
    def __init__(self, epsilon: float = 0.1, reduction='mean'):
        super().__init__()
        self.epsilon = epsilon
        self.reduction = reduction

    def forward(self, preds, target):
        n = preds.size()[-1]
        log_preds = F.log_softmax(preds, dim=-1)
        loss = reduce_loss(-log_preds.sum(dim=-1), self.reduction)
        nll = F.nll_loss(log_preds, target, reduction=self.reduction)
        return linear_combination(loss / n, nll, self.epsilon)

class LabelSmoothingCrossEntropyV3(nn.Module):
    def __init__(self, epsilon: float = 0.1, reduction='mean'):
        super().__init__()
        self.epsilon = epsilon
        self.reduction = reduction

    def forward(self, preds, target, weights):
        # weights: (B, )
        n = preds.size()[-1]
        log_preds = weights.unsqueeze(1) * F.log_softmax(preds, dim=-1)
        loss = reduce_loss(-log_preds.sum(dim=-1), self.reduction)
        nll = F.nll_loss(log_preds, target, reduction=self.reduction)
        return linear_combination(loss / n, nll, self.epsilon)

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=2, reduction="mean"):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        p = torch.sigmoid(inputs)
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        p_t = p * targets + (1 - p) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** self.gamma)

        if self.alpha >= 0:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * loss

        # Check reduction option and return loss accordingly
        if self.reduction == "none":
            pass
        elif self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
        else:
            raise ValueError(
                f"Invalid Value for arg 'reduction': '{self.reduction} \n Supported reduction modes: 'none', 'mean', 'sum'"
            )
        return loss

def sigmoid_focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.

    Args:
        inputs (Tensor): A float tensor of arbitrary shape.
                The predictions for each example.
        targets (Tensor): A float tensor with the same shape as inputs. Stores the binary
                classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha (float): Weighting factor in range (0,1) to balance
                positive vs negative examples or -1 for ignore. Default: ``0.25``.
        gamma (float): Exponent of the modulating factor (1 - p_t) to
                balance easy vs hard examples. Default: ``2``.
        reduction (string): ``'none'`` | ``'mean'`` | ``'sum'``
                ``'none'``: No reduction will be applied to the output.
                ``'mean'``: The output will be averaged.
                ``'sum'``: The output will be summed. Default: ``'none'``.
    Returns:
        Loss tensor with the reduction option applied.
    """
    # Original implementation from https://github.com/facebookresearch/fvcore/blob/master/fvcore/nn/focal_loss.py

    p = torch.sigmoid(inputs)
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    # Check reduction option and return loss accordingly
    if reduction == "none":
        pass
    elif reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()
    else:
        raise ValueError(
            f"Invalid Value for arg 'reduction': '{reduction} \n Supported reduction modes: 'none', 'mean', 'sum'"
        )
    return loss

class AsymmetricLossOptimized(nn.Module):
    ''' Notice - optimized version, minimizes memory allocation and gpu uploading,
    favors inplace operations'''

    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=False):
        super(AsymmetricLossOptimized, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

        # prevent memory allocation and gpu uploading every iteration, and encourages inplace operations
        self.targets = self.anti_targets = self.xs_pos = self.xs_neg = self.asymmetric_w = self.loss = None

    def forward(self, x, y):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """

        self.targets = y
        self.anti_targets = 1 - y

        # Calculating Probabilities
        self.xs_pos = torch.sigmoid(x)
        self.xs_neg = 1.0 - self.xs_pos

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            self.xs_neg.add_(self.clip).clamp_(max=1)

        # Basic CE calculation
        self.loss = self.targets * torch.log(self.xs_pos.clamp(min=self.eps))
        self.loss.add_(self.anti_targets * torch.log(self.xs_neg.clamp(min=self.eps)))

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            self.xs_pos = self.xs_pos * self.targets
            self.xs_neg = self.xs_neg * self.anti_targets
            self.asymmetric_w = torch.pow(1 - self.xs_pos - self.xs_neg,
                                          self.gamma_pos * self.targets + self.gamma_neg * self.anti_targets)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            self.loss *= self.asymmetric_w
        return -self.loss.mean()


class GlobalContrastiveLoss(nn.Module):
    def __init__(self, beta=0.5):
        super(GlobalContrastiveLoss, self).__init__()
        self.beta = beta

    def forward(self, x, labels):
        # x: B*embed_dim
        # labels: B*num_labels (one-hot)

        device = x.device

        # jaccard_index
        intersection = torch.matmul(labels.float(), labels.float().t()).to(device)  # B*B
        label_counts = labels.sum(dim=1).unsqueeze(1).to(device)  # B*1
        union = label_counts + label_counts.t() - intersection
        iou = (intersection / union).to(device)

        cos = F.normalize(x, p=2, dim=1).mm(F.normalize(x, p=2, dim=1).T).to(device)  # B * B
        pos = 1.0 - cos
        # pos = 1.0 - cos
        neg = F.relu(cos - iou)

        P_num = len((iou > 0.5).nonzero())
        N_num = len((iou <= 0.5).nonzero())
        if P_num == 0:
            pos_term = torch.tensor(0.0).to(x.device)
        else:
            pos_term = torch.where((iou > 0.5), pos.to(torch.float32),
                               torch.zeros_like(cos).to(torch.float32)).sum() / P_num

        if N_num == 0:
            neg_term = torch.tensor(0.0).to(x.device)
        else:
            neg_term = torch.where((iou <= 0.5), neg.to(torch.float32),
                               torch.zeros_like(cos).to(torch.float32)).sum() / N_num

        return pos_term + neg_term

class GlobalContrastiveLossV2(nn.Module):
    def __init__(self, threshold=0.5, beta=0.5, proxies=None):
        super(GlobalContrastiveLossV2, self).__init__()
        self.threshold = threshold
        self.beta = beta
        self.proxies = proxies

    def forward(self, x, labels):
        # x: B*embed_dim
        # labels: B*num_labels (one-hot)
        # proxies: num_labels, embed_dim

        P_one_hot = labels

        cos = F.normalize(x, p = 2, dim = 1).mm(F.normalize(self.proxies, p = 2, dim = 1).T)
        pos = 1 - cos
        neg = F.relu(cos - self.threshold)

        P_num = len(P_one_hot.nonzero())
        N_num = len((P_one_hot == 0).nonzero())
        pos_term = torch.where(P_one_hot  ==  1, pos.to(torch.float32), torch.zeros_like(cos).to(torch.float32)).sum() / P_num
        neg_term = torch.where(P_one_hot  ==  0, neg.to(torch.float32), torch.zeros_like(cos).to(torch.float32)).sum() / N_num
        if self.beta > 0:
            index = labels.sum(dim = 1) > 1
            y_ = labels[index].float()
            x_ = x[index]
            cos_sim = y_.mm(y_.T)
            if len((cos_sim == 0).nonzero()) == 0:
                reg_term = 0
            else:
                x_sim = F.normalize(x_, p = 2, dim = 1).mm(F.normalize(x_, p = 2, dim = 1).T)
                neg = self.beta * F.relu(x_sim - self.threshold)
                reg_term = torch.where(cos_sim == 0, neg, torch.zeros_like(x_sim)).sum() / len((cos_sim == 0).nonzero())
        else:
            reg_term = 0

        return pos_term + neg_term + reg_term

class GlobalContrastiveLossV3(nn.Module):
    def __init__(self, threshold=0.5, N_seg=5):
        super(GlobalContrastiveLossV3, self).__init__()
        self.threshold = threshold
        self.N_seg = N_seg

    def forward(self, x, labels):
        # x: B*embed_dim
        # labels: B*num_labels (one-hot)
        device = x.device
        B = x.shape[0]

        # jaccard_index
        intersection = torch.matmul(labels.float(), labels.float().t()).to(device)  # B*B
        label_counts = labels.sum(dim=1).unsqueeze(1).to(device)  # B*1
        union = label_counts + label_counts.t() - intersection
        jaccard_index = (intersection / union).to(device)

        # similarity
        norm_x = F.normalize(x, p=2, dim=1)
        sim = torch.mm(norm_x, norm_x.t())

        pos_samples, neg_samples, valid_mask = self._sample_pair(jaccard_index, device)

        rel_pos = sim[torch.arange(B), pos_samples]
        rel_neg_list = sim[torch.arange(B).unsqueeze(1), neg_samples]

        exp_pos = torch.exp(rel_pos)
        exp_neg = torch.exp(rel_neg_list)
        sum_exp_neg = torch.sum(exp_neg, dim=1)

        loss = -torch.log(exp_pos / (exp_pos + sum_exp_neg))

        valid_loss = loss * valid_mask

        return valid_loss.sum() / valid_mask.sum()

    def _sample_pair(self, jaccard_index, device):
        B = jaccard_index.shape[0]

        pos_mask = jaccard_index > self.threshold  # (B, B)

        valid_mask = pos_mask.any(dim=1).float()  # (B,)，1 表示有正样本，0 表示没有

        pos_mask = pos_mask.float()

        pos_samples = torch.zeros(B, dtype=torch.long).to(device)
        pos_samples[valid_mask.bool()] = pos_mask[valid_mask.bool()].multinomial(1).squeeze()

        pos_similarities = jaccard_index[torch.arange(B), pos_samples]  # (B,)

        neg_mask = jaccard_index < pos_similarities.unsqueeze(1)  # (B, B)

        neg_similarities = jaccard_index.masked_fill(~neg_mask, float('-inf'))  # (B, B)

        top_neg_similarities, neg_samples = torch.topk(neg_similarities, self.N_seg, dim=1)  # (B, N_seg)

        return pos_samples, neg_samples, valid_mask