import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .transformer_layers import SelfAttnLayer, DecoderBlock
from .backbone import Backbone, ResNet50, ResNet101
from .utils import weights_init, trunc_normal_
from .position_enc import positionalencoding2d
from .DictTree import TreeNode
import torch.nn.init as init
from transformers import BertTokenizer, BertModel

def get_attn_pad_mask(seq_q, seq_k, num):
    batch_size, len_q = seq_q.size() # B, N
    batch_size, len_k = seq_k.size()
    # eq(zero) is PAD token
    pad_attn_mask = torch.cat((torch.tensor([[1] for _ in range(batch_size)]).to(seq_q.device),seq_q), dim=1).data.eq(num).unsqueeze(1)  # batch_size x 1 x len_k, one is masking
    return pad_attn_mask.expand(batch_size, len_q+1, len_k+1)

class Interactor(nn.Module):
    def __init__(self, image_token, text_token, label_token, hidden=768, layers=3, heads=4, dropout=0.1):
        super(Interactor, self).__init__()

        self.image_token = image_token
        self.text_token = text_token

        # Image encoder
        # self.conv_downsample = torch.nn.Conv2d(2048, hidden, (1, 1))
        self.image_bos = torch.nn.Parameter(torch.zeros(1, 1, hidden)) # 记得初始化
        self.image_downsample = torch.nn.Conv2d(2048, hidden, (1, 1))

        # Text encoder
        # suppose output feature (B, len_token, dim)
        self.text_bos = torch.nn.Parameter(torch.zeros(1, 1, hidden)) # 记得初始化
        self.text_downsample = torch.nn.Linear(768, hidden)

        # Label encoder
        self.label_bos = torch.nn.Parameter(torch.zeros(1, 1, hidden))
        self.label_downsample = torch.nn.Linear(768, hidden)

        # Position Embeddings
        self.image_pos_embedding = nn.Parameter(torch.zeros(1, 1+image_token, hidden))
        self.text_pos_embedding = nn.Parameter(torch.zeros(1, 1+text_token, hidden))
        self.label_pos_embedding = nn.Parameter(torch.zeros(1, 1+label_token, hidden))

        # Transformer
        self.self_attn_layers = nn.ModuleList([SelfAttnLayer(hidden, heads, dropout) for _ in range(layers)])

        # Other
        self.LayerNorm = nn.LayerNorm(hidden)
        self.dropout = nn.Dropout(dropout)

        # Init all except pretrained backbone
        trunc_normal_(self.image_bos, std=.02)
        trunc_normal_(self.text_bos, std=.02)
        trunc_normal_(self.label_bos, std=.02)
        trunc_normal_(self.image_pos_embedding, std=.02)
        trunc_normal_(self.text_pos_embedding, std=.02)
        trunc_normal_(self.label_pos_embedding, std=.02)
        init.kaiming_normal_(self.image_downsample.weight, mode='fan_out', nonlinearity='relu')
        self.text_downsample.apply(weights_init)
        self.label_downsample.apply(weights_init)
        self.LayerNorm.apply(weights_init)
        self.self_attn_layers.apply(weights_init)

    def forward(self, image_feature=None, text_feature=None, label_embedding=None, padding_mask=None):
        if image_feature is not None:
            b = image_feature.size(0)
        elif text_feature is not None:
            b = text_feature.size(0)

        label_tokens = self.label_downsample(label_embedding.repeat(b, 1, 1))
        label_bos = self.label_bos.expand(b, -1, -1)
        if image_feature is None: # Text Only
            text_tokens = self.text_downsample(text_feature)
            text_bos = self.text_bos.expand(b, -1, -1)

            embeddings = torch.cat((text_bos, text_tokens, label_bos, label_tokens), dim=1) # 1+150+1+num_label
            # pos
            embeddings = embeddings + torch.cat([self.text_pos_embedding, self.label_pos_embedding], dim=1).expand(b, -1, -1)
        elif text_feature is None: # Image Only
            image_tokens = self.image_downsample(image_feature)
            image_tokens = image_tokens.view(image_tokens.size(0), image_tokens.size(1), -1).permute(0, 2, 1) # b, c, h*w -> b, h*w, c
            image_bos = self.image_bos.expand(b, -1, -1)

            embeddings = torch.cat((image_bos, image_tokens, label_bos, label_tokens), dim=1)
            embeddings = embeddings + torch.cat([self.image_pos_embedding, self.label_pos_embedding], dim=1).expand(b, -1, -1)
        elif (image_feature is not None) and (text_feature is not None): # Union
            text_tokens = self.text_downsample(text_feature)
            text_bos = self.text_bos.expand(b, -1, -1)
            image_tokens = self.image_downsample(image_feature)
            image_tokens = image_tokens.view(image_tokens.size(0), image_tokens.size(1), -1).permute(0, 2, 1)  # b, c, h*w -> b, h*w, c
            image_bos = self.image_bos.expand(b, -1, -1)

            embeddings = torch.cat((image_bos, image_tokens, text_bos, text_tokens, label_bos, label_tokens), dim=1)
            embeddings = embeddings + torch.cat([self.image_pos_embedding, self.text_pos_embedding, self.label_pos_embedding], dim=1).expand(b, -1, -1)
        else:
            raise ValueError("Interactor feature error.")

        # Feed image and label embeddings through Transformer
        embeddings = self.LayerNorm(embeddings)
        for layer in self.self_attn_layers:
            embeddings, attn = layer(embeddings, mask=None, padding_mask=padding_mask)

        # Readout each label embedding using a linear layer
        label_embeddings = embeddings[:, -label_tokens.size(1):, :]  # (B, num_labels, hidden)

        return label_embeddings

class NoiseMLP(nn.Module):
    def __init__(self, input_dim=768, forward_dim=2048, dropout_rate=0.1, num_classes=10):
        super(NoiseMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, forward_dim)
        self.activation = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(forward_dim, input_dim)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.layer_norm = nn.LayerNorm(input_dim)
        self.classifier = nn.Linear(input_dim, num_classes)

        self.fc1.apply(weights_init)
        self.fc2.apply(weights_init)
        self.layer_norm.apply(weights_init)
        self.classifier.apply(weights_init)

    def forward(self, x):
        residual = x
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        x = self.layer_norm(x + residual)
        logits = self.classifier(x)
        return logits

class SIRMM(nn.Module):

    def __init__(self, image_encoder, text_encoder, interactor,
                 label_embedding=None, num_labels=17, embed_dim=768, distance_prc=100, **kwargs):
        super().__init__()
        self.num_labels = num_labels
        self.distance_prc = distance_prc
        self.label_embedding = label_embedding # num_labels, embed_dim
        self.embed_dim = embed_dim

        # Encoder / Decoder
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder
        self.interactor = interactor
        # self.decoder = decoder

        # exp
        self.select_t = 0.4
        self.tau = 1 # temp

        # Classifier
        # Output is of size num_labels because we want a separate classifier for each label
        self.classifier_layer = torch.nn.Linear(embed_dim, num_labels)

        # Distance Preserve
        distance_dim = 768  # same with BERT output dim
        self.distance_layer = torch.nn.Linear(embed_dim, distance_dim)

        # Confidence Proj
        conf_proj_dim = 768
        # self.conf_proj_layer = torch.nn.ModuleList([torch.nn.Linear(embed_dim, conf_proj_dim), torch.nn.Linear(conf_proj_dim, distance_prc)])
        self.conf_proj_layer = NoiseMLP(input_dim=embed_dim, num_classes=distance_prc)

        # Init
        self.classifier_layer.apply(weights_init)
        self.distance_layer.apply(weights_init)
        # for layer in self.conf_proj_layer:
        #     layer.apply(weights_init)

    def _construct_id(self, classifier_output, distance_output):
        # classifier_output: (B, num_labels)
        # label_embedding: (B, num_labels, embed_dim)
        # distance_output: (B, num_labels, embed_dim)

        classifier_output = classifier_output.cuda()
        distance_output = distance_output.cuda()

        # center_embedding = label_embedding.detach().cuda()
        center_embedding = self.label_embedding.repeat(distance_output.size(0), 1, 1)

        center_embedding = F.normalize(center_embedding, p=2, dim=-1)

        # 直通估计
        # classification_res = self.straight_through_hard_sigmoid(classifier_output, threshold=0.5)
        prob = torch.sigmoid(classifier_output)
        classification_res = (prob > 0.5).float()
        # classification_res_exp = classification_res.unsqueeze(-1).expand_as(center_embedding) # (B, num_labels, embed_dim)

        cosine_sim = F.cosine_similarity(center_embedding, distance_output, dim=-1)
        # normalized_cos_sim = ((cosine_sim + 1) / 2) * self.distance_prc # [0, 100]
        normalized_cos_sim = cosine_sim + 1

        N = int(classification_res.sum().item())

        tgt = torch.zeros((N, 1)).cuda()
        tgt_id = torch.zeros((N, 2)).cuda()

        indices = torch.nonzero(classification_res, as_tuple=False).cuda()
        # tgt[:, 1] = (cosine_sim[indices[:, 0], indices[:, 1]].round(decimals=2)) * (self.distance_prc // 10)
        tgt[:, 0] = normalized_cos_sim[indices[:, 0], indices[:, 1]]

        tgt_id[:, 0] = indices[:, 1]
        tgt_id[:, 1] = normalized_cos_sim[indices[:, 0], indices[:, 1]]

        # classification score
        # cls_s = classifier_output[indices[:, 0], indices[:, 1]]

        return tgt, indices, tgt_id

    def _distanceID_STE(self, cos_sim):
        sim_norm = ((cos_sim + 1) / 2) * self.distance_prc
        id = sim_norm.long()
        return (id - sim_norm).detach() + sim_norm

    def gumbel_sample(self, cls, hard=True):
        # cls: b*num_labels
        prob = F.sigmoid(cls)
        prob_new = torch.log(torch.stack([prob, 1-prob], dim=-1))
        cls_filter = F.gumbel_softmax(prob_new, self.tau, hard=hard)[:, :, 0]

        # cls_filter = F.gumbel_softmax(cls, self.tau, hard=hard)

        return cls_filter # B*num_labels

    def _warm_learning(self, query, tgt, tgt_noise, label_embedding):
        batch_N = tgt.size(0)  # query: (b, 1, text_token)
        # Text features
        text_feature = self.text_encoder(**query).last_hidden_state  # query: (b, text_token, dim)

        # Image features
        combined_image = torch.cat([tgt, tgt_noise], dim=0)
        combined_image_feature = self.image_encoder(combined_image)

        # Interactor
        text_attention_mask = torch.cat(
            [torch.zeros((batch_N, 1), device=tgt.device, dtype=torch.bool), (query["attention_mask"] == 0)],
            dim=1)  # (b, 1, 150) False: not_mask, True: mask
        label_attention_mask = torch.zeros((batch_N, 1 + self.num_labels), device=tgt.device, dtype=torch.bool)
        text_only_key_padding_mask = torch.cat([text_attention_mask, label_attention_mask], dim=1)

        text_hidden = self.interactor(image_feature=None, text_feature=text_feature, label_embedding=label_embedding,
                                      padding_mask=text_only_key_padding_mask)
        combined_image_hidden = self.interactor(image_feature=combined_image_feature, text_feature=None,
                                                label_embedding=label_embedding)

        # CLS Task
        classifier_output = self.classifier_layer(
            torch.cat([text_hidden, combined_image_hidden], dim=0))  # b*num_labels*num_labels
        diag_mask = torch.eye(classifier_output.size(1)).unsqueeze(0).repeat(classifier_output.size(0), 1, 1).cuda()
        classifier_output = (classifier_output * diag_mask).sum(-1)

        text_cls_out = classifier_output[:batch_N, :]
        tgt_cls_out = classifier_output[batch_N:2 * batch_N, :]

        # # SIM Task
        # sim_output = self.distance_layer(torch.cat([text_hidden, combined_image_hidden], dim=0))
        # text_sim_out = sim_output[:batch_N, :, :]
        # tgt_sim_out = sim_output[batch_N:2 * batch_N, :, :]
        # noise_tgt_sim_out = sim_output[-batch_N:, :, :]
        # tgt_sim_out = F.normalize(tgt_sim_out, p=2, dim=-1)
        # noise_tgt_sim_out = F.normalize(noise_tgt_sim_out, p=2, dim=-1)

        return tgt_cls_out, text_cls_out

    def _abl_ind(self, tgt, label_embedding):
        batch_N = tgt.size(0)  # query: (b, 1, text_token)

        # Image features
        tgt_feature = self.image_encoder(tgt)

        # Interactor
        tgt_hidden = self.interactor(image_feature=tgt_feature, text_feature=None,
                                     label_embedding=label_embedding)

        # CLS Task
        classifier_output = self.classifier_layer(tgt_hidden)  # b*num_labels*num_labels
        diag_mask = torch.eye(classifier_output.size(1)).unsqueeze(0).repeat(classifier_output.size(0), 1, 1).cuda()
        classifier_output = (classifier_output * diag_mask).sum(-1)

        return classifier_output

    def _abl_ret(self, query, label_embedding):
        # Text features
        text_feature = self.text_encoder(**query).last_hidden_state  # query: (b, text_token, dim)
        batch_N = text_feature.size(0)  # query: (b, 1, text_token)
        # Interactor
        text_attention_mask = torch.cat(
            [torch.zeros((batch_N, 1), device=text_feature.device, dtype=torch.bool), (query["attention_mask"] == 0)],
            dim=1)  # (b, 1, 150) False: not_mask, True: mask
        label_attention_mask = torch.zeros((batch_N, 1 + self.num_labels), device=text_feature.device, dtype=torch.bool)
        text_only_key_padding_mask = torch.cat([text_attention_mask, label_attention_mask], dim=1)
        text_hidden = self.interactor(image_feature=None, text_feature=text_feature, label_embedding=label_embedding,
                                      padding_mask=text_only_key_padding_mask)
        # CLS Task
        classifier_output = self.classifier_layer(text_hidden)  # b*num_labels*num_labels
        diag_mask = torch.eye(classifier_output.size(1)).unsqueeze(0).repeat(classifier_output.size(0), 1, 1).cuda()
        classifier_output = (classifier_output * diag_mask).sum(-1)

        return classifier_output

    def _abl_ind_red(self, query, tgt, label_embedding):
        # pair: <q, q_noise, beta_noise, q_tgt, beta>
        # suppose query hase tokenizer

        batch_N = tgt.size(0) # query: (b, 1, text_token)
        # Text features
        text_feature = self.text_encoder(**query).last_hidden_state # query: (b, text_token, dim)

        # Image features
        image_feature = self.image_encoder(tgt)

        # Interactor
        text_attention_mask = torch.cat([torch.zeros((batch_N, 1), device=tgt.device, dtype=torch.bool), (query["attention_mask"] == 0)], dim=1)  # (b, 1, 150) False: not_mask, True: mask
        label_attention_mask = torch.zeros((batch_N, 1 + self.num_labels), device=tgt.device, dtype=torch.bool)
        text_only_key_padding_mask = torch.cat([text_attention_mask, label_attention_mask], dim=1)

        text_hidden = self.interactor(image_feature=None, text_feature=text_feature, label_embedding=label_embedding, padding_mask=text_only_key_padding_mask)
        image_hidden = self.interactor(image_feature=image_feature, text_feature=None, label_embedding=label_embedding)

        # CLS Task
        classifier_output = self.classifier_layer(torch.cat([text_hidden, image_hidden], dim=0))  # b*num_labels*num_labels
        diag_mask = torch.eye(classifier_output.size(1)).unsqueeze(0).repeat(classifier_output.size(0), 1, 1).cuda()
        classifier_output = (classifier_output * diag_mask).sum(-1)

        text_cls_out = classifier_output[:batch_N, :]
        tgt_cls_out = classifier_output[-batch_N:, :]

        return tgt_cls_out, text_cls_out

    def _text_only_joint_learning_hardway(self, query, tgt, tgt_noise, label_embedding):
        batch_N = tgt.size(0)  # query: (b, 1, text_token)
        # Text features
        text_feature = self.text_encoder(**query).last_hidden_state  # query: (b, text_token, dim)

        # Image features
        combined_image = torch.cat([tgt, tgt_noise], dim=0)
        combined_image_feature = self.image_encoder(combined_image)
        tgt_feature = combined_image_feature[:batch_N, :, :]
        tgt_noise_feature = combined_image_feature[-batch_N:, :, :]

        # Interactor
        text_attention_mask = torch.cat(
            [torch.zeros((batch_N, 1), device=tgt.device, dtype=torch.bool), (query["attention_mask"] == 0)],
            dim=1)  # (b, 1, 150) False: not_mask, True: mask
        label_attention_mask = torch.zeros((batch_N, 1 + self.num_labels), device=tgt.device, dtype=torch.bool)
        text_only_key_padding_mask = torch.cat([text_attention_mask, label_attention_mask], dim=1)

        text_hidden = self.interactor(image_feature=None, text_feature=text_feature, label_embedding=label_embedding,
                                      padding_mask=text_only_key_padding_mask)
        combined_image_hidden = self.interactor(image_feature=combined_image_feature, text_feature=None,
                                                label_embedding=label_embedding)
        tgt_hidden = combined_image_hidden[:batch_N, :, :]
        noise_hidden = combined_image_hidden[-batch_N:, :, :]

        # CLS Task
        classifier_output = self.classifier_layer(
            torch.cat([text_hidden, combined_image_hidden], dim=0))  # b*num_labels*num_labels
        diag_mask = torch.eye(classifier_output.size(1)).unsqueeze(0).repeat(classifier_output.size(0), 1, 1).cuda()
        classifier_output = (classifier_output * diag_mask).sum(-1)

        text_cls_out = classifier_output[:batch_N, :]
        tgt_cls_out = classifier_output[batch_N:2 * batch_N, :] # B * num_labels

        # SIM Task
        sim_output = self.distance_layer(combined_image_hidden)
        tgt_sim_out = sim_output[:batch_N, :, :]
        noise_tgt_sim_out = sim_output[-batch_N:, :, :]
        tgt_sim_out = F.normalize(tgt_sim_out, p=2, dim=-1)
        noise_tgt_sim_out = F.normalize(noise_tgt_sim_out, p=2, dim=-1)

        y = self.gumbel_sample(tgt_cls_out, hard=False) # B * num_labels (0/1)

        return tgt_cls_out, text_cls_out, tgt_sim_out, noise_tgt_sim_out, y

    def _text_only_joint_learning_MT(self, query, tgt, tgt_noise, label_embedding):
        batch_N = tgt.size(0)  # query: (b, 1, text_token)
        # Text features
        text_feature = self.text_encoder(**query).last_hidden_state  # query: (b, text_token, dim)

        # Image features
        combined_image = torch.cat([tgt, tgt_noise], dim=0)
        combined_image_feature = self.image_encoder(combined_image)
        tgt_feature = combined_image_feature[:batch_N, :, :]
        tgt_noise_feature = combined_image_feature[-batch_N:, :, :]

        # Interactor
        text_attention_mask = torch.cat(
            [torch.zeros((batch_N, 1), device=tgt.device, dtype=torch.bool), (query["attention_mask"] == 0)],
            dim=1)  # (b, 1, 150) False: not_mask, True: mask
        label_attention_mask = torch.zeros((batch_N, 1 + self.num_labels), device=tgt.device, dtype=torch.bool)
        text_only_key_padding_mask = torch.cat([text_attention_mask, label_attention_mask], dim=1)

        text_hidden = self.interactor(image_feature=None, text_feature=text_feature, label_embedding=label_embedding,
                                      padding_mask=text_only_key_padding_mask)
        combined_image_hidden = self.interactor(image_feature=combined_image_feature, text_feature=None,
                                                label_embedding=label_embedding)
        tgt_hidden = combined_image_hidden[:batch_N, :, :]
        noise_hidden = combined_image_hidden[-batch_N:, :, :]

        # CLS Task
        classifier_output = self.classifier_layer(
            torch.cat([text_hidden, combined_image_hidden], dim=0))  # b*num_labels*num_labels
        diag_mask = torch.eye(classifier_output.size(1)).unsqueeze(0).repeat(classifier_output.size(0), 1, 1).cuda()
        classifier_output = (classifier_output * diag_mask).sum(-1)

        text_cls_out = classifier_output[:batch_N, :]
        tgt_cls_out = classifier_output[batch_N:2 * batch_N, :] # B * num_labels

        # SIM Task
        sim_output = self.distance_layer(combined_image_hidden)
        tgt_sim_out = sim_output[:batch_N, :, :]
        noise_tgt_sim_out = sim_output[-batch_N:, :, :]
        tgt_sim_out = F.normalize(tgt_sim_out, p=2, dim=-1)
        noise_tgt_sim_out = F.normalize(noise_tgt_sim_out, p=2, dim=-1)

        return tgt_cls_out, text_cls_out, tgt_sim_out, noise_tgt_sim_out

    def _text_only_joint_learning_PP(self, query, tgt, tgt_noise, label_embedding):
        batch_N = tgt.size(0)  # query: (b, 1, text_token)
        # Text features
        text_feature = self.text_encoder(**query).last_hidden_state  # query: (b, text_token, dim)

        # Image features
        combined_image = torch.cat([tgt, tgt_noise], dim=0)
        combined_image_feature = self.image_encoder(combined_image)
        tgt_feature = combined_image_feature[:batch_N, :, :]
        tgt_noise_feature = combined_image_feature[-batch_N:, :, :]

        # Interactor
        text_attention_mask = torch.cat(
            [torch.zeros((batch_N, 1), device=tgt.device, dtype=torch.bool), (query["attention_mask"] == 0)],
            dim=1)  # (b, 1, 150) False: not_mask, True: mask
        label_attention_mask = torch.zeros((batch_N, 1 + self.num_labels), device=tgt.device, dtype=torch.bool)
        text_only_key_padding_mask = torch.cat([text_attention_mask, label_attention_mask], dim=1)

        text_hidden = self.interactor(image_feature=None, text_feature=text_feature, label_embedding=label_embedding,
                                      padding_mask=text_only_key_padding_mask)
        combined_image_hidden = self.interactor(image_feature=combined_image_feature, text_feature=None,
                                                label_embedding=label_embedding)
        tgt_hidden = combined_image_hidden[:batch_N, :, :]
        noise_hidden = combined_image_hidden[-batch_N:, :, :]

        # CLS Task
        classifier_output = self.classifier_layer(
            torch.cat([text_hidden, combined_image_hidden], dim=0))  # b*num_labels*num_labels
        diag_mask = torch.eye(classifier_output.size(1)).unsqueeze(0).repeat(classifier_output.size(0), 1, 1).cuda()
        classifier_output = (classifier_output * diag_mask).sum(-1)

        text_cls_out = classifier_output[:batch_N, :]
        tgt_cls_out = classifier_output[batch_N:2 * batch_N, :] # B * num_labels

        # SIM Task
        sim_output = self.distance_layer(combined_image_hidden)
        tgt_sim_out = sim_output[:batch_N, :, :]
        noise_tgt_sim_out = sim_output[-batch_N:, :, :]
        tgt_sim_out = F.normalize(tgt_sim_out, p=2, dim=-1)
        noise_tgt_sim_out = F.normalize(noise_tgt_sim_out, p=2, dim=-1) # b * num_labels * dim

        y = self.gumbel_sample(tgt_cls_out, hard=False)  # B * num_labels (0/1)

        # conf_out = noise_hidden
        # for layer in self.conf_proj_layer:
        #     conf_out = layer(conf_out) # B * num_labels, distance_prc
        conf_out = self.conf_proj_layer(noise_hidden)

        center_embedding = label_embedding.repeat(noise_tgt_sim_out.size(0), 1, 1) # b * num_label * dim
        conf_label = ((F.cosine_similarity(tgt_sim_out, noise_tgt_sim_out, dim=-1) + 1) / 2) * self.distance_prc # [-1, 1] -> [0, 2] / 2 * 10

        return tgt_cls_out, text_cls_out, tgt_sim_out, noise_tgt_sim_out, y, conf_out, conf_label

    def _text_only_joint_learning(self, query, tgt, tgt_noise, label_embedding):
        # pair: <q, q_noise, beta_noise, q_tgt, beta>
        # suppose query hase tokenizer

        batch_N = tgt.size(0) # query: (b, 1, text_token)
        # Text features
        text_feature = self.text_encoder(**query).last_hidden_state # query: (b, text_token, dim)

        # Image features
        combined_image = torch.cat([tgt, tgt_noise], dim=0)
        combined_image_feature = self.image_encoder(combined_image)
        tgt_feature = combined_image_feature[:batch_N, :, :]
        tgt_noise_feature = combined_image_feature[-batch_N:, :, :]

        # Interactor
        text_attention_mask = torch.cat([torch.zeros((batch_N, 1), device=tgt.device, dtype=torch.bool), (query["attention_mask"] == 0)], dim=1)  # (b, 1, 150) False: not_mask, True: mask
        label_attention_mask = torch.zeros((batch_N, 1 + self.num_labels), device=tgt.device, dtype=torch.bool)
        text_only_key_padding_mask = torch.cat([text_attention_mask, label_attention_mask], dim=1)

        text_hidden = self.interactor(image_feature=None, text_feature=text_feature, label_embedding=label_embedding, padding_mask=text_only_key_padding_mask)
        combined_image_hidden = self.interactor(image_feature=combined_image_feature, text_feature=None, label_embedding=label_embedding)
        tgt_hidden = combined_image_hidden[:batch_N, :, :]

        # CLS Task
        classifier_output = self.classifier_layer(torch.cat([text_hidden, combined_image_hidden], dim=0))  # b*num_labels*num_labels
        diag_mask = torch.eye(classifier_output.size(1)).unsqueeze(0).repeat(classifier_output.size(0), 1, 1).cuda()
        classifier_output = (classifier_output * diag_mask).sum(-1)

        text_cls_out = classifier_output[:batch_N, :]
        tgt_cls_out = classifier_output[batch_N:2*batch_N, :]
        noise_tgt_cls_out = classifier_output[-batch_N, :]

        # SIM Task
        sim_output = self.distance_layer(torch.cat([text_hidden, combined_image_hidden], dim=0))
        text_sim_out = sim_output[:batch_N, :, :]
        tgt_sim_out = sim_output[batch_N:2 * batch_N, :, :]
        noise_tgt_sim_out = sim_output[-batch_N:, :, :]
        text_sim_out = F.normalize(text_sim_out, p=2, dim=-1)
        tgt_sim_out = F.normalize(tgt_sim_out, p=2, dim=-1)
        noise_tgt_sim_out = F.normalize(noise_tgt_sim_out, p=2, dim=-1)


        # label_embedding_expand = self.label_embedding.repeat(batch_N, 1, 1)  # B, num_labels, dim
        # center_embedding = F.normalize(label_embedding_expand, p=2, dim=-1)
        #
        # # construct tgt sim id
        # tgt_sim_ctn = F.cosine_similarity(center_embedding, tgt_sim_out, dim=-1)
        # tgt_sim_id = self._distanceID_STE(tgt_sim_ctn)
        #
        # # SIM indexing
        # mask_tgt = self.gumbel_sample(tgt_cls_out).bool()
        # mask_tgt_reshape = mask_tgt.reshape(-1)
        # indexing_input = tgt_hidden.reshape(-1, self.embed_dim)[mask_tgt_reshape]
        # label_emb_indexing = label_embedding_expand.reshape(-1, self.embed_dim)[mask_tgt_reshape]
        # indexing_tgt = tgt_sim_id.reshape(-1)[mask_tgt_reshape]
        # indexing_weights = torch.ones_like(indexing_tgt)
        #
        # # SIM retireval
        # mask_text = self.gumbel_sample(text_cls_out).bool()
        # mask_text_reshape = mask_text.reshape(-1)
        # retrieval_input = text_hidden.reshape(-1, self.embed_dim)[mask_text_reshape]
        # label_emb_retrieval = label_embedding_expand.reshape(-1, self.embed_dim)[mask_text_reshape]
        # retrieval_tgt = tgt_sim_id.reshape(-1)[mask_text_reshape]
        # retrieval_weights = torch.ones_like(retrieval_tgt)
        #
        # # Decoder
        # combine_input = torch.cat((indexing_input, retrieval_input), dim=0)  # B+B, dim
        # combine_label_emb = torch.cat((label_emb_indexing, label_emb_retrieval), dim=0)  # B+B, dim
        # dec_input = torch.cat((combine_input, combine_label_emb), dim=-1)
        # combine_tgt = torch.cat((indexing_tgt, retrieval_tgt), dim=0)
        # dec_output = self.decoder(dec_input)
        # combine_weights = torch.cat((indexing_weights, retrieval_weights), dim=0)

        return tgt_cls_out, text_cls_out, tgt_sim_out, noise_tgt_sim_out

    def _ml_classification(self, src, label_embedding):
        enc_output = self.encoder(src, label_embedding)  # B, num_labels, embed_dim

        # Classifier task
        classifier_output = self.classifier_layer(enc_output)  # b*num_labels*num_labels
        # classifier_output = torch.matmul(enc_output, self.label_embedding.t())
        diag_mask = torch.eye(classifier_output.size(1)).unsqueeze(0).repeat(classifier_output.size(0), 1, 1).cuda()
        classifier_output = (classifier_output * diag_mask).sum(-1)

        # global
        # classifier_sigmoid = torch.sigmoid(classifier_output).unsqueeze(-1)
        # global_embed = (enc_output * classifier_sigmoid).sum(dim=1)
        # global_embed = self.global_feature_layer(global_embed) # B * embed_dim

        # global (cat)
        # B, _, _ = enc_output.shape
        # global_embed = self.global_feature_layer(enc_output.reshape(B, -1))

        # global (hash)
        # proxies_embed = self.proxies_proj(proxies_embedding)

        # distance task
        distance_output = self.distance_layer(enc_output)  # b*num_labels*hidden
        # distance_output = enc_output
        # normalize
        distance_output = F.normalize(distance_output, p=2, dim=-1)

        return classifier_output, distance_output

    def _tokenizer(self, src, label_embedding):
        # Image features
        image_feature = self.image_encoder(src)

        # Interactor
        image_hidden = self.interactor(image_feature=image_feature, text_feature=None,
                                                label_embedding=label_embedding)

        # label embedding
        label_embedding_expand = self.label_embedding.repeat(image_hidden.size(0), 1, 1)

        # Classifier task
        classifier_output = self.classifier_layer(image_hidden)  # b*num_labels*num_labels
        diag_mask = torch.eye(classifier_output.size(1)).unsqueeze(0).repeat(classifier_output.size(0), 1, 1).cuda()
        classifier_output = (classifier_output * diag_mask).sum(-1)

        # distance task
        distance_output = self.distance_layer(image_hidden)  # b*num_labels*hidden
        # normalize
        distance_output = F.normalize(distance_output, p=2, dim=-1)

        # Decoder task
        tgt, indices, tgt_id = self._construct_id(classifier_output, distance_output)  # tgt: (N, 1)

        # ####
        # reshape_enc_output = image_hidden[indices[:, 0], indices[:, 1]]
        #
        # #### 修改：加入embedding
        # label_embedding_expand = label_embedding_expand[indices[:, 0], indices[:, 1]]
        # reshape_enc_output = torch.cat((reshape_enc_output, label_embedding_expand), dim=-1)
        #
        # dec_output = self.decoder(reshape_enc_output)
        # _, distance_code = torch.max(F.softmax(dec_output, dim=-1), dim=-1)
        # tgt_id[:, 1] = distance_code

        return tgt_id, distance_output

    def _tokenizer_PP(self, src, label_embedding):
        # Image features
        image_feature = self.image_encoder(src)

        # Interactor
        image_hidden = self.interactor(image_feature=image_feature, text_feature=None,
                                                label_embedding=label_embedding)

        # label embedding
        label_embedding_expand = self.label_embedding.repeat(image_hidden.size(0), 1, 1)

        # Classifier task
        classifier_output = self.classifier_layer(image_hidden)  # b*num_labels*num_labels
        diag_mask = torch.eye(classifier_output.size(1)).unsqueeze(0).repeat(classifier_output.size(0), 1, 1).cuda()
        classifier_output = (classifier_output * diag_mask).sum(-1)

        classification_res = (torch.sigmoid(classifier_output) > 0.5)
        indices = torch.nonzero(classification_res, as_tuple=False).cuda()

        # conf_output = image_hidden
        # for layer in self.conf_proj_layer:
        #     conf_output = layer(conf_output) # b * num_label * distance_prc
        conf_output = self.conf_proj_layer(image_hidden)
        conf_pred = torch.argmax(F.softmax(conf_output, dim=-1), dim=-1) # B*num_label

        # distance task
        distance_output = self.distance_layer(image_hidden)  # b*num_labels*hidden
        # normalize
        distance_output = F.normalize(distance_output, p=2, dim=-1)

        # Id
        N = int(classification_res.sum().item())
        tgt_id = torch.zeros((N, 2)).cuda()
        tgt_id[:, 0] = indices[:, 1]
        tgt_id[:, 1] = conf_pred[indices[:, 0], indices[:, 1]]

        return tgt_id, distance_output

    def forward(self, query=None, tgt=None, tgt_noise=None, stage="classification", device=None):
        if tgt is not None:
            self.label_embedding = self.label_embedding.to(device)
        else:
            self.label_embedding = self.label_embedding.to(device)

        if stage == "tokenizer":
            return self._tokenizer(query, self.label_embedding)

        if stage == "tokenizer_PP":
            return self._tokenizer_PP(query, self.label_embedding)

        if stage == "warm":
            return self._warm_learning(query, tgt, tgt_noise, self.label_embedding)

        if stage == "text_only_joint_learning":
            return self._text_only_joint_learning(query, tgt, tgt_noise, self.label_embedding)

        if stage == "abl_ind":
            return self._abl_ind(tgt, self.label_embedding)

        if stage == "abl_ret":
            return self._abl_ret(query, self.label_embedding)

        if stage == "abl_ind_ret":
            return self._abl_ind_red(query, tgt, self.label_embedding)

        if stage == "abl_hard_way":
            return self._text_only_joint_learning_hardway(query, tgt, tgt_noise, self.label_embedding)

        if stage == "abl_MT":
            return self._text_only_joint_learning_MT(query, tgt, tgt_noise, self.label_embedding)

        if stage == "PP":
            return self._text_only_joint_learning_PP(query, tgt, tgt_noise, self.label_embedding)

    @torch.no_grad()
    def _get_classification(self, classifier_output):
        # classifier_output： B, num_labels
        prob = torch.sigmoid(classifier_output)
        classification_res = (prob > 0.5).float()
        indices = torch.nonzero(classification_res, as_tuple=False)
        if indices.numel() == 0:
            max_prob_idx = torch.argmax(prob)
            row = max_prob_idx // classifier_output.size(1)
            col = max_prob_idx % classifier_output.size(1)
            indices = torch.tensor([[row, col]], dtype=torch.int64)
            probs = prob[row, col].unsqueeze(0)
        else:
            sorted_indices = torch.argsort(prob.view(-1), descending=True)
            sorted_indices = sorted_indices[prob.view(-1)[sorted_indices] > 0.5]
            indices = torch.stack((sorted_indices // classifier_output.size(1),
                                   sorted_indices % classifier_output.size(1)), dim=1)
            probs = prob.view(-1)[sorted_indices]

        return indices, probs

    # Test if image query
    @torch.no_grad()
    def search_img_query(self, args, query, beam_size, dict_tree=None):
        self.label_embedding = self.label_embedding.to(args.device)
        batch_N = query.size(0)
        # Image features
        image_feature = self.image_encoder(query)

        # Interactor
        image_hidden = self.interactor(image_feature=image_feature, text_feature=None, label_embedding=self.label_embedding)

        B, num_labels, embed_dim = image_hidden.shape  # B = 1

        # label embedding expand
        label_embedding_expand = self.label_embedding.repeat(B, 1, 1)

        # classifier res
        classifier_output = self.classifier_layer(image_hidden)  # b*num_labels*num_labels
        # classifier_output = torch.matmul(enc_output, self.label_embedding.t())
        diag_mask = torch.eye(classifier_output.size(1)).unsqueeze(0).repeat(classifier_output.size(0), 1, 1).cuda()
        classifier_output = (classifier_output * diag_mask).sum(-1)
        indices, cls_score = self._get_classification(classifier_output)  # classifier_output: b * num_labels

        enc_output_reshape = image_hidden[indices[:, 0], indices[:, 1]]  # (N = 1 * labels_by_class, embed_dim)
        enc_output_reshape = torch.cat((enc_output_reshape, label_embedding_expand[indices[:, 0], indices[:, 1]]),
                                       dim=-1)
        N, _ = enc_output_reshape.shape

        ans = torch.zeros(N, beam_size, 2, dtype=torch.long)
        ans[:, :, 0] = indices[:, 1][:, np.newaxis]

        sos = self.decoder(enc_output_reshape)  # enc_output_reshape: (N, 1, embed_dim)

        # sos: [(N, class1_num)]
        sos = F.softmax(sos, dim=-1)
        pd = torch.ones(N, sos.shape[-1]).cuda() * -1
        for i in range(N):
            if int(indices[i, 1]) in dict_tree.nodes.keys():
                tmp = dict_tree.nodes[int(indices[i, 1])]
                for j in tmp.nodes.keys():
                    pd[i, j] = sos[i, j]
        v, i = torch.topk(pd, k=beam_size, dim=-1)  # v, i: (N, beam_size)

        # no constrain
        # sos = F.softmax(sos, dim=-1)
        # v, i = torch.topk(sos, k=beam_size, dim=-1)

        ans[:, :, 1] = i

        score = v * cls_score.unsqueeze(1).expand(-1, beam_size)
        # cls_s = classifier_output[indices[:, 0], indices[:, 1]]
        return ans, score  # N, beam_size

    @torch.no_grad()
    def search(self, args, query, beam_size=10, dict_tree=None):
        self.label_embedding = self.label_embedding.to(args.device)

        batch_N = query["input_ids"].size(0)
        # Text features
        text_feature = self.text_encoder(**query).last_hidden_state  # query: (b, text_token, dim)

        # Interactor
        text_attention_mask = torch.cat(
            [torch.zeros((batch_N, 1), device=args.device, dtype=torch.bool), (query["attention_mask"] == 0)],
            dim=1)  # (b, 1, 150) False: not_mask, True: mask
        label_attention_mask = torch.zeros((batch_N, 1 + self.num_labels), device=args.device, dtype=torch.bool)
        text_only_key_padding_mask = torch.cat([text_attention_mask, label_attention_mask], dim=1)

        text_hidden = self.interactor(image_feature=None, text_feature=text_feature, label_embedding=self.label_embedding,
                                      padding_mask=text_only_key_padding_mask) # 1, num_labels, embed_dim

        B, num_labels, embed_dim = text_hidden.shape # B = 1

        # label embedding expand
        label_embedding_expand = self.label_embedding.repeat(B, 1, 1)

        # classifier res
        classifier_output = self.classifier_layer(text_hidden)  # b*num_labels*num_labels
        # classifier_output = torch.matmul(enc_output, self.label_embedding.t())
        diag_mask = torch.eye(classifier_output.size(1)).unsqueeze(0).repeat(classifier_output.size(0), 1, 1).cuda()
        classifier_output = (classifier_output * diag_mask).sum(-1)
        indices, cls_score = self._get_classification(classifier_output) # classifier_output: b * num_labels

        enc_output_reshape = text_hidden[indices[:, 0], indices[:, 1]] # (N = 1 * labels_by_class, embed_dim)
        enc_output_reshape = torch.cat((enc_output_reshape, label_embedding_expand[indices[:, 0], indices[:, 1]]), dim=-1)
        N, _ = enc_output_reshape.shape

        ans = torch.zeros(N, beam_size, 2, dtype=torch.long)
        ans[:, :, 0] = indices[:, 1][:, np.newaxis]

        sos = self.decoder(enc_output_reshape)  # enc_output_reshape: (N, 1, embed_dim)

        # sos: [(N, class1_num)]
        sos = F.softmax(sos, dim=-1)
        pd = torch.ones(N, sos.shape[-1]).cuda() * -1
        for i in range(N):
            if int(indices[i, 1]) in dict_tree.nodes.keys():
                tmp = dict_tree.nodes[int(indices[i, 1])]
                for j in tmp.nodes.keys():
                    pd[i, j] = sos[i, j]
        v, i = torch.topk(pd, k=beam_size, dim=-1)  # v, i: (N, beam_size)

        # no constrain
        # sos = F.softmax(sos, dim=-1)
        # v, i = torch.topk(sos, k=beam_size, dim=-1)

        ans[:, :, 1] = i

        score = v * cls_score.unsqueeze(1).expand(-1, beam_size)
        # cls_s = classifier_output[indices[:, 0], indices[:, 1]]
        return ans, score  # N, beam_size


    @torch.no_grad()
    def searchV2(self, args, query):
        self.label_embedding = self.label_embedding.to(args.device)

        batch_N = query["input_ids"].size(0)
        # Text features
        text_feature = self.text_encoder(**query).last_hidden_state  # query: (b, text_token, dim)

        # Interactor
        text_attention_mask = torch.cat(
            [torch.zeros((batch_N, 1), device=args.device, dtype=torch.bool), (query["attention_mask"] == 0)],
            dim=1)  # (b, 1, 150) False: not_mask, True: mask
        label_attention_mask = torch.zeros((batch_N, 1 + self.num_labels), device=args.device, dtype=torch.bool)
        text_only_key_padding_mask = torch.cat([text_attention_mask, label_attention_mask], dim=1)

        text_hidden = self.interactor(image_feature=None, text_feature=text_feature,
                                      label_embedding=self.label_embedding,
                                      padding_mask=text_only_key_padding_mask)  # 1, num_labels, embed_dim

        B, num_labels, embed_dim = text_hidden.shape  # B = 1

        # label embedding expand
        label_embedding_expand = self.label_embedding.repeat(B, 1, 1)

        # classifier res
        classifier_output = self.classifier_layer(text_hidden)  # b*num_labels*num_labels
        # classifier_output = torch.matmul(enc_output, self.label_embedding.t())
        diag_mask = torch.eye(classifier_output.size(1)).unsqueeze(0).repeat(classifier_output.size(0), 1, 1).to(args.device)
        classifier_output = (classifier_output * diag_mask).sum(-1)
        indices, cls_score = self._get_classification(classifier_output)  # classifier_output: b * num_labels
        ans = indices[:, 1]
        return ans

    # 所有的label feature都进行生成
    def search_exp1(self, src, beam_size=10, dict_tree=None):
        # encoder feature
        enc_output, attns = self.encoder(src, self.label_embedding)  # 1, num_labels, embed_dim

        B, num_labels, embed_dim = enc_output.shape # B = 1

        enc_output_reshape = enc_output.reshape(-1, embed_dim)

        N, _ = enc_output_reshape.shape

        # classifier res
        classifier_output = self.classifier_layer(enc_output)  # b*num_labels*num_labels
        # classifier_output = torch.matmul(enc_output, self.label_embedding.t())
        diag_mask = torch.eye(classifier_output.size(1)).unsqueeze(0).repeat(classifier_output.size(0), 1, 1).cuda()
        cls_score = torch.sigmoid((classifier_output * diag_mask).sum(-1)).reshape(-1) # B, num_labels

        ans = torch.zeros(N, beam_size, 2, dtype=torch.long)
        ans[:, :, 0] = torch.arange(N).unsqueeze(1)

        sos = self.decoder(None, enc_output_reshape.unsqueeze(1)) # enc_output_reshape: (N, 1, embed_dim)

        # sos: [(N, 1, class1_num)]
        sos = F.softmax(sos[0], dim=-1)
        pd = torch.ones(N, sos.shape[-1]).cuda() * -1
        for i in range(N):
            if i in dict_tree.nodes.keys():
                tmp = dict_tree.nodes[i]
                for j in tmp.nodes.keys():
                    pd[i, j] = sos[i, j]
        v, i = torch.topk(pd, k=beam_size, dim=-1) # v, i: (N, beam_size)
        ans[:, :, 1] = i

        score = v * cls_score.unsqueeze(1).expand(-1, beam_size)
        return ans, score # N, beam_size

    def search_label_guild(self, src, q_label, beam_size=10, dict_tree=None, label_embedding=None):
        # q_label: N, num_labels
        # encoder feature
        enc_output, attns = self.encoder(src, label_embedding)  # 1, num_labels, embed_dim

        B, num_labels, embed_dim = enc_output.shape  # B = 1

        indices = torch.nonzero(q_label, as_tuple=False).cuda()

        enc_output_reshape = enc_output[indices[:, 0], indices[:, 1]]  # (N = 1 * labels_by_class, embed_dim)
        N, _ = enc_output_reshape.shape

        ans = torch.zeros(N, beam_size, 2, dtype=torch.long)
        ans[:, :, 0] = indices[:, 1][:, np.newaxis]

        sos = self.decoder(None, enc_output_reshape.unsqueeze(1))  # enc_output_reshape: (N, 1, embed_dim)

        # sos: [(N, 1, class1_num)]
        sos = F.softmax(sos[0], dim=-1)
        pd = torch.ones(N, sos.shape[-1]).cuda() * -1
        for i in range(N):
            if int(indices[i, 1]) in dict_tree.nodes.keys():
                tmp = dict_tree.nodes[int(indices[i, 1])]
                for j in tmp.nodes.keys():
                    pd[i, j] = sos[i, j]
        v, i = torch.topk(pd, k=beam_size, dim=-1)  # v, i: (N, beam_size)
        ans[:, :, 1] = i

        score = v
        return ans, score  # N, beam_size

def build_SIRMM(args, label_embedding):
    if args.image_encoder_model == "resnet101":
        image_encoder = ResNet101()
    else:
        image_encoder = ResNet50()
    text_encoder = BertModel.from_pretrained(args.text_encoder_model_path)
    interactor = Interactor(args.image_token, args.text_token, args.num_labels, args.hidden_dim, args.layers, args.heads, args.dropout)
    # decoder = NoiseMLP(input_dim=args.hidden_dim + label_embedding.size(-1), forward_dim=args.decoder_forward_dim, num_classes=args.distance_prc)
    sirmm = SIRMM(image_encoder, text_encoder, interactor, label_embedding, args.num_labels, args.hidden_dim, args.distance_prc)
    return sirmm

if __name__ == '__main__':
    pass