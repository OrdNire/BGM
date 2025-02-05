import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .transformer_layers import SelfAttnLayer, DecoderBlock
from .backbone import Backbone, ResNet50, ResNet101
from .utils import weights_init, trunc_normal_
from .position_enc import positionalencoding2d
from .DictTree import TreeNode

def get_attn_pad_mask(seq_q, seq_k, num):
    batch_size, len_q = seq_q.size() # B, N
    batch_size, len_k = seq_k.size()
    # eq(zero) is PAD token
    pad_attn_mask = torch.cat((torch.tensor([[1] for _ in range(batch_size)]).to(seq_q.device),seq_q), dim=1).data.eq(num).unsqueeze(1)  # batch_size x 1 x len_k, one is masking
    return pad_attn_mask.expand(batch_size, len_q+1, len_k+1)

class CTranModel(nn.Module):
    def __init__(self, num_labels, hidden=768,
                 pos_emb=False, layers=3, heads=4, dropout=0.1,
                 input_channels=3):
        super(CTranModel, self).__init__()

        # ResNet backbone
        self.backbone = ResNet101(input_channels)

        self.conv_downsample = torch.nn.Conv2d(2048, hidden, (1, 1))

        # if hidden != 2048:
        #
        # else:
        #     self.conv_downsample = torch.nn.Identity()

        # Label Embeddings
        self.label_input = torch.Tensor(np.arange(num_labels)).view(1, -1).long()
        self.label_lt = torch.nn.Embedding(num_labels, hidden, padding_idx=None)

        # Position Embeddings (for image features)
        self.use_pos_enc = pos_emb
        if self.use_pos_enc:
            # self.position_encoding = PositionEmbeddingSine(int(hidden/2), normalize=True)
            self.position_encoding = positionalencoding2d(hidden, 18, 18).unsqueeze(0)

        # Transformer
        self.self_attn_layers = nn.ModuleList([SelfAttnLayer(hidden, heads, dropout) for _ in range(layers)])

        # Other
        self.LayerNorm = nn.LayerNorm(hidden)
        self.dropout = nn.Dropout(dropout)

        # Init all except pretrained backbone
        self.label_lt.apply(weights_init)
        self.LayerNorm.apply(weights_init)
        self.self_attn_layers.apply(weights_init)

    def forward(self, images, label_embedding=None):
        # label_embedding (B, num_labels, embed_dim)
        if label_embedding is None:
            const_label_input = self.label_input.repeat(images.size(0), 1)
            self.init_label_embeddings = self.label_lt(const_label_input)
        else:
            # self.init_label_embeddings = label_embedding
            self.init_label_embeddings = label_embedding.repeat(images.size(0), 1, 1)

        features = self.backbone(images)

        features = self.conv_downsample(features)

        if self.use_pos_enc:
            pos_encoding = self.position_encoding(features,
                                                  torch.zeros(features.size(0), 18, 18, dtype=torch.bool))
            features = features + pos_encoding

        features = features.view(features.size(0), features.size(1), -1).permute(0, 2, 1)

        # (B, f_dim, 18, 18) -> (B, 18*18, f_dim)
        embeddings = torch.cat((features, self.init_label_embeddings), 1)

        # Feed image and label embeddings through Transformer
        embeddings = self.LayerNorm(embeddings)
        attns = []
        for layer in self.self_attn_layers:
            embeddings, attn = layer(embeddings, mask=None)
            attns += attn.detach().unsqueeze(0).data

        # Readout each label embedding using a linear layer
        label_embeddings = embeddings[:, -self.init_label_embeddings.size(1):, :]  # (B, num_labels, hidden)

        return label_embeddings

class Decoder(nn.Module):
    def __init__(self, embed_dim=384, num_classes=[17, 1000], id_len=0, depth=12, num_heads=12,
                 mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, decode_only=False, **kwargs):
        super().__init__()
        # assert len(num_classes) == id_len
        self.num_heads = num_heads
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        self.decode_only = decode_only
        self.enc_proj = nn.Linear(embed_dim, embed_dim * 10) if decode_only else nn.Identity()
        # proj embedding
        self.label_embed = nn.ModuleList([
            nn.Embedding(num_class, embed_dim) for num_class in num_classes
        ])

        self.start_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed_dec = nn.Parameter(torch.zeros(1, id_len + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            DecoderBlock(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])

        self.norm = nn.LayerNorm(embed_dim)

        # voc proj
        self.label_prj = nn.ModuleList([
            nn.Linear(embed_dim, num_class) for num_class in num_classes
        ])

        trunc_normal_(self.pos_embed_dec, std=.02)
        trunc_normal_(self.start_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def pos_embed(self, x):
        return self.pos_embed_dec

    def prepare_tokens(self, x, feats_num=None):
        if x is not None:
            B, N = x.shape

            embed_proj_x = torch.zeros((B, N, self.embed_dim), device=x.device)
            for i in range(N):
                embed_proj_x[:, i, :] = self.label_embed[i](x[:, i])

            # add the [CLS] token to the embed patch tokens
            start_tokens = self.start_token.expand(B, -1, -1)
            x = torch.cat((start_tokens, embed_proj_x), dim=1)
        else:
            B, N = feats_num, 0
            x = self.start_token.repeat(B, 1, 1)
        # add positional encoding to each token
        pos_emb = self.pos_embed(x)
        pos_emb = pos_emb[:, :N + 1, :]
        x = x + pos_emb
        return self.pos_drop(x)

    def forward(self, x, enc_output):
        # x: B, num_token
        if x is not None:
            mask = True
        else:
            mask = None

        x = self.prepare_tokens(x, enc_output.shape[0])
        if self.decode_only:
            enc_output = self.enc_proj(enc_output).reshape(enc_output.shape[0], 10, -1)

        for blk in self.blocks:
            x = blk(x, enc_output, mask)
        x = self.norm(x)

        return_x = [self.label_prj[i](x[:, i, :]) for i in range(min(len(self.label_prj), x.shape[1]))]
        # x = self.label_prj(x)  # B, N, emb_dim
        return return_x

class NoiseMLP(nn.Module):
    def __init__(self, input_dim=768, forward_dim=2048, dropout_rate=0.1, num_classes=10, layer_num=6):
        super(NoiseMLP, self).__init__()

        self.mlp = nn.ModuleList([NoiseMLPLayer(input_dim, forward_dim, dropout_rate) for i in range(layer_num)])

        self.classifier = nn.Linear(input_dim, num_classes)

        self.classifier.apply(weights_init)

    def forward(self, x):
        for blk in self.mlp:
            x = blk(x)
        logits = self.classifier(x)
        return logits

class NoiseMLPLayer(nn.Module):
    def __init__(self, input_dim=768, forward_dim=2048, dropout_rate=0.1):
        super(NoiseMLPLayer, self).__init__()
        self.fc1 = nn.Linear(input_dim, forward_dim)
        self.activation = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(forward_dim, input_dim)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.layer_norm = nn.LayerNorm(input_dim)

        self.fc1.apply(weights_init)
        self.fc2.apply(weights_init)
        self.layer_norm.apply(weights_init)

    def forward(self, x):
        residual = x
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        x = self.layer_norm(x + residual)
        return x

class SIR(nn.Module):

    def __init__(self, id_len=4, input_channels=3, label_embedding=None,
                 num_classes=[17, 1000], num_labels=17, embed_dim=768, distance_prc=100,
                 enc_depth=6, dec_depth=6, tau=10.0,
                 num_heads=4, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, **kwargs):
        super().__init__()
        self.num_labels = num_labels
        self.distance_prc = distance_prc
        self.label_embedding = label_embedding # num_labels, embed_dim
        self.embed_dim = embed_dim

        # exp
        self.select_t = 0.4

        self.tau = 10.0 # temp

        self.encoder = CTranModel(num_labels, hidden=embed_dim, layers=enc_depth, heads=num_heads, dropout=drop_rate,
                                  input_channels=input_channels)

        self.decoder_type =  kwargs.get("decoder_type", "Transformer")
        if self.decoder_type == "Transformer":
            self.decoder = Decoder(embed_dim=embed_dim, num_classes=num_classes, id_len=id_len, depth=dec_depth,
                                   num_heads=num_heads,
                                   mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop_rate=drop_rate,
                                   attn_drop_rate=attn_drop_rate,
                                   drop_path_rate=drop_path_rate, norm_layer=norm_layer, **kwargs)
        elif self.decoder_type == "MLP":
            self.decoder = NoiseMLP(input_dim=embed_dim+label_embedding.size(-1), forward_dim=embed_dim * 4, num_classes=distance_prc)

        # label embedding proj
        if embed_dim != 768:
            self.label_proj = torch.nn.Linear(768, embed_dim)
        else:
            self.label_proj = torch.nn.Identity()

        # Classifier
        # Output is of size num_labels because we want a separate classifier for each label
        self.classifier_layer = torch.nn.Linear(embed_dim, num_labels)

        # Distance Preserve
        distance_dim = 768  # same with BERT output dim
        self.distance_layer = torch.nn.Linear(embed_dim, distance_dim)

        # # Global Feature Layer
        # self.global_feature_layer = torch.nn.Linear(embed_dim, distance_dim)
        # self.global_feature_layer = nn.Sequential(torch.nn.Linear(embed_dim, embed_dim), torch.nn.ReLU(), torch.nn.Linear(embed_dim, embed_dim))

        # Global Feature Layer (cat)
        # self.global_feature_layer = torch.nn.Linear(embed_dim * self.num_labels, embed_dim)

        # Global Feature Layer (hash)
        # self.global_feature_layer = nn.Sequential(torch.nn.Linear(embed_dim, embed_dim), torch.nn.ReLU(), torch.nn.Linear(embed_dim, 64))
        # self.proxies_proj = torch.nn.Linear(embed_dim, 64)

        self._reset_parameters()
        self.id_len = id_len
        self.num_classes = num_classes
        self.classifier_layer.apply(weights_init)
        self.distance_layer.apply(weights_init)
        # self.global_feature_layer.apply(weights_init)
        self.label_proj.apply(weights_init)
        # self.proxies_proj.apply(weights_init)

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

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
        normalized_cos_sim = ((cosine_sim + 1) / 2) * self.distance_prc # [0, 100]

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

    def gumbel_sample(self, cls):
        # cls: b*num_labels
        prob = F.sigmoid(cls)
        prob_new = torch.stack([prob, 1-prob], dim=-1)
        cls_filter = F.gumbel_softmax(prob_new, self.tau, hard=True)[:, :, 0]
        return cls_filter

    # V2基础上修改：Indexing tgt为 tgt_image的id
    def _joint_learningV3(self, src, noise_src, tgt_src, label_onehot, label_embedding, tgt_noise_level):
        # src:query noise_src: noise query tgt_src: tgt
        # src = src + noise_src + tgt_src
        combined_src = torch.cat([src, noise_src, tgt_src], dim=0)
        batch_N = src.size(0)
        enc_output = self.encoder(combined_src, label_embedding)

        src_hidden = enc_output[:batch_N, :, :]  # N, num_labels, embed_dim
        noise_hidden = enc_output[batch_N:2 * batch_N, :, :]
        tgt_hidden = enc_output[-batch_N:, :, :]

        src_noise_hidden = enc_output[:2 * batch_N, :, :]

        # Classifier task
        src_tgt_hidden = torch.cat((src_hidden, tgt_hidden), dim=0)
        classifier_output = self.classifier_layer(src_tgt_hidden)  # b*num_labels*num_labels
        diag_mask = torch.eye(classifier_output.size(1)).unsqueeze(0).repeat(classifier_output.size(0), 1, 1).cuda()
        classifier_output = (classifier_output * diag_mask).sum(-1)

        src_cls_out = classifier_output[:batch_N, :]
        tgt_cls_out = classifier_output[-batch_N:, :]

        # Distance task
        distance_output = self.distance_layer(enc_output)  # b*num_labels*hidden
        distance_src = distance_output[:batch_N, :, :]
        distance_noise = distance_output[batch_N:2 * batch_N, :, :]
        distance_tgt = distance_output[-batch_N:, :, :]

        distance_src_out = F.normalize(distance_src, p=2, dim=-1)
        distance_noise_out = F.normalize(distance_noise, p=2, dim=-1)
        distance_tgt_out = F.normalize(distance_tgt, p=2, dim=-1)

        # Indexing&Retrieval task
        # center_embedding = label_embedding.detach().cuda()
        label_embedding_expand = self.label_embedding.repeat(batch_N, 1, 1) # B, num_labels, dim
        center_embedding = F.normalize(label_embedding_expand, p=2, dim=-1)

        s_src = F.cosine_similarity(center_embedding, distance_src_out, dim=-1)
        s_src_norm = self._distanceID_STE(s_src)

        s_tgt = F.cosine_similarity(center_embedding, distance_tgt_out, dim=-1)
        s_tgt_norm = self._distanceID_STE(s_tgt)

        # mask_src = (src_cls_out > 0).bool()  # (N, num_labels)
        # mask_tgt = (tgt_cls_out > 0).bool()
        mask_src = self.gumbel_sample(src_cls_out).bool()
        mask_tgt = self.gumbel_sample(tgt_cls_out).bool()
        mask_retrieval = (mask_tgt * mask_src * label_onehot).bool()

        mask_tgt_reshape = mask_tgt.reshape(-1)
        mask_retrieval_reshape = mask_retrieval.reshape(-1)
        indexing_input = tgt_hidden.reshape(-1, self.embed_dim)[mask_tgt_reshape]
        label_emb_indexing = label_embedding_expand.reshape(-1, self.embed_dim)[mask_tgt_reshape]
        indexing_tgt = s_tgt_norm.reshape(-1)[mask_tgt_reshape]
        indexing_weights = torch.ones_like(indexing_tgt)

        retrieval_input = src_hidden.reshape(-1, self.embed_dim)[mask_retrieval_reshape]
        label_emb_retrieval = label_embedding_expand.reshape(-1, self.embed_dim)[mask_retrieval_reshape]
        retrieval_tgt = s_tgt_norm.reshape(-1)[mask_retrieval_reshape]
        retrieval_weights = tgt_noise_level.repeat(1, self.num_labels).reshape(-1)[mask_retrieval_reshape]

        combine_input = torch.cat((indexing_input, retrieval_input), dim=0) # B+B, dim
        combine_label_emb = torch.cat((label_emb_indexing, label_emb_retrieval), dim=0) # B+B, dim
        dec_input = torch.cat((combine_input, combine_label_emb), dim=-1)
        combine_tgt = torch.cat((indexing_tgt, retrieval_tgt), dim=0)
        if self.decoder_type == "Transformer":
            dec_output = self.decoder(combine_tgt.long(), dec_input.unsqueeze(1))
        elif self.decoder_type == "MLP":
            dec_output = self.decoder(dec_input)
        combine_weights = torch.cat((indexing_weights, retrieval_weights), dim=0)

        return src_cls_out, distance_src_out, distance_noise_out, dec_output, combine_tgt, combine_weights

    def _joint_learningV2(self, src, noise_src, tgt_src, label_onehot, label_embedding):
        # src = src + noise_src + tgt_src
        combined_src = torch.cat([src, noise_src, tgt_src], dim=0)
        batch_N = src.size(0)
        enc_output = self.encoder(combined_src, label_embedding)

        src_hidden = enc_output[:batch_N, :, :]  # N, num_labels, embed_dim
        noise_hidden = enc_output[batch_N:2 * batch_N, :, :]
        tgt_hidden = enc_output[-batch_N:, :, :]

        src_noise_hidden = enc_output[:2 * batch_N, :, :]

        # Classifier task
        src_tgt_hidden = torch.cat((src_hidden, tgt_hidden), dim=0)
        classifier_output = self.classifier_layer(src_tgt_hidden)  # b*num_labels*num_labels
        diag_mask = torch.eye(classifier_output.size(1)).unsqueeze(0).repeat(classifier_output.size(0), 1, 1).cuda()
        classifier_output = (classifier_output * diag_mask).sum(-1)

        src_cls_out = classifier_output[:batch_N, :]
        tgt_cls_out = classifier_output[-batch_N:, :]

        # Distance task
        distance_output = self.distance_layer(enc_output)  # b*num_labels*hidden
        distance_src = distance_output[:batch_N, :, :]
        distance_noise = distance_output[batch_N:2 * batch_N, :, :]
        distance_tgt = distance_output[-batch_N:, :, :]

        distance_src_out = F.normalize(distance_src, p=2, dim=-1)
        distance_noise_out = F.normalize(distance_noise, p=2, dim=-1)
        distance_tgt_out = F.normalize(distance_tgt, p=2, dim=-1)

        # Indexing&Retrieval task
        # center_embedding = label_embedding.detach().cuda()
        center_embedding = self.label_embedding.repeat(batch_N, 1, 1)
        center_embedding = F.normalize(center_embedding, p=2, dim=-1)

        s_src = F.cosine_similarity(center_embedding, distance_src_out, dim=-1)
        s_src_norm = self._distanceID_STE(s_src)

        s_tgt = F.cosine_similarity(center_embedding, distance_tgt_out, dim=-1)
        s_tgt_norm = self._distanceID_STE(s_tgt)

        # mask_src = (src_cls_out > 0).bool()  # (N, num_labels)
        # mask_tgt = (tgt_cls_out > 0).bool()
        mask_src = self.gumbel_sample(src_cls_out).bool()
        mask_tgt = self.gumbel_sample(tgt_cls_out).bool()
        mask_retrieval = (mask_tgt * mask_src * label_onehot).bool()

        mask_src_reshape = mask_src.reshape(-1)
        mask_retrieval_reshape = mask_retrieval.reshape(-1)
        indexing_input = src_hidden.reshape(-1, self.embed_dim)[mask_src_reshape]
        indexing_tgt = s_src_norm.reshape(-1)[mask_src_reshape]

        retrieval_input = src_hidden.reshape(-1, self.embed_dim)[mask_retrieval_reshape]
        retrieval_tgt = s_tgt_norm.reshape(-1)[mask_retrieval_reshape]

        combine_input = torch.cat((indexing_input, retrieval_input), dim=0)
        combine_tgt = torch.cat((indexing_tgt, retrieval_tgt), dim=0)
        if self.decoder_type == "Transformer":
            dec_output = self.decoder(combine_tgt.long(), combine_input.unsqueeze(1))
        elif self.decoder_type == "MLP":
            dec_output = self.decoder(combine_input)

        return src_cls_out, distance_src_out, distance_noise_out, dec_output, combine_tgt

    def _joint_learning(self, src, noise_src, tgt_src, label_onehot, label_embedding):
        # src = src + noise_src + tgt_src
        combined_src = torch.cat([src, noise_src, tgt_src], dim=0)
        batch_N = src.size(0)
        enc_output = self.encoder(combined_src, label_embedding)

        src_hidden = enc_output[:batch_N, :, :] # N, num_labels, embed_dim
        noise_hidden = enc_output[batch_N:2*batch_N, :, :]
        tgt_hidden = enc_output[-batch_N:, :, :]

        src_noise_hidden = enc_output[:2*batch_N, :, :]

        # Classifier task
        src_tgt_hidden = torch.cat((src_hidden, tgt_hidden), dim=0)
        classifier_output = self.classifier_layer(src_tgt_hidden)  # b*num_labels*num_labels
        diag_mask = torch.eye(classifier_output.size(1)).unsqueeze(0).repeat(classifier_output.size(0), 1, 1).cuda()
        classifier_output = (classifier_output * diag_mask).sum(-1)

        src_cls_out = classifier_output[:batch_N, :]
        tgt_cls_out = classifier_output[-batch_N:, :]

        # Distance task
        distance_output = self.distance_layer(enc_output)  # b*num_labels*hidden
        distance_src = distance_output[:batch_N, :, :]
        distance_noise = distance_output[batch_N:2*batch_N, :, :]
        distance_tgt = distance_output[-batch_N:, :, :]

        distance_src_out = F.normalize(distance_src, p=2, dim=-1)
        distance_noise_out = F.normalize(distance_noise, p=2, dim=-1)
        distance_tgt_out = F.normalize(distance_tgt, p=2, dim=-1)

        # Indexing&Retrieval task
        # center_embedding = label_embedding.detach().cuda()
        center_embedding = self.label_embedding.repeat(batch_N, 1, 1)
        center_embedding = F.normalize(center_embedding, p=2, dim=-1)

        s_src = F.cosine_similarity(center_embedding, distance_src_out, dim=-1)
        s_src_norm = ((s_src + 1) / 2) * self.distance_prc # B, num_labels

        s_tgt = F.cosine_similarity(center_embedding, distance_tgt_out, dim=-1)
        s_tgt_norm = ((s_tgt + 1) / 2) * self.distance_prc

        mask_src = (src_cls_out > 0).float() # (N, num_labels)
        mask_tgt = (tgt_cls_out > 0).float()
        mask_retrieval = mask_tgt * mask_src * label_onehot.float()

        indexing_input = (src_hidden * mask_src.unsqueeze(-1)).reshape(-1, self.embed_dim)
        indexing_tgt = torch.where(mask_src.bool() == 1, s_src_norm, torch.tensor(self.distance_prc)).reshape(-1)

        retrieval_input = (src_hidden * mask_retrieval.unsqueeze(-1)).reshape(-1, self.embed_dim)
        retrieval_tgt = torch.where(mask_retrieval.bool() == 1, s_tgt_norm, torch.tensor(self.distance_prc)).reshape(-1)

        combine_input = torch.cat((indexing_input, retrieval_input), dim=0)
        combine_tgt = torch.cat((indexing_tgt, retrieval_tgt), dim=0)
        if self.decoder_type == "Transformer":
            dec_output = self.decoder(combine_tgt.long(), combine_input.unsqueeze(1))
        elif self.decoder_type == "MLP":
            dec_output = self.decoder(combine_input)

        return src_cls_out, distance_src_out, distance_noise_out, dec_output, combine_tgt

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
        enc_output = self.encoder(src, label_embedding)  # B, num_labels, embed_dim

        # label embedding
        label_embedding_expand = self.label_embedding.repeat(enc_output.size(0), 1, 1)

        # Classifier task
        classifier_output = self.classifier_layer(enc_output)  # b*num_labels*num_labels
        # classifier_output = torch.matmul(enc_output, self.label_embedding.t())
        diag_mask = torch.eye(classifier_output.size(1)).unsqueeze(0).repeat(classifier_output.size(0), 1, 1).cuda()
        classifier_output = (classifier_output * diag_mask).sum(-1)

        # global embed
        classifier_sigmoid = torch.sigmoid(classifier_output).unsqueeze(-1)
        global_embed = (enc_output*classifier_sigmoid).sum(dim=1)
        # global_embed = self.global_feature_layer(global_embed)  # B * embed_dim

        # global (cat)
        # B, _, _ = enc_output.shape
        # global_embed = self.global_feature_layer(enc_output.reshape(B, -1))

        # distance task
        distance_output = self.distance_layer(enc_output)  # b*num_labels*hidden
        # distance_output = enc_output
        # normalize
        distance_output = F.normalize(distance_output, p=2, dim=-1)

        # Decoder task
        tgt, indices, tgt_id = self._construct_id(classifier_output, distance_output)  # tgt: (N, 1)

        ####
        reshape_enc_output = enc_output[indices[:, 0], indices[:, 1]]

        #### 修改：加入embedding
        label_embedding_expand = label_embedding_expand[indices[:, 0], indices[:, 1]]
        reshape_enc_output = torch.cat((reshape_enc_output, label_embedding_expand), dim=-1)

        if self.decoder_type == "Transformer":
            dec_output = self.decoder(tgt.long(), reshape_enc_output.unsqueeze(1))
        elif self.decoder_type == "MLP":
            dec_output = self.decoder(reshape_enc_output)
        _, distance_code = torch.max(F.softmax(dec_output, dim=-1), dim=-1)
        tgt_id[:, 1] = distance_code

        return tgt_id, enc_output, global_embed

    def _indexing_learning(self, src, label_embedding):
        enc_output = self.encoder(src, label_embedding)  # B, num_labels, embed_dim

        # Classifier task
        classifier_output = self.classifier_layer(enc_output)  # b*num_labels*num_labels
        # classifier_output = torch.matmul(enc_output, self.label_embedding.t())
        diag_mask = torch.eye(classifier_output.size(1)).unsqueeze(0).repeat(classifier_output.size(0), 1, 1).cuda()
        classifier_output = (classifier_output * diag_mask).sum(-1)

        # distance task
        distance_output = self.distance_layer(enc_output)  # b*num_labels*hidden
        # distance_output = enc_output
        # normalize
        distance_output = F.normalize(distance_output, p=2, dim=-1)

        # Decoder task
        tgt, indices, tgt_id = self._construct_id(classifier_output, distance_output)  # tgt: (N, 1)
        reshape_enc_output = enc_output[indices[:, 0], indices[:, 1]]  # reshape_enc_output: (N, embed_dim)

        if self.decoder_type == "Transformer":
            dec_output = self.decoder(tgt.long(), reshape_enc_output.unsqueeze(1))
        elif self.decoder_type == "MLP":
            dec_output = self.decoder(reshape_enc_output)

        return dec_output, tgt, tgt_id

    def _retrieval_learning(self, src, tgt_src, label_onehot, label_embedding):
        combined_src = torch.cat((src, tgt_src), dim=0)

        combined_enc_out = self.encoder(combined_src, label_embedding)  # B+B, num_labels, embed_dim

        center_embedding = self.label_embedding.repeat(src.size(0), 1, 1)

        src_enc_out, tgt_enc_out = combined_enc_out[:src.shape[0]], combined_enc_out[src.shape[0]:]

        indices = torch.nonzero(label_onehot, as_tuple=False).cuda()
        src_enc_out, tgt_enc_out = src_enc_out[indices[:, 0], indices[:, 1]], tgt_enc_out[
            indices[:, 0], indices[:, 1]]  # N, embed_dim

        # construct target image id
        distance_out = self.distance_layer(tgt_enc_out)
        # distance_out = tgt_enc_out
        distance_out = F.normalize(distance_out, p=2, dim=-1)  # N, embed_dim

        center_embedding = center_embedding[indices[:, 0], indices[:, 1]].cuda()
        center_embedding = F.normalize(center_embedding, p=2, dim=-1)

        tgt = torch.zeros((tgt_enc_out.shape[0], 1)).cuda()  # N, 1
        cosine_sim = F.cosine_similarity(center_embedding, distance_out, dim=-1)
        tgt[:, 0] = ((cosine_sim + 1) / 2) * self.distance_prc  # [0, 100]

        if self.decoder_type == "Transformer":
            dec_output = self.decoder(tgt.long(), src_enc_out.unsqueeze(1))
        elif self.decoder_type == "MLP":
            dec_output = self.decoder(src_enc_out)

        return dec_output, tgt

    def forward(self, src, noise_src=None, tgt_src=None, tgt=None, label_onehot=None, tgt_noise_level=None, stage="classification"):
        self.label_embedding = self.label_embedding.to(src.device)
        label_embedding_proj = self.label_proj(self.label_embedding)
        if stage == "classification":
            return self._ml_classification(src, label_embedding_proj)

        if stage == "indexing":
            return self._indexing_learning(src, label_embedding_proj)

        if stage == "retrieval":
            return self._retrieval_learning(src, tgt_src, label_onehot, label_embedding_proj)

        if stage == "tokenizer":
            return self._tokenizer(src, label_embedding_proj)

        if stage == "joint_learning":
            return self._joint_learning(src, noise_src, tgt_src, label_onehot, label_embedding_proj)

        if stage == "joint_learningV2":
            return self._joint_learningV2(src, noise_src, tgt_src, label_onehot, label_embedding_proj)

        if stage == "joint_learningV3":
            return self._joint_learningV3(src, noise_src, tgt_src, label_onehot, label_embedding_proj, tgt_noise_level)

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

    '''
    # TODO: 完成 Beam search代码，难
    # for simple, suppose B = 1
    def search(self, src, beam_size=10, dict_tree=None, label_embedding=None):
        # encoder feature
        enc_output, attns = self.encoder(src, label_embedding)  # 1, num_labels, embed_dim

        B, num_labels, embed_dim = enc_output.shape # B = 1

        # classifier res
        classifier_output = self.classifier_layer(enc_output)  # b*num_labels*num_labels
        diag_mask = torch.eye(classifier_output.size(1)).unsqueeze(0).repeat(classifier_output.size(0), 1, 1).cuda()
        classifier_output = (classifier_output * diag_mask).sum(-1)
        indices = self._get_classification(classifier_output) # classifier_output: b * num_labels

        enc_output_reshape = enc_output[indices[:, 0], indices[:, 1]] # (N = 1 * labels_by_class, embed_dim)
        N, _ = enc_output_reshape.shape

        sos = self.decoder(None, enc_output_reshape.unsqueeze(1)) # enc_output_reshape: (N, 1, embed_dim)
        # sos: [(N, 1, class1_num)]
        sos = F.softmax(sos[0], dim=-1)
        pd = torch.ones(N, sos.shape[-1]).cuda() * -1
        for i in range(N):
            for j in dict_tree.nodes.keys():
                pd[i, j] = sos[i, j]
        v, i = torch.topk(pd, k=beam_size, dim=-1) # v, i: (N, beam_size)
        # print(f"sos v i: {v}, {i}")

        pred_label = i
        in_cur = i
        in_cur = in_cur.reshape(-1, 1).type(torch.LongTensor).cuda() # in_cur: (N * beam_size, 1)

        # expand enc_output to beam size
        enc_output_bs = enc_output_reshape.reshape(N, -1).repeat(1, beam_size).reshape(N, beam_size, -1, embed_dim)
        enc_output_bs = enc_output_bs.reshape(N * beam_size, -1, embed_dim)  # (N*beam_size, 1, embed_dim)

        out = self.decoder(in_cur, enc_output_bs)[1] # out: (N * beam_size, 1000)
        out = F.softmax(out, dim=-1)
        out = out.reshape(N, beam_size, -1) # out: (N, beam_size, 1000)
        voc_length = out.shape[-1]

        # 联合概率密度
        pd = torch.ones(N, beam_size * voc_length).cuda() * -1
        for l in range(N):
            for k in range(beam_size):
                tmp = dict_tree.nodes[int(i[l, k])]
                pr = v[l, k] * out[l, k]
                j = torch.tensor(list(tmp.nodes.keys()))
                pd[l, k * voc_length + j] = pr[j]
        v, i = torch.topk(pd, k=beam_size, dim=-1)
        # print(f"two v i: {v}, {i}")

        # pred = (v.unsqueeze(-1) * out).reshape(N, -1) # pr: (N, beam_size, voc_length) - > (N, beam_size * voc_length)
        # v, i = torch.topk(pred, k=beam_size, dim=-1) # v, i: (N, beam_size)

        # 优化循环部分
        pred_label_expanded = pred_label.unsqueeze(-1).repeat(1, 1, voc_length).reshape(N, -1).cuda()
        ans = torch.zeros(N, beam_size, 2, dtype=torch.long)
        ans[:, :, 0] = pred_label_expanded.gather(1, i)
        ans[:, :, 1] = i % voc_length

        return ans
    '''

    def search(self, src, beam_size=10, dict_tree=None):
        self.label_embedding = self.label_embedding.to(src.device)
        label_embedding_proj = self.label_proj(self.label_embedding)
        # q_label: N, num_labels
        # encoder feature
        enc_output = self.encoder(src, label_embedding_proj)  # 1, num_labels, embed_dim

        B, num_labels, embed_dim = enc_output.shape # B = 1

        # label embedding expand
        label_embedding_expand = self.label_embedding.repeat(B, 1, 1)

        # classifier res
        classifier_output = self.classifier_layer(enc_output)  # b*num_labels*num_labels
        # classifier_output = torch.matmul(enc_output, self.label_embedding.t())
        diag_mask = torch.eye(classifier_output.size(1)).unsqueeze(0).repeat(classifier_output.size(0), 1, 1).cuda()
        classifier_output = (classifier_output * diag_mask).sum(-1)
        indices, cls_score = self._get_classification(classifier_output) # classifier_output: b * num_labels

        # global embed
        classifier_sigmoid = torch.sigmoid(classifier_output).unsqueeze(-1)
        global_embed = (enc_output * classifier_sigmoid).sum(dim=1)
        # global_embed = self.global_feature_layer(global_embed)  # B * embed_dim

        # global (cat)
        # B, _, _ = enc_output.shape
        # global_embed = self.global_feature_layer(enc_output.reshape(B, -1))

        enc_output_reshape = enc_output[indices[:, 0], indices[:, 1]] # (N = 1 * labels_by_class, embed_dim)
        enc_output_reshape = torch.cat((enc_output_reshape, label_embedding_expand[indices[:, 0], indices[:, 1]]), dim=-1)
        N, _ = enc_output_reshape.shape

        ans = torch.zeros(N, beam_size, 2, dtype=torch.long)
        ans[:, :, 0] = indices[:, 1][:, np.newaxis]

        if self.decoder_type == "Transformer":
            sos = self.decoder(None, enc_output_reshape.unsqueeze(1)) # enc_output_reshape: (N, 1, embed_dim)

            # sos: [(N, 1, class1_num)]
            sos = F.softmax(sos[0], dim=-1)
            pd = torch.ones(N, sos.shape[-1]).cuda() * -1
            for i in range(N):
                if int(indices[i, 1]) in dict_tree.nodes.keys():
                    tmp = dict_tree.nodes[int(indices[i, 1])]
                    for j in tmp.nodes.keys():
                        pd[i, j] = sos[i, j]
            v, i = torch.topk(pd, k=beam_size, dim=-1) # v, i: (N, beam_size)
            ans[:, :, 1] = i

            score = v * cls_score.unsqueeze(1).expand(-1, beam_size)
            # cls_s = classifier_output[indices[:, 0], indices[:, 1]]
            return ans, score, global_embed # N, beam_size
        elif self.decoder_type == "MLP":
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
            return ans, score, global_embed  # N, beam_size


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


if __name__ == '__main__':
    # test_q = torch.rand(64, 4)
    # test_k = torch.rand(64, 4)
    # mask = get_attn_pad_mask(test_q, test_k, 100)
    # print(mask.shape, mask)
    # exit(0)

    src = torch.rand(64, 3, 224, 224)
    # # tgt = torch.randint(0, 10, (64, 4))
    label_embedding = torch.rand(64, 17, 768)
    model = SIR(num_classes=[17, 1000], id_len=2)
    dec_output, classifier_output, distance_output, tgt = model(src, None, label_embedding)
    print("model res")
    for dec_out in dec_output:
        print(dec_out.shape)
    print(classifier_output.shape, distance_output.shape, tgt.shape)
    dic_tree = TreeNode()
    dic_tree.insert_many([[0, 1], [1, 2], [2, 3]])
    ans = model.search(src, beam_size=3, dic_tree=dic_tree)
    print(ans.shape)
    # for s in sos:
    #     print(s.shape)
    # print(sos)
    torch.nn.MultiheadAttention