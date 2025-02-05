import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .transformer_layers import SelfAttnLayer, DecoderBlock
from .backbone import Backbone, ResNet50
from .utils import weights_init, trunc_normal_
from .position_enc import positionalencoding2d
from .DictTree import TreeNode
from .beam_search import BeamSearchNode
from queue import PriorityQueue
import operator
from torch import distributed as dist

def get_attn_pad_mask(seq_q, seq_k, num):
    batch_size, len_q = seq_q.size() # B, N
    batch_size, len_k = seq_k.size()
    # eq(zero) is PAD token
    pad_attn_mask = torch.cat((torch.tensor([[1] for _ in range(batch_size)]).to(seq_q.device),seq_q), dim=1).data.eq(num).unsqueeze(1)  # batch_size x 1 x len_k, one is masking
    return pad_attn_mask.expand(batch_size, len_q+1, len_k+1)

class CTranModel(nn.Module):
    def __init__(self, num_labels, hidden=768,
                 pos_emb=False, layers=3, heads=4, dropout=0.1):
        super(CTranModel, self).__init__()

        # ResNet backbone
        self.backbone = ResNet50()

        self.conv_downsample = torch.nn.Conv2d(2048, hidden, (1, 1))

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
            self.init_label_embeddings = label_embedding

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

        return label_embeddings, attns

class Decoder(nn.Module):
    def __init__(self, embed_dim=384, voc_length=1000, id_len=17,
                 depth=12, num_heads=12,
                 mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, decode_only=False, **kwargs):
        super().__init__()
        self.num_heads = num_heads
        self.voc_length = voc_length
        self.id_len = id_len
        self.num_features = self.embed_dim = embed_dim
        self.decode_only = decode_only
        self.enc_proj = nn.Linear(embed_dim, embed_dim * 10) if decode_only else nn.Identity()
        # proj embedding
        self.label_embed = nn.Embedding(voc_length, embed_dim)

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
        self.label_prj = nn.Linear(embed_dim, voc_length)

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
            x = self.label_embed(x)  # patch linear embedding

            # add the [CLS] token to the embed patch tokens
            start_tokens = self.start_token.expand(B, -1, -1)
            x = torch.cat((start_tokens, x), dim=1)
        else:
            B, N = feats_num, 0
            x = self.start_token.repeat(B, 1, 1)
        # add positional encoding to each token
        pos_emb = self.pos_embed(x)
        pos_emb = pos_emb[:, :N + 1, :]
        x = x + pos_emb
        return self.pos_drop(x)

    def forward(self, x, enc_output, padding=True):
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

        x = self.label_prj(x)  # B, id_len, emb_dim
        return x

class SIR(nn.Module):

    def __init__(self, id_len=4,
                 voc_length=1000, num_labels=17, embed_dim=768, distance_prc=100,
                 enc_depth=6, dec_depth=6,
                 num_heads=4, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, **kwargs):
        super().__init__()
        self.num_labels = num_labels
        self.distance_prc = distance_prc
        self.voc_length = voc_length
        self.PAD = voc_length - 1

        self.encoder = CTranModel(num_labels, hidden=embed_dim, layers=enc_depth, heads=num_heads, dropout=drop_rate)

        self.decoder = Decoder(embed_dim=embed_dim, voc_length=voc_length, id_len=id_len, depth=dec_depth,
                               num_heads=num_heads,
                               mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop_rate=drop_rate,
                               attn_drop_rate=attn_drop_rate,
                               drop_path_rate=drop_path_rate, norm_layer=norm_layer, **kwargs)

        # Classifier
        # Output is of size num_labels because we want a separate classifier for each label
        self.classifier_layer = torch.nn.Linear(embed_dim, num_labels)

        # Distance Preserve
        distance_dim = 768  # same with BERT output dim
        self.distance_layer = torch.nn.Linear(embed_dim, distance_dim)

        self._reset_parameters()
        self.id_len = id_len
        self.voc_length = voc_length
        self.classifier_layer.apply(weights_init)
        self.distance_layer.apply(weights_init)

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    @torch.no_grad()
    def _construct_id(self, classifier_output, distance_output, label_embedding=None):
        # classifier_output: (B, num_labels)
        # label_embedding: (B, num_labels, embed_dim)
        # distance_output: (B, num_labels, embed_dim)

        B, num_labels = classifier_output.shape

        classifier_output = classifier_output
        distance_output = distance_output

        if label_embedding is None:
            center_embedding = self.encoder.init_label_embeddings.detach().to(classifier_output.device)
        else:
            center_embedding = label_embedding.detach().to(classifier_output.device)

        center_embedding = F.normalize(center_embedding, p=2, dim=-1)

        indices = self._get_classification(classifier_output)
        indices = indices.to(classifier_output.device)

        cosine_sim = F.cosine_similarity(center_embedding, distance_output, dim=-1)
        normalized_cos_sim = torch.floor(((cosine_sim + 1) / 2) * self.distance_prc).to(classifier_output.device)

        batch_indices = indices[:, 0]
        class_indices = indices[:, 1]
        max_len = self.id_len

        tgt = torch.full((B, max_len), self.PAD, dtype=torch.float, device=classifier_output.device)

        # fill
        distance_values = normalized_cos_sim[batch_indices, class_indices]
        pad_value = class_indices * self.distance_prc + distance_values

        fill_positions = torch.arange(max_len, device=classifier_output.device).unsqueeze(0).expand(B, -1)
        fill_counts = torch.bincount(batch_indices, minlength=B).clamp(max=max_len)
        mask = (fill_positions < fill_counts.unsqueeze(1))
        tgt[mask] = pad_value[:mask.sum().item()]

        return tgt, indices

    def _tokenizer(self, src, label_embedding=None):
        enc_output, attns = self.encoder(src, label_embedding)  # B, num_labels, embed_dim
        B, num_labels, embed_dim = enc_output.shape

        # Classifier task
        classifier_output = self.classifier_layer(enc_output).to(src.device)  # b*num_labels*num_labels
        diag_mask = torch.eye(classifier_output.size(1)).unsqueeze(0).repeat(classifier_output.size(0), 1, 1).to(
            src.device)
        classifier_output = (classifier_output * diag_mask).sum(-1) # B, num_labels

        prob = torch.sigmoid(classifier_output).unsqueeze(-1)
        token_feature = torch.sum(enc_output * prob, dim=1)  # (B, embed_dim)
        return token_feature

    def _ml_classification(self, src, label_embedding=None):
        enc_output, attns = self.encoder(src, label_embedding)  # B, num_labels, embed_dim
        # Classifier task
        classifier_output = self.classifier_layer(enc_output).to(src.device)  # b*num_labels*num_labels
        diag_mask = torch.eye(classifier_output.size(1)).unsqueeze(0).repeat(classifier_output.size(0), 1, 1).to(
            src.device)
        classifier_output = (classifier_output * diag_mask).sum(-1)

        # distance task
        distance_output = self.distance_layer(enc_output)  # b*num_labels*hidden
        # normalize
        distance_output = F.normalize(distance_output, p=2, dim=-1)
        return classifier_output, distance_output

    def _ar_learning(self, src, tgt=None, label_embedding=None):
        enc_output, attns = self.encoder(src, label_embedding)  # B, num_labels, embed_dim
        dec_output = self.decoder(tgt.long(), enc_output)  # enc_output: (B, num_labels, embed_dim)
        return dec_output, tgt

    def forward(self, src, tgt=None, label_embedding=None, stage="classification"):

        if stage == "classification":
            return self._ml_classification(src, label_embedding)

        if stage == "tokenizer":
            return self._tokenizer(src, label_embedding)

        if stage == "indexing" or stage == "retrieval":
            return self._ar_learning(src, tgt, label_embedding)

        raise ValueError("stage error.")

    @torch.no_grad()
    def _get_classification(self, classifier_output):
        # classifier_output： B, num_labels
        prob = torch.sigmoid(classifier_output)
        K = prob.size(1)
        topk_values, topk_indices = torch.topk(prob, K, dim=1)
        mask = topk_values > 0.5
        max_values, max_indices = torch.max(prob, dim=1, keepdim=True)
        extended_mask = torch.cat([mask, ~mask.any(dim=1, keepdim=True)], dim=1)
        extended_topk_indices = torch.cat([topk_indices, max_indices], dim=1)
        batch_indices = torch.arange(prob.size(0)).unsqueeze(1).repeat(1, extended_topk_indices.size(1)).to(classifier_output.device)
        filtered_indices = torch.cat((batch_indices.unsqueeze(-1), extended_topk_indices.unsqueeze(-1)), dim=-1)
        indices = filtered_indices[extended_mask].view(-1, 2)

        return indices

    def beam_search_nc(self, src, beam_size=10, label_embedding=None):
        # encoder feature
        enc_output, attns = self.encoder(src, label_embedding)  # 1, num_labels, embed_dim
        B, num_labes, embed_dim = enc_output.shape

        decoded_batch = []
        for b in range(B):
            enc_out = enc_output[b, :, :].unsqueeze(0)  # 1, num_labels, embed_dim

            # Number of sentence to generate
            endnodes = []
            number_required = min((beam_size + 1), beam_size - len(endnodes))

            dec_input = None

            node = BeamSearchNode(None, None, dec_input, 0, 1)

            beam_list = [node]
            while len(beam_list) > 0:
                next_beam_list = []
                for n in beam_list:
                    decoder_input = n.wordid
                    cur_len = n.leng

                    if n.wordid is not None and ((n.wordid[0, -1].item() == self.PAD and n.prevNode != None) or (cur_len >= self.id_len)):
                        endnodes.append((-n.eval(), n))
                        number_required -= 1
                        continue

                    # decode for one step using decoder
                    dec_output = self.decoder(decoder_input, enc_out)  # 1, id_len, voc_length
                    dec_prob = F.log_softmax(dec_output, dim=-1)

                    v, i = torch.topk(dec_prob, k=beam_size, dim=-1)  # v, i: (N, id_len, beam_size)

                    for new_k in range(beam_size):
                        # decoder_input: 1, n.length
                        decoded_t = i[0][-1][new_k].view(1, 1)
                        if decoder_input is not None:
                            decoded_t = torch.cat((decoder_input, decoded_t), dim=-1)
                        log_p = v[0][-1][new_k].item()

                        node = BeamSearchNode(None, n, decoded_t, n.logp + log_p, n.leng + 1)
                        next_beam_list.append(node)
                beam_list = sorted(next_beam_list, key=lambda node: node.logp, reverse=True)[:number_required]

            # # choose nbest paths, back trace them
            # if len(endnodes) == 0:
            #     endnodes = [nodes.get() for _ in range(beam_size)]

            utterances = []
            for score, n in sorted(endnodes, key=operator.itemgetter(0)):
                utterances.append(n.wordid)  # n.wordid: 1, length
            decoded_batch.append(utterances)

        return decoded_batch

    def beam_search(self, src, beam_size=10, dict_tree=None, label_embedding=None):
        # encoder feature
        enc_output, attns = self.encoder(src, label_embedding)  # 1, num_labels, embed_dim
        B, num_labes, embed_dim = enc_output.shape

        decoded_batch = []
        for b in range(B):
            enc_out = enc_output[b, :, :].unsqueeze(0) # 1, num_labels, embed_dim

            # Number of sentence to generate
            endnodes = []
            number_required = min((beam_size + 1), beam_size - len(endnodes))

            dec_input = None

            node = BeamSearchNode(dict_tree, None, dec_input, 0, 1)

            beam_list = [node]
            while len(beam_list) > 0:
                next_beam_list = []
                for n in beam_list:
                    decoder_input = n.wordid
                    cur_dict_tree = n.dic_tree
                    cur_len = n.leng

                    if n.wordid is not None and (
                            (cur_dict_tree.is_leaf) or (cur_len >= self.id_len)):
                        endnodes.append((-n.eval(), n))
                        number_required -= 1
                        continue

                    # decode for one step using decoder
                    dec_output = self.decoder(decoder_input, enc_out)  # 1, id_len, voc_length
                    dec_prob = F.log_softmax(dec_output, dim=-1)

                    pd = torch.ones(dec_prob.shape[-1]).cuda() * -float('inf')
                    for j in cur_dict_tree.nodes.keys():
                        pd[j] = dec_prob[0, -1, j]
                    v, i = torch.topk(pd, k=beam_size, dim=-1)  # v, i: (N, beam_size)

                    for new_k in range(beam_size):
                        if int(i[new_k]) not in cur_dict_tree.nodes.keys():
                            continue
                        tmp = cur_dict_tree.nodes[int(i[new_k])]
                        # decoder_input: 1, n.length
                        decoded_t = i[new_k].view(1, 1)
                        if decoder_input is not None:
                            decoded_t = torch.cat((decoder_input, decoded_t), dim=-1)
                        log_p = v[new_k].item()

                        node = BeamSearchNode(tmp, n, decoded_t, n.logp + log_p, n.leng + 1)
                        next_beam_list.append(node)
                beam_list = sorted(next_beam_list, key=lambda node: node.logp, reverse=True)[:number_required]

            # # choose nbest paths, back trace them
            # if len(endnodes) == 0:
            #     endnodes = [nodes.get() for _ in range(beam_size)]

            utterances = []
            for score, n in sorted(endnodes, key=operator.itemgetter(0)):
                utterances.append(n.wordid) # n.wordid: 1, length
            decoded_batch.append(utterances)

        return decoded_batch


    def beam_search_exp(self, src, beam_size=10, dict_tree=None, label_embedding=None):
        # encoder feature
        enc_output, attns = self.encoder(src, label_embedding)  # 1, num_labels, embed_dim
        B, num_labes, embed_dim = enc_output.shape
        decoded_batch = []
        for b in range(B):
            enc_out = enc_output[b, :, :].unsqueeze(0) # 1, num_labels, embed_dim
            dec_input = None
            node = BeamSearchNode(dict_tree, None, dec_input, 0, 1)
            beam_list = [node]
            for idx in range(self.id_len):
                next_beam_list = []
                for n in beam_list:
                    decoder_input = n.wordid
                    cur_dict_tree = n.dic_tree

                    # decode for one step using decoder
                    dec_output = self.decoder(decoder_input, enc_out)  # 1, id_len, voc_length
                    dec_prob = F.log_softmax(dec_output, dim=-1)

                    pd = torch.ones(dec_prob.shape[-1]).cuda() * -float('inf')
                    for j in cur_dict_tree.nodes.keys():
                        pd[j] = dec_prob[0, -1, j]
                    v, i = torch.topk(pd, k=beam_size, dim=-1)  # v, i: (N, beam_size)

                    for new_k in range(beam_size):
                        if int(i[new_k]) not in cur_dict_tree.nodes.keys():
                            continue
                        tmp = cur_dict_tree.nodes[int(i[new_k])]
                        # decoder_input: 1, n.length
                        decoded_t = i[new_k].view(1, 1)
                        if decoder_input is not None:
                            decoded_t = torch.cat((decoder_input, decoded_t), dim=-1)
                        log_p = v[new_k].item()
                        node = BeamSearchNode(tmp, n, decoded_t, n.logp + log_p, n.leng + 1)
                        next_beam_list.append(node)
                beam_list = sorted(next_beam_list, key=lambda node: node.logp, reverse=True)[:beam_size]

                utterances = []
                for b in beam_list:
                    utterances.append(b.wordid)
            decoded_batch.append(utterances)

        return decoded_batch


if __name__ == '__main__':
    # test_q = torch.rand(64, 4)
    # test_k = torch.rand(64, 4)
    # mask = get_attn_pad_mask(test_q, test_k, 100)
    # print(mask.shape, mask)
    # exit(0)

    src = torch.rand(64, 3, 224, 224)
    # # tgt = torch.randint(0, 10, (64, 4))
    label_embedding = torch.rand(64, 17, 768)
    model = SIR(voc_length=17*100+3, id_len=17+2)
    dec_output, classifier_output, distance_output, x, y = model(src, None, label_embedding)
    print(dec_output.shape, classifier_output.shape, distance_output.shape, x.shape, y.shape)
