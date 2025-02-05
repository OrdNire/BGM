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

        # Label Embeddings
        self.label_input = torch.Tensor(np.arange(num_labels)).view(1, -1).long()
        self.label_lt = torch.nn.Embedding(num_labels, hidden, padding_idx=None)

        # Global Embedding
        self.global_token = nn.Parameter(torch.zeros(1, 1, hidden))

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
        global_token = self.global_token.expand(features.size(0), -1, -1)
        # (B, f_dim, 18, 18) -> (B, 18*18, f_dim)
        embeddings = torch.cat((features, global_token, self.init_label_embeddings), 1)

        # Feed image and label embeddings through Transformer
        embeddings = self.LayerNorm(embeddings)
        attns = []
        for layer in self.self_attn_layers:
            embeddings, attn = layer(embeddings, mask=None)
            attns += attn.detach().unsqueeze(0).data

        # Readout each label embedding using a linear layer
        label_embeddings = embeddings[:, -self.init_label_embeddings.size(1):, :]  # (B, num_labels, hidden)

        global_embedding = embeddings[:, features.size(1), :]

        return label_embeddings, global_embedding

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

class SIR(nn.Module):

    def __init__(self, input_channels=3, label_embedding=None, num_labels=17, embed_dim=768, distance_prc=100,
                 enc_depth=6,
                 num_heads=4, drop_rate=0.1):
        super().__init__()
        self.num_labels = num_labels
        self.distance_prc = distance_prc
        self.label_embedding = label_embedding # num_labels, embed_dim

        self.encoder = CTranModel(num_labels, hidden=embed_dim, layers=enc_depth, heads=num_heads, dropout=drop_rate,
                                  input_channels=input_channels)

        self.decoder = NoiseMLP(input_dim=embed_dim, num_classes=distance_prc)

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

        self._reset_parameters()
        self.classifier_layer.apply(weights_init)
        self.distance_layer.apply(weights_init)
        self.label_proj.apply(weights_init)
        # self.proxies_proj.apply(weights_init)

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    @torch.no_grad()
    def _construct_id(self, classifier_output, distance_output):
        # classifier_output: (B, num_labels)
        # label_embedding: (B, num_labels, embed_dim)
        # distance_output: (B, num_labels, embed_dim)

        classifier_output = classifier_output.cuda()
        distance_output = distance_output.cuda()

        # center_embedding = label_embedding.detach().cuda()
        center_embedding = self.label_embedding.repeat(distance_output.size(0), 1, 1)

        center_embedding = F.normalize(center_embedding, p=2, dim=-1)

        prob = torch.sigmoid(classifier_output)
        classification_res = (prob > 0.5).float()
        # classification_res_exp = classification_res.unsqueeze(-1).expand_as(center_embedding) # (B, num_labels, embed_dim)

        cosine_sim = F.relu(F.cosine_similarity(center_embedding, distance_output, dim=-1))
        normalized_cos_sim = cosine_sim * self.distance_prc # [0, 100]

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

    def _retrieval(self, src, label_embedding, indices):
        enc_output, global_output = self.encoder(src, label_embedding)  # B, num_labels, embed_dim

        # Classifier task
        classifier_output = self.classifier_layer(enc_output)  # b*num_labels*num_labels
        # classifier_output = torch.matmul(enc_output, self.label_embedding.t())
        diag_mask = torch.eye(classifier_output.size(1)).unsqueeze(0).repeat(classifier_output.size(0), 1, 1).cuda()
        classifier_output = (classifier_output * diag_mask).sum(-1)

        reshape_enc_output = enc_output[indices[:, 0], indices[:, 1]]

        dec_output = self.decoder(reshape_enc_output)

        return classifier_output, dec_output

    def _ml_classification(self, src, label_embedding):
        enc_output, global_output = self.encoder(src, label_embedding)  # B, num_labels, embed_dim

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

        dec_output = self.decoder(reshape_enc_output)

        return enc_output, classifier_output, distance_output, dec_output, tgt

    def _tokenizer(self, src, label_embedding):
        enc_output, global_output = self.encoder(src, label_embedding)  # B, num_labels, embed_dim

        # Classifier task
        classifier_output = self.classifier_layer(enc_output)  # b*num_labels*num_labels
        # classifier_output = torch.matmul(enc_output, self.label_embedding.t())
        diag_mask = torch.eye(classifier_output.size(1)).unsqueeze(0).repeat(classifier_output.size(0), 1, 1).cuda()
        classifier_output = (classifier_output * diag_mask).sum(-1) # B, num_labels

        # distance task
        distance_output = self.distance_layer(enc_output)  # b*num_labels*hidden
        # distance_output = enc_output
        # normalize
        distance_output = F.normalize(distance_output, p=2, dim=-1)

        # Decoder task
        tgt, indices, tgt_id = self._construct_id(classifier_output, distance_output)  # tgt: (N, 1)

        # Class label
        cls_tgt = (classifier_output > 0).float()

        return cls_tgt, indices, tgt, tgt_id

    def forward(self, src, stage="phase1", indices=None):
        self.label_embedding = self.label_embedding.to(src.device)
        label_embedding_proj = self.label_proj(self.label_embedding)
        if stage == "phase1":
            return self._ml_classification(src, label_embedding_proj)

        if stage == "phase2":
            return self._tokenizer(src, label_embedding_proj)

        if stage == "retrieval":
            return self._retrieval(src, label_embedding_proj, indices)

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

    def search(self, src, beam_size=10, dict_tree=None):
        self.label_embedding = self.label_embedding.to(src.device)
        label_embedding_proj = self.label_proj(self.label_embedding)
        # q_label: N, num_labels
        # encoder feature
        enc_output, global_output = self.encoder(src, label_embedding_proj)  # 1, num_labels, embed_dim

        B, num_labels, embed_dim = enc_output.shape # B = 1

        # classifier res
        classifier_output = self.classifier_layer(enc_output)  # b*num_labels*num_labels
        # classifier_output = torch.matmul(enc_output, self.label_embedding.t())
        diag_mask = torch.eye(classifier_output.size(1)).unsqueeze(0).repeat(classifier_output.size(0), 1, 1).cuda()
        classifier_output = (classifier_output * diag_mask).sum(-1)
        indices, cls_score = self._get_classification(classifier_output) # classifier_output: b * num_labels

        enc_output_reshape = enc_output[indices[:, 0], indices[:, 1]] # (N = 1 * labels_by_class, embed_dim)
        N, _ = enc_output_reshape.shape

        ans = torch.zeros(N, beam_size, 2, dtype=torch.long)
        ans[:, :, 0] = indices[:, 1][:, np.newaxis]

        sos = self.decoder(enc_output_reshape)

        # sos: [(N, num_classes)]
        sos = F.softmax(sos, dim=-1)
        pd = torch.ones(N, sos.shape[-1]).cuda() * -1
        for i in range(N):
            if int(indices[i, 1]) in dict_tree.nodes.keys():
                tmp = dict_tree.nodes[int(indices[i, 1])]
                for j in tmp.nodes.keys():
                    pd[i, j] = sos[i, j]
        v, i = torch.topk(pd, k=beam_size, dim=-1) # v, i: (N, beam_size)
        ans[:, :, 1] = i

        score = v
        # cls_s = classifier_output[indices[:, 0], indices[:, 1]]
        return ans, score # N, beam_size

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