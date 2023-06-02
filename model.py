import torch
import torch.nn as nn
from opt_einsum import contract
from long_seq import process_long_input
from losses import ATLoss
import torch.nn.functional as F
import numpy as np


class MultiHeadAttention(nn.Module):
    def __init__(self, query_dim, key_dim, all_head_dim, num_heads):
        super().__init__()
        self.query_dim = query_dim
        self.key_dim = key_dim
        self.all_head_dim = all_head_dim
        self.num_heads = num_heads
        
 
        self.W_query = nn.Linear(in_features=query_dim, out_features=all_head_dim)
        self.W_key = nn.Linear(in_features=key_dim, out_features=all_head_dim)
        self.W_value = nn.Linear(in_features=key_dim, out_features=all_head_dim)
 
    def forward(self, query, key, mask=None):
        querys = self.W_query(query)  # [B, N_q, all_head_dim]
        keys = self.W_key(key)  # [B, N_k, all_head_dim]
        values = self.W_value(key)
 
        head_size = self.all_head_dim // self.num_heads
        querys = torch.stack(torch.split(querys, head_size, dim=2), dim=0)  # [h, B, N_q, all_head_dim/h]
        keys = torch.stack(torch.split(keys, head_size, dim=2), dim=0)  # [h, B, N_k, all_head_dim/h]
        values = torch.stack(torch.split(values, head_size, dim=2), dim=0)  # [h, B, N_k, all_head_dim/h]
 
        ## score = softmax(QK^T / (d_k ** 0.5))
        scores = torch.matmul(querys, keys.transpose(2, 3))  # [h, B, N_q, N_k]
        scores = scores / (self.key_dim ** 0.5)
 
        ## mask
        if mask is not None:
            ## mask:  [B, N_k] --> [h, B, N_q, N_k]
            mask = mask.unsqueeze(1).unsqueeze(0).repeat(self.num_heads, 1, querys.shape[2], 1)
            scores = scores.masked_fill(mask!=1, -np.inf)
        scores = F.softmax(scores, dim=3)
 
        ## out = score * V
        out = torch.matmul(scores, values)  # [h, B, N_q, all_head_dim/h]
        out = torch.cat(torch.split(out, 1, dim=0), dim=3).squeeze(0)  # [B, N_q, all_head_dim]
 
        return out, scores
    

class DocREModel(nn.Module):
    def __init__(self, config, model, emb_size=768, block_size=64, num_labels=-1):
        super().__init__()
        self.config = config
        self.model = model
        self.hidden_size = config.hidden_size
        self.loss_fnt = ATLoss()

        self.head_extractor = nn.Linear(2 * config.hidden_size, config.hidden_size)
        self.tail_extractor = nn.Linear(2 * config.hidden_size, config.hidden_size)
        self.entity_rel_interactor = MultiHeadAttention(config.hidden_size,
                                                        config.hidden_size,
                                                        config.hidden_size,
                                                        config.num_attention_heads)
        # self.bilinear = nn.Linear(emb_size * block_size, config.num_labels)
        self.entity_pair_classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.rel_calssifier = nn.Linear(config.hidden_size, 2)

        self.emb_size = emb_size
        self.block_size = block_size
        self.num_labels = num_labels

    def encode(self, input_ids, attention_mask):
        config = self.config
        if config.transformer_type == "bert":
            start_tokens = [config.cls_token_id]
            end_tokens = [config.sep_token_id]
        elif config.transformer_type == "roberta":
            start_tokens = [config.cls_token_id]
            end_tokens = [config.sep_token_id, config.sep_token_id]
        sequence_output, attention = process_long_input(self.model, input_ids, attention_mask, start_tokens, end_tokens)
        return sequence_output, attention

    def get_hrt(self, sequence_output, attention, entity_pos, hts):
        offset = 1 if self.config.transformer_type in ["bert", "roberta"] else 0
        n, h, _, c = attention.size()
        hss, tss, rss = [], [], []
        for i in range(len(entity_pos)):
            entity_embs, entity_atts = [], []
            for e in entity_pos[i]:
                if len(e) > 1:
                    e_emb, e_att = [], []
                    for start, end in e:
                        if start + offset < c:
                            # In case the entity mention is truncated due to limited max seq length.
                            e_emb.append(sequence_output[i, start + offset])
                            e_att.append(attention[i, :, start + offset])
                    if len(e_emb) > 0:
                        e_emb = torch.logsumexp(torch.stack(e_emb, dim=0), dim=0)
                        e_att = torch.stack(e_att, dim=0).mean(0)
                    else:
                        e_emb = torch.zeros(self.config.hidden_size).to(sequence_output)
                        e_att = torch.zeros(h, c).to(attention)
                else:
                    start, end = e[0]
                    if start + offset < c:
                        e_emb = sequence_output[i, start + offset]
                        e_att = attention[i, :, start + offset]
                    else:
                        e_emb = torch.zeros(self.config.hidden_size).to(sequence_output)
                        e_att = torch.zeros(h, c).to(attention)
                entity_embs.append(e_emb)
                entity_atts.append(e_att)

            entity_embs = torch.stack(entity_embs, dim=0)  # [n_e, d]
            entity_atts = torch.stack(entity_atts, dim=0)  # [n_e, h, seq_len]

            ht_i = torch.LongTensor(hts[i]).to(sequence_output.device)
            hs = torch.index_select(entity_embs, 0, ht_i[:, 0])
            ts = torch.index_select(entity_embs, 0, ht_i[:, 1])

            h_att = torch.index_select(entity_atts, 0, ht_i[:, 0])
            t_att = torch.index_select(entity_atts, 0, ht_i[:, 1])
            ht_att = (h_att * t_att).mean(1)
            ht_att = ht_att / (ht_att.sum(1, keepdim=True) + 1e-5) 
            rs = contract("ld,rl->rd", sequence_output[i], ht_att)
            hss.append(hs)
            tss.append(ts)
            rss.append(rs)
        hss = torch.cat(hss, dim=0)
        tss = torch.cat(tss, dim=0)
        rss = torch.cat(rss, dim=0)
        return hss, rss, tss

    def forward(self,
                input_ids=None,
                attention_mask=None,
                labels=None,
                entity_pos=None,
                hts=None,
                instance_mask=None,
                rel_ids=None,
                rel_attention_mask=None,
                rel_pos=None,
                ):

        sequence_output, sequence_attention = self.encode(input_ids, attention_mask)
        hs, rs, ts = self.get_hrt(sequence_output, sequence_attention, entity_pos, hts)

        hs = torch.tanh(self.head_extractor(torch.cat([hs, rs], dim=1)))
        ts = torch.tanh(self.tail_extractor(torch.cat([ts, rs], dim=1)))
        hts = hs + ts
        # b1 = hs.view(-1, self.emb_size // self.block_size, self.block_size) # [1310, 12, 64]
        # b2 = ts.view(-1, self.emb_size // self.block_size, self.block_size) # [1310, 12, 64]
        # bl = (b1.unsqueeze(3) * b2.unsqueeze(2)).view(-1, self.emb_size * self.block_size) # [1310, 12*64*64]
        # logits = self.bilinear(bl) # 1310 * 97
        rels = self.encode(rel_ids, rel_attention_mask)[rel_pos]
        
        hts = self.entity_rel_interactor(hts, rels)
        rels = self.entity_rel_interactor(rels, hts)

        hts_logits = self.entity_pair_classifier(hts)

        output = (self.loss_fnt.get_label(hts_logits, num_labels=self.num_labels),)
        if labels is not None:
            labels = [torch.tensor(label) for label in labels]
            labels = torch.cat(labels, dim=0).to(hts_logits)
            loss = self.loss_fnt(hts_logits.float(), labels.float())
            output = (loss.to(sequence_output),) + output
        return output
