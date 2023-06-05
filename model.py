from typing import Callable, Optional, Union
import torch
import torch.nn as nn
from torch.nn.modules.normalization import LayerNorm
from torch.nn.modules.dropout import Dropout
from opt_einsum import contract
from long_seq import process_long_input
from losses import ATLoss
import torch.nn.functional as F
import numpy as np


class MultiHeadAttention(nn.Module):
    def __init__(self, config, query_dim, key_dim, all_head_dim, num_heads):
        super().__init__()
        self.config = config
        self.query_dim = query_dim
        self.key_dim = key_dim
        self.all_head_dim = all_head_dim
        self.num_heads = num_heads
        
        self.W_query = nn.Linear(in_features=query_dim, out_features=all_head_dim)
        self.W_key = nn.Linear(in_features=key_dim, out_features=all_head_dim)
        self.W_value = nn.Linear(in_features=key_dim, out_features=all_head_dim)

        self.dropout1 = nn.Dropout(config.attention_probs_dropout_prob)
        self.dropout2 = nn.Dropout(config.attention_probs_dropout_prob)
 
    def forward(self, query, key, query_mask=None, key_mask=None):
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
 
        ## key_mask
        if key_mask is not None:
            ## key_mask:  [B, N_k] --> [h, B, N_q, N_k]
            key_mask = key_mask.unsqueeze(1).unsqueeze(0).repeat(self.num_heads, 1, querys.size(2), 1)
            scores = scores.masked_fill(key_mask!=1, -np.inf)
        scores = self.dropout1(F.softmax(scores, dim=3))
        out = torch.matmul(scores, values)  # [h, B, N_q, all_head_dim/h]
        #out = torch.tanh(out)

        ## htr交互
        scores = torch.matmul(out, out.transpose(2, 3))  # [h, B, N_q, N_k]
        scores = scores / (self.query_dim ** 0.5)
        if query_mask is not None:
            ## query_mask:  [B, N_k] --> [h, B, N_q, N_k]
            query_mask = query_mask.unsqueeze(1).unsqueeze(0).repeat(self.num_heads, 1, querys.size(2), 1)
            scores = scores.masked_fill(query_mask!=1, -np.inf)
        scores = self.dropout2(F.softmax(scores, dim=3))
        out = torch.matmul(scores, out)  # [h, B, N_q, all_head_dim/h]

        out = torch.cat(torch.split(out, 1, dim=0), dim=3).squeeze(0)  # [B, N_q, all_head_dim]
        out = torch.tanh(out)
        return out, scores
    
class TransformerEncoderLayer(nn.Module):
    __constants__ = ['batch_first', 'norm_first']

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: Union[str, Callable[[torch.Tensor], torch.Tensor]] = F.relu,
                 layer_norm_eps: float = 1e-5, batch_first: bool = False, norm_first: bool = False,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
                                            **factory_kwargs)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, **factory_kwargs)

        self.norm_first = norm_first
        self.norm1 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        # Legacy string support for activation function.
        if isinstance(activation, str):
            activation = self._get_activation_fn(activation)

        # We can't test self.activation in forward() in TorchScript,
        # so stash some information about it instead.
        if activation is F.relu or isinstance(activation, torch.nn.ReLU):
            self.activation_relu_or_gelu = 1
        elif activation is F.gelu or isinstance(activation, torch.nn.GELU):
            self.activation_relu_or_gelu = 2
        else:
            self.activation_relu_or_gelu = 0
        self.activation = activation
    
    def _get_activation_fn(activation: str) -> Callable[[torch.Tensor], torch.Tensor]:
        if activation == "relu":
            return F.relu
        elif activation == "gelu":
            return F.gelu

        raise RuntimeError("activation should be relu/gelu, not {}".format(activation))

    def __setstate__(self, state):
        super(TransformerEncoderLayer, self).__setstate__(state)
        if not hasattr(self, 'activation'):
            self.activation = F.relu


    def forward(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None,
                src_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if src_key_padding_mask is not None:
            _skpm_dtype = src_key_padding_mask.dtype
            if _skpm_dtype != torch.bool and not torch.is_floating_point(src_key_padding_mask):
                raise AssertionError(
                    "only bool and floating types of key_padding_mask are supported")
        
        x = src
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), src_mask, src_key_padding_mask)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block(x, src_mask, src_key_padding_mask))
            x = self.norm2(x + self._ff_block(x))

        return x

    # self-attention block
    def _sa_block(self, x: torch.Tensor,
                  attn_mask: Optional[torch.Tensor], key_padding_mask: Optional[torch.Tensor]) -> torch.Tensor:
        x = self.self_attn(x, x, x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           need_weights=False)[0]
        return self.dropout1(x)

    # feed forward block
    def _ff_block(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)
    

class DocREModel(nn.Module):
    def __init__(self, config, model, emb_size=768, block_size=64, num_labels=-1):
        super().__init__()
        self.config = config
        self.model = model
        self.hidden_size = config.hidden_size
        self.loss_fnt = ATLoss()

        # self.head_extractor = nn.Linear(2 * config.hidden_size, config.hidden_size)
        # self.tail_extractor = nn.Linear(2 * config.hidden_size, config.hidden_size)
        self.extractor = nn.Linear(3 * config.hidden_size, config.hidden_size)
        # self.entity_pair_interactor = TransformerEncoderLayer(d_model=config.hidden_size, nhead=16, batch_first=True)
        self.entity_rel_interactor = MultiHeadAttention(config,
                                                        config.hidden_size,
                                                        config.hidden_size,
                                                        config.hidden_size,
                                                        config.num_attention_heads)
        
        self.interactor_norm = LayerNorm(config.hidden_size, eps=1e-5)
        self.bilinear = nn.Linear(emb_size * block_size, config.num_labels)
        self.entity_pair_classifier = nn.Linear(config.hidden_size, config.num_labels)
        #self.rel_calssifier = nn.Linear(config.hidden_size, 1)

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
        # hss = torch.cat(hss, dim=0)
        # tss = torch.cat(tss, dim=0)
        # rss = torch.cat(rss, dim=0)
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
                rel_labels=None,
                ):

        sequence_output, sequence_attention = self.encode(input_ids, attention_mask)
        hss, rss, tss = self.get_hrt(sequence_output, sequence_attention, entity_pos, hts)
        hts = []
        for i in range(input_ids.size(0)):
            # hs = torch.tanh(self.head_extractor(torch.cat([hss[i], rss[i]], dim=1)))
            # ts = torch.tanh(self.tail_extractor(torch.cat([tss[i], rss[i]], dim=1)))
            htr = torch.tanh(self.extractor(torch.cat([hss[i], tss[i], rss[i]], dim=1)))
            hts.append(htr)
        # b1 = hs.view(-1, self.emb_size // self.block_size, self.block_size) # [1310, 12, 64]
        # b2 = ts.view(-1, self.emb_size // self.block_size, self.block_size) # [1310, 12, 64]
        # bl = (b1.unsqueeze(3) * b2.unsqueeze(2)).view(-1, self.emb_size * self.block_size) # [1310, 12*64*64]
        # logits = self.bilinear(bl) # 1310 * 97
        hts_len = [i.size(0) for i in hts]
        hts = [torch.cat([item, torch.zeros(max(hts_len)-item.size(0), item.size(1), dtype=item.dtype, device=item.device)],
                          dim=0) for item in hts]
        hts = torch.stack(hts, dim=0)
        hts_attn_mask = [torch.cat([torch.ones(i, dtype=torch.float, device=hts.device),
                                    torch.zeros(max(hts_len)-i, dtype=torch.float, device=hts.device)],
                                    dim=0) for i in hts_len]
        hts_attn_mask = torch.stack(hts_attn_mask, dim=0)

        # residual = hts
        # hts = self.entity_pair_interactor(src=hts,
        #                                   src_key_padding_mask=(hts_attn_mask!=1))
        # hts = hts + residual
        
    
        rels, _ = self.encode(rel_ids, rel_attention_mask)
        rel_pos = torch.LongTensor(rel_pos).to(rels.device)
        rels = torch.index_select(rels.squeeze(0), 0, rel_pos)

        rels = rels.repeat(input_ids.size(0), 1, 1)
        rel_attention_mask = torch.ones(rels.size(0), rels.size(1), dtype=torch.float, device=rels.device)


        hts_1 = self.entity_rel_interactor(hts, rels, hts_attn_mask, rel_attention_mask)[0]
        #rels_1 = self.entity_rel_interactor(rels, hts, rel_attention_mask, hts_attn_mask)[0]


        hts = self.interactor_norm(hts + hts_1)
        #rels = self.interactor_norm(rels + rels_1)

        hts = hts.view(-1, hts.size(-1))[hts_attn_mask.view(-1)==1]
        #rels = rels.view(-1, rels.size(-1))[rel_attention_mask.view(-1)==1]

        hts_logits = self.entity_pair_classifier(hts)
        #rels_logits = self.rel_calssifier(rels)


        output = (self.loss_fnt.get_label(hts_logits, num_labels=self.num_labels),)
        if labels is not None and rel_labels is not None:
            labels = [torch.tensor(label) for label in labels]
            labels = torch.cat(labels, dim=0).to(hts_logits)
            hts_loss = self.loss_fnt(hts_logits.float(), labels.float())
            
            # rels_labels = torch.tensor(rel_labels).view(-1).to(rels_logits)
            # rels_loss_func = nn.BCEWithLogitsLoss(reduction="mean")
            # rels_loss = rels_loss_func(input = rels_logits.squeeze(1), target = rels_labels)

            # consis_loss_func = nn.MSELoss(reduction="sum")
            # consis_loss = consis_loss_func(input = hts_logits.mean(0), target = rels_logits.view(input_ids.size(0), -1).mean(0))

            total_loss = hts_loss #+ 0.5*rels_loss
            
            #print("hts_loss:{}, rels_loss:{}".format(hts_loss.item(), 0.5*rels_loss.item()))
            
            output = (total_loss.to(sequence_output),) + output

        return output
