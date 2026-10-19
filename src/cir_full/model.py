from __future__ import annotations
import hashlib, random
import torch
from torch import nn
class FeatureReranker(nn.Module):
 def __init__(self,input_dim,hidden_dim,dropout,base_score_scale=1.):
  super().__init__();self.query_proj=nn.Linear(input_dim,hidden_dim);self.cand_proj=nn.Linear(input_dim,hidden_dim)
  self.score_head=nn.Sequential(nn.Linear(hidden_dim*4+1,hidden_dim),nn.ReLU(),nn.Dropout(dropout),nn.Linear(hidden_dim,1));self.base_scale=nn.Parameter(torch.tensor(float(base_score_scale)))
 def forward(self,query,candidates,base_scores):
  q=self.query_proj(query);c=self.cand_proj(candidates);q=q.unsqueeze(1).expand_as(c)
  return self.score_head(torch.cat((q,c,q*c,torch.abs(q-c),base_scores.unsqueeze(-1)),dim=-1)).squeeze(-1)+self.base_scale*base_scores

def training_item(bundle,index,epoch,cfg):
 """IDs define mapping. All candidate scores are true dot products of frozen unit features."""
 q,im,top=bundle['query'],bundle['image'],bundle['topk'];ids=im['image_ids'];lookup=bundle['index']
 target=q['target_ids'][index];source=q['source_ids'][index]
 qid=bundle['query_ids'][index]
 seed=int.from_bytes(hashlib.sha256(f"{cfg['seed']}:{epoch}:{qid}".encode()).digest()[:8],'big');rng=random.Random(seed)
 pool=[x for x in top['sorted_ids'][index][:cfg['train_pool']] if x not in {target,source}]
 if len(pool)<cfg['train_k']-1:raise ValueError(f'insufficient distinct negatives for {qid}: {len(pool)}')
 negatives=rng.sample(pool,cfg['train_k']-1);label=rng.randrange(cfg['train_k']);chosen=negatives[:];chosen.insert(label,target)
 candidates=im['features'][[lookup[x] for x in chosen]].float();query=q['query_features'][index].float()
 scores=torch.mv(candidates,query)
 if cfg.get('score_rule')=='legacy-positive-constant-ablation':scores=scores.clone();scores[label]=1.0
 elif cfg.get('score_rule','true-similarity')!='true-similarity':raise ValueError('unknown RR score rule')
 return query,candidates,scores,label,chosen
