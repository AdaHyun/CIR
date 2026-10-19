import torch
import torch.nn as nn
import torch.nn.functional as F
from .raw_data import create_backbone


def _split_text_query(query: str) -> tuple[str, str]:
    marker = ", but "
    if marker in query:
        reference_caption, modification = query.split(marker, 1)
        return reference_caption.strip(), modification.strip()
    return query.strip(), query.strip()


class BaseCIRModel(nn.Module):
    def __init__(
        self,
        hidden_dim=1024,
        dropout=0.5,
        backbone="ViT-H-14",
        pretrained="laion2B-s32B-b79K",
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.clip, self.preprocess_train, self.preprocess_val, self.tokenizer = create_backbone(backbone, pretrained)
        self.clip = self.clip.float()

        self.loss_weight = torch.nn.Parameter(torch.FloatTensor((10.0,)))

        self.combiner_fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
        )
        self.dropout = nn.Dropout(dropout)
        self.scaler_fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def extract_img_fea(self, x):
        return self.clip.encode_image(x)

    def extract_text_fea(self, txt):
        txt = self.tokenizer(txt).to(next(self.parameters()).device)
        return self.clip.encode_text(txt)

    def fuse_query_features(self, textual_query, visual_query):
        combined_feature = self.combiner_fc(torch.cat([textual_query, visual_query], dim=-1))
        dynamic_scaler = self.scaler_fc(self.dropout(combined_feature))
        query = dynamic_scaler * textual_query + (1 - dynamic_scaler) * visual_query
        return F.normalize(query, p=2, dim=-1)

    def encode_textual_query(self, textual_query, visual_query=None):
        return F.normalize(self.extract_text_fea(textual_query), p=2, dim=-1)

    def extract_query(self, textual_query, visual_query):
        textual_query = self.encode_textual_query(textual_query, visual_query)
        visual_query = F.normalize(self.extract_img_fea(visual_query), p=2, dim=-1)
        return self.fuse_query_features(textual_query, visual_query)

    def extract_target(self, target_img):
        target_img_fea = self.extract_img_fea(target_img)
        return F.normalize(target_img_fea, p=2, dim=-1)

    def compute_loss(self, textual_query, visual_query, target_img):
        query_feature = self.extract_query(textual_query, visual_query)
        target_feature = self.extract_target(target_img)
        return {"ranking": self.ranking_nce_loss(query_feature, target_feature)}

    def ranking_nce_loss(self, query, target):
        x = torch.mm(query, target.t())
        labels = torch.arange(x.shape[0], device=x.device).long()
        return F.cross_entropy(self.loss_weight * x, labels)


class DQU_CIR(BaseCIRModel):
    pass


class DQU_PFP_CIR(BaseCIRModel):
    def __init__(
        self,
        hidden_dim=1024,
        dropout=0.5,
        backbone="ViT-H-14",
        pretrained="laion2B-s32B-b79K",
        prompt_templates=None,
    ):
        super().__init__(
            hidden_dim=hidden_dim,
            dropout=dropout,
            backbone=backbone,
            pretrained=pretrained,
        )
        self.prompt_templates = prompt_templates or [
            "{reference}",
            "{reference}, but {modification}",
            "Reference item: {reference}. Target change: {modification}.",
            "Find a fashion item keeping {reference} while changing {modification}.",
        ]
        self.prompt_router = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, len(self.prompt_templates)),
        )
        self.prompt_residual = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def build_prompt_texts(self, textual_query):
        prompt_groups = [[] for _ in self.prompt_templates]
        for query in textual_query:
            reference_caption, modification = _split_text_query(query)
            for idx, template in enumerate(self.prompt_templates):
                prompt_groups[idx].append(
                    template.format(
                        reference=reference_caption,
                        modification=modification,
                    )
                )
        return prompt_groups

    def encode_textual_query(self, textual_query, visual_query=None):
        base_text = F.normalize(self.extract_text_fea(textual_query), p=2, dim=-1)
        if visual_query is None:
            visual_query = torch.zeros_like(base_text)
        else:
            visual_query = F.normalize(self.extract_img_fea(visual_query), p=2, dim=-1)

        prompt_groups = self.build_prompt_texts(textual_query)
        prompt_features = []
        for prompts in prompt_groups:
            prompt_feature = F.normalize(self.extract_text_fea(prompts), p=2, dim=-1)
            prompt_features.append(prompt_feature.unsqueeze(1))
        prompt_features = torch.cat(prompt_features, dim=1)

        prompt_logits = self.prompt_router(torch.cat([base_text, visual_query], dim=-1))
        prompt_weights = torch.softmax(prompt_logits, dim=-1).unsqueeze(-1)
        prompt_mixture = torch.sum(prompt_features * prompt_weights, dim=1)
        residual = self.prompt_residual(torch.cat([base_text, prompt_mixture], dim=-1))
        return F.normalize(base_text + residual, p=2, dim=-1)


class DQU_PFP_REFCOND_CIR(BaseCIRModel):
    def __init__(
        self,
        hidden_dim=1024,
        dropout=0.5,
        backbone="ViT-H-14",
        pretrained="laion2B-s32B-b79K",
        num_prompt_queries=8,
        num_heads=8,
    ):
        super().__init__(
            hidden_dim=hidden_dim,
            dropout=dropout,
            backbone=backbone,
            pretrained=pretrained,
        )
        self.prompt_queries = nn.Parameter(torch.randn(num_prompt_queries, hidden_dim) * 0.02)
        self.prompt_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.prompt_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.prompt_gate = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def encode_textual_query(self, textual_query, visual_query=None):
        reference_captions = []
        modifications = []
        for query in textual_query:
            reference_caption, modification = _split_text_query(query)
            reference_captions.append(reference_caption)
            modifications.append(modification)

        reference_text = F.normalize(self.extract_text_fea(reference_captions), p=2, dim=-1)
        modification_text = F.normalize(self.extract_text_fea(modifications), p=2, dim=-1)
        if visual_query is None:
            visual_feature = torch.zeros_like(modification_text)
        else:
            visual_feature = F.normalize(self.extract_img_fea(visual_query), p=2, dim=-1)

        conditioning_tokens = torch.stack([reference_text, modification_text, visual_feature], dim=1)
        prompt_queries = self.prompt_queries.unsqueeze(0).expand(conditioning_tokens.size(0), -1, -1)
        prompt_tokens, _ = self.prompt_attention(prompt_queries, conditioning_tokens, conditioning_tokens)
        prompt_context = prompt_tokens.mean(dim=1)

        fused_context = self.prompt_mlp(torch.cat([prompt_context, reference_text, visual_feature], dim=-1))
        gate = self.prompt_gate(torch.cat([modification_text, prompt_context, visual_feature], dim=-1))
        conditioned_text = modification_text + gate * fused_context + (1.0 - gate) * reference_text
        return F.normalize(conditioned_text, p=2, dim=-1)
