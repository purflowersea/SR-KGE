import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class KGC(nn.Module):

    def __init__(self, num_ent: int, num_rel: int, dim: int = 128):
        super().__init__()
        self.dim = dim
        self.num_ent = num_ent
        self.num_rel = num_rel

        self.ent_embeddings = nn.Embedding(self.num_ent, self.dim)
        self.rel_embeddings = nn.Embedding(self.num_rel, self.dim)

        self.linear_1 = nn.Linear(self.dim * 2, 512)
        self.linear_2 = nn.Linear(512, 256)
        self.linear_pre = nn.Linear(256, 1)

        self.bce_loss = nn.BCELoss(reduction="mean")

        self._parameter_init(normalize=True)

    def _parameter_init(self, normalize: bool = False):
        nn.init.xavier_uniform_(self.ent_embeddings.weight.data)
        nn.init.xavier_uniform_(self.rel_embeddings.weight.data)
        if normalize:
            self._normalize_embeddings_()

    def _normalize_embeddings_(self):
        # L2 normalize entity & relation embeddings (row-wise)
        ent = self.ent_embeddings.weight.detach().cpu().numpy()
        ent = ent / (np.linalg.norm(ent, axis=1, keepdims=True) + 1e-12)
        self.ent_embeddings.weight.data.copy_(torch.from_numpy(ent))

        rel = self.rel_embeddings.weight.detach().cpu().numpy()
        rel = rel / (np.linalg.norm(rel, axis=1, keepdims=True) + 1e-12)
        self.rel_embeddings.weight.data.copy_(torch.from_numpy(rel))

    def _score_triples(self, triples: torch.Tensor) -> torch.Tensor:

        batch_h = triples[:, 0]
        batch_t = triples[:, 1]
        batch_r = triples[:, 2]

        h = self.ent_embeddings(batch_h)
        r = self.rel_embeddings(batch_r)
        t = self.ent_embeddings(batch_t)

        x = F.normalize(torch.cat([h, r * t], dim=-1))
        x = torch.relu(self.linear_1(x))
        x = torch.relu(self.linear_2(x))
        score = torch.sigmoid(self.linear_pre(x))  # [B, 1]
        return score

    def forward(self, data, eval: bool = False, cf_train: bool = False):

        if eval:
            triples = data if cf_train else data["hr_pair"]  # [B, 3]
            return self._score_triples(triples)

        batch = data["hr_pair"]              # [B, 4]
        triples = batch[:, :3]               # (h,t,r)
        labels = batch[:, 3].float()         # [B]
        scores = self._score_triples(triples).squeeze(-1)  # [B]
        loss = self.bce_loss(scores, labels)
        return loss


class Recommender(nn.Module):

    def __init__(self, data_config, args_config, graph=None, ui_sp_graph=None, item_rel_mask=None):
        super().__init__()
        self.n_relations = data_config["n_relations"]
        self.n_entities = data_config["n_entities"]
        self.n_nodes = data_config["n_nodes"]

        if isinstance(args_config, dict):
            dim = int(args_config.get("dim", args_config.get("embed_dim", 128)))
        else:
            dim = int(getattr(args_config, "dim", getattr(args_config, "embed_dim", 128)))

        self.kgc = KGC(num_ent=self.n_entities, num_rel=self.n_relations, dim=dim)

    def forward(self, batch, mode=None):
        return self.kgc(batch)

    def generate(self, for_kgc: bool = False):
        ent_emb = self.kgc.ent_embeddings.weight  # [n_entities, dim]
        if for_kgc:
            return ent_emb, None
        return ent_emb, None
