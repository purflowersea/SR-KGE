import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm

def topk_indices_1d(row: np.ndarray, k: int, self_id: Optional[int] = None, min_th: Optional[float] = None) -> np.ndarray:
    
    if (k is None or k <= 0) and (min_th is None):
        raise ValueError("Invalid: k<=0 and min_th=None would create full connection. "
                         "Set min_th (similarity_threshold) or set k>0.")

    if k is None or k <= 0:
        # 纯阈值模式
        idx = np.where(row >= min_th)[0].astype(np.int64)
        if self_id is not None:
            idx = idx[idx != self_id]
        return idx

    row2 = row
    if self_id is not None:
        row2 = row.copy()
        row2[self_id] = -1e9

    n = row2.shape[0]
    kk = min(k, n)

    idx = np.argpartition(-row2, kk - 1)[:kk]
    idx = idx[np.argsort(-row2[idx])]

    if min_th is not None:
        idx = idx[row2[idx] >= min_th]

    return idx.astype(np.int64)


def topk_global(scores: np.ndarray, k: int) -> np.ndarray:

    if k is None or k <= 0 or scores.size == 0:
        return np.arange(scores.size, dtype=np.int64)

    kk = min(k, scores.size)
    idx = np.argpartition(-scores, kk - 1)[:kk]
    idx = idx[np.argsort(-scores[idx])]
    return idx.astype(np.int64)

class CandidateGenerator:
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device = torch.device('cpu'),
        verbose: bool = True
    ):
        
        self.model = model
        self.device = device
        self.verbose = verbose
    
    def compute_entity_neighbors(
        self,
        training_data: Dict,
        similarity_metric: str = 'cosine',
        similarity_threshold: float = 0.88,
        block_size: int = 512,
        use_heads_only: bool = True,
        kg_dict_for_heads: Dict = None,
        topk_neighbors: int = 20, 
        exclude_self: bool = True,
    ) -> Tuple[Dict[int, np.ndarray], int]:
        
        if self.verbose:
            print("\n第1步：分块计算实体相似度（稀疏存储）")

        with torch.no_grad():
            entity_embeddings = None

            if hasattr(self.model, 'kgc') and hasattr(self.model.kgc, 'ent_embeddings'):
                entity_embeddings = self.model.kgc.ent_embeddings.weight    # [n_entities, dim]

            if entity_embeddings is None:
                raise ValueError("无法从模型中获取实体嵌入")

            ent_emb = entity_embeddings.detach().cpu().numpy().astype(np.float32)

        n_entities_total, dim = ent_emb.shape

        if self.verbose:
            print(f"  实体嵌入形状: {ent_emb.shape}")
            if 'n_entities' in training_data:
                print(f"  training_data['n_entities']: {training_data['n_entities']}")

        if similarity_metric == 'cosine':
            norms = np.linalg.norm(ent_emb, axis=1, keepdims=True) + 1e-8
            ent_emb = ent_emb / norms
        else:
            raise ValueError(f"不支持的相似度度量方式: {similarity_metric}")

        kg_dict = kg_dict_for_heads if kg_dict_for_heads is not None else training_data.get("kg_dict")
        if kg_dict is None:
            raise ValueError("training_data 必须包含 'kg_dict'")

        if use_heads_only:
            head_entities = sorted(kg_dict.keys())
        else:
            head_entities = list(range(n_entities_total))

        n_heads = len(head_entities)
        if self.verbose:
            print(f"  将对 {n_heads} 个实体计算相似度（use_heads_only={use_heads_only}）")

        neighbors: Dict[int, np.ndarray] = {}
        total_pairs = 0

        for start in range(0, n_heads, block_size):
            end = min(start + block_size, n_heads)
            batch_heads = head_entities[start:end]             
            block = ent_emb[batch_heads]                        

            # [B, N] = [B, dim] @ [dim, N]
            sims = np.matmul(block, ent_emb.T)

            for i_row, head_id in enumerate(batch_heads):
                row = sims[i_row]                            
                idx = topk_indices_1d(
                    row=row,
                    k=topk_neighbors,                 
                    self_id=head_id if exclude_self else None,
                    min_th=similarity_threshold       
                )

                if idx.size > 0:
                    neighbors[head_id] = idx
                    total_pairs += idx.size

            if self.verbose and (start // block_size) % 10 == 0:
                print(f"  已处理 head 实体 {end}/{n_heads}，当前相似实体对数: {total_pairs}")

        if self.verbose:
            print(f"  最终相似实体对数（满足阈值）: {total_pairs}")

        return neighbors, n_entities_total
    
    def propagate_relations(
        self,
        neighbors: Dict[int, np.ndarray],
        kg_dict: Dict,
    ) -> List[Tuple[int, int, int]]:
        
        if self.verbose:
            print("\n第2步：关系传播")

        candidates = []

        for entity_a, similar_entities in neighbors.items():
            if entity_a not in kg_dict:
                continue

            relations_of_a = kg_dict[entity_a]

            for entity_b in similar_entities:
                for relation, tail in relations_of_a:
                    candidates.append((int(entity_b), int(tail), int(relation)))

        candidates = list(set(candidates))

        if self.verbose:
            print(f"  生成候选三元组: {len(candidates)}条")

        return candidates

    
    def score_candidates(
        self,
        candidates: List[Tuple[int, int, int]],
        batch_size: int = 32,
        score_threshold: float = 0.92,
        topk_candidates: int = 80000,
    ) -> Tuple[List[Tuple[int, int, int]], np.ndarray]:
        
        if self.verbose:
            print("\n第3步：KGC评分")
        
        if len(candidates) == 0:
            return [], np.array([])
        
        # 转换为张量
        candidates_array = np.array(candidates)
        candidates_tensor = torch.LongTensor(candidates_array).to(self.device)
        
        scores = []
        
        # 分批评分
        n_batches = (len(candidates_tensor) + batch_size - 1) // batch_size
        
        with torch.no_grad():
            for batch_idx in tqdm(range(n_batches), desc="评分候选", disable=not self.verbose):
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, len(candidates_tensor))
                
                batch_candidates = candidates_tensor[start_idx:end_idx]
                
                batch_data = {
                    'hr_pair': batch_candidates
                }
                
                batch_scores = self.model.kgc(batch_data, eval=True)
                
                batch_scores = batch_scores.squeeze(-1).cpu().numpy()
                scores.extend(batch_scores)
        
        scores = np.array(scores)
        
        mask = np.isfinite(scores)
        if not np.all(mask):
            scores = scores[mask]
            candidates = [candidates[i] for i in np.where(mask)[0]]

        keep = np.where(scores >= score_threshold)[0]
        scores2 = scores[keep]
        cands2 = [candidates[i] for i in keep]

        idx2 = topk_global(scores2, topk_candidates)
        filtered_candidates = [cands2[i] for i in idx2]
        filtered_scores = scores2[idx2]

        if self.verbose:
            print(f"  评分后候选三元组: {len(filtered_candidates)}条")
            if scores.size > 0:
                print(f"  平均评分: {np.mean(scores):.4f}")
                print(f"  最高评分: {np.max(scores):.4f}")
        
        return filtered_candidates, filtered_scores
    
    def verify_candidates(
        self,
        candidates: List[Tuple[int, int, int]],
        existing_triplets: np.ndarray,
        verify_method: str = 'single'
    ) -> List[Tuple[int, int, int]]:
        
        if self.verbose:
            print("\n第4步：验证筛选")
        
        # 构建已存在三元组集合（用于快速查询）
        existing_set = set()
        for triplet in existing_triplets:
            existing_set.add(tuple(triplet))
        
        verified_candidates = []
        
        if verify_method == 'single':
            for head, tail, relation in candidates:
                candidate_tuple = (head, tail, relation)
                reverse_tuple = (tail, head, relation)
                
                if candidate_tuple in existing_set:
                    continue
                
                if reverse_tuple in existing_set:
                    continue
                
                verified_candidates.append((head, tail, relation))
        
        else:
            for head, tail, relation in candidates:
                candidate_tuple = (head, tail, relation)
                if candidate_tuple not in existing_set:
                    verified_candidates.append((head, tail, relation))
        
        if self.verbose:
            print(f"  验证后候选三元组: {len(verified_candidates)}条")
        
        return verified_candidates
    
    def generate_candidates(
        self,
        training_data: Dict,
        similarity_threshold: float = 0.88,
        score_threshold: float = 0.92,
        batch_size: int = 32,
        verify_method: str = 'single',
        block_size: int = 512,
        kg_source: str = 'train',
        topk_neighbors: int = 20,      
        topk_candidates: int = 80000,     
    ) -> List[Tuple[int, int, int]]:
        
        if self.verbose:
            print("\n" + "="*60)
            print("开始候选生成")
            print("="*60)

        if kg_source == "full":
            kg_dict = training_data.get("full_kg_dict")
        else:
            kg_dict = training_data.get("kg_dict")  # train_kg_dict

        if kg_dict is None:
            raise ValueError("training_data缺少对应kg_dict")

        if self.verbose:
            print(f"[KG SOURCE] kg_source={kg_source} | #heads(with out edges)={len(kg_dict)}")
        
        # 第1步：分块计算实体相似邻居
        neighbors, n_entities = self.compute_entity_neighbors(
            training_data,
            similarity_threshold=similarity_threshold,
            block_size=block_size,
            use_heads_only=True,   # 只对有出边的实体算行相似度
            kg_dict_for_heads = kg_dict,
            topk_neighbors=topk_neighbors
        )

        if len(neighbors) == 0:
            if self.verbose:
                print("  没有任何实体找到相似邻居，候选生成终止")
            return []

        # 第2步：关系传播
        candidates = self.propagate_relations(
            neighbors=neighbors,
            kg_dict=kg_dict,
        )

        if len(candidates) == 0:
            if self.verbose:
                print("未生成任何候选三元组")
            return []

        # 第3步：KGC评分
        candidates, scores = self.score_candidates(
            candidates,
            batch_size=batch_size,
            score_threshold=score_threshold,
            topk_candidates=topk_candidates
        )

        if len(candidates) == 0:
            if self.verbose:
                print("评分后无有效候选三元组")
            return []

        # 第4步：双重验证筛选
        if kg_source == "full":
            existing_triplets = np.vstack([training_data["original_triplets"],training_data["train_triplets"],])
        else:
            existing_triplets = training_data.get("train_triplets")  # 只看训练边，避免 test 泄漏

        if existing_triplets is not None:
            candidates = self.verify_candidates(
                candidates=candidates,
                existing_triplets=existing_triplets,
                verify_method=verify_method
            )

        if self.verbose:
            print("\n" + "="*60)
            print(f"候选生成完成：{len(candidates)}条新三元组")
            print("="*60 + "\n")

        return candidates

def generate_candidates(
    model: nn.Module,
    training_data: Dict,
    kg_dict: Optional[Dict] = None,
    device: torch.device = torch.device('cpu'),
    similarity_threshold: float = 0.88,
    score_threshold: float = 0.92,
    batch_size: int = 32,
    verify_method: str = 'single',
    verbose: bool = True,
    block_size: int = 512,
    kg_source: str = 'train',
    topk_neighbors: int = 20,
    topk_candidates: int = 80000,     
) -> List[Tuple[int, int, int]]:
    
    if kg_dict is not None:
        if kg_source == "full":
            training_data["full_kg_dict"] = kg_dict
        else:
            training_data["kg_dict"] = kg_dict

    
    generator = CandidateGenerator(
        model=model,
        device=device,
        verbose=verbose
    )
    
    candidates = generator.generate_candidates(
        training_data=training_data,
        similarity_threshold=similarity_threshold,
        score_threshold=score_threshold,
        batch_size=batch_size,
        verify_method=verify_method,
        kg_source=kg_source,
        topk_neighbors=topk_neighbors,
        topk_candidates=topk_candidates,    
    )
    
    return candidates
