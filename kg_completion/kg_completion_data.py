import numpy as np
from collections import defaultdict
from typing import Dict, Tuple, List


class KGCompletionDataLoader:
    
    def __init__(self, data_dir: str, verbose: bool = True):

        self.data_dir = data_dir
        self.verbose = verbose
        
        self.entity2id = {}
        self.id2entity = {}
        self.relation2id = {}
        self.id2relation = {}
        self.triplets = None
        
        self.kg_dict = {}  # 方式1：按头实体索引
        self.kg_dict_by_relation = {}  # 方式2：按关系索引
        self.kg_dict_reverse = {}  # 方式3：反向索引
        
        self.n_entities = 0
        self.n_relations = 0
        self.n_triplets = 0
        self.out_degree = None
        self.in_degree = None
        
        # 训练/测试集
        self.train_triplets = None
        self.test_triplets = None
        # train-only / full 的结构
        self.train_kg_dict = None
        self.full_kg_dict = None
        self.full_triplets = None
    
    def load_entity2id(self) -> Dict[str, int]:

        entity2id = {}
        entity_file = f"{self.data_dir}/entity2id.txt"
        
        try:
            with open(entity_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split('\t')
                    if len(parts) == 2:
                        entity_name, entity_id = parts
                        entity2id[entity_name] = int(entity_id)
        except FileNotFoundError:
            if self.verbose:
                print(f" 未找到{entity_file}，将从networks.txt推断实体ID")
            return None
        
        self.entity2id = entity2id
        self.id2entity = {v: k for k, v in entity2id.items()}
        self.n_entities = len(entity2id)
        
        if self.verbose:
            print(f" 加载entity2id.txt: {self.n_entities}个实体")
        
        return entity2id
    
    def load_networks(self) -> np.ndarray:

        triplets = []
        relation_ids = set()
        entity_ids = set()
        
        network_file = f"{self.data_dir}/networks.txt"
        
        with open(network_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) == 3:
                    head, tail, relation = map(int, parts)
                    triplets.append([head, tail, relation])
                    entity_ids.add(head)
                    entity_ids.add(tail)
                    relation_ids.add(relation)
        
        triplets = np.array(triplets, dtype=np.int32)
        
        if len(triplets) == 0:
            raise ValueError("networks.txt 没解析出任何三元组，请检查文件内容和格式")

        
        self.triplets = triplets
        self.n_triplets = len(triplets)
        self.n_entities = max(entity_ids) + 1
        self.n_relations = max(relation_ids) + 1
        
        if self.verbose:
            print(f" 加载networks.txt: {self.n_triplets}条三元组")
            print(f" 实体数: {self.n_entities}, 关系数: {self.n_relations}")
        
        return triplets
    
    def build_kg_dict(self) -> Tuple[Dict, Dict, Dict]:
    
        # 方式1：按头实体索引
        kg_dict = defaultdict(list)
        for head, tail, relation in self.triplets:
            kg_dict[head].append((relation, tail))
        self.kg_dict = dict(kg_dict)
        
        # 方式2：按关系索引
        kg_dict_by_relation = defaultdict(list)
        for head, tail, relation in self.triplets:
            kg_dict_by_relation[relation].append((head, tail))
        self.kg_dict_by_relation = dict(kg_dict_by_relation)
        
        # 方式3：反向索引
        kg_dict_reverse = defaultdict(list)
        for head, tail, relation in self.triplets:
            kg_dict_reverse[tail].append((relation, head))
        self.kg_dict_reverse = dict(kg_dict_reverse)
        
        if self.verbose:
            print(f" 构建KG字典（3种方式）")
            print(f"   方式1（按头实体）: {len(self.kg_dict)}个实体有出边")
            print(f"   方式2（按关系）: {len(self.kg_dict_by_relation)}个关系")
            print(f"   方式3（反向索引）: {len(self.kg_dict_reverse)}个实体有入边")
        
        return self.kg_dict, self.kg_dict_by_relation, self.kg_dict_reverse

    @staticmethod
    def _build_kg_dict_from_triplets(triplets: np.ndarray) -> Dict[int, List[Tuple[int, int]]]:
        kg_dict = defaultdict(list)
        for h, t, r in triplets:
            kg_dict[int(h)].append((int(r), int(t)))
        return dict(kg_dict)
    
    def compute_degree_statistics(self) -> Tuple[np.ndarray, np.ndarray]:

        # 出度：使用kg_dict
        out_degree = np.zeros(self.n_entities, dtype=np.int32)
        for entity, neighbors in self.kg_dict.items():
            out_degree[entity] = len(neighbors)
        
        # 入度：使用kg_dict_reverse
        in_degree = np.zeros(self.n_entities, dtype=np.int32)
        for entity, in_neighbors in self.kg_dict_reverse.items():
            in_degree[entity] = len(in_neighbors)
        
        self.out_degree = out_degree
        self.in_degree = in_degree
        
        if self.verbose:
            print(f" 计算度数统计")
            print(f"   平均出度: {out_degree.mean():.2f}")
            print(f"   平均入度: {in_degree.mean():.2f}")
            print(f"   最大出度: {out_degree.max()}")
            print(f"   最大入度: {in_degree.max()}")
            print(f"   出度为0的实体: {(out_degree == 0).sum()}")
            print(f"   入度为0的实体: {(in_degree == 0).sum()}")
        
        return out_degree, in_degree
    
    def compute_relation_statistics(self) -> Dict[int, int]:

        relation_counts = {}
        for relation, triplets in self.kg_dict_by_relation.items():
            relation_counts[relation] = len(triplets)
        
        if self.verbose:
            print(f" 计算关系统计")
            # 按频率排序
            sorted_relations = sorted(relation_counts.items(), key=lambda x: x[1], reverse=True)
            print(f"   关系分布（前10个）:")
            for rel_id, count in sorted_relations[:10]:
                print(f"      关系{rel_id}: {count}条 ({count/self.n_triplets*100:.2f}%)")
        
        return relation_counts 
    
    def split_train_test(self, test_ratio: float = 0.2, random_seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:

        np.random.seed(random_seed)
        
        # 随机打乱
        indices = np.random.permutation(len(self.triplets))
        
        # 划分
        split_point = int(len(self.triplets) * (1 - test_ratio))
        train_indices = indices[:split_point]
        test_indices = indices[split_point:]
        
        self.train_triplets = self.triplets[train_indices]
        self.test_triplets = self.triplets[test_indices]

        # 保存 full 视图
        self.full_triplets = self.triplets
        self.full_kg_dict = self.kg_dict  # 这是 build_kg_dict() 用全量 triplets 建好的

        # 构建 train-only 的 kg_dict（关键：训练/候选传播/verify 用它，避免看到 test）
        self.train_kg_dict = self._build_kg_dict_from_triplets(self.train_triplets)

        
        if self.verbose:
            print(f" 划分训练/测试集")
            print(f"   训练集: {len(self.train_triplets)}条 ({len(self.train_triplets)/len(self.triplets)*100:.1f}%)")
            print(f"   测试集: {len(self.test_triplets)}条 ({len(self.test_triplets)/len(self.triplets)*100:.1f}%)")
        
        return self.train_triplets, self.test_triplets
    
    def preprocess(self, test_ratio: float = 0.2) -> Dict:

        print("\n" + "="*60)
        print("开始数据加载和预处理")
        print("="*60 + "\n")
        
        self.load_entity2id()
        self.load_networks()
        
        self.build_kg_dict()
        
        self.compute_degree_statistics()
        self.compute_relation_statistics()
        
        self.split_train_test(test_ratio=test_ratio)
        
        print("\n" + "="*60)
        print("数据加载和预处理完成")
        print("="*60 + "\n")
        
        # 返回所有结果
        return {
            'triplets': self.triplets,
            'train_triplets': self.train_triplets,
            'test_triplets': self.test_triplets,
            'kg_dict': self.kg_dict,
            'kg_dict_by_relation': self.kg_dict_by_relation,
            'kg_dict_reverse': self.kg_dict_reverse,
            'n_entities': self.n_entities,
            'n_relations': self.n_relations,
            'n_triplets': self.n_triplets,
            'out_degree': self.out_degree,
            'in_degree': self.in_degree,
            'full_triplets': self.full_triplets,
            'full_kg_dict': self.full_kg_dict,
            'train_kg_dict': self.train_kg_dict,
        }


def load_and_preprocess_data(data_dir: str, test_ratio: float = 0.2, verbose: bool = True) -> Dict:

    loader = KGCompletionDataLoader(data_dir, verbose=verbose)
    return loader.preprocess(test_ratio=test_ratio)