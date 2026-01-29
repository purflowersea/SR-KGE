import os
import json
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import sys
import traceback
from contextlib import contextmanager
import torch

class KGCompletionOutputGenerator:

    def __init__(
        self,
        output_dir: str = './kg_completion_output/drugbank',
        verbose: bool = True
    ):

        self.output_dir = output_dir
        self.verbose = verbose
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'kg'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'plots'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'reports'), exist_ok=True)
    
    def save_kg(
        self,
        triplets: np.ndarray,
        filename: str = 'kg_extended 14 sim 0.88 score 0.92 topk 20 80000.txt'
    ) -> str:

        if self.verbose:
            print(f"\n保存知识图谱")
        
        filepath = os.path.join(self.output_dir, 'kg', filename)
        
        with open(filepath, 'w') as f:
            for triplet in triplets:
                head, tail, relation = int(triplet[0]), int(triplet[1]), int(triplet[2])
                f.write(f"{head} {tail} {relation}\n")
        
        if self.verbose:
            print(f"  保存到: {filepath}")
            print(f"  三元组数: {len(triplets)}")
        
        return filepath
    
    def save_kg_dict(
        self,
        kg_dict: Dict,
        filename: str = 'kg_dict.json'
    ) -> str:

        if self.verbose:
            print(f"\n保存KG字典")
        
        filepath = os.path.join(self.output_dir, 'kg', filename)
        
        kg_dict_serializable = {}
        for head, neighbors in kg_dict.items():
            kg_dict_serializable[str(head)] = neighbors
        
        with open(filepath, 'w') as f:
            json.dump(kg_dict_serializable, f, indent=2)
        
        if self.verbose:
            print(f"  保存到: {filepath}")
            print(f"  头实体数: {len(kg_dict)}")
        
        return filepath
    
    def save_training_history(
        self,
        history: Dict,
        filename: str = 'training_history 14 sim 0.88 score 0.92 topk 20 80000.txt.json'
    ) -> str:

        if self.verbose:
            print(f"\n保存训练历史")
        
        filepath = os.path.join(self.output_dir, 'reports', filename)
        
        history_serializable = {}
        for key, value in history.items():
            if isinstance(value, np.ndarray):
                history_serializable[key] = value.tolist()
            elif isinstance(value, list):
                history_serializable[key] = value
            else:
                history_serializable[key] = value
        
        with open(filepath, 'w') as f:
            json.dump(history_serializable, f, indent=2)
        
        if self.verbose:
            print(f"  保存到: {filepath}")
        
        return filepath
    
    def plot_training_curves(
        self,
        history: Dict,
        filename: str = 'training_curves 14 sim 0.88 score 0.92 topk 20 80000.txt.png'
    ) -> str:

        if self.verbose:
            print(f"\n绘制训练曲线")
        
        filepath = os.path.join(self.output_dir, 'plots', filename)
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        if 'train_loss' in history and len(history['train_loss']) > 0:
            axes[0, 0].plot(history['epoch'], history['train_loss'], 'b-', linewidth=2)
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].set_title('Training Loss')
            axes[0, 0].grid(True, alpha=0.3)
        
        if 'train_time' in history and len(history['train_time']) > 0:
            axes[0, 1].plot(history['epoch'], history['train_time'], 'g-', linewidth=2)
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Time (s)')
            axes[0, 1].set_title('Training Time per Epoch')
            axes[0, 1].grid(True, alpha=0.3)
        
        if 'kg_update_time' in history and len(history['kg_update_time']) > 0:
            axes[1, 0].plot(history['epoch'], history['kg_update_time'], 'r-', linewidth=2)
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Time (s)')
            axes[1, 0].set_title('KG Update Time per Epoch')
            axes[1, 0].grid(True, alpha=0.3)
        
        if 'total_time' in history and len(history['total_time']) > 0:
            axes[1, 1].plot(history['epoch'], history['total_time'], 'm-', linewidth=2)
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Time (s)')
            axes[1, 1].set_title('Total Time per Epoch')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        
        if self.verbose:
            print(f"  保存到: {filepath}")
        
        return filepath
    
    def generate_statistics_report(
        self,
        original_triplets: np.ndarray,
        extended_triplets: np.ndarray,
        training_data: Dict,
        history: Dict,
        metrics: Optional[Dict] = None,
        filename: str = 'statistics_report 14 sim 0.88 score 0.92 topk 20 80000.txt.txt'
    ) -> str:

        if self.verbose:
            print(f"\n生成统计报告")
        
        filepath = os.path.join(self.output_dir, 'reports', filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("="*60 + "\n")
            f.write("知识图谱补全 - 统计报告\n")
            f.write("="*60 + "\n\n")
            
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("="*60 + "\n")
            f.write("数据统计\n")
            f.write("="*60 + "\n")
            f.write(f"原始三元组数: {len(original_triplets)}\n")
            f.write(f"扩展后三元组数: {len(extended_triplets)}\n")
            f.write(f"新增三元组数: {len(extended_triplets) - len(original_triplets)}\n")
            f.write(f"增长率: {(len(extended_triplets) - len(original_triplets)) / len(original_triplets) * 100:.2f}%\n")
            f.write(f"实体数: {training_data.get('n_entities', 'N/A')}\n")
            f.write(f"关系数: {training_data.get('n_relations', 'N/A')}\n\n")
            
            f.write("="*60 + "\n")
            f.write("训练统计\n")
            f.write("="*60 + "\n")
            if 'epoch' in history and len(history['epoch']) > 0:
                f.write(f"训练轮数: {len(history['epoch'])}\n")
                f.write(f"最终损失: {history['train_loss'][-1]:.6f}\n")
                f.write(f"总训练时间: {np.sum(history['train_time']):.2f}秒\n")
                f.write(f"平均每轮时间: {np.mean(history['train_time']):.2f}秒\n")
                f.write(f"总KG更新时间: {np.sum(history['kg_update_time']):.2f}秒\n")
            f.write("\n")
            
            if metrics is not None:
                f.write("="*60 + "\n")
                f.write("评估指标\n")
                f.write("="*60 + "\n")
                for key, value in metrics.items():
                    if isinstance(value, float):
                        f.write(f"{key}: {value:.4f}\n")
                    else:
                        f.write(f"{key}: {value}\n")
                f.write("\n")
            
            f.write("="*60 + "\n")
            f.write("候选生成统计\n")
            f.write("="*60 + "\n")
            if 'candidate_stats' in history:
                candidate_stats = history['candidate_stats']
                if 'total_new_triplets' in candidate_stats:
                    f.write(f"总新增三元组: {candidate_stats['total_new_triplets']}\n")
                if 'epoch' in candidate_stats:
                    f.write(f"候选生成轮数: {len(candidate_stats['epoch'])}\n")
            f.write("\n")
            
            f.write("="*60 + "\n")
            f.write("报告结束\n")
            f.write("="*60 + "\n")
        
        if self.verbose:
            print(f"  保存到: {filepath}")
        
        return filepath
    
    def generate_summary(
        self,
        original_triplets: np.ndarray,
        extended_triplets: np.ndarray,
        training_data: Dict,
        history: Dict,
        metrics: Optional[Dict] = None,
        filename: str = 'summary 14 sim 0.88 score 0.92 topk 20 80000.txt.json'
    ) -> str:

        if self.verbose:
            print(f"\n生成结果摘要")
        
        filepath = os.path.join(self.output_dir, 'reports', filename)
        
        summary = {
            'timestamp': datetime.now().isoformat(),
            'data_statistics': {
                'original_triplets': int(len(original_triplets)),
                'extended_triplets': int(len(extended_triplets)),
                'new_triplets': int(len(extended_triplets) - len(original_triplets)),
                'growth_rate': float((len(extended_triplets) - len(original_triplets)) / len(original_triplets) * 100),
                'n_entities': int(training_data.get('n_entities', 0)),
                'n_relations': int(training_data.get('n_relations', 0))
            },
            'training_statistics': {
                'epochs': int(len(history.get('epoch', []))),
                'final_loss': float(history['train_loss'][-1]) if 'train_loss' in history and len(history['train_loss']) > 0 else None,
                'total_training_time': float(np.sum(history['train_time'])) if 'train_time' in history else None,
                'avg_time_per_epoch': float(np.mean(history['train_time'])) if 'train_time' in history and len(history['train_time']) > 0 else None,
                'total_kg_update_time': float(np.sum(history['kg_update_time'])) if 'kg_update_time' in history else None
            }
        }
        
        if metrics is not None:
            summary['evaluation_metrics'] = {
                key: float(value) if isinstance(value, (float, np.floating)) else value
                for key, value in metrics.items()
            }
        
        with open(filepath, 'w') as f:
            json.dump(summary, f, indent=2)
        
        if self.verbose:
            print(f"  保存到: {filepath}")
        
        return filepath
    
    def generate_all_outputs(
        self,
        original_triplets: np.ndarray,
        extended_triplets: np.ndarray,
        training_data: Dict,
        history: Dict,
        metrics: Optional[Dict] = None,
        kg_dict: Optional[Dict] = None
    ) -> Dict[str, str]:

        if self.verbose:
            print("\n" + "="*60)
            print("生成所有输出")
            print("="*60)
        
        outputs = {}
        
        outputs['kg'] = self.save_kg(extended_triplets)
        
        outputs['history'] = self.save_training_history(history)
        
        outputs['curves'] = self.plot_training_curves(history)
        
        outputs['report'] = self.generate_statistics_report(
            original_triplets,
            extended_triplets,
            training_data,
            history,
            metrics
        )
        
        outputs['summary'] = self.generate_summary(
            original_triplets,
            extended_triplets,
            training_data,
            history,
            metrics
        )
        
        if self.verbose:
            print("\n" + "="*60)
            print("所有输出已生成")
            print("="*60 + "\n")
        
        return outputs


def generate_kg_completion_outputs(
    original_triplets: np.ndarray,
    extended_triplets: np.ndarray,
    training_data: Dict,
    history: Dict,
    metrics: Optional[Dict] = None,
    kg_dict: Optional[Dict] = None,
    output_dir: str = './kg_completion_output',
    verbose: bool = True
) -> Dict[str, str]:

    generator = KGCompletionOutputGenerator(
        output_dir=output_dir,
        verbose=verbose
    )
    
    outputs = generator.generate_all_outputs(
        original_triplets=original_triplets,
        extended_triplets=extended_triplets,
        training_data=training_data,
        history=history,
        metrics=metrics,
        kg_dict=kg_dict
    )
    
    return outputs

if __name__ == '__main__':
    
    from KGC import Recommender
    from kg_completion_model import initialize_kg_completion_model
    from kg_completion_train import train_kg_completion
    from kg_completion_eval import evaluate_kg_completion
    from kg_completion_candidate import generate_candidates
    import argparse
    import os
    import random
    import numpy as np
    import torch
    import torch.nn.functional as F

    def set_seed(seed: int = 42, deterministic: bool = False):
        # Python / numpy
        os.environ["PYTHONHASHSEED"] = str(seed)
        random.seed(seed)
        np.random.seed(seed)

        # Torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

            try:
                torch.use_deterministic_algorithms(True)
            except Exception:
                pass

            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    
    parser = argparse.ArgumentParser(description='KG补全输出生成')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--deterministic', action='store_true', default=False)
    parser.add_argument('--data_dir', type=str, default='biomedical_kg/drugbank',
                        help='数据目录')
    parser.add_argument('--epochs', type=int, default=14,
                        help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='批处理大小')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='学习率')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='计算设备')
    parser.add_argument('--output_dir', type=str, default='kg_completion_output/drugbank',
                        help='输出目录')
    
    args = parser.parse_args()

    set_seed(args.seed, deterministic=args.deterministic)
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    print("初始化模型...")
    model, optimizer, training_data = initialize_kg_completion_model(
        data_dir=args.data_dir,
        model_class=Recommender,
        args_config={'embed_dim': 128, 'context_hops': 2},
        device=device,
        lr=args.lr, 
        verbose=True
    )
    
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    
    print("\n开始训练...")
    history, trainer = train_kg_completion(
        model=model,
        optimizer=optimizer,
        training_data=training_data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        scheduler=scheduler,
        candidate_generator=generate_candidates,
        kg_dict=training_data['kg_dict'],
        triplets=training_data['triplets'],
        device=device,
        save_dir='./checkpoints/drugbank',
        verbose=True
    )
    
    with torch.no_grad():
        ent_emb = model.kgc.ent_embeddings.weight          # [n_entities, dim]
        ent_emb = F.normalize(ent_emb, dim=1)              # L2 归一化

        n_entities = ent_emb.size(0)
        num_pairs = 50000                                  # 随机抽 5 万对实体
        idx1 = torch.randint(0, n_entities, (num_pairs,), device=ent_emb.device)
        idx2 = torch.randint(0, n_entities, (num_pairs,), device=ent_emb.device)

        v1 = ent_emb[idx1]                                 # [num_pairs, dim]
        v2 = ent_emb[idx2]
        sims = (v1 * v2).sum(dim=1)                        # 余弦相似度

        print("\n当前实体嵌入相似度统计（随机抽样）:")
        print(f"  mean = {sims.mean().item():.4f}")
        print(f"  max  = {sims.max().item():.4f}")
        print(f"  min  = {sims.min().item():.4f}")
        high_sim = (sims > 0.8).sum().item()
        print(f"  > 0.8 的实体对数量: {high_sim}")

    print("\n训练结束后做一次简单评估...")
    if trainer.last_candidates is not None and len(trainer.last_candidates) > 0:
        metrics_train = evaluate_kg_completion(
            model=model,
            test_triplets=training_data["test_triplets"],
            training_data=training_data,
            candidates=trainer.last_candidates,
            device=device,
            verbose=True,
        )
        print("[Eval@TrainCandidates]", metrics_train)
    else:
        metrics_train = None
        print("没有 last_candidates，跳过最终评估")

    print("\n[Final Generate] 用全量图(full_kg_dict)传播生成最终候选...")

    print("\n[DEBUG] keys:", sorted(training_data.keys())[:50], " ...")
    print("[DEBUG] full_kg_dict:", "full_kg_dict" in training_data, type(training_data.get("full_kg_dict", None)))
    print("[DEBUG] full_triplets:", "full_triplets" in training_data, type(training_data.get("full_triplets", None)))

    final_candidates_full = generate_candidates(
        model=model,
        training_data=training_data,
        device=device,
        similarity_threshold=0.88,
        score_threshold=0.92,
        batch_size=args.batch_size,
        verify_method="single",   
        block_size=512,
        kg_source="full",       
    )

    print(f"[Final Generate] final_candidates_full = {len(final_candidates_full)}")


    print("\n[Final Eval] 用最终(full)候选命中测试集（可选）...")
    metrics_full = evaluate_kg_completion(
        model=model,
        test_triplets=training_data["test_triplets"],
        training_data=training_data,
        candidates=final_candidates_full,
        device=device,
        verbose=True,
    )
    print("[Eval@FullCandidates]", metrics_full)

    print("\n生成输出...")
    full_triplets = training_data['full_triplets']  # 这里应该是扩展后的三元组
    cand_arr = np.array(final_candidates_full, dtype=np.int64) if len(final_candidates_full) > 0 \
        else np.zeros((0, 3), dtype=np.int64)

    extended_triplets = np.concatenate([full_triplets, cand_arr], axis=0)
    extended_triplets = np.unique(extended_triplets, axis=0)  # 去重

    
    outputs = generate_kg_completion_outputs(
        original_triplets=training_data['original_triplets'],
        extended_triplets=extended_triplets,
        training_data=training_data,
        history=history,
        metrics=metrics_full,
        kg_dict=None,
        output_dir=args.output_dir,
        verbose=True
    )
    
    print("\n生成的输出文件:")
    for key, filepath in outputs.items():
        print(f"  {key}: {filepath}")
