import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple, List, Optional
from tqdm import tqdm
from time import time
import os


class KGCompletionTrainer:
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device: torch.device = torch.device('cpu'),
        verbose: bool = True
    ):

        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.verbose = verbose
        self.last_candidates = None

        self.history = {
            'epoch': [],
            'train_loss': [],
            'train_time': [],
            'kg_update_time': [],
            'total_time': []
        }
        
        self.candidate_stats = {
            'epoch': [],
            'n_candidates': [],
            'total_new_triplets': 0
        }
    
    def train_one_epoch(
        self,
        training_data: Dict,
        batch_size: int = 32,
        num_neg: int = 8,   # 每个正样本采多少个负样本，可以先用 1
    ) -> float:

        self.model.train()
        
        train_triplets = training_data.get('train_triplets')
        if train_triplets is None:
            raise ValueError("training_data必须包含'train_triplets'")

        if isinstance(train_triplets, np.ndarray):
            train_triplets = torch.LongTensor(train_triplets)
        
        if isinstance(train_triplets, torch.Tensor):
            train_triplets = train_triplets.to(self.device)   # [N, 3] = (head, tail, rel)

        n_triplets = train_triplets.size(0)
        n_batches = (n_triplets + batch_size - 1) // batch_size

        n_entities = training_data.get('n_entities', None)
        if n_entities is None:
            raise ValueError("training_data必须包含'n_entities'用于负采样")

        total_loss = 0.0
        
        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, n_triplets)
            
            pos_triplets = train_triplets[start_idx:end_idx]
            B = pos_triplets.size(0)

            # pos_triplets: [B, 3] -> [B, num_neg, 3] -> [B*num_neg, 3]
            pos_repeat = pos_triplets.unsqueeze(1).repeat(1, num_neg, 1)   # [B, num_neg, 3]
            pos_repeat = pos_repeat.view(-1, 3)                            # [B*num_neg, 3]

            neg_tails = torch.randint(
                low=0,
                high=n_entities,
                size=(pos_repeat.size(0),),
                device=self.device
            )

            neg_triplets = pos_repeat.clone()
            neg_triplets[:, 1] = neg_tails                                 # [B*num_neg, 3]

            pos_labels = torch.ones(B, 1, dtype=torch.long, device=self.device)
            neg_labels = torch.zeros(neg_triplets.size(0), 1, dtype=torch.long, device=self.device)

            all_triplets = torch.cat([pos_triplets, neg_triplets], dim=0)   # [B + B*num_neg, 3]
            all_labels = torch.cat([pos_labels, neg_labels], dim=0)         # [B + B*num_neg, 1]

            batch_triplets_with_labels = torch.cat([all_triplets, all_labels], dim=1)  # [*, 4]

            if batch_idx % 1000 == 0 and self.verbose:
                print(f"  batch {batch_idx}/{n_batches} | 正样本 {B}, 负样本 {neg_triplets.size(0)}")

            self.optimizer.zero_grad()
            
            batch_data = {
                'hr_pair': batch_triplets_with_labels,  # [batch_size*(1+num_neg), 4]
            }
            
            if hasattr(self.model, 'kgc'):
                loss = self.model.kgc(batch_data)
            else:
                loss = self.model(batch_data)
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / n_batches
        
        return avg_loss

    def train(
        self,
        training_data: Dict,
        epochs: int = 100,
        batch_size: int = 32,
        candidate_generator = None,
        kg_dict: Optional[Dict] = None,
        triplets: Optional[np.ndarray] = None,
        candidate_interval: int = 3,
        save_dir: Optional[str] = None,
        save_interval: int = 10
    ) -> Dict:

        if self.verbose:
            print("\n" + "="*60)
            print("开始训练知识图谱补全模型")
            print("="*60 + "\n")
        
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        best_loss = float('inf')
        patience_counter = 0
        max_patience = 20
        
        for epoch in range(epochs):
            epoch_start_time = time()
            
            if self.verbose:
                print(f"Epoch {epoch+1}/{epochs}")
            
            train_start_time = time()
            avg_loss = self.train_one_epoch(training_data, batch_size)
            train_time = time() - train_start_time
            
            self.history['epoch'].append(epoch)
            self.history['train_loss'].append(avg_loss)
            self.history['train_time'].append(train_time)
            
            if self.verbose:
                print(f"  训练损失: {avg_loss:.6f} (耗时: {train_time:.2f}s)")
            
            if epoch % candidate_interval == 0 and candidate_generator is not None:
                kg_start_time = time()

                if self.verbose:
                    print(f"  生成候选三元组...")

                candidates = candidate_generator(
                    model=self.model,
                    training_data=training_data,
                    kg_dict=training_data["kg_dict"],
                    device=self.device,
                    kg_source="train",
                )

                self.last_candidates = candidates

                if candidates is not None and len(candidates) > 0:

                    try:
                        from kg_completion_eval import evaluate_kg_completion

                        eval_metrics = evaluate_kg_completion(
                            model=self.model,
                            test_triplets=training_data["test_triplets"],
                            training_data=training_data,
                            candidates=candidates,
                            device=self.device,
                            verbose=self.verbose,
                        )

                        self.history.setdefault("eval_precision_on_test", []).append(eval_metrics["precision_on_test"])
                        self.history.setdefault("eval_n_hit_test", []).append(eval_metrics["n_hit_test"])
                        self.history.setdefault("eval_n_candidates", []).append(eval_metrics["n_candidates"])

                        if self.verbose:
                            print(
                                f"  [Eval] hit@test = {eval_metrics['n_hit_test']}/{eval_metrics['n_candidates']} "
                                f"({eval_metrics['precision_on_test']*100:.4f}%)"
                            )

                    except Exception as e:
                        if self.verbose:
                            print(f"  [Warn] 简单评估失败: {e}")

                    self._update_kg(training_data, candidates)

                    self.candidate_stats['epoch'].append(epoch)
                    self.candidate_stats['n_candidates'].append(len(candidates))
                    self.candidate_stats['total_new_triplets'] += len(candidates)

                    if self.verbose:
                        print(f"  新增三元组: {len(candidates)}条")

                kg_update_time = time() - kg_start_time
                self.history['kg_update_time'].append(kg_update_time)
            else:
                self.last_candidates = None 
                self.history['kg_update_time'].append(0.0)


            if self.scheduler is not None:
                self.scheduler.step()
            
            if save_dir and (epoch + 1) % save_interval == 0:
                self._save_checkpoint(save_dir, epoch, avg_loss)
            
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
                
                if save_dir:
                    self._save_checkpoint(save_dir, epoch, avg_loss, is_best=True)
            else:
                patience_counter += 1
                if patience_counter >= max_patience:
                    if self.verbose:
                        print(f"\n早停：损失在{max_patience}轮内未改进")
                    break
            
            epoch_time = time() - epoch_start_time
            self.history['total_time'].append(epoch_time)
            
            if self.verbose:
                print(f"  总耗时: {epoch_time:.2f}s\n")
        
        if self.verbose:
            print("="*60)
            print("训练完成")
            print("="*60 + "\n")
        
        return self.history
    
    def _update_kg(self, training_data: Dict, new_triplets: List) -> None:

        if isinstance(new_triplets, list):
            new_triplets = np.array(new_triplets, dtype=np.int32)
        elif isinstance(new_triplets, np.ndarray):
            new_triplets = new_triplets.astype(np.int32)
        
        if new_triplets.shape[1] != 3:
            raise ValueError(f"new_triplets应该有3列，但得到{new_triplets.shape[1]}列") 
        
        if 'kg_dict' in training_data:
            kg_dict = training_data['kg_dict']

            touched_heads = set()
            for head, tail, relation in new_triplets:
                head, tail, relation = int(head), int(tail), int(relation)
                if head not in kg_dict:
                    kg_dict[head] = []
                kg_dict[head].append((relation, tail))
                touched_heads.add(head)
            
            for h in touched_heads:
                kg_dict[h] = list(set(kg_dict[h]))
        
        if "full_kg_dict" in training_data:
            full_kg_dict = training_data["full_kg_dict"]
            for head, tail, relation in new_triplets:
                head, tail, relation = int(head), int(tail), int(relation)
                if head not in full_kg_dict:
                    full_kg_dict[head] = []
                full_kg_dict[head].append((relation, tail))

        if 'triplets' in training_data:
            old_triplets = training_data['triplets']
            if isinstance(old_triplets, torch.Tensor):
                old_triplets = old_triplets.cpu().numpy()
            training_data['triplets'] = np.vstack([old_triplets, new_triplets])
            training_data['triplets'] = np.unique(training_data['triplets'], axis=0) # ✅去重

        if "full_triplets" in training_data:
            ft = training_data["full_triplets"]
            if isinstance(ft, torch.Tensor):
                ft = ft.cpu().numpy()
            training_data["full_triplets"] = np.vstack([ft, new_triplets]).astype(np.int32)
            training_data["full_triplets"] = np.unique(training_data["full_triplets"], axis=0)

    
    def _save_checkpoint(
        self,
        save_dir: str,
        epoch: int,
        loss: float,
        is_best: bool = False
    ) -> None:

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        checkpoint_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch}.pt')
        torch.save(checkpoint, checkpoint_path)
        
        if is_best:
            best_path = os.path.join(save_dir, 'best_model.pt')
            torch.save(checkpoint, best_path)
            if self.verbose:
                print(f"  保存最佳模型: {best_path}")
    
    def load_checkpoint(self, checkpoint_path: str) -> int:

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        epoch = checkpoint['epoch']
        
        if self.verbose:
            print(f" 加载检查点: {checkpoint_path} (Epoch {epoch})")
        
        return epoch
    
    def get_history(self) -> Dict:
        return self.history
    
    def get_candidate_stats(self) -> Dict:
        return self.candidate_stats


def train_kg_completion(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    training_data: Dict,
    epochs: int = 100,
    batch_size: int = 32,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    candidate_generator = None,
    kg_dict: Optional[Dict] = None,
    triplets: Optional[np.ndarray] = None,
    device: torch.device = torch.device('cpu'),
    save_dir: Optional[str] = None,
    verbose: bool = True
) -> Tuple[Dict, KGCompletionTrainer]:
    
    trainer = KGCompletionTrainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        verbose=verbose
    )
    
    history = trainer.train(
        training_data=training_data,
        epochs=epochs,
        batch_size=batch_size,
        candidate_generator=candidate_generator,
        kg_dict=kg_dict,
        triplets=triplets,
        save_dir=save_dir
    )
    
    return history, trainer
