import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional
from kg_completion_data import load_and_preprocess_data


class KGCompletionModelInitializer:
    
    def __init__(self, data_dir: str, args_config, device: torch.device, verbose: bool = True):

        self.data_dir = data_dir
        self.args_config = args_config
        self.device = device
        self.verbose = verbose
        
        self.data = None
        self.n_entities = 0
        self.n_relations = 0
        self.n_nodes = 0
        
        self.kg_dict = None
        self.triplets = None
        self.train_triplets = None
        self.test_triplets = None
        
        self.n_params = None
        self.model = None
        self.optimizer = None
        self.scheduler = None
    
    def load_data(self) -> Dict:

        if self.verbose:
            print("\n" + "="*60)
            print("第1步：加载和预处理数据")
            print("="*60 + "\n")
        
        self.data = load_and_preprocess_data(
            self.data_dir,
            test_ratio=0.2,
            verbose=self.verbose
        )
        
        self.n_entities = self.data['n_entities']
        self.n_relations = self.data['n_relations']
        self.n_nodes = self.n_entities
        
        self.kg_dict = self.data['kg_dict']
        self.triplets = self.data['triplets']
        self.train_triplets = self.data['train_triplets']
        self.test_triplets = self.data['test_triplets']
        
        if self.verbose:
            print(f" 数据加载完成")
            print(f"   实体数: {self.n_entities}")
            print(f"   关系数: {self.n_relations}")
            print(f"   训练集: {len(self.train_triplets)}")
            print(f"   测试集: {len(self.test_triplets)}\n")
        
        return self.data
    
    def build_model_config(self) -> Dict:

        if self.verbose:
            print("="*60)
            print("第2步：构建模型配置")
            print("="*60 + "\n")
        
        self.n_params = {
            'n_entities': self.n_entities,   
            'n_relations': self.n_relations, 
            'n_nodes': self.n_nodes          
        }
        
        if self.verbose:
            print(f" 模型配置构建完成")
            print(f"   n_entities: {self.n_params['n_entities']}")
            print(f"   n_relations: {self.n_params['n_relations']}")
            print(f"   n_nodes: {self.n_params['n_nodes']}\n")
        
        return self.n_params   
    
    def initialize_model(self, model_class) -> nn.Module:

        if self.verbose:
            print("="*60)
            print("第4步：初始化模型")
            print("="*60 + "\n")
        
        self.model = model_class(
            data_config=self.n_params,
            args_config=self.args_config,
            graph=None,               
            ui_sp_graph=None,           
            item_rel_mask=None           
        ).to(self.device)
        
        if self.verbose:
            print(f" 模型初始化完成")
            print(f"   模型类: {model_class.__name__}")
            print(f"   设备: {self.device}")
            print(f"   参数数量: {sum(p.numel() for p in self.model.parameters())}\n")
        
        return self.model
    
    def initialize_optimizer(self, lr: float = 0.001, weight_decay: float = 0.0) -> torch.optim.Optimizer:

        if self.verbose:
            print("="*60)
            print("第5步：初始化优化器")
            print("="*60 + "\n")
        
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        
        if self.verbose:
            print(f" 优化器初始化完成")
            print(f"   优化器: Adam")
            print(f"   学习率: {lr}")
            print(f"   权重衰减: {weight_decay}\n")
        
        return self.optimizer
    
    def initialize_scheduler(self, step_size: int = 10, gamma: float = 0.1) -> Optional[torch.optim.lr_scheduler._LRScheduler]:

        if self.verbose:
            print("="*60)
            print("第6步：初始化学习率调度器")
            print("="*60 + "\n")
        
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=step_size,
            gamma=gamma
        )
        
        if self.verbose:
            print(f" 学习率调度器初始化完成")
            print(f"   调度器: StepLR")
            print(f"   step_size: {step_size}")
            print(f"   gamma: {gamma}\n")
        
        return self.scheduler
    
    def prepare_training_data(self) -> Dict:

        if self.verbose:
            print("="*60)
            print("第7步：准备训练数据")
            print("="*60 + "\n")
        
        training_data = {
            'kg_dict': self.data['train_kg_dict'],
            'triplets': self.train_triplets,
            "full_triplets": self.data["full_triplets"],
            "full_kg_dict": self.data["full_kg_dict"],
            'train_triplets': self.train_triplets,
            'test_triplets': self.test_triplets,
            'n_entities': self.n_entities,
            'n_relations': self.n_relations,
            'original_triplets': self.data["full_triplets"].copy(),
        }
        
        if self.verbose:
            print(f" 训练数据准备完成")
            print(f"   train_triplets: {training_data['train_triplets'].shape}")
            print(f"   test_triplets: {training_data['test_triplets'].shape}\n")
        
        return training_data
    
    def get_summary(self) -> Dict:

        summary = {
            'data': self.data,
            'model_config': self.n_params,
            'model': self.model,
            'optimizer': self.optimizer,
            'scheduler': self.scheduler,
            'kg_dict': self.kg_dict,
            'triplets': self.triplets,
            'train_triplets': self.train_triplets,
            'test_triplets': self.test_triplets,
            'n_entities': self.n_entities,
            'n_relations': self.n_relations,
            'device': self.device,
        }
        
        return summary


def initialize_kg_completion_model(
    data_dir: str,
    model_class,
    args_config,
    device: torch.device,
    lr: float = 0.001,
    weight_decay: float = 0.0,
    verbose: bool = True
) -> Tuple[nn.Module, torch.optim.Optimizer, Dict]:

    initializer = KGCompletionModelInitializer(
        data_dir=data_dir,
        args_config=args_config,
        device=device,
        verbose=verbose
    )
    
    initializer.load_data()
    initializer.build_model_config()
    model = initializer.initialize_model(model_class)
    optimizer = initializer.initialize_optimizer(lr=lr, weight_decay=weight_decay)
    scheduler = initializer.initialize_scheduler()
    training_data = initializer.prepare_training_data()
    
    if verbose:
        print("="*60)
        print(" 模型初始化完成！")
        print("="*60 + "\n")
    
    return model, optimizer, training_data