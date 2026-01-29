import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional


class KGCompletionEvaluatorSimple:
    def __init__(
        self,
        model: Optional[nn.Module] = None, 
        device: torch.device = torch.device("cpu"),
        verbose: bool = True
    ):
        self.model = model
        self.device = device
        self.verbose = verbose

    @staticmethod
    def _to_set(triplets: np.ndarray) -> set:
        """np.ndarray [N,3] -> set((h,t,r), ...)"""
        if triplets is None or len(triplets) == 0:
            return set()
        if not isinstance(triplets, np.ndarray):
            triplets = np.array(triplets)
        triplets = triplets.astype(np.int64)
        return set(map(tuple, triplets.tolist()))

    def evaluate_candidates_hit_test(
        self,
        candidates: List[Tuple[int, int, int]],
        test_triplets: np.ndarray,
        return_matches: bool = False
    ) -> Dict:
        """
        candidates: List[(h,t,r)]
        test_triplets: np.ndarray [N,3]
        """
        if candidates is None:
            candidates = []

        cand_set = set(candidates)
        test_set = self._to_set(test_triplets)

        hit = cand_set & test_set

        metrics = {
            "n_candidates": int(len(candidates)),
            "n_hit_test": int(len(hit)),
            "precision_on_test": float(len(hit) / len(candidates)) if len(candidates) > 0 else 0.0,
        }

        if self.verbose:
            print("\n" + "=" * 60)
            print("简单评估：候选命中测试集")
            print("=" * 60)
            print(f"候选数 n_candidates: {metrics['n_candidates']}")
            print(f"命中测试集 n_hit_test: {metrics['n_hit_test']}")
            print(f"precision_on_test: {metrics['precision_on_test']:.6f}")
            print("=" * 60 + "\n")

        if return_matches:
            metrics["matches"] = list(hit)
        return metrics


def evaluate_kg_completion(
    model: nn.Module,
    test_triplets: np.ndarray,
    training_data: Dict,
    candidates: Optional[List[Tuple[int, int, int]]] = None,
    device: torch.device = torch.device("cpu"),
    verbose: bool = True,
    return_matches: bool = False
) -> Dict:

    evaluator = KGCompletionEvaluatorSimple(model=model, device=device, verbose=verbose)

    if candidates is None:
        candidates = training_data.get("last_candidates", [])

    return evaluator.evaluate_candidates_hit_test(
        candidates=candidates,
        test_triplets=test_triplets,
        return_matches=return_matches
    )
