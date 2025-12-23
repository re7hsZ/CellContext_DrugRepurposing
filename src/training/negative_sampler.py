"""
Negative Sampler for Drug-Disease Link Prediction.
Generates negative (non-existent) drug-disease pairs for training.
"""
import torch
import random

class NegativeSampler:
    def __init__(self, num_drugs, num_diseases, positive_edges):
        """
        Args:
            num_drugs: Total number of drug nodes.
            num_diseases: Total number of disease nodes.
            positive_edges: Tensor of shape [2, num_pos_edges] containing known positive pairs.
        """
        self.num_drugs = num_drugs
        self.num_diseases = num_diseases
        
        # Store positive edges as a set for fast lookup
        self.positive_set = set()
        for i in range(positive_edges.shape[1]):
            drug_idx = positive_edges[0, i].item()
            disease_idx = positive_edges[1, i].item()
            self.positive_set.add((drug_idx, disease_idx))
            
    def sample(self, num_samples):
        """
        Sample negative drug-disease pairs.
        Returns tensor of shape [2, num_samples].
        """
        neg_samples = []
        attempts = 0
        max_attempts = num_samples * 10
        
        while len(neg_samples) < num_samples and attempts < max_attempts:
            drug_idx = random.randint(0, self.num_drugs - 1)
            disease_idx = random.randint(0, self.num_diseases - 1)
            
            if (drug_idx, disease_idx) not in self.positive_set:
                neg_samples.append([drug_idx, disease_idx])
                
            attempts += 1
            
        if len(neg_samples) < num_samples:
            print(f"Warning: Could only sample {len(neg_samples)} negatives out of {num_samples} requested.")
            
        neg_tensor = torch.tensor(neg_samples, dtype=torch.long).t()
        return neg_tensor
