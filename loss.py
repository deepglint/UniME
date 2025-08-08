import torch
from torch import Tensor
import torch.distributed as dist
from torch.nn import functional as F
from typing import List, Dict, Any, Optional

class SimpleContrastiveLossBatchRank:
    def __init__(self, temperature: float = 0.02):
        self.temperature = temperature

    def _select_rows_by_indices(self, matrix: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        return matrix[indices]

    def _topk_filtered_cosine_similarity(self, similarity_matrix: torch.Tensor, k: int, threshold_value: float = 0.1) -> torch.Tensor:
        """
        Calculates the top-k indices of a similarity matrix, filtering out
        negatives whose scores are too close to their corresponding positive score.

        Args:
            similarity_matrix (torch.Tensor): A square matrix of shape (N, N) where similarity_matrix[i, j] is the similarity between query i and key j.
            k (int): The number of top-ranked negatives to select.
            threshold_value (float): The value added to the diagonal (positive) similarity score to set the filtering threshold.

        Returns:
            torch.Tensor: The indices of the selected top-k negatives of shape (N, k).
        """
        n = similarity_matrix.size(0)
        
        # Clone the matrix to avoid modifying the original similarity_matrix
        modified_matrix = similarity_matrix.clone()

        # Get the diagonal values (positive similarities)
        diag_values = torch.diag(similarity_matrix)

        # Create a mask for values to filter. A value is filtered if it's
        threshold = diag_values.view(-1, 1) + threshold_value
        threshold_mask = (modified_matrix > threshold) & (~torch.eye(n, dtype=torch.bool, device=modified_matrix.device))

        # Set filtered elements and diagonal to -inf to ensure they are not selected by topk
        modified_matrix[threshold_mask] = -float('inf')
        modified_matrix.fill_diagonal_(-float('inf'))

        # Calculate the top-k values and indices from the modified matrix
        _, topk_indices = torch.topk(modified_matrix, k, dim=1)

        return topk_indices

    def __call__(self, x: Tensor, y: Tensor, reduction: str = 'mean', topk: int = 8, threshold_value: float = 0.1) -> Tensor:
        """
        Computes the batch-ranked contrastive loss.

        Args:
            x (Tensor): Query embeddings, shape (batch_size, feature_dim).
            y (Tensor): Key embeddings, shape (batch_size, feature_dim).
            reduction (str): Specifies the reduction to apply to the output.
                             'mean' or 'sum'.
            topk (int): Number of hard negatives to select.
            threshold_value (float): The threshold for filtering hard negatives.

        Returns:
            Tensor: The computed contrastive loss.
        """
        # 1. Calculate the full similarity matrix between queries and keys
        # x is (bs, fs), y is (bs, fs) -> x @ y.T is (bs, bs)
        similarity_matrix = x @ y.T

        # 2. Select top-k hard negatives based on the filtered similarity matrix
        # indices has shape (bs, topk)
        neg_indices = self._topk_filtered_cosine_similarity(similarity_matrix, topk, threshold_value)

        # 3. Gather the embeddings for the selected hard negatives
        # neg_keys has shape (bs, topk, fs)
        neg_keys = self._select_rows_by_indices(y, neg_indices)

        # 4. Combine the positive key (y) and the negative keys
        # The positive key for query i is y[i]. We need to align it with its negatives.
        # pos_key has shape (bs, 1, fs)
        pos_key = y.unsqueeze(1)
        
        # key has shape (bs, 1 + topk, fs)
        all_keys = torch.cat([pos_key, neg_keys], dim=1)
        
        # 5. Reshape query for batch matrix multiplication
        # query has shape (bs, 1, fs)
        query = x.unsqueeze(1)

        # 6. Calculate scores for cross-entropy loss
        # scores = query @ all_keys.T -> (bs, 1, fs) @ (bs, fs, 1 + topk) -> (bs, 1, 1 + topk)
        scores = torch.bmm(query, all_keys.transpose(1, 2)).squeeze(1) # Final scores shape (bs, 1 + topk)

        # 7. Create labels and compute loss
        # The label for cross-entropy is the index of the positive class.
        # Since we put the positive key at the start of all_keys, the label is 0.
        labels = torch.zeros(query.size(0), device=query.device, dtype=torch.long)
        
        # Compute the loss using a temperature-scaled cross-entropy
        loss = F.cross_entropy(scores / self.temperature, labels, reduction=reduction)

        return loss

class DistributedContrastiveLossBatchRank(SimpleContrastiveLossBatchRank):
    def __init__(
        self,
        scale_loss: bool = True,
        temperature: float = 0.02,
        topk: int = 8,
        threshold_value: float = 0.1,
    ):
        """
        Initializes the distributed loss function.

        Args:
            scale_loss (bool): If True, scales the loss by the world size. This is
                               standard practice to ensure the loss magnitude is
                               consistent regardless of the number of GPUs.
            temperature (float): The temperature for the softmax calculation.
            topk (int): Number of hard negatives to select from the combined batch.
            threshold_value (float): The threshold for filtering hard negatives.
        """
        assert dist.is_initialized(), "Distributed training has not been properly initialized."
        super().__init__(temperature=temperature)

        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()
        self.scale_loss = scale_loss
        self.temperature = temperature
        self.topk = topk
        self.threshold_value = threshold_value

    def _gather_tensor(self, t: Tensor) -> Tensor:
        gathered_tensors = [torch.empty_like(t) for _ in range(self.world_size)]
        dist.all_gather(gathered_tensors, t)
        return torch.cat(gathered_tensors, dim=0)

    def __call__(self, x: Tensor, y: Tensor, **kwargs: Any) -> Tensor:
        """
        Computes the distributed batch-ranked contrastive loss.

        Args:
            x (Tensor): Query embeddings from the current GPU, shape (local_bs, feature_dim).
            y (Tensor): Key embeddings from the current GPU, shape (local_bs, feature_dim).
            **kwargs: Additional keyword arguments passed to the base class's __call__ method.

        Returns:
            Tensor: The computed loss for the current GPU, potentially scaled.
        """
        dist_x = self._gather_tensor(x)
        dist_y = self._gather_tensor(y)

        loss = super().__call__(
            x=dist_x,
            y=dist_y,
            topk=self.topk,
            threshold_value=self.threshold_value,
            **kwargs
        )

        if self.scale_loss:
            loss = loss * self.world_size

        return loss
