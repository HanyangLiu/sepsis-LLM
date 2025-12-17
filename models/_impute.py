import torch
from torch import nn
import faiss
import torch.nn.functional as F
import numpy as np
from ._modules import MLP


class kNNDecayImputation(nn.Module):
    def __init__(self, embed_size, k_neighbors):
        super().__init__()
        self.k_neighbors = k_neighbors
        self.decay_layer = nn.Linear(1, embed_size)  # Learnable mapping from distance to decay
        self.fallback_decay = nn.Linear(1, embed_size)  # For fallback weighting
        self.predictor = MLP(dropout=0.3, in_dim=2 * embed_size, post_dim=embed_size, out_dim=embed_size)

    def forward(self, knn_emb, knn_dist, fallback_emb):
        """
        knn_emb: (B, K, D) - kNN embeddings
        knn_dist: (B, K, 1) - kNN distances
        fallback_emb: (B, D) - Fallback embedding

        Returns:
        imputed_embedding: (B, D) - Final imputed embedding
        """
        # Compute learnable decay factor γ from distances
        knn_dist_flat = knn_dist.view(-1, 1)  # Flatten knn_dist to (B*K, 1) for Linear
        gamma = torch.exp(-F.relu(self.decay_layer(knn_dist_flat)))  # Shape: (B*K, D)
        gamma = gamma.view(knn_dist.shape[0], self.k_neighbors, -1)  # Reshape to (B, K, D)

        # Normalize γ across neighbors
        gamma_sum = gamma.sum(dim=1, keepdim=True) + 1e-8  # Prevent divide-by-zero
        gamma_normalized = gamma / gamma_sum  # Ensure proper weighting

        # Compute weighted k-NN embedding sum
        imputed_knn = (gamma_normalized * knn_emb).sum(dim=1)  # Shape: (B, D)

        # Compute learnable decay factor γ for fallback embedding
        fallback_weight = torch.exp(-F.relu(self.fallback_decay(knn_dist.mean(dim=1).view(-1, 1))))  # (B, D)

        # Compute final imputed embedding
        # imputed_embedding = fallback_weight * self.predictor(fallback_emb) + (1 - fallback_weight) * imputed_knn  # (B, D)
        # imputed_embedding = self.predictor(imputed_knn)
        imputed_embedding = self.predictor(torch.cat([imputed_knn, fallback_emb], dim=-1))

        return imputed_embedding


class kNNTransformerImputation(nn.Module):
    def __init__(self, embed_size, k_neighbors, num_heads=4, num_layers=2):
        super().__init__()
        self.k_neighbors = k_neighbors

        # Transformer encoder to process kNN embeddings
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_size, nhead=num_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Final linear projection (instead of MLP)
        self.predictor = nn.Linear(embed_size, embed_size)

    def forward(self, knn_emb, knn_dist, fallback_emb):
        """
        knn_emb: (B, K, D) - kNN embeddings
        knn_dist: (B, K, 1) - kNN distances (currently unused)
        fallback_emb: (B, D) - fallback embedding (optional / unused)

        Returns:
        imputed_embedding: (B, D) - Final imputed embedding
        """
        # Encode kNN embeddings via Transformer
        encoded = self.transformer(knn_emb)  # Shape: (B, K, D)

        # Mean pooling across K neighbors
        pooled = encoded.mean(dim=1)  # Shape: (B, D)

        # Linear projection
        imputed_embedding = self.predictor(pooled)  # Shape: (B, D)

        return imputed_embedding, pooled


class FixedSizeFaissIndex:
    def __init__(self, d, max_size=10000, nlist=100, nprobe=10, min_training_samples=4000):
        """
        d: Dimension of vectors
        max_size: Maximum number of vectors to store in the index
        nlist: Number of clusters (higher = faster but needs good clustering)
        nprobe: Number of clusters to search over (higher = better recall, but slower)
        min_training_samples: Minimum number of samples before training FAISS properly
        """
        self.d = d
        self.max_size = max_size
        self.nlist = nlist
        self.nprobe = nprobe
        self.min_training_samples = min_training_samples
        self.trained = False  # Track if FAISS is trained
        self.pending_vectors = []  # Store vectors before training

        # Initialize FAISS index
        self._reset_faiss()
        self.train_cnt = 0

    def _reset_faiss(self):
        """ Resets FAISS index and initializes it with a new quantizer. """
        self.id_list = []  # Track inserted vector IDs
        self.embedding_dict = {}  # Store embeddings explicitly for retrieval
        quantizer = faiss.IndexFlatIP(self.d)  # Base index for clustering
        self.index = faiss.IndexIVFFlat(quantizer, self.d, self.nlist, faiss.METRIC_INNER_PRODUCT)
        self.trained = False  # Reset training status

    def train(self):
        """
        Train the FAISS IVF index once enough data is accumulated.
        """
        if len(self.pending_vectors) < self.min_training_samples or self.train_cnt > 5:
            return  # Wait until we have enough samples

        # Convert to numpy array
        training_vectors = np.vstack(self.pending_vectors).astype(np.float32)

        # 🔍 Filter out NaN or Inf vectors
        is_finite = np.isfinite(training_vectors).all(axis=1)
        num_filtered = np.sum(~is_finite)
        training_vectors = training_vectors[is_finite]
        if training_vectors.shape[0] < self.min_training_samples:
            print(f"[WARN] Only {training_vectors.shape[0]} valid vectors after filtering. Waiting for more.")
            return
        print(
            f"[INFO] Training FAISS index with {training_vectors.shape[0]} vectors (filtered out {num_filtered} invalid).")

        # Reset FAISS index before training
        self._reset_faiss()

        # Train FAISS with the new accumulated data
        self.index.train(training_vectors)
        self.trained = True
        self.train_cnt += 1
        print("[INFO] FAISS training complete!")

        # 🔹 Move all pending vectors to FAISS and clear pending list
        self._add_vectors(training_vectors)
        self.pending_vectors = []  # Clear stored vectors after adding

    def add(self, vectors):
        """
        Add new vectors to the index while ensuring a fixed size.
        """
        num_new = vectors.shape[0]

        self.pending_vectors.extend(vectors)
        self.train()

        if not self.trained:
            return

        # Skip adding to FAISS if over capacity
        if len(self.id_list) + num_new > self.max_size:
            return  # Skip adding to FAISS until retraining

        self._add_vectors(vectors)

    def _add_vectors(self, vectors):
        # 🔹 Add new vectors to FAISS
        new_ids = np.arange(len(self.id_list), len(self.id_list) + vectors.shape[0])  # Generate unique IDs
        self.index.add_with_ids(vectors, new_ids)
        self.id_list.extend(new_ids)  # Track inserted IDs

        # Store embeddings for retrieval
        for i, vec in zip(new_ids, vectors):
            self.embedding_dict[i] = vec  # Store the corresponding vector

    def search(self, queries, k=5):
        """
        Perform kNN search to retrieve the k nearest neighbors for each query.
        Returns:
        - distances: The distances of k-nearest neighbors
        - indices: The indices of k-nearest neighbors
        - knn_embeddings: The actual embeddings of k-nearest neighbors
        """
        if not self.trained:
            raise ValueError("Cannot search in FAISS before training!")

        self.index.nprobe = self.nprobe  # Set number of clusters to search
        distances, indices = self.index.search(queries, k)

        # Retrieve actual embeddings using stored dictionary
        knn_embeddings = np.array([
            [self.embedding_dict.get(idx, np.zeros(queries.shape[1])) for idx in row]
            for row in indices
        ])

        return distances, indices, knn_embeddings

    def size(self):
        """
        Return the number of vectors currently stored in FAISS.
        """
        return self.index.ntotal


# class FixedSizeFaissIndex:
#     def __init__(self, d, max_size=10000, nlist=100, nprobe=10, min_training_samples=4000):
#         """
#         d: Dimension of vectors
#         max_size: Maximum number of vectors to store in the index
#         nlist: Number of clusters (higher = faster but needs good clustering)
#         nprobe: Number of clusters to search over (higher = better recall, but slower)
#         min_training_samples: Minimum number of samples before training FAISS properly
#         """
#         self.max_size = max_size
#         self.nlist = nlist
#         self.nprobe = nprobe
#         self.min_training_samples = min_training_samples
#         self.trained = False  # Track if FAISS is trained
#         self.pending_vectors = []  # Store vectors before training
#         self.id_list = []  # Track inserted vector IDs
#         self.embedding_dict = {}  # Store embeddings explicitly for retrieval
#
#         # Define the FAISS index
#         quantizer = faiss.IndexFlatIP(d)  # Base index for clustering
#         self.index = faiss.IndexIVFFlat(quantizer, d, self.nlist, faiss.METRIC_INNER_PRODUCT)
#
#     def train(self):
#         """
#         Train the FAISS IVF index once enough data is accumulated.
#         """
#         if len(self.pending_vectors) < self.min_training_samples:
#             return  # Wait until we have enough samples
#
#         # Convert to numpy array
#         training_vectors = np.vstack(self.pending_vectors).astype(np.float32)
#         print(f"[INFO] Training FAISS index with {training_vectors.shape[0]} vectors...")
#
#         # Reset FAISS index
#         quantizer = faiss.IndexFlatIP(training_vectors.shape[1])
#         self.index = faiss.IndexIVFFlat(quantizer, training_vectors.shape[1], self.nlist, faiss.METRIC_INNER_PRODUCT)
#
#         # Train FAISS with the new accumulated data
#         self.index.train(training_vectors)
#         self.trained = True
#         print("[INFO] FAISS training complete!")
#
#         # 🔹 Now, add all pending vectors after training
#         self.add(training_vectors)
#         self.pending_vectors = []  # Clear stored vectors after adding
#
#     def add(self, vectors):
#         """
#         Add new vectors to the index while ensuring a fixed size.
#         """
#         num_new = vectors.shape[0]
#
#         # 🔹 If FAISS is not trained, store vectors for later training
#         if not self.trained:
#             self.pending_vectors.extend(vectors)
#             self.train()  # Try training FAISS when enough data accumulates
#             return  # Skip adding until FAISS is trained
#
#         # 🔥 FIX: If FAISS reaches max capacity, start accumulating fresh pending vectors
#         if len(self.id_list) + num_new > self.max_size:
#             return  # Skip adding to FAISS until retraining
#
#         # Add new vectors
#         new_ids = np.arange(len(self.id_list), len(self.id_list) + num_new)  # Generate new unique IDs
#         self.index.add_with_ids(vectors, new_ids)
#         self.id_list.extend(new_ids)  # Track inserted IDs
#
#         # Store embeddings for retrieval
#         for i, vec in zip(new_ids, vectors):
#             self.embedding_dict[i] = vec  # Store the corresponding vector
#
#     def search(self, queries, k=5):
#         """
#         Perform kNN search to retrieve the k nearest neighbors for each query.
#         Returns:
#         - distances: The distances of k-nearest neighbors
#         - indices: The indices of k-nearest neighbors
#         - knn_embeddings: The actual embeddings of k-nearest neighbors
#         """
#         if not self.trained:
#             raise ValueError("Cannot search in FAISS before training!")
#
#         self.index.nprobe = self.nprobe  # Set number of clusters to search
#         distances, indices = self.index.search(queries, k)
#
#         # Retrieve actual embeddings using stored dictionary
#         knn_embeddings = np.array([
#             [self.embedding_dict.get(idx, np.zeros(queries.shape[1])) for idx in row]
#             for row in indices
#         ])
#
#         return distances, indices, knn_embeddings
#
#     def size(self):
#         """
#         Return the number of vectors currently stored in FAISS.
#         """
#         return self.index.ntotal

