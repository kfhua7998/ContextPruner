"""
ContextPruner: Remove irrelevant sentences from LLM context
"""

import re
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List


class ContextPruner:
    """
    Prune context by keeping only sentences most relevant to the question.
    
    Example:
        >>> pruner = ContextPruner()
        >>> pruned = pruner.prune(
        ...     question="What is the largest planet?",
        ...     context="Jupiter is the largest. Earth is third.",
        ...     keep_ratio=0.5
        ... )
    """
    
    def __init__(self, embedding_model: str = "BAAI/bge-base-en-v1.5"):
        """
        Args:
            embedding_model: Sentence transformer model name for computing similarity
        """
        self.embedder = SentenceTransformer(embedding_model)
    
    def split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def get_importance_scores(self, question: str, sentences: List[str]) -> List[float]:
        """
        Compute semantic similarity between question and each sentence.
        
        Returns:
            List of similarity scores (higher = more relevant)
        """
        if not sentences:
            return []
        q_emb = self.embedder.encode([question])
        s_embs = self.embedder.encode(sentences)
        return cosine_similarity(q_emb, s_embs)[0].tolist()
    
    def prune(self, question: str, context: str, keep_ratio: float = 0.5) -> str:
        """
        Prune context to keep only top-k sentences by relevance.
        
        Args:
            question: The question to answer
            context: The full context text
            keep_ratio: Fraction of sentences to keep (0.0 to 1.0)
        
        Returns:
            Pruned context string
        """
        sentences = self.split_sentences(context)
        
        if not sentences or keep_ratio >= 1.0:
            return context
        
        scores = self.get_importance_scores(question, sentences)
        n_keep = max(1, int(len(sentences) * keep_ratio))
        indices = np.argsort(scores)[::-1][:n_keep]
        kept_indices = sorted(indices)
        
        return " ".join([sentences[i] for i in kept_indices])
