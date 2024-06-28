from typing import Dict, List
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from torch import Tensor


class SimilarityFetch:
    def __init__(self,model_name:str="paraphrase-MiniLM-L6-v2") -> None:
        self.model = SentenceTransformer(model_name)    
    
    
    
    def get_top_k_similar_instances(self,
        sentence: str, doc_embeddings,docs,
        k: int, threshold: float
    ) -> List[Dict]:
        """get top k neighbours for a sentence.

        Args:
            sentence (str): input
            data_emb (Tensor): corpus embeddings
            data (List[Dict]): corpus
            k (int): top_k to return
            threshold (float):

        Returns:
            List[Dict]: list of top_k data points
        """
        sent_emb = self.model.encode(sentence)
        # doc_embeddings = self.model.encode(docs)
        # data_emb = self.get_embeddings_for_data(transfer_questions)
        print("new_emb", sent_emb.shape, doc_embeddings.shape)
        text_sims = cosine_similarity(doc_embeddings, [sent_emb]).tolist()
        results_sims = zip(range(len(text_sims)), text_sims)
        sorted_similarities = sorted(
            results_sims, key=lambda x: x[1], reverse=True)
        print("text_sims", sorted_similarities[:2])
        top_questions = []
        for idx, item in sorted_similarities[:k]:
            if item[0] > threshold:
                top_questions.append(list(docs)[idx]) 
        return top_questions