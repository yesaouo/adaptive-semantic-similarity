from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
import numpy as np

class AggregateEmbeddings:
    def __init__(self, model_name='sentence-transformers/all-mpnet-base-v2', chunk_size=512):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.chunk_size = chunk_size

    def get_tokens(self, text):
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        return tokens

    def split_tokens(self, tokens):
        n = len(tokens) // self.chunk_size + 1
        avg_len = len(tokens) // n
        extras = len(tokens) % n
        result = []
        start = 0
        for i in range(n):
            end = start + avg_len + (1 if i < extras else 0)
            result.append(tokens[start:end])
            start = end
        return result

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def get_embeddings(self, text):
        tokens = self.get_tokens(text)
        chunks = self.split_tokens(tokens)
        embeddings = []
        for chunk in chunks:
            window_text = self.tokenizer.decode(chunk)
            encoded_input = self.tokenizer(window_text, padding=True, truncation=True, return_tensors='pt').to(self.device)
            with torch.no_grad():
                model_output = self.model(**encoded_input)
            chunk_embedding = self.mean_pooling(model_output, encoded_input['attention_mask'])
            embeddings.append(chunk_embedding)
        return torch.cat(embeddings, dim=0)

    def aggregate_embeddings(self, embeddings):
        return torch.mean(embeddings, dim=0)

    def cosine_similarity(self, embedding1, embedding2):
        return F.cosine_similarity(embedding1, embedding2, dim=0).item()

    def get_similarity(self, text1, text2):
        if text1 == "" or text2 == "":
            return 0

        embeddings_a = self.get_embeddings(text1)
        embeddings_b = self.get_embeddings(text2)

        agg_embedding_a = self.aggregate_embeddings(embeddings_a)
        agg_embedding_b = self.aggregate_embeddings(embeddings_b)

        similarity = self.cosine_similarity(agg_embedding_a, agg_embedding_b)
        return similarity

class SlidingWindow:
    def __init__(self, model_name='sentence-transformers/all-mpnet-base-v2', chunk_size=512, N=2):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.chunk_size = chunk_size
        self.N = N
        self.stride = (chunk_size - 2) // N

    def get_tokens(self, text):
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        return tokens

    def split_tokens(self, tokens):
        n = len(tokens) // self.chunk_size + 1
        avg_len = len(tokens) // n
        extras = len(tokens) % n
        result = []
        start = 0
        for i in range(n):
            end = start + avg_len + (1 if i < extras else 0)
            result.append(tokens[start:end])
            start = end
        return result

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def get_embeddings(self, text):
        tokens = self.get_tokens(text)
        chunks = self.split_tokens(tokens)
        embeddings = []
        for chunk in chunks:
            window_text = self.tokenizer.decode(chunk)
            encoded_input = self.tokenizer(window_text, padding=True, truncation=True, return_tensors='pt').to(self.device)
            with torch.no_grad():
                model_output = self.model(**encoded_input)
            chunk_embedding = self.mean_pooling(model_output, encoded_input['attention_mask'])
            embeddings.append(chunk_embedding)
        return torch.cat(embeddings, dim=0)

    def cosine_similarity(self, embedding1, embedding2):
        return F.cosine_similarity(embedding1, embedding2, dim=0).item()

    def get_similarity(self, text1, text2, method="mean"):
        if text1 == "" or text2 == "":
            return 0

        embeddings_a = self.get_embeddings(text1)
        embeddings_b = self.get_embeddings(text2)
        
        similarities = []
        for emb1 in embeddings_a:
            for emb2 in embeddings_b:
                sim = self.cosine_similarity(emb1, emb2)
                similarities.append(sim)

        if method == "mean":
            return np.mean(similarities)
        elif method == "max":
            return np.max(similarities)
        else:
            raise ValueError("Unsupported aggregation method: choose 'mean' or 'max'")
