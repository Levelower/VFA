import torch
import numpy as np
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel


class semantics_model(object):
    def __init__(self, model_path, device):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path).to(device)
        self.device = device


    # Calculate sentence embedding from mean-pooling
    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]  
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        numerator = torch.sum(token_embeddings * input_mask_expanded, 1)
        denominator = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return numerator / denominator


    # Calculate the similarity between two sentences based on sentence embedding
    def get_sentence_similarity(self, sentences):
        encoded_input = self.tokenizer(
            sentences, padding=True, truncation=True, return_tensors='pt'
        ).to(self.device)

        with torch.no_grad():
            model_output = self.model(**encoded_input)

        sentence_embeddings = self.mean_pooling(
            model_output, encoded_input['attention_mask']
        )

        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

        cosine_sim = np.dot(
            sentence_embeddings.cpu()[0].numpy().reshape(1, -1),
            sentence_embeddings.cpu()[1].numpy().reshape(1, -1).T
        ).diagonal()
        return cosine_sim.item()


