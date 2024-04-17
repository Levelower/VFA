import torch
import numpy as np
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel


class semantics_model(object):
    def __init__(self, model_path, device):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path).to(device)
        self.device = device

    def mean_pooling(self, model_output, attention_mask):
        # First element of model_output contains all token embeddings
        token_embeddings = model_output[0]  
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def get_sentence_similarity(self, sents):
        encoded_input = self.tokenizer(sents, padding=True, truncation=True, return_tensors='pt').to(self.device)

        # Compute token embeddings
        with torch.no_grad():
            model_output = self.model(**encoded_input)

        # Perform pooling
        sentence_embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'])

        # Normalize embeddings
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

        cosine_sim = np.dot(sentence_embeddings.cpu()[0].numpy().reshape(1, -1),
                            sentence_embeddings.cpu()[1].numpy().reshape(1, -1).T).diagonal()
        return cosine_sim.item()


