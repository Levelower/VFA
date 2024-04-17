import torch
import numpy as np
from torch.nn import Softmax
from transformers import BertTokenizer, BertForMaskedLM


class saliency_model:
    def __init__(self, model_path, device):
        self.tokenizer = BertTokenizer.from_pretrained(model_path)
        self.model = BertForMaskedLM.from_pretrained(model_path)
        
        self.device = device
        self.model = self.model.to(self.device)

        self.model.eval()
        self.norm = Softmax(dim=1)

    def subword_split(self, words):
        tokenized_text = []
        indexDic = {}
        word_count = 0
        char_count = 0
        for word in words:
            charsList = self.tokenizer.tokenize(word)
            charIdxes = []
            for c in charsList:
                tokenized_text.append(c)
                charIdxes.append(char_count)
                char_count += 1

            indexDic[word_count] = charIdxes
            word_count += 1

        return tokenized_text, indexDic
    
    def mask_and_predict(self, tokenized_text, indexDic, word_idx):
        _tokenized_text = tokenized_text.copy()
        maskCharIdxes = indexDic[word_idx]
        for masked_idx in maskCharIdxes:
            _tokenized_text[masked_idx] = '[MASK]'

        indexed_tokens = self.tokenizer.convert_tokens_to_ids(_tokenized_text)
        segments_ids = [0 for i in range(len(_tokenized_text))]

        tokens_tensor = torch.tensor([indexed_tokens])
        segments_tensors = torch.tensor([segments_ids])

        tokens_tensor = tokens_tensor.to(self.device)
        segments_tensors = segments_tensors.to(self.device)

        with torch.no_grad():
            outputs = self.model(tokens_tensor, token_type_ids=segments_tensors)
            predictions = outputs[0]

        return maskCharIdxes, predictions
    
    def calc_wordProb(self, tokenized_text, maskCharIdxes, predictions):
        wordProb = 1
        for masked_idx in maskCharIdxes:
            confidence_scores = predictions[:, masked_idx, :]
            confidence_scores = self.norm(confidence_scores)

            masked_token = tokenized_text[masked_idx]
            masked_token_id = self.tokenizer.convert_tokens_to_ids([masked_token])[0]
            orig_prob = confidence_scores[0, masked_token_id].item()

            wordProb = wordProb*orig_prob

        return wordProb

    def scores(self, words):
        tokenized_text, indexDic = self.subword_split(words)

        saliency = np.zeros(len(words), dtype=float)

        for i in range(len(words)):
            maskCharIdxes, predictions = self.mask_and_predict(tokenized_text, indexDic, i)

            wordProb = self.calc_wordProb(tokenized_text, maskCharIdxes, predictions)

            saliency[i] = 1 - wordProb

        return saliency

