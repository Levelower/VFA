import torch
import numpy as np
from torch.nn import Softmax
from transformers import BertTokenizer, BertForMaskedLM


class importance_model(object):
    # Choose different models based on different languages
    # For Chinese, we use hgl/chinese-bert-wwm-ext
    # For others, we use google-bert/bert-base-multilingual-cased
    def __init__(self, model_path, device):
        self.tokenizer = BertTokenizer.from_pretrained(model_path)
        self.model = BertForMaskedLM.from_pretrained(model_path)
        
        self.device = device
        self.model = self.model.to(self.device)

        self.model.eval()
        self.norm = Softmax(dim=1)


    # Further segmentation of each word using the current model
    def subword_split(self, words):
        tokenized_text = []
        index_dict = {}
        word_count = 0
        character_count = 0
        for word in words:
            character_list = self.tokenizer.tokenize(word)
            character_ids = []
            for c in character_list:
                tokenized_text.append(c)
                character_ids.append(character_count)
                character_count += 1

            index_dict[word_count] = character_ids
            word_count += 1

        return tokenized_text, index_dict
    

    # Mask out the characters 
    # Predict the probability of all words appearing at that position
    def mask_and_predict(self, tokenized_text, index_dict, word_idx):
        _tokenized_text = tokenized_text.copy()
        mask_character_ids = index_dict[word_idx]
        for masked_idx in mask_character_ids:
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

        return mask_character_ids, predictions
    

    # Due to the presence of many characters in a word
    # The word probability is obtained from the character probability
    def calc_word_probability(self, tokenized_text, mask_character_ids, predictions):
        word_probability = 1
        for masked_idx in mask_character_ids:
            confidence_scores = predictions[:, masked_idx, :]
            confidence_scores = self.norm(confidence_scores)

            masked_token = tokenized_text[masked_idx]
            masked_token_id = self.tokenizer.convert_tokens_to_ids([masked_token])[0]
            character_probability = confidence_scores[0, masked_token_id].item()

            word_probability = np.sqrt(word_probability * character_probability)

        return word_probability


    # Overall importance score function
    def scores(self, words):
        tokenized_text, index_dict = self.subword_split(words)

        importance = np.zeros(len(words), dtype=float)

        for i in range(len(words)):
            mask_character_ids, predictions = self.mask_and_predict(
                tokenized_text, index_dict, i
            )

            word_probability = self.calc_word_probability(
                tokenized_text, mask_character_ids, predictions
            )

            importance[i] = 1 - word_probability

        return importance


if __name__ == '__main__':
    model_path = './model/chinese-bert-wwm-ext'

    device = torch.device('cuda:1')

    model = importance_model(model_path, device)

    words = ['白天', '天气', '很好',]
    importance = model.scores(words)

    print(importance)