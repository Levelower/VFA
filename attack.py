import torch
import pickle
import numpy as np
from transformers import MarianMTModel, MarianTokenizer
from pixel import pixel_model
from importance import importance_model
from semantics import semantics_model


class Attacker(object):
    def __init__(
        self, src, tgt, model_tgt_path, model_aux_path, 
        importance_model_path, semantics_model_path, device,
        r='0.2', theta=0.95, S='all', beta=0.9, vision_constraint=True
    ):
        self.src = src
        self.tgt = tgt
        
        self.tokenizer_tgt = MarianTokenizer.from_pretrained(model_tgt_path)
        self.model_tgt = MarianMTModel.from_pretrained(model_tgt_path)
        self.tokenizer_aux = MarianTokenizer.from_pretrained(model_aux_path)
        self.model_aux = MarianMTModel.from_pretrained(model_aux_path)

        self.device = device
        self.model_tgt = self.model_tgt.to(self.device)
        self.model_aux = self.model_aux.to(self.device)
        
        # Load S_pix and S_rad
        pixel_file = './utils/characters/pixel_level_similar_characters_with_local_constraint.pkl'
        with open(pixel_file, 'rb') as file:
            self.S_pix = pickle.load(file)

        radicals_file = './utils/characters/radical_level_similar_characters_with_local_constraint.pkl'
        with open(radicals_file, 'rb') as file:
            self.S_rad = pickle.load(file)

        # Load importance model
        # Mainly used to calculate the importance of each word in a sentence
        self.importance_model = importance_model(importance_model_path, device=device)

        # Load semantic similarity calculation model
        # Mainly used to calculate the semantic similarity between two sentences
        self.semantics_model = semantics_model(semantics_model_path, device=device)

        # Load a visual similarity calculation model
        # Mainly using LPIPS to construct perceptual constraints
        self.pixel_model = pixel_model()

        # Semantic search section
        self.token_vocab = self.tokenizer_tgt.get_vocab()
        self.text_map = {label: text for text, label in self.token_vocab.items()}
        self.token_embeddings = self.model_tgt.get_input_embeddings().weight.data

        # Hyperparameter
        self.r = r
        self.theta = theta
        self.S = S
        self.vision_constraint = vision_constraint
        self.beta = beta


    # Translate source sentence to target sentence
    def translate_to_tgt(self, text, length=None):
        input_ids = self.tokenizer_tgt.encode(text, return_tensors="pt").to(self.device)
        if length is None:
            output = self.model_tgt.generate(input_ids)
        else:
            output = self.model_tgt.generate(input_ids, max_length=length, min_length=length)
            
        output_text = self.tokenizer_tgt.decode(output[0], skip_special_tokens=True)
        return output_text


    # Translate target sentence to source sentence
    def translate_to_src(self, text, length=None):
        input_ids = self.tokenizer_aux.encode(text, return_tensors="pt").to(self.device)
        if length is None:
            output = self.model_aux.generate(input_ids)
        else:
            output = self.model_aux.generate(input_ids, max_length=length, min_length=length)
        output_text = self.tokenizer_aux.decode(output[0], skip_special_tokens=True)
        return output_text
    

    # Calculate the importance of each word in a sentence
    def get_word_importance(self, words):
        importance = self.importance_model.scores(words)
        return importance
    

    # Find visually similar words for a certain word
    def get_visually_similar_characters(self, text):
        candidate_character_list = []
        candidate_score_list = []
        for idx, character in enumerate(text):
            if character != '▁':
                r_tmp = []
                s_tmp = []

                if self.S == 'pix':
                    if character in self.S_pix:
                        r_tmp = [i[0] for i in self.S_pix[character]]
                        s_tmp = [i[1][1] for i in self.S_pix[character]]

                elif self.S == 'rad':
                    if character in self.S_rad.keys():
                        r_tmp = [i[0] for i in self.S_rad[character]]
                        s_tmp = [i[1][1] for i in self.S_rad[character]]
                else:
                    if character in self.S_rad.keys():
                        r_tmp += [i[0] for i in self.S_rad[character]]
                        s_tmp += [i[1][1] for i in self.S_rad[character]]

                    elif character in self.S_pix.keys():
                        r_tmp = [i[0] for i in self.S_pix[character]]
                        s_tmp = [i[1][1] for i in self.S_pix[character]]
                         
                if len(r_tmp) > 0:     
                    candidate_character_list.extend([text[: idx] + c + text[idx+1:] for c in r_tmp])
                    candidate_score_list = candidate_score_list + s_tmp

        return candidate_character_list, candidate_score_list
    

    # Calculate the overall visual similarity between two sentences
    def get_sentence_visual_similarity(self, text1, text2):
        similarity_lpips = self.pixel_model.get_lpips_similarity(text1, text2)

        return similarity_lpips


    # Calculate the overall semantic similarity between two sentences
    def get_sentence_semantic_similarity(self, text1, text2):
        cosine_sim = self.semantics_model.get_sentence_similarity([text1, text2])
        return cosine_sim


    # Using visual similarity replacement to obtain adversarial examples
    def search_examples_by_vision(self, text):
        # Remove the beginning, end, and excess whitespace to ensure alignment
        text = ' '.join(text.strip().split())

        # The token result does not contain an ending identifier, 
        # while the tokens result has an ending identifier. Remove it
        tokens = self.tokenizer_tgt(text, return_tensors='pt').input_ids.numpy()[0][:-1]
        attack_res = self.tokenizer_tgt.tokenize(text)

        # Calculate token importance
        importance = self.get_word_importance([i.replace('▁', '') for i in attack_res])

        # Sort by importance from highest to lowest
        token_ids = np.arange(0, tokens.shape[0])
        tokens_order = sorted(zip(importance, token_ids), reverse=True)
        tokens_order = [pair[1] for pair in tokens_order]

        # Calculate the maximum number of replacements proportionally based on the length of the text
        num_changed = round(len(text) * self.r)

        # Replacement Count
        cnt = 0 

        # Replace in order of importance. If the similarity after replacement is lower than the threshold, 
        # abandon the replacement and stop after reaching the maximum number of replacements
        for token_idx in tokens_order:
            token = attack_res[token_idx]

            candidates, scores = self.get_visually_similar_characters(token)
            
            # Obtain the adversarial sample with the smallest perceived change after replacement
            res = []
            if len(candidates) > 0:
                sim_thresh = 0
                for candidate, score in zip(candidates, scores):
                    atk_text = ''.join(
                        attack_res[:token_idx] + [candidate] + attack_res[token_idx+1:]
                    ).replace('▁', ' ').strip()

                    # The overall similarity of the sentence, 
                    # the local similarity between the substitute word and the original word, 
                    # constitute the similarity score
                    sim_score = (self.get_sentence_visual_similarity(text, atk_text) + score) / 2
                    
                    res.append((candidate, sim_score))
                    sim_thresh = max(sim_thresh, sim_score)

                # Obtain the highest overall similarity
                max_token = max(res, key=lambda x: x[1])[0] 

                # The overall visual similarity should not be lower than the threshold, 
                # otherwise it will not be replaced
                if self.vision_constraint:
                    if sim_thresh > self.theta:
                        attack_res[token_idx] = max_token
                        cnt = cnt + 1
                else:
                    attack_res[token_idx] = max_token
                    cnt = cnt + 1

                # Stop when the replacement ratio is reached
                if cnt >= num_changed:
                    break

        return (''.join(attack_res).replace('▁', ' ').strip())
    

    # Attacks using only visual replacement methods
    def attack_by_vision_only(self, text):
        attack_res_vision = self.search_examples_by_vision(text)
        return attack_res_vision


    # Attack using a combination of visual replacement methods and reverse translation
    def attack_by_vision_with_reverse(self, text_src, text_tgt):
        # Whether to use semantics as the sample to be replaced
        semantics_flag = False

        # Set the original sample as the sample to be replaced
        source = text_src

        # Translate the reference translation back to obtain semantically similar samples to be replaced
        text_tgt_translation = self.translate_to_src(text_tgt)

        # Review whether semantically similar samples to be replaced meet the semantic similarity condition
        semantic_similarity = self.get_sentence_semantic_similarity(text_tgt_translation, text_src)

        if text_tgt_translation != text_src and semantic_similarity > self.beta:
            source = text_tgt_translation
            semantics_flag = True
        else:
            return False, None

        # Visual replacement using the sample to be replaced to obtain the final adversarial sample
        attack_res_semantics = self.search_examples_by_vision(source)

        return semantics_flag, attack_res_semantics

    
    # Our whole attack method using visual search and reverse translation
    # Select the translation result with the lowest semantic similarity to the reference translation
    def attack_by_vision_search(self, text_src, text_tgt):
        vision_result = self.attack_by_vision_only(text_src)
        flag, semantics_result = self.attack_by_vision_with_reverse(text_src, text_tgt)

        if not flag:
            return flag, vision_result

        vision_translation = self.translate_to_tgt(vision_result)
        semantics_translation = self.translate_to_tgt(semantics_result)

        vision_similar = self.semantics_model.get_sentence_similarity([
            text_tgt, 
            vision_translation
        ])
        semantics_similar = self.semantics_model.get_sentence_similarity([
            text_tgt, 
            semantics_translation
        ])

        if vision_similar < semantics_similar:
            adversarial_example = vision_result
            flag = False
        else:
            adversarial_example = semantics_result

        return flag, adversarial_example
    
    
    # Attack Function
    def attack(self, text_src, text_tgt):
        flag, adversarial_example = self.attack_by_vision_search(text_src, text_tgt)

        return flag, adversarial_example


if __name__ == '__main__':
    src = 'zh'
    tgt = 'en'

    model_tgt_path = './model/opus-mt-zh-en'
    model_aux_path = './model/opus-mt-en-zh'

    importance_model_path='./model/chinese-bert-wwm-ext'
    semantics_model_path='./model/all-MiniLM-L6-v2'

    r=0.2
    theta=0.95
    S='all'
    beta=0.9
    vision_constraint=True

    device = torch.device('cuda:0')

    attacker = Attacker(
        src, tgt,
        model_tgt_path, model_aux_path, 
        importance_model_path=importance_model_path,
        semantics_model_path=semantics_model_path,
        device=device, r=r, theta=theta, S=S, beta=beta,
        vision_constraint=vision_constraint
    )

    text_src = '他们从未那样做过。'
    text_tgt = 'They really don\'t do that.'

    flag, attack_res = attacker.attack(text_src, text_tgt)
    print(flag, attack_res)

    translation_res = attacker.translate_to_tgt(attack_res)
    print(translation_res)

    translation_res = attacker.translate_to_tgt('他们从未这样做。')
    print(translation_res)