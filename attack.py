import copy
import torch
import pickle
import string
import numpy as np
from transformers import MarianMTModel, MarianTokenizer
from pixel import vision_model
from importance import saliency_model
from semantics import semantics_model 
from evaluate import calc_bleu


class Attacker(object):
    def __init__(
        self, model_src_path, model_tgt_path, device, method,
        percent='0.2', thresh=0.95, sc='all', search_method='vision',
        vision_constraint=True
    ):
        self.tokenizer_src = MarianTokenizer.from_pretrained(model_src_path)
        self.model_src = MarianMTModel.from_pretrained(model_src_path)
        self.tokenizer_tgt = MarianTokenizer.from_pretrained(model_tgt_path)
        self.model_tgt = MarianMTModel.from_pretrained(model_tgt_path)

        self.device = device
        self.model_src = self.model_src.to(self.device)
        self.model_tgt = self.model_tgt.to(self.device)
        
        # Load a list of visually similar words
        vision_file = './utils/chinese_characters/vision_similar_char_score.pkl'
        with open(vision_file, 'rb') as file:
            self.vision_similar_chars = pickle.load(file)

        radicals_file = './utils/chinese_characters/radicals_similar_char_score.pkl'
        with open(radicals_file, 'rb') as file:
            self.radicals_similar_chars = pickle.load(file)

        # Load importance calculation model
        model_saliency_path = './model/chinese-bert-wwm-ext'
        self.saliency_model = saliency_model(model_saliency_path, device=device)

        # Load visual similarity model
        self.vision_model = vision_model()

        # Load semantic similarity calculation model
        model_semantics_path = './model/all-MiniLM-L6-v2'
        self.semantics_model = semantics_model(model_semantics_path, device=device)

        # Semantic search section
        self.token_vocab = self.tokenizer_src.get_vocab()
        self.text_map = {label: text for text, label in self.token_vocab.items()}
        self.token_embeddings = self.model_src.get_input_embeddings().weight.data

        # Merge criteria, the main experiment uses merge3
        self.set_merge_method(method)

        # Hyperparameter
        self.percent = percent
        self.thresh = thresh
        self.sc = sc
        self.search_method = search_method
        self.vision_constraint = vision_constraint

    def set_merge_method(self, method):
        self.attack_method = [
            self.attack1, self.attack2, 
            self.attack3, self.attack4,
            self.attack5, self.attack6,
            self.attack7
        ][method-1]

    # Translate src to tgt
    def translate_src(self, text, length=None):
        input_ids = self.tokenizer_src.encode(text, return_tensors="pt").to(self.device)
        if length is None:
            output = self.model_src.generate(input_ids)
        else:
            output = self.model_src.generate(input_ids, max_length=length, min_length=length)
            
        output_text = self.tokenizer_src.decode(output[0], skip_special_tokens=True)
        return output_text

    # Translate tgt to src
    def translate_tgt(self, text, length=None):
        input_ids = self.tokenizer_tgt.encode(text, return_tensors="pt").to(self.device)
        if length is None:
            output = self.model_tgt.generate(input_ids)
        else:
            output = self.model_tgt.generate(input_ids, max_length=length, min_length=length)
        output_text = self.tokenizer_tgt.decode(output[0], skip_special_tokens=True)
        return output_text
    
    # Calculate the importance of each token in a sentence
    def get_token_importance(self, words):
        importance = self.saliency_model.scores(words)
        return importance
    
    # Find visually similar tokens for a certain token
    def get_token_vision_similarity(self, text):
        candidate_char_list = []
        candidate_score_list = []
        for idx, char in enumerate(text):
            r_tmp = []
            s_tmp = []

            if self.sc == 'glyph':
                if char in self.vision_similar_chars:
                    for first in self.vision_similar_chars[char]:
                        r_tmp.append(first[0])
                        s_tmp.append(first[1])

            elif self.sc == 'radicals':
                if char in self.radicals_similar_chars.keys():
                    for first in self.radicals_similar_chars[char]:
                        r_tmp.append(first[0])
                        s_tmp.append(first[1])
            else:
                if char in self.radicals_similar_chars.keys():
                    for first in self.radicals_similar_chars[char]:
                        r_tmp.append(first[0])
                        s_tmp.append(first[1])
                
                elif char in self.vision_similar_chars:
                    for first in self.vision_similar_chars[char]:
                        r_tmp.append(first[0])
                        s_tmp.append(first[1])

            if len(r_tmp) > 0:     
                candidate_char_list = candidate_char_list + [text[: idx] + c + text[idx+1:] for c in r_tmp]
                candidate_score_list = candidate_score_list + s_tmp

        return candidate_char_list, candidate_score_list
    
    # Find semantically similar tokens for a certain token, 
    # that is, the token with the highest cosine similarity
    def get_most_similar_topN(self, text, topn=10, thresh=0.9):
        input_token = text

        input_vector_matrix = torch.unsqueeze(
            self.token_embeddings[self.token_vocab[input_token]], 0
        ).repeat(len(self.token_embeddings), 1)
        
        cos_sim_list = torch.cosine_similarity(
            input_vector_matrix, self.token_embeddings, 1
        ).tolist()

        cos_sim_list = [(self.text_map[i], cos_sim_list[i]) for i in range(len(cos_sim_list))]
        cos_sim_list.sort(key=lambda x: x[1], reverse=True)

        cnt = 0
        topn_tokens = []
        for i in range(1, len(cos_sim_list)):
            if cos_sim_list[i][1] > thresh:
                topn_tokens.append(cos_sim_list[i][0])
                cnt += 1
            if cnt >= topn:
                break

        return topn_tokens
    
    # Calculate the overall visual similarity between two sentences
    def get_sentence_vision_similarity(self, text1, text2):
        score_overall = self.vision_model.get_lpips_similarity(text1, text2)
        return score_overall

    # Calculate the overall semantic similarity between two sentences
    def get_sentence_semantic_similarity(self, text1, text2):
        cosine_sim = self.semantics_model.get_sentence_similarity([text1, text2])
        return cosine_sim

    # Visual replacement generates adversarial samples
    def search_samples_by_vision(self, text):    
        # The token result does not contain an ending identifier, 
        # while the tokens result has an ending identifier. Remove it
        tokens = self.tokenizer_src(text, return_tensors='pt').input_ids.numpy()[0][:-1]
        attack_res = [i.replace('▁','') for i in self.tokenizer_src.tokenize(text)]

        # Calculate token importance
        importance = self.get_token_importance(attack_res)

        # Sort by importance from highest to lowest
        token_ids = np.arange(0, tokens.shape[0])
        tokens_order = sorted(zip(importance, token_ids), reverse=True)
        tokens_order = [pair[1] for pair in tokens_order]

        # Calculate the maximum number of replacements proportionally based on the length of the text
        if self.percent == '1':
            num_changed = 1
        elif self.percent == '2':
            num_changed = 2
        else:
            num_changed = round(len(text) * float(self.percent))

        # Replacement Count
        cnt = 0 

        # Replace in order of importance. If the similarity after replacement is lower than the threshold, 
        # abandon the replacement and stop after reaching the maximum number of replacements
        for token_idx in tokens_order:
            token = attack_res[token_idx]

            candidates, scores = self.get_token_vision_similarity(token)
            
            # Obtain the adversarial sample with the smallest perceived change after replacement
            res = []
            if len(candidates) > 0:
                sim_thresh = 0
                for candidate, score in zip(candidates, scores):
                    atk = copy.deepcopy(attack_res)
                    atk[token_idx] = candidate
                    atk_text = ''.join(atk)

                    # The overall similarity of the sentence, 
                    # the local similarity between the substitute word and the original word, 
                    # constitute the similarity score
                    sim_score = (self.get_sentence_vision_similarity(text, atk_text) + score) / 2
                    
                    res.append((candidate, sim_score))
                    sim_thresh = max(sim_thresh, sim_score)

                # Obtain the highest overall similarity
                max_token = max(res, key=lambda x: x[1])[0] 

                # The overall visual similarity should not be lower than the threshold, 
                # otherwise it will not be replaced
                if self.vision_constraint:
                    if sim_thresh > self.thresh:
                        attack_res[token_idx] = max_token
                        cnt = cnt + 1
                else:
                    attack_res[token_idx] = max_token
                    cnt = cnt + 1

                # Stop when the replacement ratio is reached
                if cnt >= num_changed:
                    break

        return (''.join(attack_res))
    
    def search_samples_by_semantics(self, text):
        # The token result does not contain an ending identifier, 
        # while the tokens result has an ending identifier. Remove it
        tokens = self.tokenizer_src(text, return_tensors='pt').input_ids.numpy()[0][:-1]
        attack_res = [i.replace('▁','') for i in self.tokenizer_src.tokenize(text)]

        # Calculate token importance
        importance = self.get_token_importance(attack_res)

        # Sort by importance from highest to lowest
        token_ids = np.arange(0, tokens.shape[0])
        tokens_order = sorted(zip(importance, token_ids), reverse=True)
        tokens_order = [pair[1] for pair in tokens_order]

        # Calculate the maximum number of replacements proportionally based on the length of the text
        if self.percent == '1':
            num_changed = 1
        elif self.percent == '2':
            num_changed = 2
        else:
            num_changed = round(len(text) * float(self.percent))

        # Replacement Count
        cnt = 0 

        # Replace based on semantic similarity
        res_attack = [self.text_map.get(t, "") for t in tokens]
        for idx, i in enumerate(tokens_order):
            # Do not replace punctuation marks
            context = self.text_map.get(tokens[i], "")
            if context == "" or context in string.punctuation:
                res_attack[i] = context
                continue

            # Calculate the top N of semantic similarity based on cosine similarity
            candidate = self.get_most_similar_topN(context, 20, 0.8)
            if len(candidate) <= 0:
                res_attack[i] = context
                continue

            # Replace with a candidate word that reduces the overall semantic similarity of the sentence the most
            res = []
            for candi in candidate:
                atk_tmp = [self.token_vocab[j] for j in res_attack]
                atk_tmp[idx] = self.token_vocab.get(candi)
                atk_text = ''.join([self.text_map.get(t, '') for t in atk_tmp])
                atk_text = atk_text.replace('▁', ' ').replace('</s>', '')
                semantics_score = self.semantics_model.get_sentence_similarity([text, atk_text])

                # Global similarity constraint
                lpips_score = self.get_sentence_vision_similarity(text, atk_text)
                sim_score = (lpips_score + self.get_sentence_vision_similarity(res_attack[i], candi)) / 2

                res.append((candi, semantics_score, sim_score))
            
            max_token_info = min(res, key=lambda x: x[1])

            # The overall visual similarity should not be lower than the threshold, otherwise it will not be replaced
            if self.vision_constraint:
                if max_token_info[2] > self.thresh:
                    res_attack[i] = max_token_info[0]
                    cnt += 1
            else:
                res_attack[i] = max_token_info[0]
                cnt += 1

            # Stop when the replacement ratio is reached
            if cnt >= num_changed:
                break

        return (''.join(res_attack)).replace('▁', '').replace('</s>', '')
    
    # Attacks using only visual substitution methods
    def attack_by_vision(self, text):
        attack_res_vision = self.search_samples_by_vision(text)
        return attack_res_vision

    # Attack using a combination of visual and semantic methods
    def attack_by_semantics(self, text_src, text_tgt):
        # Whether to use semantics as the sample to be replaced
        semantics_flag = False

        # Set the original sample as the sample to be replaced
        source = text_src

        # Translate the reference translation back to obtain semantically similar samples to be replaced
        text_tgt_translation = self.translate_tgt(text_tgt)

        # Review whether semantically similar samples to be replaced meet the semantic similarity condition
        semantic_similarity = self.get_sentence_semantic_similarity(text_tgt_translation, text_src)

        if text_tgt_translation != text_src and semantic_similarity > 0.9:
            source = text_tgt_translation
            semantics_flag = True
        else:
            return False, None

        # Visual replacement using the sample to be replaced to obtain the final adversarial sample
        attack_res_semantics = self.search_samples_by_vision(source)

        return semantics_flag, attack_res_semantics
    
    # Choose the most aggressive one from the results of the above two methods
    def attack1(self, text_src, text_tgt):
        vision_result = self.attack_by_vision(text_src)
        flag, semantics_result = self.attack_by_semantics(text_src, text_tgt)

        if not flag:
            return flag, vision_result
        
        vision_translation = self.translate_src(vision_result)
        semantics_translation = self.translate_src(semantics_result)

        vision_bleu = calc_bleu(vision_translation, text_tgt)
        semantics_bleu = calc_bleu(semantics_translation, text_tgt)

        if vision_bleu < semantics_bleu:
            adversarial_example = vision_result
            flag = False
        else:
            adversarial_example = semantics_result

        return flag, adversarial_example
    
    # Select the translation result with the lowest semantic similarity to the original translation from two adversarial samples
    def attack2(self, text_src, text_tgt):
        vision_result = self.attack_by_vision(text_src)
        flag, semantics_result = self.attack_by_semantics(text_src, text_tgt)

        if not flag:
            return flag, vision_result

        vision_translation = self.translate_src(vision_result)
        semantics_translation = self.translate_src(semantics_result)

        origin_translation = self.translate_src(text_src)

        vision_similar = self.semantics_model.get_sentence_similarity([
            origin_translation, 
            vision_translation
        ])
        semantics_similar = self.semantics_model.get_sentence_similarity([
            origin_translation, 
            semantics_translation
        ])

        if vision_similar < semantics_similar:
            adversarial_example = vision_result
            flag = False
        else:
            adversarial_example = semantics_result

        return flag, adversarial_example
    
    # Select the translation result with the lowest semantic similarity to the reference translation from two adversarial samples
    def attack3(self, text_src, text_tgt):
        vision_result = self.attack_by_vision(text_src)
        flag, semantics_result = self.attack_by_semantics(text_src, text_tgt)

        if not flag:
            return flag, vision_result

        vision_translation = self.translate_src(vision_result)
        semantics_translation = self.translate_src(semantics_result)

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
    
    # Select one of the two adversarial samples with the highest semantic similarity to the original sample
    def attack4(self, text_src, text_tgt):
        vision_result = self.attack_by_vision(text_src)
        flag, semantics_result = self.attack_by_semantics(text_src, text_tgt)

        if not flag:
            return flag, vision_result

        vision_similar = self.semantics_model.get_sentence_similarity([
            text_src, 
            vision_result
        ])
        semantics_similar = self.semantics_model.get_sentence_similarity([
            text_src, 
            semantics_result
        ])

        if vision_similar > semantics_similar:
            adversarial_example = vision_result
            flag = False
        else:
            adversarial_example = semantics_result

        return flag, adversarial_example

    # Using only visual attacks
    def attack5(self, text_src, text_tgt):
        adversarial_example = self.attack_by_vision(text_src)
        return False, adversarial_example
    
    # Use only semantic attacks, but with constraints, replace with visual ones that do not meet the constraints
    def attack6(self, text_src, text_tgt):
        # Whether to use semantics as the sample to be replaced
        semantics_flag = False

        # Set the original sample as the sample to be replaced
        source = text_src

        # Translate the reference translation back to obtain semantically similar samples to be replaced
        text_tgt_translation = self.translate_tgt(text_tgt)

        # Review whether semantically similar samples to be replaced meet the semantic similarity condition
        semantic_similarity = self.get_sentence_semantic_similarity(text_tgt_translation, text_src)

        if text_tgt_translation != text_src and semantic_similarity > 0.9:
            source = text_tgt_translation
            semantics_flag = True

        # Visual replacement using the sample to be replaced to obtain the final adversarial sample
        attack_res_semantics = self.search_samples_by_vision(source)

        return semantics_flag, attack_res_semantics
    
    # Only semantic attacks, but unconstrained
    def attack7(self, text_src, text_tgt):
        # Translate the reference translation back to obtain semantically similar samples to be replaced
        text_tgt_translation = self.translate_tgt(text_tgt)

        source = text_tgt_translation
        semantics_flag = True

        # Visual replacement using the sample to be replaced to obtain the final adversarial sample
        attack_res_semantics = self.search_samples_by_vision(source)

        return semantics_flag, attack_res_semantics
    
    # Select the translation result with the lowest semantic similarity to the reference translation from two adversarial samples
    def attack_by_vision_search(self, text_src, text_tgt):
        flag, adversarial_example = self.attack_method(text_src, text_tgt)
        return flag, adversarial_example
    
    # Attack using semantic search
    def attack_by_semantics_search(self, text):
        attack_res_semantics = self.search_samples_by_semantics(text)
        return None, attack_res_semantics
    
    def attack(self, text_src, text_tgt):
        if self.search_method == 'vision':
            flag, adversarial_example = self.attack_by_vision_search(text_src, text_tgt)
        else:
            flag, adversarial_example = self.attack_by_semantics_search(text_src)

        return flag, adversarial_example


if __name__ == '__main__':
    model_src_path = './model/opus-mt-zh-en'
    model_tgt_path = './model/opus-mt-en-zh'

    percent='0.2'
    thresh=0.95
    sc='all'
    search_method='semantics'
    vision_constraint=False

    device = torch.device('cuda:0')

    attacker = Attacker(
        model_src_path, model_tgt_path, device, method=1,
        percent=percent, thresh=thresh, sc=sc,
        search_method=search_method,
        vision_constraint=vision_constraint
    )

    text_src = '他们从未那样做过。'
    text_tgt = 'They really don\'t do that.'

    flag, attack_res = attacker.attack(text_src, text_tgt)
    print(flag, attack_res)

    translation_res = attacker.translate_src('他们从未这样做。')

    print(translation_res)