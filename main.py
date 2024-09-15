import os
import json
import torch
import argparse
import pandas as pd
from tqdm import tqdm
from attack import Attacker


def get_data(in_path, src, tgt):
    with open(in_path, 'r', encoding='utf-8') as json_file:
        json_data = json.load(json_file)

    translation_data = json_data.get("translation", [])

    tgt_list = [item.get(tgt, '') for item in translation_data]
    src_list = [item.get(src, '') for item in translation_data]

    df = pd.DataFrame({tgt: tgt_list, src: src_list})

    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='VFA')
    parser.add_argument('--src', type=str, default='zh')
    parser.add_argument('--tgt', type=str, default='en')
    parser.add_argument('--dataset', type=str, default='wmt19')

    parser.add_argument('--device', type=int, default=0)

    parser.add_argument('--r', type=float, default=0.2, help='0.x')
    parser.add_argument('--theta', type=float, default=0.95, help='0.x')
    parser.add_argument('--S', type=str, default='all', help='all、pix or rad')
    parser.add_argument('--beta', type=float, default=0.9, help='0.x')
    parser.add_argument('--vision_constraint', action='store_true', help='add or not')

    args = parser.parse_args()
    if args.src == 'zh':
        source_language = 'Chinese'
    else:
        source_language = 'Japanese'

    data_path = f'./data/{source_language}/{args.dataset}.json'

    data_eval = get_data(data_path, args.src, args.tgt)

    # Translation models for various languages
    if args.src == 'zh':
        model_tgt_path = './model/opus-mt-zh-en'
        model_aux_path = './model/opus-mt-en-zh'
    else:
        model_tgt_path = './model/opus-mt-ja-en'
        model_aux_path = './model/opus-tatoeba-en-ja'

    # Selected a specific importance ranking model for Chinese 
    # And a multilingual adaptation model for other languages
    if args.src == 'zh':
        importance_model_path='./model/chinese-bert-wwm-ext'
    else:
        importance_model_path='./model/bert-base-multilingual-cased'

    # Chinese can use radical similarity modules
    # while other languages only use pixel similarity modules
    if args.src == 'zh':
        S=args.S
    else:
        S='pix'

    # Only used for testing English translation results
    semantics_model_path='./model/all-MiniLM-L6-v2'

    device = torch.device(f'cuda:{str(args.device)}')

    attacker = Attacker(
        args.src, args.tgt,
        model_tgt_path, model_aux_path, 
        importance_model_path=importance_model_path,
        semantics_model_path=semantics_model_path,
        device=device, r=args.r, theta=args.theta, S=S, 
        beta=args.beta, vision_constraint=args.vision_constraint
    )

    atk_contents = []
    ori_translations = []
    atk_translations = []
    signals = []

    pbar = tqdm(total=len(data_eval[args.src]), desc='attack')
    for text_src, text_tgt in zip(data_eval[args.src], data_eval[args.tgt]):
        flag, atk_content = attacker.attack(text_src, text_tgt)
        atk_contents.append(atk_content)

        ori_translation = attacker.translate_to_tgt(text_src)
        ori_translations.append(ori_translation)

        atk_translation = attacker.translate_to_tgt(atk_content)
        atk_translations.append(atk_translation)

        if flag is None:
            signal = 2
        elif flag:
            signal = 1
        else:
            signal = 0

        signals.append(signal)

        pbar.update(1)

    data_eval['atk_content'] = atk_contents
    data_eval['ori_translate'] = ori_translations
    data_eval['atk_translate'] = atk_translations
    data_eval['signals'] = signals

    result_dir = f'./result/{source_language}'
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    result_parameters = [
        args.src, args.tgt,
        args.dataset, str(args.r), str(args.theta),
        S, str(args.beta), str(args.vision_constraint)    
    ]
    result_name = '-'.join(result_parameters)
    data_eval.to_csv(f'{result_dir}/{result_name}.csv', sep='\t', index=False)
