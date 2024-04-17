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
    parser = argparse.ArgumentParser(description='argparse learning')
    parser.add_argument('--src', type=str, default='zh')
    parser.add_argument('--tgt', type=str, default='en')
    parser.add_argument('--dataset', type=str, default='wmt19')

    parser.add_argument('--device', type=int, default=0)

    parser.add_argument('--merge', type=int, default=3)

    parser.add_argument('--percent', type=str, default='0.2', help='1, 2, 0.x')
    parser.add_argument('--thresh', type=float, default=0.95, help='0.x')
    parser.add_argument('--sc', type=str, default='all', help='all、glyph and radicals')
    parser.add_argument('--search_method', type=str, default='vision', help='vision and semantics')
    parser.add_argument('--vision_constraint', action='store_true', help='add or not')

    args = parser.parse_args()

    if args.src == 'zh':
        language = 'chinese'
    else:
        language = 'japanese'


    data_path = f'./data/{language}/{args.dataset}/{args.dataset}.json'

    data_eval = get_data(data_path, args.src, args.tgt)

    if args.src == 'zh':
        model_src_path = './model/opus-mt-zh-en'
        model_tgt_path = './model/opus-mt-en-zh'
    else:
        model_src_path = './model/opus-mt-ja-en'
        model_tgt_path = './model/opus-tatoeba-en-ja'

    device = torch.device(f'cuda:{str(args.device)}')
    
    attacker = Attacker(
        model_src_path, model_tgt_path, device=device, method=args.merge,
        percent=args.percent, thresh=args.thresh, sc=args.sc,
        search_method=args.search_method,
        vision_constraint=args.vision_constraint
    )

    atk_contents = []
    ori_translations = []
    atk_translations = []
    signals = []

    pbar = tqdm(total=len(data_eval[args.src]), desc='attack')
    for text_src, text_tgt in zip(data_eval[args.src], data_eval[args.tgt]):
        flag, atk_content = attacker.attack(text_src, text_tgt)
        atk_contents.append(atk_content)

        ori_translation = attacker.translate_src(text_src)
        ori_translations.append(ori_translation)

        atk_translation = attacker.translate_src(atk_content)
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

    result_dir = f'./result/{language}/{args.dataset}'
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    data_eval.to_csv(f'{result_dir}/{args.dataset}_merge_{str(args.merge)}_{args.percent}_{str(args.thresh)}_{args.sc}_{args.search_method}_{str(args.vision_constraint)}.csv', sep='\t', index=False)
