import pickle
from tqdm import tqdm
from pixel import vision_model


if __name__ == '__main__':
    chinese_utils_dir = './utils/chinese_characters'
    chinese_characters_file = f'{chinese_utils_dir}/unicode_characters.txt'
    chinese_chaizi_file = f'{chinese_utils_dir}/chaizi-jt-subset.txt'

    chinese_characters_list = []
    with open(chinese_characters_file, 'r', encoding='utf-8') as f:
        chinese_characters_list = [i.replace('\n','') for i in f.readlines()]

    chinese_chaizi_dict = {}
    with open(chinese_chaizi_file, 'r', encoding='utf-8') as f:
        radicals_data = f.readlines()
        pbar = tqdm(total=len(radicals_data), desc='get chaizi data')
        for char in radicals_data:
            char = char.strip().split('\t')
            char_key = char[0]
            if char_key =='□':
                pbar.update(1)
                continue
            char_radical = [e for c in char[1:] for e in c.split(' ') if e != '□']
            chinese_chaizi_dict[char_key] = char_radical
            pbar.update(1)

    radicals_dict = {}
    pbar = tqdm(total=len(chinese_chaizi_dict.items()), desc='get radicals to char')
    for character, radicals in chinese_chaizi_dict.items():
        radical_list = list(set(radicals))
        for radical in radical_list:
            if radical in radicals_dict:
                radicals_dict[radical].append(character)
            else:
                radicals_dict[radical] = [character]
        pbar.update(1)

    similar_char_dict ={}
    pbar = tqdm(total=len(chinese_characters_list), desc='get similar char')
    for char_zh in chinese_characters_list:
        if char_zh in chinese_chaizi_dict:
            sim_list = []
            for ra_res in chinese_chaizi_dict[char_zh]:
                if ra_res in radicals_dict:
                    for i in radicals_dict[ra_res]:
                        sim_list.append(i)
            similar_char_dict[char_zh] = list(set(sim_list))
        pbar.update(1)

    filter_res = {}
    model = vision_model()
    pbar = tqdm(total=len(similar_char_dict), desc='filter by mse')
    for k_char in similar_char_dict.keys():
        v_list = similar_char_dict[k_char]
        v_dict = {}
        for c_char in v_list:
            if c_char != k_char and c_char in chinese_characters_list:
                score = model.get_mse_similarity(k_char, c_char)
                v_dict[c_char] = score
        v_dict = dict(sorted(v_dict.items(), key=lambda item: item[1], reverse=True))
        top_10 = dict(list(v_dict.items())[:10])
        filter_res[k_char] = top_10
        pbar.update(1)

    with open(f'{chinese_utils_dir}/radicals_similar_char.pkl', 'wb') as file:
        pickle.dump(filter_res, file)
    
