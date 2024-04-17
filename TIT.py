import pickle
from tqdm import tqdm
from pixel import vision_model


if __name__ == '__main__':
    chinese_utils_dir = './utils/chinese_characters'
    vision_char_file = f'{chinese_utils_dir}/vision_similar_char.pkl'
    radicals_char_file = f'{chinese_utils_dir}/radicals_similar_char.pkl'
    
    with open(vision_char_file, 'rb') as file:
        vision_similar_chars = pickle.load(file)

    with open(radicals_char_file, 'rb') as file:
        radicals_similar_chars = pickle.load(file)

    model = vision_model()

    vision_dict = {}
    pbar = tqdm(total=len(vision_similar_chars.keys()), desc='sort vision similar char')
    for char in vision_similar_chars.keys():
        tmp = []
        for s_char in vision_similar_chars[char]:
            tmp.append((s_char[0], model.get_lpips_similarity(s_char[0], char)))
        tmp = sorted(tmp, key=lambda x: x[1], reverse=True)
        vision_dict[char] = tmp
        pbar.update(1)

    with open(f'{chinese_utils_dir}/vision_similar_char_score.pkl', 'wb') as file:
        pickle.dump(vision_dict, file)

    radicals_dict = {}
    pbar = tqdm(total=len(radicals_similar_chars.keys()), desc='sort radicals similar char')
    for char in radicals_similar_chars.keys():
        tmp = []
        for s_char in radicals_similar_chars[char].keys():
            tmp.append((s_char, model.get_lpips_similarity(s_char,char)))
        tmp = sorted(tmp, key=lambda x: x[1], reverse=True)
        radicals_dict[char] = tmp
        pbar.update(1)

    with open(f'{chinese_utils_dir}/radicals_similar_char_score.pkl', 'wb') as file:
        pickle.dump(radicals_dict,file)

