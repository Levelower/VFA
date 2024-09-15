import pickle
from tqdm import tqdm
from pixel import pixel_model
from utils.characters.glyph_zh import glyph_similar


# Calculate visual similarity based on character shapes
def get_glyph_similarity(text1, text2):
    similarity_glyph = glyph_similar(text1 + ' ' + text2, text2 + ' ' + text1)

    return similarity_glyph


if __name__ == '__main__':
    # Utils
    utils_dir = './utils/characters'
    characters_file = f'{utils_dir}/unicode_characters.txt'
    characters_to_radicals_file = f'{utils_dir}/chaizi-jt-subset.txt'

    # Read meaningful Unicode characters
    characters_list = []
    with open(characters_file, 'r', encoding='utf-8') as f:
        characters_list = [i.replace('\n','') for i in f.readlines()]

    # Get the correspondence between characters and radicals
    characters_to_radicals = {}
    with open(characters_to_radicals_file, 'r', encoding='utf-8') as f:
        data = f.readlines()
        pbar = tqdm(total=len(data), desc='characters to radicals')
        for line in data:
            items = line.strip().split('\t')
            character = items[0]
            if character =='□':
                pbar.update(1)
                continue
            radicals = [e for c in items[1:] for e in c.split(' ') if e != '□']
            characters_to_radicals[character] = radicals
            pbar.update(1)

    # Get all characters corresponding to radicals
    radicals_dict = {}
    pbar = tqdm(total=len(characters_to_radicals.items()), desc='radicals to characters')
    for character, radicals in characters_to_radicals.items():
        radical_list = list(set(radicals))
        for radical in radical_list:
            if radical in radicals_dict:
                radicals_dict[radical].append(character)
            else:
                radicals_dict[radical] = [character]
        pbar.update(1)

    # Get similar characters
    similar_character_dict ={}
    pbar = tqdm(total=len(characters_list), desc='get similar characters')
    for character in characters_list:
        if character in characters_to_radicals:
            similar_list = []
            for radical in characters_to_radicals[character]:
                if radical in radicals_dict:
                    for similar_character in radicals_dict[radical]:
                        similar_list.append(similar_character)
            similar_character_dict[character] = list(set(similar_list))
        pbar.update(1)

    # Filter the characters using mse
    filter_res = {}
    model = pixel_model()
    pbar = tqdm(total=len(similar_character_dict), desc='filter by mse')
    for character in similar_character_dict.keys():
        tmp_list = similar_character_dict[character]
        tmp_dict = {}
        for tmp_character in tmp_list:
            if tmp_character != character and tmp_character in characters_list:
                score = model.get_mse_similarity(character, tmp_character)
                tmp_dict[tmp_character] = score
        tmp_dict = dict(sorted(tmp_dict.items(), key=lambda item: item[1], reverse=True))
        top_similar = dict(list(tmp_dict.items())[:50])
        filter_res[character] = top_similar
        pbar.update(1)

    with open(f'{utils_dir}/radical_level_similar_characters.pkl', 'wb') as file:
        pickle.dump(filter_res, file)
    
