import faiss
import pickle
import torch
import lpips
import numpy as np
from tqdm import tqdm
from PIL import Image, ImageFont, ImageDraw
from utils.chinese_stoke.glyph_zh import glyph_similar
from skimage.metrics import structural_similarity as ssim


class vision_model(object):
    def __init__(self):
        self.loss_fn = lpips.LPIPS(net='alex')
        self.gpu = False


    def read_characters_from_file(self, filename):
        characters = []
        with open(filename, 'r', encoding='utf-8') as file:
            for line in file:
                line = line.strip()
                characters.extend(line)
        return characters


    def simVec(self, text):
        img = Image.new("RGB", (40, 40), (255, 255, 255))
        font = ImageFont.truetype('./utils/chinese_fonts/simsun.ttc', 18)
        draw = ImageDraw.Draw(img)
        draw.text((10, 5), text, font=font, fill="#000000")
        arr = np.array(img.convert('L'))
        return arr


    def get_mse_similarity(self, text1, text2):
        arr1 = self.simVec(text1)
        arr2 = self.simVec(text2)
        
        similarity_mse = -np.mean((arr1 - arr2) ** 2)
        
        return similarity_mse


    def get_ssim_similarity(self, text1, text2):
        arr1 = self.simVec(text1)
        arr2 = self.simVec(text2)
        
        similarity_ssim = ssim(arr1, arr2, multichannel=True)
        
        return similarity_ssim


    def get_lpips_similarity(self, text1, text2):
        arr1 = self.simVec(text1 + ' ' + text2)
        arr2 = self.simVec(text2 + ' ' + text1)

        similarity_lpips = 1 - self.loss_fn(torch.tensor(arr1), torch.tensor(arr2)).item()
        similarity_lpips = 0 if similarity_lpips < 0.9 else (similarity_lpips - 0.9) * 10

        return similarity_lpips


    def get_glyph_similarity(self, text1, text2):
        similarity_glyph = glyph_similar(text1 + ' ' + text2, text2 + ' ' + text1)

        return similarity_glyph


    def get_my_similarity(self, text1, text2):
        similarity_lpips = self.get_lpips_similarity(text1, text2)

        similarity_glyph = self.get_glyph_similarity(text1, text2)

        similarity = 0.8 * similarity_lpips + 0.2 * similarity_glyph

        return similarity


    def calculate_top_similar_chars(self, simvecs, similar_function):
        top_similar_chars = {}

        simvecs_matrix = np.array(list(simvecs.values()), dtype=np.float32)
        
        simvecs_matrix = simvecs_matrix.reshape(simvecs_matrix.shape[0], -1)
        faiss.normalize_L2(simvecs_matrix)

        index = faiss.IndexFlatIP(simvecs_matrix.shape[1])

        if self.gpu:
            gpu_resource = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(gpu_resource, 0, index)

        index.add(simvecs_matrix)
        
        pbar = tqdm(total=len(simvecs), desc='get top10 similar characters')
        for char, sim_array in simvecs.items():
            query = np.array([sim_array.reshape(1600)], dtype=np.float32)
            faiss.normalize_L2(query)

            _, indices = index.search(query, k=101)
            top_similar = indices[0][1:]
            top_similar_char_score = [(j, similar_function(list(simvecs.keys())[j], char)) for j in top_similar]

            top_similar_char_score.sort(key=lambda x: x[1], reverse=True)
            top_similar_chars[char] = [(list(simvecs.keys())[j], score) for j, score in top_similar_char_score[:10]]

            pbar.update(1)

        return top_similar_chars


if __name__ == '__main__':
    model = vision_model()

    filename = './utils/chinese_characters/unicode_characters.txt'
    characters = model.read_characters_from_file(filename)

    charVec = {}
    for char in characters:
        charVec[char] = model.simVec(char)

    top_similar_chars = model.calculate_top_similar_chars(charVec, model.get_glyph_similarity)

    filename = './utils/chinese_characters/vision_similar_char.pkl'
    with open(filename, 'wb') as file:
        pickle.dump(top_similar_chars, file)

