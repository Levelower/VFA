import faiss
import pickle
import torch
import lpips
import numpy as np
from tqdm import tqdm
from PIL import Image, ImageFont, ImageDraw
from skimage.metrics import structural_similarity as ssim


class pixel_model(object):
    def __init__(self):
        self.loss_fn = lpips.LPIPS(net='alex')
        self.gpu = False


    # Read meaningful Unicode characters
    def read_characters_from_file(self, filename):
        characters = []
        with open(filename, 'r', encoding='utf-8') as file:
            for line in file:
                line = line.strip()
                characters.extend(line)
        return characters


    # Convert "text" to vector
    # "text" includes both individual characters and the entire sentence
    def vector(self, text):
        img = Image.new("RGB", (40, 40), (255, 255, 255))
        font = ImageFont.truetype('./utils/fonts/simsun.ttc', 18)
        draw = ImageDraw.Draw(img)
        draw.text((10, 5), text, font=font, fill="#000000")
        arr = np.array(img.convert('L'))
        return arr


    # Calculate the MSE similarity of text
    # When the text is a sentence, it is necessary to ensure length alignment
    def get_mse_similarity(self, text1, text2):
        arr1 = self.vector(text1 + ' ' + text2)
        arr2 = self.vector(text2 + ' ' + text1)
        
        similarity_mse = -np.mean((arr1 - arr2) ** 2)
        
        return similarity_mse


    # Calculate the SSIM similarity of text
    # When the text is a sentence, it is necessary to ensure length alignment
    def get_ssim_similarity(self, text1, text2):
        arr1 = self.vector(text1 + ' ' + text2)
        arr2 = self.vector(text2 + ' ' + text1)
        
        similarity_ssim = ssim(arr1, arr2, multichannel=True)
        
        return similarity_ssim


    # Calculate the LPIPS similarity of text and scale them forward
    # When the text is a sentence, it is necessary to ensure length alignment
    # This is also the calculation method for local constraint scores constructed using LPIPS
    def get_lpips_similarity(self, text1, text2):
        arr1 = self.vector(text1 + ' ' + text2)
        arr2 = self.vector(text2 + ' ' + text1)

        similarity_lpips = 1 - self.loss_fn(torch.tensor(arr1), torch.tensor(arr2)).item()
        similarity_lpips = 0 if similarity_lpips < 0.9 else (similarity_lpips - 0.9) * 10

        return similarity_lpips


    # Calculate the similar characters for each character by using different similarity functions
    # Firstly, use the faiss tool and cos similarity to perform coarse-grained screening
    # Then use the selected similarity function for fine-grained screening
    def calculate_top_similar_chars(self, character_vectors, similar_function):
        top_similar_characters = {}

        vector_matrix = np.array(list(character_vectors.values()), dtype=np.float32)
        vector_matrix = vector_matrix.reshape(vector_matrix.shape[0], -1)
        faiss.normalize_L2(vector_matrix)
        index = faiss.IndexFlatIP(vector_matrix.shape[1])

        if self.gpu:
            gpu_resource = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(gpu_resource, 0, index)

        index.add(vector_matrix)
        
        pbar = tqdm(total=len(character_vectors), desc='get top similar characters')
        for character, vector in character_vectors.items():
            query = np.array([vector.reshape(1600)], dtype=np.float32)
            faiss.normalize_L2(query)

            _, indices = index.search(query, k=101)
            top_similar = indices[0][1:]
            top_similar_character_score = [
                (j, similar_function(list(character_vectors.keys())[j], character)) for j in top_similar
            ]

            top_similar_character_score.sort(key=lambda x: x[1], reverse=True)
            top_similar_characters[character] = [
                (list(character_vectors.keys())[j], score) for j, score in top_similar_character_score[:50]
            ]

            pbar.update(1)

        return top_similar_characters


if __name__ == '__main__':
    model = pixel_model()

    filename = './utils/characters/unicode_characters.txt'
    characters = model.read_characters_from_file(filename)

    character_vectors = {}
    for character in characters:
        character_vectors[character] = model.vector(character)

    top_similar_characters = model.calculate_top_similar_chars(
        character_vectors, model.get_mse_similarity
    )

    filename = './utils/characters/pixel_level_similar_characters.pkl'
    with open(filename, 'wb') as file:
        pickle.dump(top_similar_characters, file)

