import re
import time
import nltk
import numpy as np
import pandas as pd
from nltk.translate.bleu_score import SmoothingFunction
from PIL import Image, ImageFont, ImageDraw
from skimage.metrics import structural_similarity as ssim


smooth_func = SmoothingFunction()


def calc_bleu_asr(source, candidate, reference):
    if type(source) is str:
        source = [source]
    if type(candidate) is str:
        candidate = [candidate]
    if type(reference) is str:
        reference = [reference]

    bleu_scores_atk = []
    bleu_scores_ori = []

    all = 0
    suc = 0

    for ori, cand, ref in zip(source, candidate, reference):
        try:
            source_list = nltk.word_tokenize(ori)
            candidate_list = nltk.word_tokenize(cand)
            reference_list = nltk.word_tokenize(ref)

            references = [reference_list]

            bleu_score_atk = nltk.translate.bleu_score.sentence_bleu(
                references, candidate_list, 
                smoothing_function=smooth_func.method1
            )

            bleu_score_ori = nltk.translate.bleu_score.sentence_bleu(
                references, source_list, 
                smoothing_function=smooth_func.method1
            )

            bleu_scores_atk.append(bleu_score_atk)
            bleu_scores_ori.append(bleu_score_ori)

            if bleu_score_ori > 0 and (bleu_score_ori - bleu_score_atk) / bleu_score_ori > 0.5:
                suc += 1

        except Exception as e:
            bleu_scores_atk.append(0)
            bleu_scores_ori.append(0)

        all += 1

    avg_bleu_score_atk = sum(bleu_scores_atk) / len(bleu_scores_atk)
    avg_bleu_score_ori = sum(bleu_scores_ori) / len(bleu_scores_ori)
    asr = suc / all
    return avg_bleu_score_atk, avg_bleu_score_ori, asr


def get_similarity(text1, text2):
    font = ImageFont.truetype('./utils/fonts/simsun.ttc', 18)

    _, _, w1, _ = font.getbbox(text1)
    _, _, w2, _ = font.getbbox(text2)

    margin = 20
    im_w = w1 + w2 + margin

    im_h = 50
    im1 = Image.new("RGB", (im_w, im_h), (255, 255, 255))
    im2 = Image.new("RGB", (im_w, im_h), (255, 255, 255))
    dr1 = ImageDraw.Draw(im1)
    dr2 = ImageDraw.Draw(im2)

    offset = margin // 2
    dr1.text((offset, offset), text1 + " " + text2, font=font, fill="#000000")
    dr2.text((offset, offset), text2 + " " + text1, font=font, fill="#000000")
    arr1 = np.array(im1.convert('L'))
    arr2 = np.array(im2.convert('L'))

    similarity = ssim(arr1, arr2, multichannel=True)

    return similarity


def calc_ssim(candidate, reference):
    if type(candidate) is str:
        candidate = [candidate]
    if type(reference) is str:
        reference = [reference]

    ssim_score = 0
    cnt = 0
    for i in range(len(candidate)):
        s_t = get_similarity(
            re.sub(r'[^\w\s]', '', candidate[i].replace(' ','')), 
            re.sub(r'[^\w\s]', '', reference[i].replace(' ',''))
        )
        ssim_score += s_t
        cnt += 1
    return ssim_score / cnt


def get_data(in_path):
    data = pd.read_csv(in_path, sep='\t')
    return data


if __name__ == '__main__':
    res_path = './result/Japanese/ja-en-opus100-0.2-0.95-pix-0.9-True.csv'
    src_l = 'ja'
    tgt_l = 'en'

    data_eval = get_data(res_path)

    source = data_eval['ori_translate']
    candidate = data_eval['atk_translate']
    reference = data_eval[tgt_l]

    refs = []
    sys = []
    ori = []

    for src, cand, ref in zip(source, candidate, reference):
        references = [ref]
        refs.append(references)
        sys.append(cand)
        ori.append(src)

    ssim_score = calc_ssim(data_eval['atk_content'], data_eval[src_l])

    bleu_atk, bleu_ori, asr = calc_bleu_asr(source, candidate, reference)

    print('SSIM:\t\t', str(ssim_score), flush=True)
    print('BLEU:\t\t', str(bleu_ori) + '\t->\t' + str(bleu_atk) + '\tdown: ' + str((bleu_ori-bleu_atk)/bleu_ori), flush=True)
    print('ASR:\t\t', str(asr), flush=True)