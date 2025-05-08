# from typing import List
# import sys
# import os
# import pandas as pd
# import torch
# import matplotlib.pyplot as plt
# import numpy as np
# import cv2
# import json
# from PIL import Image
# from tqdm import tqdm
# from torch.utils.data import DataLoader
# from transformers import ViltConfig
# from transformers import ViltProcessor
# from transformers import ViltForQuestionAnswering
# from inference_utils import *
#
# # 修改 sys.path.append 的路径
# sys.path.append('/root/autodl-tmp/VizWiz2024-VQA-AnswerTherapy-main/src/polygon-transformer')
# sys.path.append('/root/autodl-tmp/VizWiz2024-VQA-AnswerTherapy-main/src/polygon-transformer/fairseq')
# from demo import visual_grounding
#
# # vizwiz_data_base_path = '../dataset'
# #
# # viz_wiz_data_train_image_dir = os.path.join(vizwiz_data_base_path, 'val')
# # viz_wiz_data_train_annotation_path = os.path.join(vizwiz_data_base_path, 'VizWiz_val.json')
# viz_wiz_data_train_image_dir = '/root/autodl-tmp/VizWiz2024-VQA-AnswerTherapy-main/dataset/train'
# viz_wiz_data_train_annotation_path = '/root/autodl-tmp/VizWiz2024-VQA-AnswerTherapy-main/dataset/Annotations/VizWiz_train.json'
#
# # pretrained_model = 'vilt-b32-finetuned-vqa'
# # model_folder = os.path.join('../ViLT', pretrained_model)
# # finetune_folder = '../ViLT/my_finetune/custom_vqa_vilt-b32-finetuned-vqa'
# # 直接指定预训练模型文件夹路径
# model_folder = '/root/autodl-tmp/VizWiz2024-VQA-AnswerTherapy-main/src/ViLT/vilt-b32-finetuned-vqa-main'
# # 直接指定微调模型文件夹路径
# finetune_folder = '/root/autodl-tmp/VizWiz2024-VQA-AnswerTherapy-main/src/ViLT/custom_vqa_vilt-b32-finetuned-vqa'
#
# processor = ViltProcessor.from_pretrained(model_folder)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
# missing_words_in_vocab = []
#
# COMBINED_IMAGE_DIR = '/root/autodl-tmp/VizWiz2024-VQA-AnswerTherapy-main/dataset/test'
# question_image_data_path = '/root/autodl-tmp/VizWiz2024-VQA-AnswerTherapy-main/dataset/Annotations/VizWiz_test_new.json'
# question_image_data = json.load(open(question_image_data_path))
#
# # 直接从字典中获取对应的列表
# formatted_data = {
#     "questions": question_image_data.get("questions", []),
#     "images": question_image_data.get("images", [])
# }
# question_image_data = formatted_data
#
# print("question_image_data 的数据类型:", type(question_image_data))
# if isinstance(question_image_data, list):
#     print("question_image_data 是列表，列表长度:", len(question_image_data))
#     if len(question_image_data) > 0:
#         print("列表第一个元素的类型:", type(question_image_data[0]))
#         if isinstance(question_image_data[0], dict):
#             print("第一个元素的键:", question_image_data[0].keys())
# elif isinstance(question_image_data, dict):
#     print("question_image_data 是字典，字典的键:", question_image_data.keys())
#
# try:
#     print(len(question_image_data['questions']))
# except TypeError as e:
#     print(f"出现错误: {e}")
#
# # 定义函数将图片id转换为12位格式
# def format_image_id(image_id):
#     return str(image_id).zfill(12)
#
# num = 111
# question_list_length = len(question_image_data['questions'])
# max_index = min(question_list_length, num + 20)
# for i in range(num, max_index):
#     print("=====Input information: {}".format(i))
#     question = question_image_data['questions'][i]
#     print(question)
#     # 转换图片id为12位格式
#     formatted_image_id = format_image_id(question_image_data['images'][i])
#     # 检查文件名是否已经包含 .jpg 后缀
#     if not formatted_image_id.endswith('.jpg'):
#         formatted_image_id += '.jpg'
#     image_path = os.path.join(COMBINED_IMAGE_DIR, formatted_image_id)
#     print("尝试读取的图片路径:", image_path)
#     image = cv2.imread(image_path)
#     if image is None:
#         print(f"无法读取图片 {image_path}，跳过当前样本")
#         continue
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     plt.imshow(image)
#     plt.show()
#     print("Inference using finetune model")
#     pred_answers: List[str] = inference_vilt(input_image=image,
#                                              input_text=question,
#                                              print_preds=False, finetune_folder=finetune_folder)
#     processor = ViltProcessor.from_pretrained("/root/autodl-tmp/VizWiz2024-VQA-AnswerTherapy-main/src/ViLT/custom_vqa_vilt-b32-finetuned-vqa")
#     predicted_grounding_masks: list = []
#     image = Image.open(image_path)
#     for i, ans in enumerate(pred_answers):
#         # input
#         input_text = question + " answer:" + ans
#         # input_text = ans
#
#         # input_text = question + " answer: sweet potato" + ans
#         print(input_text)
#         pred_overlayed, pred_mask = visual_grounding(image=image, text=input_text)
#         plt.imshow(pred_mask)
#         plt.show()
#         predicted_grounding_masks.append(pred_mask)
#         # save to mask
#     predicted_label = 'single' if is_single_groundings(predicted_grounding_masks) else 'multiple'
#     print("Final predicted label: ", predicted_label)
#     # prediction
#     break
# print("Done")
#
# ## INFERENCE FOR SUBMISSION
# naive_path = '/root/autodl-tmp/VizWiz2024-VQA-AnswerTherapy-main/src/submission.json'
# naive_data = json.load(open(naive_path))
# results = []
# maximum_length = 35
# question_list_length_submission = len(question_image_data['questions'])
# for i in tqdm(range(question_list_length_submission)):
#     raw_question = question_image_data['questions'][i]
#     number_of_chars = len(raw_question)
#     if number_of_chars > maximum_length:
#         question = raw_question[:maximum_length]
#     else:
#         question = raw_question
#     # 转换图片id为12位格式
#     formatted_image_id = format_image_id(question_image_data['images'][i])
#     # 检查文件名是否已经包含 .jpg 后缀
#     if not formatted_image_id.endswith('.jpg'):
#         formatted_image_id += '.jpg'
#     image_path = os.path.join(COMBINED_IMAGE_DIR, formatted_image_id)
#     print("尝试读取的图片路径:", image_path)
#     image = cv2.imread(image_path)
#     if image is None:
#         print(f"无法读取图片 {image_path}，跳过当前样本")
#         continue
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#
#     # visualize
#     # plt.imshow(image)
#     # plt.show()
#     #
#     pred_answers: List[str] = inference_vilt(input_image=image,
#                                              input_text=question,
#                                              print_preds=False, finetune_folder=finetune_folder)
#     predicted_grounding_masks: list = []
#     image = Image.open(image_path)
#     for j, ans in enumerate(pred_answers):
#         # input
#         input_text = question + " answer:" + ans
#         pred_overlayed, pred_mask = visual_grounding(image=image, text=input_text)
#         predicted_grounding_masks.append(pred_mask)
#         # print(input_text)
#         # plt.imshow(pred_overlayed)
#         # plt.show()
#         # save to mask
#     predicted_label = 'single' if is_single_groundings(predicted_grounding_masks) else 'multiple'
#     temp = {}
#     temp['question_id'] = question_image_data['images'][i]
#     temp['single_grounding'] = 1 if predicted_label == 'single' else 0
#     results.append(temp)
#
#     # break
# # save to json
# with open('submission.json', 'w') as f:
#     json.dump(results, f)






# 86.06所采用的方法
from typing import List
import sys
import os
import pandas as pd
import torch
import matplotlib.pyplot as plt
import numpy as np
import cv2
import json
from PIL import Image
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import ViltConfig
from transformers import ViltProcessor
from transformers import ViltForQuestionAnswering
from inference_utils import *

# 修改 sys.path.append 的路径
sys.path.append('/root/autodl-tmp/VizWiz2024-VQA-AnswerTherapy-main/src/polygon-transformer')
sys.path.append('/root/autodl-tmp/VizWiz2024-VQA-AnswerTherapy-main/src/polygon-transformer/fairseq')
from demo import visual_grounding

# vizwiz_data_base_path = '../dataset'
#
# viz_wiz_data_train_image_dir = os.path.join(vizwiz_data_base_path, 'val')
# viz_wiz_data_train_annotation_path = os.path.join(vizwiz_data_base_path, 'VizWiz_val.json')
viz_wiz_data_train_image_dir = '/root/autodl-tmp/VizWiz2024-VQA-AnswerTherapy-main/dataset/train'
viz_wiz_data_train_annotation_path = '/root/autodl-tmp/VizWiz2024-VQA-AnswerTherapy-main/dataset/Annotations/VizWiz_train.json'

# pretrained_model = 'vilt-b32-finetuned-vqa'
# model_folder = os.path.join('../ViLT', pretrained_model)
# finetune_folder = '../ViLT/my_finetune/custom_vqa_vilt-b32-finetuned-vqa'
# 直接指定预训练模型文件夹路径
model_folder = '/root/autodl-tmp/VizWiz2024-VQA-AnswerTherapy-main/src/ViLT/vilt-b32-finetuned-vqa-main'
# 直接指定微调模型文件夹路径
finetune_folder = '/root/autodl-tmp/VizWiz2024-VQA-AnswerTherapy-main/src/ViLT/custom_vqa_vilt-b32-finetuned-vqa'

processor = ViltProcessor.from_pretrained(model_folder)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

missing_words_in_vocab = []

COMBINED_IMAGE_DIR = '/root/autodl-tmp/VizWiz2024-VQA-AnswerTherapy-main/dataset/test'
question_image_data_path = '/root/autodl-tmp/VizWiz2024-VQA-AnswerTherapy-main/dataset/Annotations/VizWiz_test_new.json'
try:
    question_image_data = json.load(open(question_image_data_path))
except FileNotFoundError:
    print(f"错误：未找到文件 {question_image_data_path}。")
    sys.exit(1)
except json.JSONDecodeError:
    print(f"错误：无法解析 {question_image_data_path} 中的 JSON 数据。")
    sys.exit(1)

print("question_image_data 的数据类型:", type(question_image_data))
if isinstance(question_image_data, list):
    print("question_image_data 是列表，列表长度:", len(question_image_data))
    if len(question_image_data) > 0:
        print("列表第一个元素的类型:", type(question_image_data[0]))
        if isinstance(question_image_data[0], dict):
            print("第一个元素的键:", question_image_data[0].keys())
elif isinstance(question_image_data, dict):
    print("question_image_data 是字典，字典的键:", question_image_data.keys())

try:
    print(len(question_image_data))
except TypeError as e:
    print(f"出现错误: {e}")


# 定义函数将图片id转换为12位格式
def format_image_id(image_id):
    return str(image_id).zfill(12)


num = 111
question_list_length = len(question_image_data)
max_index = min(question_list_length, num + 20)
for i in range(num, max_index):
    item = question_image_data[i]
    question = item.get('question')
    image_id = item.get('image_id')
    question_id = item.get('question_id')
    if question_id == "":
        question_id = image_id
    print("=====Input information: {}".format(i))
    print(question)
    # 转换图片id为12位格式
    formatted_image_id = format_image_id(image_id)
    # 检查文件名是否已经包含 .jpg 后缀
    if not formatted_image_id.endswith('.jpg'):
        formatted_image_id += '.jpg'
    image_path = os.path.join(COMBINED_IMAGE_DIR, formatted_image_id)
    print("尝试读取的图片路径:", image_path)
    image = cv2.imread(image_path)
    if image is None:
        print(f"无法读取图片 {image_path}，跳过当前样本")
        continue
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(image)
    plt.show()
    print("Inference using finetune model")
    pred_answers: List[str] = inference_vilt(input_image=image,
                                             input_text=question,
                                             print_preds=False, finetune_folder=finetune_folder)
    processor = ViltProcessor.from_pretrained(
        "/root/autodl-tmp/VizWiz2024-VQA-AnswerTherapy-main/src/ViLT/custom_vqa_vilt-b32-finetuned-vqa")
    predicted_grounding_masks: list = []
    image = Image.open(image_path)
    for i, ans in enumerate(pred_answers):
        # input
        input_text = question + " answer:" + ans
        print(input_text)
        pred_overlayed, pred_mask = visual_grounding(image=image, text=input_text)
        plt.imshow(pred_mask)
        plt.show()
        predicted_grounding_masks.append(pred_mask)
        # save to mask
    predicted_label = 'single' if is_single_groundings(predicted_grounding_masks) else 'multiple'
    print("Final predicted label: ", predicted_label)
    # prediction
    break
print("Done")

## INFERENCE FOR SUBMISSION
naive_path = '/root/autodl-tmp/VizWiz2024-VQA-AnswerTherapy-main/src/submission.json'
try:
    naive_data = json.load(open(naive_path))
except FileNotFoundError:
    print(f"错误：未找到文件 {naive_path}。")
    sys.exit(1)
except json.JSONDecodeError:
    print(f"错误：无法解析 {naive_path} 中的 JSON 数据。")
    sys.exit(1)
results = []
maximum_length = 35
question_list_length_submission = len(question_image_data)
for i in tqdm(range(question_list_length_submission)):
    item = question_image_data[i]
    raw_question = item.get('question')
    number_of_chars = len(raw_question)
    if number_of_chars > maximum_length:
        question = raw_question[:maximum_length]
    else:
        question = raw_question
    image_id = item.get('image_id')
    question_id = item.get('question_id')
    if question_id == "":
        question_id = image_id
    # 转换图片id为12位格式
    formatted_image_id = format_image_id(image_id)
    # 检查文件名是否已经包含 .jpg 后缀
    if not formatted_image_id.endswith('.jpg'):
        formatted_image_id += '.jpg'
    image_path = os.path.join(COMBINED_IMAGE_DIR, formatted_image_id)
    print("尝试读取的图片路径:", image_path)
    image = cv2.imread(image_path)
    if image is None:
        print(f"无法读取图片 {image_path}，跳过当前样本")
        continue
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # visualize
    # plt.imshow(image)
    # plt.show()
    #
    pred_answers: List[str] = inference_vilt(input_image=image,
                                             input_text=question,
                                             print_preds=False, finetune_folder=finetune_folder)
    predicted_grounding_masks: list = []
    image = Image.open(image_path)
    for j, ans in enumerate(pred_answers):
        # input
        input_text = question + " answer:" + ans
        pred_overlayed, pred_mask = visual_grounding(image=image, text=input_text)
        predicted_grounding_masks.append(pred_mask)
        # print(input_text)
        # plt.imshow(pred_overlayed)
        # plt.show()
        # save to mask
    predicted_label = 'single' if is_single_groundings(predicted_grounding_masks) else 'multiple'
    temp = {}
    temp['question_id'] = question_id
    temp['single_grounding'] = 1 if predicted_label == 'single' else 0
    results.append(temp)

# save to json
with open('submission.json', 'w') as f:
    json.dump(results, f)





#加入了tta
# import torch
# import pandas as pd
# from PIL import Image
# import os
# import copy
# import json
# from transformers import ViltConfig, ViltProcessor, ViltForQuestionAnswering
# import numpy as np
# import cv2
# import matplotlib.pyplot as plt
# from torch.utils.data import Dataset, DataLoader
# from torch.optim import AdamW
# from tqdm import tqdm
# import datetime
# from typing import List
# from torchvision import transforms
# from typing import List
# import sys
# import os
# import pandas as pd
# import torch
# import matplotlib.pyplot as plt
# import numpy as np
# import cv2
# import json
# from PIL import Image
# from tqdm import tqdm
# from torch.utils.data import DataLoader
# from transformers import ViltConfig
# from transformers import ViltProcessor
# from transformers import ViltForQuestionAnswering
# from inference_utils import *
#
#
#
#
# sys.path.append('/root/autodl-tmp/VizWiz2024-VQA-AnswerTherapy-main/src/polygon-transformer')
# sys.path.append('/root/autodl-tmp/VizWiz2024-VQA-AnswerTherapy-main/src/polygon-transformer/fairseq')
# from demo import visual_grounding
#
# # 路径设置
# viz_wiz_data_train_image_dir = '/root/autodl-tmp/VizWiz2024-VQA-AnswerTherapy-main/dataset/train'
# viz_wiz_data_train_annotation_path = '/root/autodl-tmp/VizWiz2024-VQA-AnswerTherapy-main/dataset/Annotations/VizWiz_train.json'
# model_folder = '/root/autodl-tmp/VizWiz2024-VQA-AnswerTherapy-main/src/ViLT/vilt-b32-finetuned-vqa-main'
# finetune_folder = '/root/autodl-tmp/VizWiz2024-VQA-AnswerTherapy-main/src/ViLT/custom_vqa_vilt-b32-finetuned-vqa'
# COMBINED_IMAGE_DIR = '/root/autodl-tmp/VizWiz2024-VQA-AnswerTherapy-main/dataset/test'
# question_image_data_path = '/root/autodl-tmp/VizWiz2024-VQA-AnswerTherapy-main/dataset/Annotations/VizWiz_test_new.json'
# naive_path = '/root/autodl-tmp/VizWiz2024-VQA-AnswerTherapy-main/src/submission.json'
#
# processor = ViltProcessor.from_pretrained(model_folder)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
# missing_words_in_vocab = []
#
# try:
#     question_image_data = json.load(open(question_image_data_path))
# except FileNotFoundError:
#     print(f"错误：未找到文件 {question_image_data_path}。")
#     import sys
#     sys.exit(1)
# except json.JSONDecodeError:
#     print(f"错误：无法解析 {question_image_data_path} 中的 JSON 数据。")
#     import sys
#     sys.exit(1)
#
# print("question_image_data 的数据类型:", type(question_image_data))
# if isinstance(question_image_data, list):
#     print("question_image_data 是列表，列表长度:", len(question_image_data))
#     if len(question_image_data) > 0:
#         print("列表第一个元素的类型:", type(question_image_data[0]))
#         if isinstance(question_image_data[0], dict):
#             print("第一个元素的键:", question_image_data[0].keys())
# elif isinstance(question_image_data, dict):
#     print("question_image_data 是字典，字典的键:", question_image_data.keys())
#
# try:
#     print(len(question_image_data))
# except TypeError as e:
#     print(f"出现错误: {e}")
#
#
# # 定义函数将图片id转换为12位格式
# def format_image_id(image_id):
#     return str(image_id).zfill(12)
#
#
# # 定义 TTA 数据增强变换
# tta_transforms = [
#     transforms.RandomRotation(10),
#     transforms.RandomHorizontalFlip(),
#     transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
#     transforms.GaussianBlur(kernel_size=3)
# ]
#
#
# def inference_vilt_tta(input_image, input_text, print_preds=False, finetune_folder=finetune_folder,
#                        tta_transforms=tta_transforms):
#     processor = ViltProcessor.from_pretrained(finetune_folder)
#     model = ViltForQuestionAnswering.from_pretrained(finetune_folder)
#     model.to(device)
#     model.eval()
#
#     all_pred_answers = []
#     original_image = input_image.copy()  # 保存原始图像
#
#     # 对原始图像进行推理
#     encoding = processor(original_image, input_text, return_tensors="pt")
#     encoding = {k: v.to(device) for k, v in encoding.items()}
#     with torch.no_grad():
#         outputs = model(**encoding)
#         logits = outputs.logits
#         predicted_classes = torch.sigmoid(logits)
#         probs, classes = torch.topk(predicted_classes, 5)
#         results = []
#         for prob, class_idx in zip(probs.squeeze().tolist(), classes.squeeze().tolist()):
#             if prob >= 0.2:
#                 results.append(model.config.id2label[class_idx])
#         all_pred_answers.extend(results)
#
#     # 对增强后的图像进行推理
#     for transform in tta_transforms:
#         augmented_image = transform(Image.fromarray(original_image))
#         encoding = processor(augmented_image, input_text, return_tensors="pt")
#         encoding = {k: v.to(device) for k, v in encoding.items()}
#         with torch.no_grad():
#             outputs = model(**encoding)
#             logits = outputs.logits
#             predicted_classes = torch.sigmoid(logits)
#             probs, classes = torch.topk(predicted_classes, 5)
#             results = []
#             for prob, class_idx in zip(probs.squeeze().tolist(), classes.squeeze().tolist()):
#                 if prob >= 0.2:
#                     results.append(model.config.id2label[class_idx])
#             all_pred_answers.extend(results)
#
#     # 融合推理结果（这里简单使用投票法）
#     label_counts = {}
#     for label in all_pred_answers:
#         label_counts[label] = label_counts.get(label, 0) + 1
#     final_results = [k for k, v in label_counts.items() if v == max(label_counts.values())]
#
#     if print_preds:
#         for label in final_results:
#             print(label)
#
#     return final_results
#
#
# def is_single_groundings(predicted_grounding_masks):
#     # 这里需要你根据实际情况实现判断逻辑
#     pass
#
#
# num = 111
# question_list_length = len(question_image_data)
# max_index = min(question_list_length, num + 20)
# for i in range(num, max_index):
#     item = question_image_data[i]
#     question = item.get('question')
#     image_id = item.get('image_id')
#     question_id = item.get('question_id')
#     if question_id == "":
#         question_id = image_id
#     print("=====Input information: {}".format(i))
#     print(question)
#     # 转换图片id为12位格式
#     formatted_image_id = format_image_id(image_id)
#     # 检查文件名是否已经包含 .jpg 后缀
#     if not formatted_image_id.endswith('.jpg'):
#         formatted_image_id += '.jpg'
#     image_path = os.path.join(COMBINED_IMAGE_DIR, formatted_image_id)
#     print("尝试读取的图片路径:", image_path)
#     image = cv2.imread(image_path)
#     if image is None:
#         print(f"无法读取图片 {image_path}，跳过当前样本")
#         continue
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     plt.imshow(image)
#     plt.show()
#     print("Inference using finetune model")
#     pred_answers: List[str] = inference_vilt_tta(input_image=image,
#                                                  input_text=question,
#                                                  print_preds=False, finetune_folder=finetune_folder)
#     processor = ViltProcessor.from_pretrained(
#         "/root/autodl-tmp/VizWiz2024-VQA-AnswerTherapy-main/src/ViLT/custom_vqa_vilt-b32-finetuned-vqa")
#     predicted_grounding_masks: list = []
#     image = Image.open(image_path)
#     for i, ans in enumerate(pred_answers):
#         # input
#         input_text = question + " answer:" + ans
#         print(input_text)
#         pred_overlayed, pred_mask = visual_grounding(image=image, text=input_text)
#         plt.imshow(pred_mask)
#         plt.show()
#         predicted_grounding_masks.append(pred_mask)
#         # save to mask
#     predicted_label = 'single' if is_single_groundings(predicted_grounding_masks) else 'multiple'
#     print("Final predicted label: ", predicted_label)
#     # prediction
#     break
# print("Done")
#
# ## INFERENCE FOR SUBMISSION
# try:
#     naive_data = json.load(open(naive_path))
# except FileNotFoundError:
#     print(f"错误：未找到文件 {naive_path}。")
#     import sys
#     sys.exit(1)
# except json.JSONDecodeError:
#     print(f"错误：无法解析 {naive_path} 中的 JSON 数据。")
#     import sys
#     sys.exit(1)
# results = []
# maximum_length = 35
# question_list_length_submission = len(question_image_data)
# for i in tqdm(range(question_list_length_submission)):
#     item = question_image_data[i]
#     raw_question = item.get('question')
#     number_of_chars = len(raw_question)
#     if number_of_chars > maximum_length:
#         question = raw_question[:maximum_length]
#     else:
#         question = raw_question
#     image_id = item.get('image_id')
#     question_id = item.get('question_id')
#     if question_id == "":
#         question_id = image_id
#     # 转换图片id为12位格式
#     formatted_image_id = format_image_id(image_id)
#     # 检查文件名是否已经包含 .jpg 后缀
#     if not formatted_image_id.endswith('.jpg'):
#         formatted_image_id += '.jpg'
#     image_path = os.path.join(COMBINED_IMAGE_DIR, formatted_image_id)
#     print("尝试读取的图片路径:", image_path)
#     image = cv2.imread(image_path)
#     if image is None:
#         print(f"无法读取图片 {image_path}，跳过当前样本")
#         continue
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#
#     # visualize
#     # plt.imshow(image)
#     # plt.show()
#
#     pred_answers: List[str] = inference_vilt_tta(input_image=image,
#                                                  input_text=question,
#                                                  print_preds=False, finetune_folder=finetune_folder)
#     predicted_grounding_masks: list = []
#     image = Image.open(image_path)
#     for j, ans in enumerate(pred_answers):
#         # input
#         input_text = question + " answer:" + ans
#         pred_overlayed, pred_mask = visual_grounding(image=image, text=input_text)
#         predicted_grounding_masks.append(pred_mask)
#         # print(input_text)
#         # plt.imshow(pred_overlayed)
#         # plt.show()
#         # save to mask
#     predicted_label = 'single' if is_single_groundings(predicted_grounding_masks) else 'multiple'
#     temp = {}
#     temp['question_id'] = question_id
#     temp['single_grounding'] = 1 if predicted_label == 'single' else 0
#     results.append(temp)
#
# # save to json
# with open('submission.json', 'w') as f:
#     json.dump(results, f)