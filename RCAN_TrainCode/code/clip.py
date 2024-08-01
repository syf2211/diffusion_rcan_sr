from transformers import CLIPTokenizer, CLIPTextModel
import torch

# 加载预训练的 CLIP 模型
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
text_model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")

# 示例文本
text = ["A photo of a cat", "A photo of a dog"]

# 将文本编码为输入ID
inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)

# 使用文本编码器获取文本特征
with torch.no_grad():
    text_features = text_model(**inputs).last_hidden_state

print(text_features.shape)  # 输出特征的形状
#查看特征在什么设备上
print(text_features.device)  # 输出特征的设备