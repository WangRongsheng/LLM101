# llm

1. [检查 Tokenizer 词表大小与 LLM 的 Embedding 和 LM_head 输入大小是否匹配](https://github.com/KnowAIHub/llm/blob/main/embedding/check_embedding.ipynb)
2. [对 SBert 进行训练、预测、评估使其进行相似度计算](https://github.com/KnowAIHub/llm/tree/main/huggingface/Sentence-BERT-Similarity)
3. [对 BERT 进行训练、预测、评估使其进行文本分类](https://github.com/KnowAIHub/llm/tree/main/huggingface/bert-text-classification)
4. [使用 CLIP 模型进行文本图像匹配](https://github.com/KnowAIHub/llm/tree/main/huggingface/clip-text-image-matching)
5. [对 JoinBERT 进行训练、预测使其进行对话意图和槽位联合识别](https://github.com/KnowAIHub/llm/tree/main/huggingface/intent-reg)
6. [对比LoRA微调、模型Last Layers微调以及模型全参数微调对比，并且使用网格搜索 LoRA 最佳参数设置](https://github.com/KnowAIHub/llm/tree/main/huggingface/lora-comparison)
7. [对 Qwen2-0.5B 模型进行 LoRA 微调](https://github.com/KnowAIHub/llm/tree/main/huggingface/peft-lora)
8. [对 RoBERTa 进行训练、预测使其进行中文/英文文本分类](https://github.com/KnowAIHub/llm/tree/main/huggingface/roberta-text-classification)
9. [利用 SBert 进行Embedding、文本相似度计算、语义检索、检索ReRank、图像检索等](https://github.com/KnowAIHub/llm/tree/main/huggingface/sbert)
10. [简单的文本分类实现](https://github.com/KnowAIHub/llm/tree/main/huggingface/text-classification)
11. [LLM 不同精度（FP16，FP32，BF16）下显存占用、精度转换](https://github.com/KnowAIHub/llm/tree/main/memory_precision)
12. [使用 Sentencepiece 进行LLM词表的扩展与中文化](https://github.com/KnowAIHub/llm/tree/main/sentencepiece)
13. [扩展LLM词表后对 Embedding 以及 LM_head 进行随机初始化](https://github.com/KnowAIHub/llm/blob/main/embedding/code.ipynb)

# Reference

1. https://github.com/taishan1994/sentencepiece_chinese_bpe
2. https://github.com/Glanvery/LLM-Travel
3. https://github.com/hyunwoongko/transformer
4. https://github.com/zyds/transformers-code
5. https://github.com/leeguandong/MiniLLaMA3
6. https://github.com/datawhalechina/self-llm
7. [使用huggingface的PEFT库在千问2基础上进行Lora指令微调](https://www.ethanzhang.xyz/2024/07/09/%E3%80%90%E4%B8%AA%E4%BA%BA%E5%8D%9A%E5%AE%A2%E3%80%91%E4%BD%BF%E7%94%A8huggingface%E5%9C%A8%E5%8D%83%E9%97%AE2%E5%9F%BA%E7%A1%80%E4%B8%8A%E8%BF%9B%E8%A1%8CLora%E6%8C%87%E4%BB%A4%E5%BE%AE%E8%B0%83/)
8. https://github.com/monologg/JointBERT
9. https://github.com/Linear95/bert-intent-slot-detector
10. https://github.com/Coobiw/MPP-LLaVA
