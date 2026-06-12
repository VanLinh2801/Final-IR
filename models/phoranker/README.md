---
language:
- vi
license: apache-2.0
library_name: transformers
tags:
- cross-encoder
- rerank
datasets:
- unicamp-dl/mmarco
widget:
- text: Trường UIT là gì ?.
  output:
  - label: >-
      Trường Đại_học Công_nghệ Thông_tin có tên tiếng Anh là University of
      Information_Technology ( viết tắt là UIT ) là thành_viên của Đại_học
      Quốc_Gia TP. HCM.
    score: 0.9819
  - label: >-
      Trường Đại_học Kinh_tế – Luật ( tiếng Anh : University of Economics and
      Law – UEL ) là trường đại_học đào_tạo và nghiên_cứu khối ngành kinh_tế ,
      kinh_doanh và luật hàng_đầu Việt_Nam .
    score: 0.2444
  - label: >-
      Quĩ_uỷ_thác đầu_tư ( tiếng Anh : Unit Investment_Trusts ; viết tắt : UIT )
      là một công_ty đầu_tư mua hoặc nắm giữ một danh_mục đầu_tư cố_định
    score: 0.9253
pipeline_tag: text-classification
---

#### Table of contents
1. [Installation](#installation)
2. [Pre-processing](#pre-processing)
3. [Usage with `sentence-transformers`](#usage-with-sentence-transformers)
4. [Usage with `transformers`](#usage-with-transformers)
5. [Performance](#performance)
6. [Support me](#support-me)
7. [Citation](#citation)


## Installation
  - Install `VnCoreNLP` to word segment:

	- `pip install py_vncorenlp`
      
 -  Install `sentence-transformers` (recommend) - [Usage](#usage-with-sentence-transformers):
	
	- `pip install sentence-transformers`

 -  Install `transformers` (optional) - [Usage](#usage-with-transformers):

	- `pip install transformers`

## Pre-processing

```python
import py_vncorenlp
py_vncorenlp.download_model(save_dir='/absolute/path/to/vncorenlp')
rdrsegmenter = py_vncorenlp.VnCoreNLP(annotators=["wseg"], save_dir='/absolute/path/to/vncorenlp')

query = "Trường UIT là gì?"
sentences = [
    "Trường Đại học Công nghệ Thông tin có tên tiếng Anh là University of Information Technology (viết tắt là UIT) là thành viên của Đại học Quốc Gia TP.HCM.",
    "Trường Đại học Kinh tế – Luật (tiếng Anh: University of Economics and Law – UEL) là trường đại học đào tạo và nghiên cứu khối ngành kinh tế, kinh doanh và luật hàng đầu Việt Nam.",
    "Quĩ uỷ thác đầu tư (tiếng Anh: Unit Investment Trusts; viết tắt: UIT) là một công ty đầu tư mua hoặc nắm giữ một danh mục đầu tư cố định"
]

tokenized_query = rdrsegmenter.word_segment(query)
tokenized_sentences = [rdrsegmenter.word_segment(sent) for sent in sentences]

tokenized_pairs = [[tokenized_query, sent] for sent in tokenized_sentences]

MODEL_ID = 'itdainb/PhoRanker'
MAX_LENGTH = 256
```

## Usage with sentence-transformers

```python
from sentence_transformers import CrossEncoder
model = CrossEncoder(MODEL_ID, max_length=MAX_LENGTH)

# For fp16 usage
model.model.half()

scores = model.predict(tokenized_pairs)

# 0.982, 0.2444, 0.9253
print(scores)
```

## Usage with transformers

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# For fp16 usage
model.half()

features = tokenizer(tokenized_pairs, padding=True, truncation="longest_first", return_tensors="pt", max_length=MAX_LENGTH)

model.eval()
with torch.no_grad():
    model_predictions = model(**features, return_dict=True)

    logits = model_predictions.logits
    logits = torch.nn.Sigmoid()(logits)
    scores = [logit[0] for logit in logits]

# 0.9819, 0.2444, 0.9253
print(scores)
```

## Performance
In the following table, we provide various pre-trained Cross-Encoders together with their performance on the [MS MMarco Passage Reranking - Vi - Dev](https://huggingface.co/datasets/unicamp-dl/mmarco) dataset. 

| Model-Name                                            | NDCG@3 | MRR@3 | NDCG@5 | MRR@5 | NDCG@10 | MRR@10 | Docs / Sec |
| ----------------------------------------------------- |:------ | :---- |:------ | :---- |:------ | :----| :--- |
|itdainb/PhoRanker                       |**0.6625**|**0.6458**|**0.7147**|**0.6731**|**0.7422**|**0.6830**|15
|[amberoad/bert-multilingual-passage-reranking-msmarco](https://huggingface.co/amberoad/bert-multilingual-passage-reranking-msmarco)   |0.4634|0.5233|0.5041|0.5383|0.5416|0.5523|**22**
|[kien-vu-uet/finetuned-phobert-passage-rerank-best-eval](https://huggingface.co/kien-vu-uet/finetuned-phobert-passage-rerank-best-eval) |0.0963|0.0883|0.1396|0.1131|0.1681|0.1246|15
|[BAAI/bge-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3)                                |0.6087|0.5841|0.6513|0.6062|0.6872|0.62091|3.51
|[BAAI/bge-reranker-v2-gemma](https://huggingface.co/BAAI/bge-reranker-v2-gemma)                          |0.6088|0.5908|0.6446|0.6108|0.6785|0.6249|1.29

 Note: Runtime was computed on a A100 GPU with fp16.

## Support me
If you find this work useful and would like to support its continued development, here are a few ways you can help:

1. **Star the Repository**: If you appreciate this work, please give it a star. Your support encourages continued development and improvement.
2. **Contribute**: Contributions are always welcome! You can help by reporting issues, submitting pull requests, or suggesting new features.
3. **Share**: Share this project with your colleagues, friends, or community. The more people know about it, the more feedback and contributions it can attract.
4. **Buy me a coffee**: If you’d like to provide financial support, consider making a donation. You can donate via
   - Momo: 0948798843
   - BIDV Bank: DAINB
   - Paypal: 0948798843

## Citation

 Please cite as

```Plaintext
@misc{PhoRanker,
  title={PhoRanker: A Cross-encoder Model for Vietnamese Text Ranking},
  author={Dai Nguyen Ba ({ORCID:0009-0008-8559-3154})},
  year={2024},
  publisher={Huggingface},
  journal={huggingface repository},
  howpublished={\url{https://huggingface.co/itdainb/PhoRanker}},
}
```