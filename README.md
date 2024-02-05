# BiLLM: Pushing the Limit of Post-Training Quantization for LLMs

![intuition](imgs/author.png)

**^1^The University of Hong Kong ^2^Beihang University ^3^ETH  Zürich **

![intuition](imgs/main_2.pdf)

## Abstract

Pretrained large language models (LLMs) exhibit exceptional general language processing capabilities but come with significant demands on memory and computational resources. As a powerful compression technology, binarization can extremely reduce model weights to a mere 1 bit, lowering the expensive computation and memory requirements. However, existing quantization techniques fall short of maintaining LLM performance under ultra-low bit-widths. In response to this challenge, we present *BiLLM*, a groundbreaking 1-bit post-training quantization scheme tailored for pretrained LLMs. Based on the weight distribution of LLMs, *BiLLM* first identifies and structurally selects salient weights, and minimizes the compression loss through an effective *binary residual approximation* strategy. Moreover, considering the bell-shaped distribution of the non-salient weights, we propose an *optimal splitting search* to group and binarize them accurately. *BiLLM* achieving for the first time high-accuracy inference (e.g. 8.41 perplexity on LLaMA2-70B) with only 1.08-bit weights across various LLMs families and evaluation metrics, outperforms SOTA quantization methods of LLM by significant margins. Moreover, *BiLLM* enables the binarization process of the LLM with 7 billion weights within 0.5 hours on a single GPU, demonstrating satisfactory time efficiency.

## News

- [2024/2] *BiLLM* source code is open now!

## Dependences

* `torch`: tested on v1.10.1+cu111
* `transformers`: tested on v4.21.2 (the LLaMa integration currently requires a main install from source and `sentencepiece`)
* `datasets`: tested on v1.17.0
* `Huggingface`:

All binarization process and experiments were run on a single 80GB NVIDIA A100. However, all the process can also be conducted on a single 24GB NVIDIA 3090 Ti when the model's parameter is under 70B.

## Installation



## LLMs Binarization

#### Binarization for OPT families

```

```



#### Binarization for LLaMA families

```

```



#### Binarization for Vicuna families (Instruction Fine-tuning Models)

```

```

#### 

## Results

- BiLLM  achieve superior perplexity performance on Wikitext2 datasets  within only an average of **1.11** bit-width weights OPT families.

![intuition](imgs/opt_wiki_results.png)

- BiLLM  achieve superior perplexity performance on Wikitext2 datasets  within only an average of **1.09** bit-width weights LLaMA families and **1.08** bit-width weights LLaMA2 families.

![intuition](imgs/llama_wiki_results.png)

- We also evaluated the performance of *BiLLM* on PTB and C4 datasets. 

![intuition](imgs/ptbandc4.pdf)

![intuition](imgs/ptbandc4_1.pdf)

- We further evaluated *BiLLM* on 7 zero-shot dataset to give extensive insight on  binarization LLMs

  ![intuition](imgs/zero_shot.png)

- BiLLM  achieve superior perplexity performance on Wikitext2 datasets  within only an average of **1.10** bit-width weights Vicuna families (instruction fine-tune models).

![intuition](imgs/vicuna.png)

## Citation

If you find *BiLLM* is useful and helpful to your work, please kindly cite this paper:

```

```
