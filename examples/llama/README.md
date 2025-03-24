## Pre-Training LLaMA on C4 dataset

This example provides scripts for pre-training LLaMA models on the C4 dataset using COAP optimizer.


### Environment Setup

```bash
pip install -r requirements.txt
```


### Download C4
Download the C4 dataset locally by running:

```bash
bash scripts/download_c4.sh
```


### Pre-Training LLaMA on C4

To pre-train the 1B LLaMA model on C4, execute:

```bash
# LLaMA-1B, COAP-AdamW, 8xH100, 1 Node
bash scripts/llama_1b.sh
```

To pre-train the 7B LLaMA model with an 8-bit COAP optimizer, execute:

```bash
# LLaMA-7B, COAP-AdamW (8-bit), 8xH100, 1 Node
bash scripts/llama_7b.sh
```
