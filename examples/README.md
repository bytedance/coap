# Examples


| Model            | GPU  | Optimizer       | Optimizer Memory (MB) | Time (h)   | Evaluation  |
|:----------------:|:----:|:---------------:|:---------------------:|:----------:|:-----------:|
| DDPM (CIFAR-10)  | V100 | COAP-AdamW      | 213.99                | -          | FID = 5.67  |
| DDPM (CELEBAHQ)  | V100 | COAP-AdamW      | 522.77                | -          | FID = 15.59 |
| ControlNet-XL    | A100 | COAP-Adafactor  | 3704.91 (3.62G)       | 10         | -           |
| LLAMA-1B         | H100 | COAP-AdamW      | 1988.34 (1.94G)       | 24.8 (89k) | PPL = 15.62 |
| LLAMA-7B         | H100 | COAP-AdamW-8bit | 5380.51 (5.25G)       | 55.2 (89k) | PPL = 15.08 |


**Credits to the following repositories:**
[DDPM](https://github.com/huggingface/diffusers/tree/main/examples/unconditional_image_generation), 
[ControlNet-XL](https://github.com/huggingface/diffusers/tree/main/examples/controlnet), 
[LLAMA](https://github.com/jiaweizzhao/GaLore).