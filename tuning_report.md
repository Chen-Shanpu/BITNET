# BitNet Kernel Tuning Report

Generated on 2026-03-12 01:04:36 UTC

## Hardware Configuration

| Parameter | Value |
|-----------|-------|
| CPU Model | AMD EPYC 7763 64-Core Processor |
| CPU Architecture | x86_64 |
| Physical Cores | 4 |
| Logical Cores | 4 |
| RAM | 15.62 GB |
| OS | Linux #17~24.04.1-Ubuntu SMP Mon Dec  1 20:10:50 UTC 2025 |
| Python | 3.12.3 |

## Models

- [Llama-80M](#llama-80m)
- [Llama-120M](#llama-120m)
- [Llama-300M](#llama-300m)
- [Llama-500M](#llama-500m)
- [Llama-700M](#llama-700m)
- [Llama-1B](#llama-1b)

## Llama-80M

### Model Configuration

| Parameter | Value |
|-----------|-------|
| hidden_size | 512 |
| intermediate_size | 1408 |
| num_hidden_layers | 8 |
| num_attention_heads | 8 |
| num_key_value_heads | 8 |
| vocab_size | 32000 |

### Unique Linear Layer Shapes

| # | M (out) | K (in) |
|---|---------|--------|
| 0 | 512 | 512 |
| 1 | 1408 | 512 |
| 2 | 512 | 1408 |
| 3 | 32000 | 512 |

### TL1 (ARM NEON) Tuning Results

| M | K | BM | BK | bm | Candidates |
|---|---|----|----|-----|------------|
| 512 | 512 | 128 | 64 | 64 | 36 |
| 1408 | 512 | 128 | 64 | 64 | 20 |
| 512 | 1408 | 128 | 64 | 64 | 27 |
| 32000 | 512 | 128 | 64 | 64 | 40 |

Config saved to: `tuned_kernels/Llama-80M/kernel_config_tl1.ini`

### TL2 (x86 AVX2) Tuning Results

| M | K | BM | BK | bm | Candidates |
|---|---|----|----|-----|------------|
| 512 | 512 | 128 | 96 | 32 | 10 |
| 1408 | 512 | 128 | 96 | 32 | 6 |
| 512 | 1408 | 128 | 96 | 32 | 10 |
| 32000 | 512 | 128 | 96 | 32 | 12 |

Config saved to: `tuned_kernels/Llama-80M/kernel_config_tl2.ini`

---

## Llama-120M

### Model Configuration

| Parameter | Value |
|-----------|-------|
| hidden_size | 768 |
| intermediate_size | 2048 |
| num_hidden_layers | 12 |
| num_attention_heads | 12 |
| num_key_value_heads | 12 |
| vocab_size | 32000 |

### Unique Linear Layer Shapes

| # | M (out) | K (in) |
|---|---------|--------|
| 0 | 768 | 768 |
| 1 | 2048 | 768 |
| 2 | 768 | 2048 |
| 3 | 32000 | 768 |

### TL1 (ARM NEON) Tuning Results

| M | K | BM | BK | bm | Candidates |
|---|---|----|----|-----|------------|
| 768 | 768 | 128 | 64 | 64 | 44 |
| 2048 | 768 | 128 | 64 | 64 | 36 |
| 768 | 2048 | 128 | 128 | 64 | 44 |
| 32000 | 768 | 128 | 64 | 64 | 40 |

Config saved to: `tuned_kernels/Llama-120M/kernel_config_tl1.ini`

### TL2 (x86 AVX2) Tuning Results

| M | K | BM | BK | bm | Candidates |
|---|---|----|----|-----|------------|
| 768 | 768 | 128 | 96 | 32 | 12 |
| 2048 | 768 | 128 | 96 | 32 | 10 |
| 768 | 2048 | 128 | 96 | 32 | 12 |
| 32000 | 768 | 128 | 96 | 32 | 12 |

Config saved to: `tuned_kernels/Llama-120M/kernel_config_tl2.ini`

---

## Llama-300M

### Model Configuration

| Parameter | Value |
|-----------|-------|
| hidden_size | 1024 |
| intermediate_size | 2816 |
| num_hidden_layers | 16 |
| num_attention_heads | 16 |
| num_key_value_heads | 16 |
| vocab_size | 32000 |

### Unique Linear Layer Shapes

| # | M (out) | K (in) |
|---|---------|--------|
| 0 | 1024 | 1024 |
| 1 | 2816 | 1024 |
| 2 | 1024 | 2816 |
| 3 | 32000 | 1024 |

### TL1 (ARM NEON) Tuning Results

| M | K | BM | BK | bm | Candidates |
|---|---|----|----|-----|------------|
| 1024 | 1024 | 128 | 64 | 64 | 36 |
| 2816 | 1024 | 256 | 64 | 32 | 28 |
| 1024 | 2816 | 128 | 128 | 64 | 36 |
| 32000 | 1024 | 128 | 64 | 64 | 40 |

Config saved to: `tuned_kernels/Llama-300M/kernel_config_tl1.ini`

### TL2 (x86 AVX2) Tuning Results

| M | K | BM | BK | bm | Candidates |
|---|---|----|----|-----|------------|
| 1024 | 1024 | 128 | 96 | 32 | 10 |
| 2816 | 1024 | 256 | 96 | 32 | 8 |
| 1024 | 2816 | 128 | 96 | 32 | 10 |
| 32000 | 1024 | 128 | 96 | 32 | 12 |

Config saved to: `tuned_kernels/Llama-300M/kernel_config_tl2.ini`

---

## Llama-500M

### Model Configuration

| Parameter | Value |
|-----------|-------|
| hidden_size | 1280 |
| intermediate_size | 3456 |
| num_hidden_layers | 20 |
| num_attention_heads | 20 |
| num_key_value_heads | 20 |
| vocab_size | 32000 |

### Unique Linear Layer Shapes

| # | M (out) | K (in) |
|---|---------|--------|
| 0 | 1280 | 1280 |
| 1 | 3456 | 1280 |
| 2 | 1280 | 3456 |
| 3 | 32000 | 1280 |

### TL1 (ARM NEON) Tuning Results

| M | K | BM | BK | bm | Candidates |
|---|---|----|----|-----|------------|
| 1280 | 1280 | 128 | 64 | 64 | 40 |
| 3456 | 1280 | 128 | 64 | 64 | 36 |
| 1280 | 3456 | 128 | 128 | 64 | 30 |
| 32000 | 1280 | 128 | 64 | 64 | 40 |

Config saved to: `tuned_kernels/Llama-500M/kernel_config_tl1.ini`

### TL2 (x86 AVX2) Tuning Results

| M | K | BM | BK | bm | Candidates |
|---|---|----|----|-----|------------|
| 1280 | 1280 | 128 | 96 | 32 | 12 |
| 3456 | 1280 | 128 | 96 | 32 | 10 |
| 1280 | 3456 | 128 | 96 | 32 | 12 |
| 32000 | 1280 | 128 | 96 | 32 | 12 |

Config saved to: `tuned_kernels/Llama-500M/kernel_config_tl2.ini`

---

## Llama-700M

### Model Configuration

| Parameter | Value |
|-----------|-------|
| hidden_size | 1536 |
| intermediate_size | 4096 |
| num_hidden_layers | 24 |
| num_attention_heads | 12 |
| num_key_value_heads | 12 |
| vocab_size | 32000 |

### Unique Linear Layer Shapes

| # | M (out) | K (in) |
|---|---------|--------|
| 0 | 1536 | 1536 |
| 1 | 4096 | 1536 |
| 2 | 1536 | 4096 |
| 3 | 32000 | 1536 |

### TL1 (ARM NEON) Tuning Results

| M | K | BM | BK | bm | Candidates |
|---|---|----|----|-----|------------|
| 1536 | 1536 | 128 | 64 | 64 | 52 |
| 4096 | 1536 | 256 | 64 | 32 | 36 |
| 1536 | 4096 | 128 | 128 | 64 | 52 |
| 32000 | 1536 | 128 | 64 | 64 | 40 |

Config saved to: `tuned_kernels/Llama-700M/kernel_config_tl1.ini`

### TL2 (x86 AVX2) Tuning Results

| M | K | BM | BK | bm | Candidates |
|---|---|----|----|-----|------------|
| 1536 | 1536 | 128 | 96 | 32 | 14 |
| 4096 | 1536 | 256 | 96 | 32 | 10 |
| 1536 | 4096 | 128 | 96 | 32 | 14 |
| 32000 | 1536 | 128 | 96 | 32 | 12 |

Config saved to: `tuned_kernels/Llama-700M/kernel_config_tl2.ini`

---

## Llama-1B

### Model Configuration

| Parameter | Value |
|-----------|-------|
| hidden_size | 2048 |
| intermediate_size | 5632 |
| num_hidden_layers | 22 |
| num_attention_heads | 32 |
| num_key_value_heads | 32 |
| vocab_size | 32000 |

### Unique Linear Layer Shapes

| # | M (out) | K (in) |
|---|---------|--------|
| 0 | 2048 | 2048 |
| 1 | 5632 | 2048 |
| 2 | 2048 | 5632 |
| 3 | 32000 | 2048 |

### TL1 (ARM NEON) Tuning Results

| M | K | BM | BK | bm | Candidates |
|---|---|----|----|-----|------------|
| 2048 | 2048 | 128 | 128 | 64 | 36 |
| 5632 | 2048 | 256 | 128 | 32 | 36 |
| 2048 | 5632 | 128 | 128 | 64 | 36 |
| 32000 | 2048 | 128 | 128 | 64 | 40 |

Config saved to: `tuned_kernels/Llama-1B/kernel_config_tl1.ini`

### TL2 (x86 AVX2) Tuning Results

| M | K | BM | BK | bm | Candidates |
|---|---|----|----|-----|------------|
| 2048 | 2048 | 128 | 96 | 32 | 10 |
| 5632 | 2048 | 256 | 96 | 32 | 10 |
| 2048 | 5632 | 128 | 96 | 32 | 10 |
| 32000 | 2048 | 128 | 96 | 32 | 12 |

Config saved to: `tuned_kernels/Llama-1B/kernel_config_tl2.ini`

---
