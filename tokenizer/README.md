# Ludwig 2's tokenizer (64K)

Fixed on November 14, 2026. **`tokenizer.json` does not change after the trunk's first step** (design §3): sha1 `0b7da8fdb0622fe185f8c7117b0eadd7c35568ba`.

| File | What it is |
|---|---|
| `tokenizer.json` | The tokenizer (Hugging Face `tokenizers`): vocabulary, merges, pre-tokenizer and control tokens |
| `tokenizer_config.json`, `special_tokens_map.json` | For transformers: EOS, padding and the chat template |
| `tokenizer_meta.json` | The recipe, the 23 sources of the sample with their files and licenses, versions and metrics |

Trained by [`scripts/train_tokenizer.py`](../scripts/train_tokenizer.py) (v1.2) on a Kaggle CPU in 56 minutes. The raw output of that run, with its report, is in [`results/tokenizador_64k/con_transweb/`](../results/tokenizador_64k/con_transweb/).

## Recipe

- **Model:** byte-level BPE with **65,536 tokens**, so every id fits in a uint16. All 256 bytes are in the vocabulary, and there is no normalizer.
- **Pre-tokenizer:** Qwen2's split regex, which splits digits one by one. llama.cpp implements this regex as `qwen2`.
- **Training:**
  - Trained on whole documents (up to 20,000 characters), with a minimum frequency of 2.
  - The sample is 1.2 GB from 23 sources of the trunk mix (v3):
    - English web;
    - Spanish: FineWeb2-HQ, TransWeb-Edu and Wikipedia;
    - permissively licensed code from Stack-Edu, fetched from Software Heritage, plus HTML and CSS from The Stack v1;
    - math and reasoning traces.
  - Each source's last file is held out for validation and never enters training.
- **Not sampled:** three sources of the mix: general StackExchange, notebooks and English–Spanish parallel text.
- **Control-token strings** are removed from the sample. The pipeline places those ids, so the strings get no merges.

## Control tokens

| id | Token | Use |
|---|---|---|
| 0 | `<\|endoftext\|>` | EOS: the end of every document |
| 1 | `<\|pad\|>` | Padding |
| 2, 3 | `<\|im_start\|>`, `<\|im_end\|>` | Chat turns (ChatML) |
| 4, 5 | `<think>`, `</think>` | Reasoning |
| 6–9 | `<tool_call>`, `</tool_call>`, `<tool_response>`, `</tool_response>` | Tool calls |
| 10–13 | `<\|fim_prefix\|>`, `<\|fim_middle\|>`, `<\|fim_suffix\|>`, `<\|fim_pad\|>` | Fill in the middle |
| 14, 15 | `<\|repo_name\|>`, `<\|file_sep\|>` | Repository-level code |
| 16, 17 | `<\|reserved_0\|>`, `<\|reserved_1\|>` | Start and end of a MeCo metadata span in the trunk's shards (T1, P21); out of the loss |
| 18–255 | `<\|reserved_2\|>` … `<\|reserved_239\|>` | Reserved |

`<think>` and the tool tags (ids 4–9) are added but not special, as in Qwen3. Decoding with `skip_special_tokens=True` keeps them, so a client still sees the reasoning and the calls.

The chat template in `tokenizer_config.json` is plain ChatML. It is not part of the vocabulary, so SFT can extend it without touching `tokenizer.json`.

## Two modes

- **Data mode** (pre-training) is `ludwig_lm.data.load_tokenizer("tokenizer/tokenizer.json")`.
  - No text produces a control id: `<|endoftext|>` or `<think>` written in a web page stay text.
  - The pipeline adds the ids: EOS after each document now; in T1, FIM on code and the chat format of instruction and reasoning data.
- **Chat mode** (SFT and serving) is `AutoTokenizer.from_pretrained("tokenizer")`. Control tokens written in the text are tokens, and `apply_chat_template` uses the template above.

## Compression

Bytes per token on each source's held-out text ([`compresion.csv`](../results/tokenizador_64k/con_transweb/compresion.csv)). The trunk figure weighs each domain by its share of the trunk's tokens: English 36%, code 28%, Spanish 24%, math 10%, reasoning 2%.

| Bytes per token | Ludwig 2 (64K) | SmolLM2 (49K) | OLMoE (50K) | Qwen3 (151K) |
|---|---|---|---|---|
| English | 4.496 | 4.414 | 4.499 | 4.579 |
| Spanish | **4.598** | 2.934 | 3.350 | 3.790 |
| Code | 3.708 | 3.189 | 3.177 | 3.869 |
| Math | 3.413 | 3.124 | 3.483 | 3.367 |
| Reasoning traces | 2.877 | 2.623 | 2.851 | 2.803 |
| **Trunk mix** | **4.159** | 3.551 | 3.719 | 4.034 |
| Tokens per word, English / Spanish | 1.37 / 1.33 | 1.38 / 2.08 | 1.36 / 1.83 | 1.35 / 1.61 |

- **Unseen text:** on this repository's documents and code, Ludwig 2 uses 20% fewer tokens than SmolLM2 in Spanish, 7% fewer in English and 12% fewer in Python.
- **Unused tokens:** 236 tokens (0.36%) do not appear in 100 MB of text.
- **Sample size:** going from 0.3 GB to 1.2 GB of sample moves the trunk figure only from 4.151 to 4.159.

## llama.cpp

llama.cpp recognizes a BPE pre-tokenizer by hashing how it encodes a fixed test string. This vocabulary is new, so llama.cpp's stock converter stops on it.

The split regex is Qwen2's, so [`scripts/convert_gguf.py`](../scripts/convert_gguf.py) converts it with `LUDWIG_TOKENIZER_PRE=qwen2`. T0.1b ([`scripts/smoke_tokenizer.py`](../scripts/smoke_tokenizer.py), [`results/t0_1b_tokenizer.json`](../results/t0_1b_tokenizer.json)) checked the result in Ludwig 2's real format on November 14, 2026, with llama.cpp `61dbf49`:

- **Tokenization:** 17 of 17 strings tokenize as in Hugging Face: prose in English and Spanish, code, LaTeX, and the chat, reasoning, tool-call and FIM markup.
- **Special tokens on load:**
  - BOS 0, EOS 0, end of turn 3 and padding 1.
  - FIM: prefix 10, suffix 12, middle 11, padding 13, repository 14, file separator 15.
  - Generation stops at 0, 3, 13, 14 and 15.
- **The hash** llama.cpp does not know yet is `b01d3ee07b08b215bfe2dbf0240a1b214920f74417cbfb91ba88028fe85a0481`.
