"""Cómo cargar el tokenizador 64K de Ludwig 2.

para_datos(): para TEXTO CRUDO (preentrenamiento). Ningún texto produce tokens de control:
    un "<|endoftext|>" o un "<think>" escritos en una página web quedan como texto normal.
    Los ids de control los pone el pipeline: EOS al final de cada documento, FIM, chat.
para_chat(): el normal, para SFT, servir y chatear. Ahí "<|im_start|>" sí es un token.
"""
import json
from tokenizers import Tokenizer

IDS = {
    "<|endoftext|>": 0,
    "<|pad|>": 1,
    "<|im_start|>": 2,
    "<|im_end|>": 3,
    "<think>": 4,
    "</think>": 5,
    "<tool_call>": 6,
    "</tool_call>": 7,
    "<tool_response>": 8,
    "</tool_response>": 9,
    "<|fim_prefix|>": 10,
    "<|fim_middle|>": 11,
    "<|fim_suffix|>": 12,
    "<|fim_pad|>": 13,
    "<|repo_name|>": 14,
    "<|file_sep|>": 15
}
EOS, PAD = IDS["<|endoftext|>"], IDS["<|pad|>"]

def para_chat(ruta="tokenizer.json"):
    return Tokenizer.from_file(ruta)

def para_datos(ruta="tokenizer.json"):
    j = json.loads(open(ruta, encoding="utf-8").read())
    for a in j["added_tokens"]:
        a["special"] = True
    tk = Tokenizer.from_str(json.dumps(j, ensure_ascii=False))
    tk.encode_special_tokens = True
    return tk

def documento(tk_datos, texto):
    """Un documento de preentrenamiento: sus tokens y el EOS al final."""
    return tk_datos.encode(texto, add_special_tokens=False).ids + [EOS]
