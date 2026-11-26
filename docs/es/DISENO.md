# Ludwig 2 — Documento de diseño

**Versión:** 1.2 · 13 de noviembre de 2026 (el cuerpo del documento todavía describe la 1.1: ver los recuadros de la v1.2 y de la T0.2 v2)
**Estado:** diseño cerrado para la v1 (25B tokens). La v1.2 cambió el formato a AFMoE. Fases hasta 1T planificadas, sin obligación de llegar.

> **Cambios de la v1.1** (tras la crítica externa y la primera prueba real):
> 1. La T0.1 encontró que llama.cpp **sin parche se cae** con el formato Qwen3-Next de atención completa; con un parche de 3 líneas pasan todas las pruebas, MTP incluido (2.5).
> 2. El formato se decide con un A/B en la T2: A = Qwen3-Next (compuerta + softmax) contra B = GLM-4.5 (sin compuerta, router sigmoide). El A/B también resuelve la duda del router.
> 3. Momento de Muon en bf16 por defecto; plan B de 32 expertos si no queda margen de memoria (5.2).
> 4. Compuerta G0: ninguna hora de tronco sin una prueba sostenida de varias horas con guardado y reanudación (5.7).
> 5. Español del 4% al 10% en el tronco y las ramas, y 25–30% en el SFT (4.9).
> 6. Topes y diversidad para los datos sintéticos (4.10); auditoría de licencias antes de publicar pesos (10); compuertas para seguir de una fase a la siguiente (7.1).

> **✅ v1.2, aprobada por Pipe el 12 de noviembre de 2026** (detalle y evidencia en [`BITACORA.md`](BITACORA.md) y [`../DECISIONS.md`](../DECISIONS.md#v12--approved-by-pipe-on-november-12-2026)):
> 1. **Banco v1 en 2×T4:** 26 × 1024 con 40 expertos va a **3,856 tokens/s**, unas 1,801 h para 25B (planear ~2,000 h). La forma y los 40 expertos quedan medidos. La memoria está justa: 13.8 GiB reservados de ~14.5. Entre las GPUs hay 3.2 GB/s.
> 2. **Formato AFMoE** (`AfmoeForCausalLM`, Arcee Trinity) en lugar de Qwen3-Next. Pasa la T0.1-C en llama.cpp **sin parche**. Tiene compuerta, router sigmoide con bias, normas sándwich, 3 capas locales (ventana 2048) por 1 global sin RoPE, y 1 capa densa. Son 1.906B totales y 0.491B activos. Pierde el MTP.
> 3. **Balanceo con SMEBU** en lugar del bias de signo fijo. **AttnRes y KDA quedan fuera de la v1**: solo existen en el formato Kimi-K3.
> 4. **Entrenamiento:** la clase de transformers más nuestras piezas de velocidad (bmm con relleno, residuo en FP32, gradientes en FP16, prefetch, momento en la RAM opcional). Lo mide la T0.2 v2.
>
> El resto de este documento todavía describe la v1.1 y se reescribe con los números de la T0.2 v2. Donde se contradigan, manda este recuadro, junto con `DECISIONS.md`.

> **Después de la T0.2 v2 (13 de noviembre de 2026)** (evidencia en `results/t0_2v2_intento2/` y la bitácora; decisiones P2a, P5a, D16a, P6–P8 de [`../DECISIONS.md`](../DECISIONS.md)):
> 1. **Configuración del entrenamiento:** un matmul por experto (el loop, +17.3% frente al bmm con relleno), residuo en FP32, gradientes en FP16, prefetch, **momento de Muon en la RAM** (−1.57 GiB reservados por −3.4%) y 8 secuencias de 2K por GPU. Medidas por separado, juntas deberían dar ~3,300 tokens/s (~2,100 h para 25B) con ~12.7 GiB reservados. 4K y 16K caben.
> 2. **SMEBU con λ = 0.01:** dejó al experto más cargado en 2.41× el promedio, contra 6.86× con 0.001 y 7.57× con el signo fijo. Se revisa con texto real en la T0.3 + T2.
> 3. **El tope de memoria de la G0 (13.5 GiB) es de memoria reservada** (5.7, punto 3).
> 4. **La T0.3 y la T2 van juntas y antes del tokenizador:** tres corridas de 400 pasos (~105M tokens) con LR ×0.5, ×1 y ×2 alrededor de la receta del Ludwig 1 (Muon 1.5e-3, AdamW 5e-4, weight decay con el LR de AdamW), con FineWeb-Edu y el tokenizador de SmolLM2 como provisionales, comparadas en bits por byte. La regla de decisión está en `DECISIONS.md` (P6) y en el notebook `t0_3_t2_lr_2xT4.ipynb`.
> 5. **El bucle de entrenamiento existe** (`scripts/train.py`, P7), con reanudación verificada bit a bit. Al escribirlo aparecieron dos errores que ya están corregidos (P8): transformers dejaba fijo el embedding del EOS (id 0) y la carga de checkpoints pasaba el momento de Muon a FP32.

> **Después de la primera ronda de la T0.3 + T2 (14 de noviembre de 2026)** (`results/t0_3_t2/`; decisiones P6a, P2b y P9):
> 1. **Las tres corridas son estables** y ×2 gana en el horizonte corto (1.1516 bits por byte, 6.1% por debajo de ×1).
> 2. **Las corridas cortas favorecen los LR altos:** el mejor LR baja con la duración del entrenamiento, con exponente ~0.32 (medido para AdamW y para Muon). La segunda ronda lo mide aquí: ×4 a 400 pasos, y ×1 y ×2 seguidas hasta 1,200. `scripts/lr_horizon.py` aplica la regla P9; con lo que hay hoy da ×1, la receta del Ludwig 1.
> 3. **El balanceo se juzga en la ventana de 50 pasos**, por capa, y **λ escala con el LR** (0.01 × el multiplicador). En ×2 hubo un episodio de ~4× que SMEBU corrigió en ~30 pasos.
> 4. **La diferencia de velocidad entre corridas es de las máquinas**, no del desbalance: dentro de cada corrida la velocidad no cambia con la carga.

> **Datos del tronco (15 de noviembre de 2026)** ([`results/datos_tronco_v3/`](../../results/datos_tronco_v3/); decisiones P13–P17 de [`../DECISIONS.md`](../DECISIONS.md)). La sección 4 ya está reescrita con esto:
> 1. **Mezcla v3,** elegida por Pipe el 13 de noviembre: inglés 36%, español 24%, código 28%, matemática 10% e instrucción 2%, dimensionada para 100B.
> 2. **El código** son los archivos permisivos de Stack-Edu, con el contenido de `common-pile/stackv2_edu_filtered`, que trae ~90% de los de Python, JS y TS sin pasar por Software Heritage. HTML y CSS salen de The Stack v1.
> 3. **Fuentes y repartos cerrados:** DCLM se queda, y StackExchange, los notebooks, el texto paralelo y la instrucción ya tienen fuente.
> 4. **Topes por tipo** para el texto escrito por modelos. La v3 queda en 18.5% del tronco.
> 5. **Sin prueba proxy de mezcla antes del tronco:** la reemplaza la evaluación por dominio desde el paso 0.

> **El tronco desde cero (25 de noviembre de 2026)** (`results/tronco/`; decisiones P29–P31 de [`../DECISIONS.md`](../DECISIONS.md)):
> 1. **El primer tronco se rompió en el paso ~1250.** En las primeras 5 capas MoE, el experto más cargado de cada paso pasó de ~1.5 a ~7.8 veces la media, y los acaparadores se turnaban. Se detuvo por picos de pérdida en el paso 3348.
> 2. **La causa (P29):** el z-loss del router, pensado para routers softmax, empujó los scores sigmoide hacia la cola. Ahí se achica la diferencia entre el 4.º y el 5.º experto de cada token, y el paso del sesgo de SMEBU (λ 0.01) la supera: un solo paso re-enruta a casi todos los tokens a la vez. El quiebre empieza en el mismo nivel del router en dos corridas con calendarios muy distintos, y un modelo de juguete del lazo lo reproduce (`scripts/toy_router_loop.py`).
> 3. **El tronco empieza de cero como `tronco2`** (P30, `configs/tronco_v2.json`):
>    - sin z-loss del router;
>    - SMEBU con λ 0.01 hasta el paso 500 y 0.001 después;
>    - `seq_aux` 1e-3;
>    - cada paso registra λ contra la brecha más chica entre las capas, y la corrida se detiene y guarda si el balanceo se rompe.
>
>    El LR, los datos y el resto no cambian.
> 4. **La sonda (P31)** mide el router del checkpoint 3270 para confirmar el diagnóstico, sin frenar al `tronco2`.

> **El tronco desde cero otra vez, con 3 capas densas (25 de noviembre de 2026, en la noche)** (decisiones P32–P33 de [`../DECISIONS.md`](../DECISIONS.md)):
> 1. **El `tronco2` estaba sano en el paso 220,** pero el nivel de su router baja igual, aun sin z-loss, a ~45% del ritmo del primer tronco.
> 2. **La literatura señala las primeras capas MoE:**
>    - Sigma-MoE-Tiny vio ahí crecer el sesgo hasta que decidía solo, y lo arregló activando más expertos en esas capas;
>    - Trinity Large pasó de 3 a 6 capas densas como parte del arreglo de un colapso igual al nuestro;
>    - DeepSeek-V3 usa 3.
> 3. **El `tronco3` empieza de cero con 3 capas densas en vez de 1, y 44 expertos por capa MoE en vez de 40:** 1.938B parámetros totales (antes 1.906B) y los mismos 0.491B activos. Los 44 expertos son la condición de Pipe para reiniciar: no menos parámetros totales que el `tronco2`. El resto de la receta del `tronco2` no cambia, y el formato sigue sirviendo en llama.cpp sin cambios (`results/t0_1c_afmoe_d3.json`).
> 4. **El piso para el nivel del router** (P33) queda listo y apagado.

> **El piso del router, encendido (26 de noviembre de 2026)** (decisión P34 de [`../DECISIONS.md`](../DECISIONS.md)):
> 1. **En el turno 1 del `tronco3`, el nivel del router siguió bajando** aun sin z-loss: `z_router` pasó de 15.84 a 10.78 en 480 pasos, cada vez más rápido mientras sube el LR.
> 2. **El piso vale cero mientras el lse medio de cada capa MoE esté sobre τ = 1.5,** así que no cambia nada hasta que hace falta, y actúa solo, también de noche.
> 3. **Con τ = 1.5, cada capa queda en z ≥ ~2.1,** sobre los quiebres que vimos con λ de 0.01 y 0.02. La alarma de `z_router` (1.5) queda como aviso de que el piso falló.

**Leyenda:** ✅ decidido · ❓ decide Pipe · 📏 se mide antes de fijarlo · 🧪 se prueba en corrida corta · 🟢🟡🔴 licencia de datos (ver 4.8)

---

## 0. Resumen

- **Qué es:** un MoE de **1.96B parámetros totales y 0.49B activos**: 26 capas × ancho 1024, con atención completa y compuerta en todas las capas, entrenado desde cero en 2×T4 de Kaggle con relevo entre cuentas. Es una arquitectura propia que se guarda en el formato de la implementación de Qwen3-Next (`Qwen3NextForCausalLM` con las 26 capas `full_attention`); no es una reproducción de Qwen3-Next, que mezcla Gated DeltaNet con atención.
- **De dónde sale:** la forma de MobileMoE-M (Meta, 2026), con la atención de Qwen3-Next (QK-norm + compuerta), RoPE parcial, Muon, WSD, un módulo MTP para decodificación especulativa y 16K de contexto.
- **v1 (25B tokens):** ~1,750–2,800 h de 2×T4 de preentrenamiento, o sea 12–20 semanas de relevo 24/7.
- **Programa:** 25B → 50B → 100B → 250B → 500B → 750B → 1T sobre **un solo tronco**. Cada versión es una rama de enfriamiento. Hasta 100B cabe en Kaggle (~1–1.7 años acumulados). Desde 250B hace falta TRC o pagar GPU (sección 7).
- **Razonamiento:** sí, desde el día uno: `<think>` en el tokenizador, matemática y Python en las ramas, SFT con trazas cortas. Expectativas honestas en la sección 8.
- **Destilación:** de datos sí; de logits no (sección 9).
- **Todo abierto:** pesos, código, receta y manifiestos de datos, logs, checkpoints intermedios y registros de producción de entropía (sección 10).

---

## 1. Objetivo y criterios

**Objetivo:** el mejor modelo de ~0.5B activos que se pueda entrenar gratis en 2×T4, que corra en cualquier PC y que muestre todos los pasos.

**Objetivos de capacidad (Pipe, 13 de noviembre de 2026):** "totalmente bilingüe" en inglés y español; que "domine como un nivel principiante" Python y JS/TS; y que "razone y sepa matemáticas básicas".

**Criterios para publicar cada versión:**
1. **Formato:** transformers y llama.cpp dan los mismos logits (prueba de la sección 12).
2. **Progreso:** mejora a la versión anterior en pérdida de validación y en los benchmarks grandes (HellaSwag, MMLU, GSM8K) por encima del ruido (5.5).
3. **Comparación justa, por tokens:** contra checkpoints intermedios públicos con un presupuesto parecido, por ejemplo OLMoE-1B-7B y Pythia-410M/1B en su checkpoint de ~25B tokens 📏. No contra MobileMoE-M de tú a tú: Meta entrenó ~260 veces más tokens.

**Fuera de alcance en la v1:** competir en benchmarks crudos con MobileMoE-M, Qwen3-0.6B o MobileLLM-R1; contexto de más de 32K; multimodal.

---

## 2. Arquitectura ✅

### 2.1 Ficha

| Parte | Valor | Por qué |
|---|---|---|
| Formato | `Qwen3NextForCausalLM`, `layer_types` = 26 × `full_attention` (candidato A; ver 2.5) | Compuerta + QK-norm + MoE con experto compartido en un formato que carga transformers; llama.cpp necesita un parche (2.5) |
| Ancho × capas | 1024 × 26 | La forma de MobileMoE-M; mismo ancho que Ludwig 1 |
| Atención | GQA: 16 cabezas Q, 4 KV, de 64 dimensiones | Caché chica; igual que Ludwig 1 |
| QK-norm | RMSNorm por cabeza en q y k | Logits de atención acotados en FP16 |
| Compuerta | Sigmoide por elemento a la salida de la atención (q_proj saca el doble: query + compuerta) | Sin attention sinks ni activaciones masivas; permite LR más alto |
| Posiciones | RoPE parcial al 50% (32 de 64 dimensiones), θ = 500,000 | Extensión de contexto más suave |
| FFN | MoE en las 26 capas: 40 expertos SwiGLU de 512, top-4, + 1 compartido SwiGLU de 2048 con compuerta sigmoide | Expansión 5.5 (la zona útil de MobileMoE es 4–8). llama.cpp exige expertos en todas las capas de este formato |
| Router | Softmax en FP32, top-4 normalizado | Lo que fija el formato |
| Balanceo | Pérdida de balanceo de lote global (coef. 1e-3) + router z-loss 1e-4; sin descartar tokens | |
| Normas | RMSNorm centrada en cero (escala 1 + w), eps 1e-6, pre-norm | Formato Qwen3-Next |
| Embeddings | 65,536 tokens, atados a la cabeza de salida | La cabeza es el 12% del cómputo (en MobileMoE-M, ~23%) |
| MTP | 1 módulo: proyección 2048→1024, 2 normas, 1 capa igual a las demás y la cabeza compartida | Decodificación especulativa en llama.cpp |
| Contexto | 2K → 4K → 16K (v1) → 32K (desde 250B) | |

### 2.2 Números

| | Valor |
|---|---|
| Parámetros totales | 1.963B (+75M del MTP = 2.04B) |
| Activos por token | 0.491B (0.424B sin la cabeza) |
| Expansión de la FFN | (40×512 + 2048) / (4×512 + 2048) = 5.5 |
| FLOPs por token de entrenamiento | 3.27G a 2K; 3.60G a 4K |
| Caché KV | 13,312 valores por token (26.6 KB en fp16): 436 MB a 16K, 232 MB en q8_0 |
| Tamaño en GGUF | ~2.2 GB en Q8_0; ~1.15 GB en Q4_0 |
| Memoria por GPU (2×T4) | ~9.1 GiB de estados (momento en bf16) + 3–4.5 GiB de activaciones, buffers de FSDP y contexto CUDA ≈ 12.2–13.6 GiB de 15. Se mide en la T0.2 (5.2) |

Comprobación: con esta misma fórmula, MobileMoE-M (60 expertos, 128K de vocabulario, sin compuerta) da exactamente 2.82B / 528M, sus cifras publicadas.

### 2.3 `config.json` (borrador validado con transformers 5.22.2)

```json
{
  "architectures": ["Qwen3NextForCausalLM"],
  "model_type": "qwen3_next",
  "vocab_size": 65536,
  "hidden_size": 1024,
  "num_hidden_layers": 26,
  "layer_types": [
    "full_attention", "full_attention", "full_attention", "full_attention", "full_attention",
    "full_attention", "full_attention", "full_attention", "full_attention", "full_attention",
    "full_attention", "full_attention", "full_attention", "full_attention", "full_attention",
    "full_attention", "full_attention", "full_attention", "full_attention", "full_attention",
    "full_attention", "full_attention", "full_attention", "full_attention", "full_attention",
    "full_attention"
  ],
  "num_attention_heads": 16,
  "num_key_value_heads": 4,
  "head_dim": 64,
  "rope_parameters": {"rope_type": "default", "rope_theta": 500000.0, "partial_rotary_factor": 0.5},
  "num_experts": 40,
  "num_experts_per_tok": 4,
  "moe_intermediate_size": 512,
  "shared_expert_intermediate_size": 2048,
  "norm_topk_prob": true,
  "router_aux_loss_coef": 0.001,
  "decoder_sparse_step": 1,
  "mlp_only_layers": [],
  "tie_word_embeddings": true,
  "rms_norm_eps": 1e-6,
  "max_position_embeddings": 16384,
  "attention_bias": false,
  "hidden_act": "silu",
  "eos_token_id": 0
}
```

- `eos_token_id` = 0 es el `<|endoftext|>` del tokenizador de 64K 📏 (confirmar los ids).
- Los campos `linear_*` quedan en sus valores por defecto: no se usan porque no hay capas lineales.
- **Pesos del MTP:** `mtp.fc`, `mtp.pre_fc_norm_embedding`, `mtp.pre_fc_norm_hidden`, `mtp.layers.0.*` y `mtp.norm`. transformers los ignora al cargar. El conversor de llama.cpp los pasa al GGUF como capa NextN, y se usan con `--spec-type draft-mtp`. **Ya está probado corriéndolo** (T0.1, sección 2.5).
- **Candidato B de formato:** GLM-4.5 (`Glm4MoeForCausalLM`) sin compuerta: atención pura en llama.cpp, MTP soportado y router sigmoide con sesgo (la receta de MobileMoE).

### 2.4 Frente a MobileMoE-M

| | MobileMoE-M (Meta) | Ludwig 2 | Gana |
|---|---|---|---|
| Forma | 1024 × 26, GQA 16/4 | Igual | Igual |
| Totales / activos | 2.82B / 528M | 1.96B / 0.49B | Meta en capacidad; el tuyo cabe en la T4 |
| Vocabulario | 128K (Llama 3) | 64K propio | Tú: cabeza a la mitad |
| QK-norm y compuerta | No las menciona | Las dos | Tú |
| Router | Sigmoide sin pérdida auxiliar | Softmax + balanceo global | Meta (receta validada) |
| Optimizador | AdamW, coseno | Muon, WSD | Tú |
| Precisión | BF16 | FP16 (la T4 no tiene BF16) | Meta |
| Tokens | ~6T + 500B de mid-training | 25B (v1) → 1T | Meta |
| Contexto | 8K | 16K → 32K | Tú |
| Decodificación especulativa | No | MTP | Tú |
| Dónde corre | ExecuTorch con op propia | transformers y llama.cpp | Tú |
| Publicación | El paper no lo dice | Todo abierto | Tú |

### 2.5 Formato: lo que encontró la T0.1 y cómo se decide

La T0.1 (`scripts/smoke_format.py`) arma un modelo diminuto con exactamente la estructura de Ludwig 2, lo exporta, lo convierte a GGUF y compara llama.cpp contra transformers. Resultados del 11 nov 2026 (llama.cpp `dd6b945`, transformers 5.22.2):

| Prueba | llama.cpp sin parche | Con el parche de Ludwig |
|---|---|---|
| A1 Exportar y recargar en transformers | ✅ logits idénticos | ✅ |
| A2 Convertir a GGUF con MTP | ✅ | ✅ |
| A3 Probabilidades del siguiente token | ❌ **llama-server se cae al cargar** (`GGML_ASSERT(buffer)`) | ✅ diferencia máx. 1.9e-6 |
| A4 Greedy token a token | ❌ | ✅ 4/4 idénticos |
| A5 Decodificación especulativa con MTP | ❌ | ✅ 4/4 idénticos; 58 de 91 borradores aceptados (64%) en el modelo diminuto |
| T1–T2 Tokenizador (pre-tokenizador `qwen2`, 9 textos) | ✅ | ✅ |

- **La causa:** llama.cpp trata la arquitectura `qwen3next` como híbrida aunque ninguna capa sea recurrente, y el tensor `s_copy` del estado recurrente queda sin memoria asignada. El parche (`patches/llama-cpp-hybrid-without-recurrent-layers.patch`, 3 líneas) solo salta ese paso cuando no hay estado que copiar. Se puede proponer a llama.cpp como PR, pero sus reglas exigen que quien lo envíe entienda cada línea, escriba él mismo el issue, la descripción y el commit, y declare el uso de IA (`patches/README.md`).
- **Lo que el parche no arregla:** llama.cpp sigue tratando el modelo como híbrido, así que `llama-server` no puede desplazar el contexto (context shift) y reutiliza prompts con checkpoints en vez de con la caché completa. Y otros motores (vLLM, MLX, Ollama) podrían tener supuestos parecidos.
- **La decisión, con datos (T2):** dos corridas cortas iguales (~0.3B tokens, mismo orden de datos y mismo LR) del modelo real:
  - **A** = formato Qwen3-Next: compuerta + router softmax con balanceo global.
  - **B** = formato GLM-4.5: sin compuerta, router sigmoide sin pérdida auxiliar (sesgo de DeepSeek), QK-norm, RoPE parcial, MTP.
  - **Regla:** si A le gana a B por ≥1.5% de pérdida de validación (o B se vuelve inestable en FP16), va A y se empuja el PR a llama.cpp. Si no, va B, que corre en todas partes sin parche.
  - Se miden también: balanceo de expertos, entropía del router, expertos infrautilizados, picos de pérdida y tokens/s.

---

## 3. Tokenizador ✅

- 📏 **Fijado el 14 nov 2026** en `tokenizer/` (sha1 `0b7da8fdb062`; P10 en `docs/DECISIONS.md`): 4.159 bytes por token en la mezcla del tronco (Qwen3 4.034, OLMoE 3.719, SmolLM2 3.551) y 1.33 tokens por palabra en español (SmolLM2 2.08). Receta, tokens de control y compresión por dominio en `tokenizer/README.md`.
- 📏 **T0.1b** (14 nov 2026; `scripts/smoke_tokenizer.py`, `results/t0_1b_tokenizer.json`): pasa en el formato AFMoE con llama.cpp `61dbf49`.
  - El pre-tokenizador es `qwen2`.
  - 17 de 17 textos tokenizan igual que en Hugging Face, con chat, `<think>`, herramientas y FIM.
  - llama.cpp detecta todos los ids especiales.
  - El hash del vocabulario es nuevo para llama.cpp: se convierte con `scripts/convert_gguf.py` hasta registrarlo en llama.cpp (un PR que escribe Pipe).
  - Sin `bos_token_id`, llama.cpp tomaba como BOS el id 11 (`<|fim_middle|>`); el config dice ahora 0, el EOS, que nunca se añade (P12).
- **La receta de 64K del plan del modelo grande**, con la mezcla v3 del tronco de Ludwig 2 (`scripts/train_tokenizer.py`, v1.2): los 256 bytes siempre en el vocabulario, sin normalización, entrenado con documentos enteros, con val aparte y en modo datos (el texto crudo nunca produce tokens de control).
- **Tokens especiales desde el día uno:** plantilla de chat, `<think>` y `</think>` (id 4…), FIM (ids 10–13), llamadas a herramientas y fin de texto (id 0) 📏.
- **Ventaja de compartirlo con el modelo grande:** si algún día existe Ludwig 3 con el mismo tokenizador, podrá destilarle logits a Ludwig 2, y Ludwig 2 le servirá de borrador para decodificación especulativa.
- 📏 **Bytes por token** medidos por fuente y por dominio contra SmolLM2, OLMoE y Qwen3 (`results/tokenizador_64k/con_transweb/compresion.csv`). Llama 3 no entró en la comparación.
- ❗ **El tokenizador muestrea el código de la misma fuente que el tronco.** Si no, quedan tokens que casi no salen en el entrenamiento (glitch tokens). El fijado usó Stack-Edu (solo licencia permisiva, bajado de Software Heritage) y HTML y CSS de The Stack v1.
- **Si el tokenizador del plan grande cambia, Ludwig 2 usa la versión final.** Después del primer paso del tronco ya no se puede cambiar.

---

## 4. Datos

> **Mezcla v3,** elegida por Pipe el 13 de noviembre de 2026. Fuentes y repartos cerrados el 15 de noviembre con una revisión independiente ([`results/datos_tronco_v3/`](../../results/datos_tronco_v3/); decisiones P13–P17 de [`../DECISIONS.md`](../DECISIONS.md)).

### 4.1 Qué se reutiliza del plan del modelo grande y qué cambia

**Se reutiliza tal cual (PLAN_MODELO_NUEVO v0.5):**
- **Tronco + ramas.** El tronco lleva una mezcla estacionaria. Lo escaso y lo mejor van solo en las ramas de enfriamiento.
- **Reglas de licencias** 🟢🟡🔴 (4.8).

**Cambia porque el modelo es ~4 veces más chico, y por los objetivos de Pipe (sección 1):**
1. **Menos dominios.** Con menos capacidad, cada dominio diluye a los demás. El código se concentra en Python y JS/TS, con los mismos tokens para los dos.
2. **Más matemática:** 10% en el tronco (antes 6%) y 22% en las ramas (antes 18%), porque la meta es razonar. MobileLLM-R1 subió la matemática a ~35% en su fase 2 de preentrenamiento, y le funcionó.
3. **Bilingüe:** español al 24% del tronco (60/40 con el inglés en lenguaje natural), al 10% en las ramas y al 25–30% en el SFT (4.9).
4. **Horizonte de 100B:** las fuentes están dimensionadas para que ninguna pase de ~4 pasadas hasta un tronco de 90B. Pasar de 100B pide revisarlas.
5. **Trazas más cortas:** los modelos chicos aprenden mejor de cadenas cortas (sección 9).
6. **Contexto de 16K** en la v1 (antes 32K).

### 4.2 Fuentes

| Bloque | Fuente (id en Hugging Face) | Licencia | Notas |
|---|---|---|---|
| Web educativa | FineWeb-Edu sin duplicados (`HuggingFaceTB/smollm-corpus`, `fineweb-edu-dedup`) | 🟢 ODC-By + términos de Common Crawl | |
| Web diversa | DCLM sin duplicados (`Zyphra/dclm-dedup`) | 🟢 CC-BY-4.0 | La ficha original dice "intended for research use only" en sus usos fuera de alcance, y lo justifica por rendimiento. Se anota en la ficha de datos (P15) |
| Web reescrita | FinePhrase (`HuggingFaceFW/finephrase`), formatos faq, tutorial, math y table | 🟢 ODC-By; reescrita con SmolLM2-1.7B-Instruct (Apache 2.0) | La reescritura está en `rollout_results`; `text` es la página original |
| Conocimiento | Wikipedia en inglés (`wikimedia/wikipedia`, 20231101.en) y StackExchange sin Stack Overflow (`HuggingFaceH4/stack-exchange-preferences`, volcado de febrero de 2023) | CC-BY-SA | StackExchange guarda el autor, su perfil y el link de cada respuesta, que es lo que pide la atribución |
| Libros sintéticos | Cosmopedia v2 (`HuggingFaceTB/smollm-corpus`) | 🟢 ODC-By; generado con Mixtral-8x7B | |
| Español nativo | FineWeb2-HQ (`epfml/FineWeb2-HQ`, `spa_Latn`) | 🟢 ODC-By + términos de Common Crawl | |
| Español traducido | TransWeb-Edu (`britllm/TransWeb-Edu-Spanish`): FineWeb-Edu traducido con Mistral-7B-Instruct-v0.1 | 🟢 ODC-By; Mistral es Apache 2.0 (P10) | No confundir con TransWebEdu, traducido con NLLB-200 y de uso no comercial: ese no entra |
| Español enciclopédico | Wikipedia en español (`wikimedia/wikipedia`, 20231101.es) | CC-BY-SA | |
| Texto paralelo | Corpus paralelo de la ONU y Europarl, inglés–español, sin sus sets de prueba | ONU: pide citar la fuente. Europarl: sin restricciones conocidas | Traducción humana; 25.2M y 2.0M de pares |
| Código | Stack-Edu (`HuggingFaceTB/stack-edu`), solo los archivos con licencia permisiva, con el contenido de `common-pile/stackv2_edu_filtered` cruzado por `blob_id` | La de cada archivo | Trae ~90% de los permisivos de Python, JS y TS (4.6). Son los mismos archivos que vio el tokenizador (P14) |
| HTML y CSS | The Stack v1 sin duplicados (`bigcode/the-stack-dedup`, `data/html` y `data/css`) | La de cada archivo + términos de The Stack | Es lo que vio el tokenizador. 51.2 GB y 8.5 GB comprimidos |
| Notebooks | StarCoderData (`bigcode/starcoderdata`), subconjuntos `jupyter-*` | La de cada archivo | Sale de The Stack v1, sin duplicados y con los datos personales quitados; ~13 GB |
| Stack Overflow | `mikex86/stackoverflow-posts`, volcado del 14 de junio de 2023 | CC BY-SA 2.5, 3.0 o 4.0, la de cada fila | Anterior al cambio de condiciones de julio de 2024 |
| Matemática | FineMath 4+ e InfiWebMath 4+ (`HuggingFaceTB/finemath`) | 🟢 ODC-By + términos de Common Crawl | |
| Instrucción y razonamiento corto | OpenMathReasoning (`nvidia/OpenMathReasoning`), solo las soluciones que quepan en 2K; OpenAssistant | 🟢 CC-BY-4.0 (soluciones de DeepSeek-R1 y QwQ-32B); Apache-2.0 (texto escrito por personas) | OpenThoughts3 queda fuera hasta auditarlo |
| Razonamiento (ramas y SFT) | OpenMathReasoning: 3.2M soluciones con cadena de pensamiento + 1.7M con Python (TIR) | 🟢 CC-BY-4.0 | |
| Matemática escolar (ramas y SFT) | Train de GSM8K y MATH, con soluciones cortas generadas en la fábrica (9.3) | 🟢 | |

- **Salen del tronco:** arXiv, OpenWebMath, C#, Go, Rust, Java, C/C++ y los issues de GitHub. De Algebraic Stack solo entraría el Python, si su licencia pasa.
- **Software Heritage queda opcional.** Solo haría falta para el ~10% de los permisivos que Common Pile no trae, y bajar en masa pide un acuerdo con Software Heritage e INRIA.
- **Cada fuente lleva su ficha** con licencia, tamaño, versión y hash.

### 4.3 Procesamiento

1. **Pre-tokenizado una sola vez (T1):**
   - Todo se convierte a ids una vez y se guarda en arreglos planos uint16, en archivos de tamaño fijo con un manifiesto de hashes. Así lo hacen nanoGPT, Megatron-LM y OLMo.
   - Se construye en sesiones de CPU de Kaggle, que no gastan cuota de GPU. La sesión de GPU no descarga, no filtra ni tokeniza: reanudar exacto es un offset, y reproducir es comparar hashes.
   - Para la v1 son ~50 GB. Los archivos quedan privados, porque tienen texto y código de terceros. Se publican la receta, los manifiestos y los hashes.
   - EOS al final de cada documento y ventanas de largo fijo sin botar colas, como en el lector actual.
   - ✅ **Hecho** (P18–P24): `scripts/build_shards.py`, con la receta en `configs/data_v3.json`, y el lector en `ludwig_lm/shards.py`.
     - Shards de 2^27 tokens (256 MiB): el tronco de la v1 son 168 (`notebooks/t1_shards_cpu.ipynb`). La primera sesión hizo 28 en 10.4 h, así que son unas 6 sesiones de ~11 h, no 3 de 56 (`results/t1/sesion1_reconstruccion_resumen_salida_kaggle.txt`).
     - Construidos en una sesión o en varias, salen idénticos byte a byte (`scripts/test_shards_cpu.py`).
2. **Duplicados:** FineWeb-Edu y DCLM entran en sus versiones sin duplicados, porque según el informe de Zyda-2 las originales traen ~80% de duplicados internos. Entre FineWeb-Edu y DCLM se quitan además las URL repetidas, con un hash de 64 bits.
3. **Un documento, una sola forma:** el original en inglés, su reescritura en FinePhrase y su traducción en TransWeb-Edu no entran juntos. Se cruzan por el id o la URL de Common Crawl, que las dos conservan. La excepción es el 0.5% de texto paralelo.
4. **Código:**
   - El filtro es que el `blob_id` esté entre los archivos permisivos de Stack-Edu, nunca el puntaje de Common Pile, que no es comparable con el de Stack-Edu.
   - Fuera los encabezados de licencia y los archivos generados o minificados.
   - Common Pile ya reemplaza correos, teléfonos e IP. Aquí se limpian llaves y contraseñas; en HTML y CSS, que vienen de The Stack v1, también los datos personales.
   - El manifiesto guarda `repo_name`, `path`, `license`, `url` y `blob_id`, y en Stack Overflow, `Id` y `OwnerUserId`: es la atribución.
5. **Descontaminación** en el momento, con los 13-gramas de GSM8K, MATH (y MATH-500), HumanEval, MBPP, MultiPL-E, MMLU, ARC, HellaSwag, PIQA y WinoGrande, y de los de español: MGSM, Belebele, Global-MMLU, HellaSwag en español y Flores.
   - ✅ Son 13-gramas de tokens con hashes exactos de 64 bits, en vez de un filtro de Bloom: caben en 80 MB y no dan falsos positivos (P20).
   - Un 13-grama que aparece en más de 10 ítems es una plantilla y no cuenta.
6. **Etiquetas de fuente** al estilo MeCo 🧪: la fuente antepuesta en el tronco y nada en las ramas, para que el modelo final funcione sin etiquetas.
   - ✅ Forma (P21): `<|reserved_0|>` + "URL: dominio" + `<|reserved_1|>`, fuera de la pérdida como en MeCo.
   - `--meta strip` las quita al leer, así que se puede entrenar sin ellas con los mismos shards.
7. **FIM** en el 10% del código. Los ids 10–13 los pone el lector, no el texto. ✅ En el orden prefijo-sufijo-medio del Qwen2.5-Coder, cortado en dos caracteres al azar (P24).
8. **Val:** los primeros documentos del último archivo de cada fuente nunca se entrenan; el resto de ese archivo sí (P19). Así el código no pierde un shard entero de Common Pile por lenguaje. La pérdida se reporta en bits por byte, por dominio, desde el paso 0 (P17).
9. **Trazas y chat** con la plantilla de chat y `<think>` como token (id 4), nunca como texto plano. Si no, el modelo aprende dos formatos para lo mismo.

### 4.4 Mezcla del tronco v3 (estacionaria) ✅

| Bloque | Fuente | % | Tronco de la v1 (22.5B) | Tronco a 90B |
|---|---|---|---|---|
| **Inglés, 36%** | FineWeb-Edu sin duplicados | 14 | 3.15B | 12.6B |
| | DCLM sin duplicados | 11 | 2.5B | 9.9B |
| | FinePhrase, 1.5 por formato | 6 | 1.35B | 5.4B |
| | Wikipedia en inglés 1.5 + StackExchange 1.5 | 3 | 0.68B | 2.7B |
| | Cosmopedia v2 | 2 | 0.45B | 1.8B |
| **Español, 24%** | FineWeb2-HQ | 13 | 2.9B | 11.7B |
| | TransWeb-Edu | 8.5 | 1.9B | 7.7B |
| | Wikipedia en español | 2 | 0.45B | 1.8B |
| | Texto paralelo: ONU 0.35 + Europarl 0.15 | 0.5 | 0.11B | 0.45B |
| **Código, 28%** | Python | 11.2 | 2.5B | 10.1B |
| | JavaScript 7.3 + TypeScript 3.9 | 11.2 | 2.5B | 10.1B |
| | Notebooks 1.1 + Markdown 1.0 + Stack Overflow 1.0 | 3.1 | 0.7B | 2.8B |
| | HTML 0.7 + CSS 0.7 | 1.4 | 0.32B | 1.26B |
| | SQL 0.55 + Shell 0.55 | 1.1 | 0.25B | 1.0B |
| **Matemática, 10%** | FineMath 4+ e InfiWebMath 4+ | 10 | 2.25B | 9B |
| **Instrucción, 2%** | OpenMathReasoning 1.9 + OpenAssistant 0.1 | 2 | 0.45B | 1.8B |

- **Inglés y español:** 60/40 en lenguaje natural.
- **Contexto:** 2K hasta 50B; 4K desde 50B.
- **Pasadas:** ninguna fuente pasa de ~4 hasta un tronco de 90B. Las que más se repiten son las de código: Python 1.9, JavaScript 2.7 y TypeScript 2.5 a 90B (4.6). Wikipedia en inglés queda en 0.37.
- **Por qué la web a gran escala:** en MobileLLM-R1, quitar FineWeb-Edu, su única fuente de web general, fue lo que más degradó a la vez el conocimiento, la matemática y el código. Los autores lo dicen de la web a gran escala, no de una proporción: aquí, la web en inglés (FineWeb-Edu, DCLM y FinePhrase) es el 31% del tronco.
- **Por qué FinePhrase:** en su estudio, con un modelo de 1.2B y ~21B tokens, DCLM sola saca 13.77, lo mejor de FineWeb-Edu solo 11.82 y una mezcla con 10–20% de web reescrita, 15.1–15.2. Meta encontró que la buena dosis de texto reescrito es ~30% de la mezcla (4.10).
- **Por qué el código ayuda a razonar:** en MobileLLM-R1, StarCoder subió la matemática más de lo que OpenWebMath subió el código.
- **Punto débil:** la web en inglés escrita por personas (FineWeb-Edu y DCLM) queda en 25% del tronco, contra 50% en la mezcla anterior. Es un riesgo para el conocimiento general y los benchmarks en inglés, y lo vigila la evaluación por dominio (4.9).

### 4.5 Mezcla de las ramas de enfriamiento (~10% de cada hito) ✅

| Dominio | % | Qué lleva |
|---|---|---|
| Web (inglés) | 26% | FineWeb-Edu ≥4 y lo mejor de DCLM |
| Código | 25% | Stack-Edu de puntaje alto, notebooks; mitad Python |
| Matemática | 22% | FineMath 4+, problemas con pasos, soluciones con Python (TIR) |
| Razonamiento e instrucción | 13% | Trazas cortas con plantilla de chat y `<think>`; instrucciones generales; textos de identidad ("soy Ludwig…") |
| Conocimiento | 4% | Wikipedia y Cosmopedia |
| Español | 10% | Lo mejor de FineWeb2 y TransWeb-Edu en español; incluye ~2 puntos de matemática y razonamiento en español |

- Sin etiquetas de fuente. El LR baja hasta 0 con curva 1−√. Contexto de 4K.
- **Contexto largo:** el último ~0.5B tokens de la rama de la v1 va a 16K, con la misma mezcla más documentos largos (repositorios concatenados por archivo, papers completos de arXiv, artículos largos y libros de dominio público). Desde 250B, ~2% de cada rama va a 32K.
- **MTP:** se entrena en el último ~1B tokens de la rama de la v1 (5.1).

### 4.6 Código por dentro

| Lenguaje | % del tronco | Tronco de la v1 | Tronco a 90B | Disponible: permisivo y en Common Pile | Pasadas a 22.5B / 90B |
|---|---|---|---|---|---|
| Python | 11.2 | 2.5B | 10.1B | ≈5.30B | 0.48 / 1.9 |
| JavaScript | 7.3 | 1.6B | 6.6B | ≈2.39B | 0.69 / 2.7 |
| TypeScript | 3.9 | 0.88B | 3.5B | ≈1.39B | 0.63 / 2.5 |
| SQL | 0.55 | 0.12B | 0.5B | ≈1.18B | 0.10 / 0.4 |
| Shell | 0.55 | 0.12B | 0.5B | ≈0.94B | 0.13 / 0.5 |
| Markdown | 1.0 | 0.23B | 0.9B | ≈1.90B | 0.12 / 0.5 |
| Notebooks (StarCoderData) | 1.1 | 0.25B | 1.0B | ~13 GB | — |
| Stack Overflow | 1.0 | 0.23B | 0.9B | — | — |
| HTML y CSS (The Stack v1) | 1.4 | 0.32B | 1.26B | 51.2 + 8.5 GB comprimidos | — |

- **De dónde salen los números:** [`results/datos_tronco_v3/common_pile.txt`](../../results/datos_tronco_v3/common_pile.txt), cruzando por `blob_id`. Los tokens están contados con la proporción de la ficha de Stack-Edu: con el tokenizador de 64K serán ~10% menos, y las pasadas, ~10% más.
- **Common Pile trae ~90% de los permisivos:** el 89.2% de los archivos de Python, el 88.3% de JavaScript y el 90.4% de TypeScript. Bajando todo lo permisivo de Software Heritage, las pasadas a 90B serían 1.7 / 2.4 / 2.4; con The Stack v3, 2.1 / 4.3 / 4.4.
- **Para la T1:** los shards de Common Pile vienen ordenados por lenguaje, así que no hace falta bajar los 83 GB: los de estos lenguajes son ~45 GB.
- **Reserva:** Common Pile trae además ~1.5M archivos de JavaScript con puntaje ≥2.5 que no están en Stack-Edu (≈1.8B tokens), por si a 90B se quieren bajar sus pasadas.
- Más de ~4 pasadas por los datos escasos ya no valen como datos nuevos (Muennighoff). Pasado eso, se reformula al estilo Kimi en vez de repetir.

### 4.7 Lo que ve la v1 en total (25B)

| | Tronco (22.5B) | Rama (2.5B) | Total |
|---|---|---|---|
| Web en inglés | 6.98B | 0.65B | ~7.6B |
| Conocimiento y libros | 1.13B | 0.1B | ~1.2B |
| Español | 5.4B | 0.25B | ~5.7B |
| Código | 6.3B | 0.63B | ~6.9B |
| Matemática | 2.25B | 0.55B | ~2.8B |
| Razonamiento e instrucción | 0.45B | 0.33B | ~0.8B |

### 4.8 Licencias 🟢🟡🔴

- 🟢 **Entra:** texto humano con licencia abierta, y datos sintéticos generados con modelos abiertos de licencia permisiva: DeepSeek-R1/V3 (MIT), Qwen3 y QwQ (Apache 2.0), gpt-oss (Apache 2.0), Mixtral y Mistral-7B (Apache 2.0) y SmolLM2 (Apache 2.0). En cada dataset se revisa qué modelo generó cada subconjunto.
- 🟡 **Solo si aceptas la condición:** datos generados con modelos cuya licencia pone condiciones. Por ejemplo, Llama pide que el modelo entrenado con sus salidas lleve "Llama" en el nombre. Por defecto, fuera.
- 🔴 **No entra:** salidas de APIs cerradas, cuyos términos prohíben entrenar modelos competidores. Ejemplo: TinyGSM, generado con GPT-3.5. La regla es "ante la duda, fuera".
- **Atribución:** una fila por fuente en la ficha de datos (sección 10). Wikipedia, StackExchange y Stack Overflow son CC-BY-SA, y el código lleva la licencia de cada archivo, con su repo y su ruta en el manifiesto.
- **Common Crawl:** FineWeb-Edu, FineWeb2-HQ y FineMath también quedan sujetos a sus términos de uso.

### 4.9 Español: por qué 24% (60/40 en lenguaje natural)

- **El objetivo:** Pipe quiere un modelo "totalmente bilingüe" (sección 1). Con 24% del tronco, la v1 ve ~5.4B tokens de español en el tronco, y la proporción con el inglés en lenguaje natural es 60/40.
- **No 90/10:** que un idioma dominante ayude al otro, y que 90/10 le ganara a 50/50, viene de experimentos con idiomas clonados. El precedente de un modelo chico bilingüe de verdad es CroissantLLM (1.3B), entrenado con inglés y francés 1:1.
- **Referencia:** SmolLM3 (3B) puso 12% de web multilingüe repartida entre 5 idiomas, y en HellaSwag en español le ganó a Qwen2.5-3B y Llama 3.2 3B.
- **Calidad sobre cantidad:** un estudio de 2025 atribuye el daño en los modelos bilingües a la calidad desigual de los datos más que a la cantidad. Por eso el español sale de FineWeb2-HQ y de TransWeb-Edu, que trae el mismo texto educativo de la base en inglés, más un poco de texto paralelo para alinear los dos idiomas.
- **Costo para el inglés:** la web en inglés escrita por personas baja a 25% del tronco (4.4).
- **Qué se mide:** bits por byte por dominio desde el paso 0, revisados a 1B y a 2.25B tokens del tronco (P17). Además, la compresión del tokenizador en español, MGSM, Belebele, Global-MMLU, HellaSwag y Flores (inglés↔español). Si el español se queda atrás, se sube en las ramas, que sí se pueden cambiar sin tocar el tronco.
- **Sin prueba proxy de mezcla antes del tronco (P17):** su regla casi no podía fallar, y si fallaba, movía ~4 puntos que las ramas pueden compensar.

### 4.10 Texto escrito por modelos: topes y diversidad

- **Topes por tipo en el tronco (P16):**
  - Generación libre: ≤5%.
  - Reescritura de páginas reales: ≤10% del tronco y ≤30% de la web en inglés.
  - Traducción automática: ≤10% del tronco y ≤40% del español.
  - En total, ≤20%, y ningún modelo generador por encima de 10%.
  - En las ramas, ≤35%.
- **La v3 queda en** ~4% de generación libre (Cosmopedia y la instrucción), 6% de reescritura (FinePhrase) y 8.5% de traducción (TransWeb-Edu): 18.5% en total. El generador más grande es Mistral-7B, con 8.5%. Con Mixtral, la familia Mistral suma 10.5%.
- **Evidencia:** Meta (la buena dosis de reescritura es ~30%), FinePhrase (con 10% a 90% de reescritura, todo supera a la web sola) y TransWeb-Edu (un modelo entrenado solo con texto traducido fue competitivo). El tope de traducción es criterio, no medición.
- **Al menos 3 familias de maestros:** hoy son Mistral (Mistral-7B y Mixtral), SmolLM2, DeepSeek-R1 y QwQ.
- **Deduplicación** de casi-duplicados dentro de cada fuente sintética.
- **Vigilancia de estilo:** en cada hito, la diversidad de n-gramas de muestras generadas y la frecuencia de muletillas típicas de maestros ("Certainly!", "As an AI language model").

---

## 5. Entrenamiento de la v1 en 2×T4

### 5.1 Receta

| Parte | Valor |
|---|---|
| Optimizador | **Muon** en todas las matrices (atención, cada experto por separado, experto compartido y MTP): momento Nesterov 0.95, 5 pasos de Newton–Schulz y la escala de Moonlight (0.2·√máx(m, n)) para reutilizar el LR y el weight decay de AdamW. **AdamW** (β 0.9/0.95, ε 1e-8) en embeddings, normas, router y compuerta del experto compartido |
| Learning rate | Pico: el de Ludwig 1, que tiene el mismo ancho y la misma atención. 🧪 Se confirma con 3 corridas cortas (×0.5, ×1 y ×2, ~0.2B tokens cada una, ~40–65 h en total). MobileLLM-R1 encontró que un LR más alto en el preentrenamiento dio mejor MMLU |
| Schedule | WSD: calentamiento de 2,000 pasos; constante en el tronco; en cada rama baja hasta 0 con curva 1−√ |
| Weight decay | 0.1 en las matrices; 0 en las normas |
| Batch | 0.25M tokens por paso los primeros ~2B tokens, después 0.5M: ~54,000 pasos en la v1. No copiar los 4.2M de Meta, que con 25B serían apenas ~6,000 pasos |
| Precisión | FP16 con escalado dinámico de pérdida y pesos maestros en FP32. Normas, softmax, router y pérdida en FP32. Newton–Schulz en FP16 con la entrada normalizada; si aparece un NaN, en FP32 (cuesta menos del 2% del paso) |
| Estabilidad | QK-norm + compuerta; recorte de gradiente en 1.0; z-loss de salida 1e-4 como seguro en FP16; saltar lotes no finitos |
| Inicialización | Normal(0, 0.02); o_proj y down_proj divididas por √(2·26); w de las normas en 0 (escala 1 + w). La T0.2 usa la inicialización por defecto de transformers (no cambia velocidad ni memoria); la escalada entra con el script de entrenamiento (T0.3) |
| MoE | Sin descartar tokens (la implementación de transformers no tiene tope de capacidad). 📏 La T0.2 compara sus tres kernels de expertos (`eager`, `batched_mm` y `grouped_mm`); ScatterMoE solo si ninguno llega a la meta de tokens/s |
| Pérdida | Por bloques: los logits de 16K tokens × 64K de vocabulario pesan 2 GB en fp16 y 4.3 GB en fp32 |
| Compilación | torch.compile en normas, SwiGLU y compuertas 📏 |
| Paralelismo | **FSDP2 con todo fragmentado** (una unidad por capa; la raíz no se refragmenta para que la pérdida por bloques use la cabeza atada). Es lo que mide la T0.2. La alternativa, expertos repartidos (20 por GPU) + DDP en lo denso, deja replicados ~0.33B parámetros densos y pide ~1.7 GiB más por GPU: con 40 expertos probablemente no cabe, así que solo se implementa si FSDP2 no llega a los tokens/s de la G0 (5.7) |
| MTP | En el último ~1B tokens de la rama de la v1, con *stop-gradient* hacia el tronco, los embeddings y la cabeza: aprende de las representaciones del modelo pero no puede empeorarlo. Cuesta ~11–17 h |
| Momento de Muon | En bf16 por defecto (rango de fp32, la mitad de memoria). Si la T0.2 deja margen, se puede probar fp32 |

### 5.2 Memoria por GPU (T4 de 15 GiB)

| Concepto | GiB |
|---|---|
| Pesos FP32 + gradientes FP32 + momento bf16 (10 bytes × 1.963B ÷ 2 GPUs) | 9.1 |
| AdamW (2 estados) en embeddings, normas y router (~67M parámetros) | +0.1 |
| Activaciones, buffers de FSDP (una capa reunida + la siguiente), ruteo y contexto CUDA | 3–4.5 |
| **Total** | **~12.2–13.6** (+0.4 con el MTP) |
| Plan B si no queda ≥1.5 GiB libre: 32 expertos (1.64B totales, expansión 4.5, aún dentro de la zona 4–8) | −1.5 |

Esto sigue siendo una estimación: la cifra que vale es la que mida la T0.2 (`scripts/bench_train.py`), con FSDP2 real, el kernel de MoE elegido y cada largo de secuencia.

### 5.3 Relevo y checkpoints

- **Cada amigo corre su turno en su propia cuenta**, con el notebook del relevo: toma el candado, reanuda, entrena, guarda, sube y suelta el candado. El candado vence solo si una sesión muere sin soltarlo.
- **Sesiones** de hasta 12 h 📏.
- **Checkpoint cada 1–2 h,** con el estado completo: pesos, momento, estados de AdamW, escalador de pérdida, RNG (torch, numpy y python), estado del lector, paso y posición del LR. Pesa ~16 GB (~12 GB con el momento en bf16).
- **Dos checkpoints privados que se turnan,** más hitos públicos (solo pesos en bf16, ~4 GB). Hugging Face da 100 GB de almacenamiento privado gratis 📏.
- **La reanudación se verifica:** se hashea el primer lote después de reanudar y se compara con el esperado.
- Con ~10 cuentas hay ~300 h de T4 por semana. El relevo 24/7 usa ~168, y el resto queda para pruebas, evaluación, SFT y la fábrica de datos.
- ✅ **Cómo quedó** (P26; `ludwig_lm/relay.py`, `notebooks/tronco_relevo_2xT4.ipynb`):
  - **Principal:** un repo privado de Hugging Face de la organización del proyecto. El candado es un *lease* de 2 h en `lock.json`, que se renueva cada 30 min y con cada subida.
  - **La valla:** cada escritura es un commit cuyo padre es el último que vio la sesión. Una sesión cuyo candado venció y tomó otra falla en su próximo commit y para sola.
  - **Checkpoint cada 3 h:** cada uno se sube en un solo commit, con el índice, la rotación (quedan los 2 más nuevos) y la renovación del candado.
  - Al final de cada turno se aplasta el historial, así que nunca pasa de ~68 GB de los 100 GB privados.
  - **Respaldo:** el último checkpoint del turno va al de los dos datasets privados de Kaggle que tenga el más viejo, como un tar con su meta.
  - **Disco:** en `/tmp` nunca hay más de dos checkpoints a la vez (~23 GiB de los ~73 GB de Kaggle). El de partida se borra después del primer guardado del turno, localmente queda solo el periódico más nuevo y, antes del respaldo, solo el último.
  - **Reanudación:** desde el checkpoint completo más nuevo de los tres lugares; si empatan, gana la copia que ya está en el disco.
  - **Credenciales:** cada cuenta usa las suyas: su token de Hugging Face con escritura en el repo y su token de la API de Kaggle.
  - **Prueba:** `scripts/test_relay_cpu.py`. Tres turnos en tres cuentas dan bit a bit la corrida sin cortes.

### 5.4 Alarmas automáticas

| Señal | Umbral | Acción |
|---|---|---|
| Pico de pérdida | > media + 4σ de los últimos 200 pasos | Saltar el lote; si se repite 3 veces, volver al checkpoint anterior y bajar el LR un 20% |
| Gradientes no finitos | Más de 10 en 1,000 pasos | Alarma; revisar el escalador |
| Escalador de pérdida | Por debajo de 1 | Alarma (desborde de FP16) |
| Carga por experto | Máx/media > 3 durante 500 pasos | Subir el coeficiente de balanceo |
| Balanceo roto | Experto más cargado > 3× la media en el 80% de 50 pasos, desde el paso 200 | Detener y guardar (P30) |
| Brecha del router | λ de SMEBU > 5× la brecha más chica entre las capas (la mediana entre el 4.º y el 5.º score de cada token) durante 50 pasos | Alarma: revisar λ y el nivel del router (P29) |
| Experto muerto | < 10% de la carga media durante 1,000 pasos | Alarma; revisar |
| Logit máximo de atención | Crece sin parar | Revisar QK-norm |
| Tokens/s | < ~2,500 | Revisar kernels y sincronización |
| Memoria pico | > 14.3 GiB | Momento en bf16 o micro-batch menor |

Además se registran: norma del gradiente, entropía del router, pérdida de balanceo, z-loss, activación media de la compuerta y producción de entropía por capa y experto (5.6).

### 5.5 Evaluación

- **Durante el tronco, cada ~0.5B tokens:** bits por byte en val por dominio (inglés, Python, JS, matemática y español), más HellaSwag, ARC-Easy y PIQA. La pérdida cruda nunca se compara entre dominios: cada uno tiene su propio piso.
- **En cada rama:** HellaSwag, ARC-E/C, PIQA, WinoGrande, MMLU, GSM8K (8-shot en la base, 0-shot con cadena de pensamiento en la instruct), MATH500, HumanEval, MBPP, IFEval y MGSM en español.
- **En español:** MGSM, Belebele, Global-MMLU (formato cloze), HellaSwag en español y Flores (inglés↔español).
- **Ruido (±1σ):** ARC-Challenge ±1.3 puntos; WinoGrande ±1.4; PIQA ±1.1; HellaSwag ±0.5; MMLU ±0.4. Una diferencia en ARC-C o WinoGrande solo cuenta si pasa de ~4 puntos.

### 5.6 Producción de entropía (para los papers) 🧪

- **Solo se mide; no cambia nada del entrenamiento.**
- El estimador está validado con momentum-SGD, no con Muon: se valida primero con Muon antes de presentarlo como sello.
- Debe costar menos del 1% del paso. Los registros por capa y por experto se publican con cada versión.

### 5.7 Compuerta G0: antes de la primera hora de tronco

No se arranca el tronco hasta cumplir todo esto con el modelo real y el lector de datos real:
1. **Corrida sostenida de ≥6 h** sin NaN, con el escalador de pérdida estable.
2. **Tokens/s sostenidos ≥ 2,600** (el extremo bajo del plan), medidos en la última hora, no en el arranque.
3. **Memoria pico ≤ 13.5 GiB reservada** (la que tiene tomada el asignador de PyTorch, como en el banco v1) durante toda la corrida, evaluación y guardado incluidos.
4. **Al menos 2 guardados** subidos a Hugging Face y **una reanudación en otra cuenta**, con el hash del primer lote igual al esperado.
5. **16K probado:** 50 pasos a 16K de contexto sin falta de memoria.

**Plan (P27, propuesta de Pipe más un ensayo previo):** todas las partes corren con `notebooks/tronco_relevo_2xT4.ipynb`.
- **G0-0, `prueba-relevo` (~15 min):** el relevo completo con el modelo diminuto, contra un repo de prueba. Prueba la subida, el candado, el respaldo, el aplastado y una reanudación exacta.
- **G0-1, `g0-16k` (~30–60 min, en una cuenta libre):** la condición 5.
- **G0-2, el primer turno del tronco:** las condiciones 1–4, menos la reanudación. Si pasa, sus checkpoints cuentan.
- **G0-3, el segundo turno, en otra cuenta:** la reanudación, con el estado y el primer lote exactos.
- **Antes de la G0-2:** leer `muestras.txt` de la construcción de la T1.
- El no determinismo de la GPU no afecta la G0 (P28): la reanudación compara el estado y el hash del primer lote, no la curva.

---

## 6. Post-entrenamiento (en cada versión)

1. **SFT general:** instrucciones, chat e identidad. 2 épocas.
2. **SFT de razonamiento, después del general.** En MobileLLM-R1, hacerlos en orden le ganó a mezclarlos.
   - Mayoría de cadenas cortas y soluciones con Python, más una parte de cadenas largas: es la "destilación mixta" (sección 9).
   - En la v1, ejemplos de ≤4K tokens; desde 250B, ≤8K.
   - `<think>` como token y la plantilla de chat.
3. **DPO:** opcional, desde 100B.
4. **GRPO:** opcional, desde 250B. En MobileLLM-R1, el SFT le ganó al RL en modelos chicos.
5. **Reajuste del MTP** con generaciones del propio modelo (~0.1–0.2B tokens), para que el borrador se parezca a como genera.
6. **Fusión:** promedio de los últimos checkpoints del SFT.
7. **Exportación:** HF (safetensors en bf16) y GGUF (Q8_0 y Q4_K_M) con el MTP; prueba de logits; `llama-server --spec-type draft-mtp`.

**Costo en 2×T4:** ~45–120 h para 0.3–0.5B tokens de SFT en 2 épocas. A eso se suman la evaluación y la fábrica de datos (9.3).

---

## 7. Fases hasta 1T tokens

### 7.1 Reglas

- **La arquitectura y el tokenizador no cambian en todo el programa.** Cambiarlos es empezar de cero.
- **Un solo tronco,** con LR constante y la mezcla estacionaria de 4.4.
- **Cada versión es una rama de enfriamiento** de ~10% del hito, que sale del último checkpoint del tronco. El tronco sigue desde antes de la rama, no desde la rama.
- **Versiones previas gratis entre hitos:** promediar checkpoints del tronco (ByteDance mostró que eso mejora el modelo y predice cómo quedará el enfriamiento).
- **Cada versión publica:** base, instruct, GGUF con MTP, logs, manifiesto de datos y reporte de evaluación.
- **Si cambia el hardware** (T4 → TPU o H100), pasar de FP16 a BF16 está bien: el tronco sigue igual.
- **1T no es obligación.** Antes de cada fase nueva se revisa:
  1. La versión anterior está publicada.
  2. La pérdida de validación sigue bajando al ritmo esperado contra el log de los tokens (ajuste de una ley de potencia). Si se queda muy por debajo de lo esperado, se investiga antes de seguir.
  3. Los expertos siguen sanos: ninguno muerto y la carga balanceada.
  4. No hay señales de memorización en las fuentes que se repiten: la pérdida en su val no se separa de la de un val fresco.
  5. Hay hardware y horas para la fase completa.

### 7.2 Tabla

Supuestos: 2×T4 a 8.3–13.2 TFLOPS efectivos (calibrado con los ~13k tokens/s de Ludwig 1); relevo con ~143 h útiles por semana; TPU v5e-8 con 20–35% de MFU; H100 con 25–40% de MFU, a USD 2–3 la hora.

| Versión | Hito | Tronco hasta | Rama | Tokens nuevos | 2×T4 (h) | Semanas de relevo | TPU v5e-8 (h) | H100 (h) | USD en H100 | Contexto |
|---|---|---|---|---|---|---|---|---|---|---|
| Ludwig 2 · 25B | 25B | 22.5B | 2.5B | 25B | 1,750–2,800 | 12–20 | 41–72 | 57–92 | 115–276 | 16K |
| Ludwig 2 · 50B | 50B | 45B | 5B | 27.5B | 1,900–3,000 | 13–21 | 45–79 | 63–101 | 126–303 | 16K |
| Ludwig 2 · 100B | 100B | 90B | 10B | 55B | 4,200–6,600 | 29–46 | 100–174 | 139–222 | 278–667 | 16K |
| Ludwig 2 · 250B | 250B | 225B | 25B | 160B | 12,100–19,300 ❌ | 85–135 | 290–508 | 404–647 | 809–1,941 | 32K |
| Ludwig 2 · 500B | 500B | 450B | 50B | 275B | ❌ | — | 499–872 | 695–1,112 | 1,390–3,337 | 32K |
| Ludwig 2 · 750B | 750B | 675B | 75B | 300B | ❌ | — | 544–952 | 758–1,213 | 1,517–3,640 | 32K |
| Ludwig 2 · 1T | 1T | 900B | 100B | 325B | ❌ | — | 589–1,031 | 822–1,314 | 1,643–3,943 | 32K |
| **Total** | | | | **1,167.5B** | 88K–140K (12–19 años) | | **2,100–3,700** | 2,900–4,700 | **5,900–14,100** | |

**Lo que dice la tabla:**
- **Kaggle llega hasta 100B:** ~7,800–12,400 h acumuladas, o sea 54–87 semanas (1–1.7 años) de relevo. 1T solo en T4 tomaría 12–19 años.
- **TRC:** 1T completo son ~2,100–3,700 h de v5e-8. En los 30 días de TRC eso equivale a tener 3–5 TPUs v5e-8 a la vez. Con una sola v5e-8 durante 30 días cabe el tramo de 100B a 250B (290–508 h). Pasar a TPU implica portar el código a JAX o torch_xla 📏 (1–2 semanas).
- **Con plata:** USD 1,000 en H100 compran ~80–200B tokens. Alcanza para llevarlo de 100B a ~180–300B, no a 1T.
- **Si TRC no llega,** el programa se detiene con dignidad en 100B: cada versión hasta ahí es publicable.

### 7.3 Qué esperar de cada versión (estimación propia; incertidumbre alta)

| Versión | Qué debería hacer |
|---|---|
| 25B | Inglés fluido; funciones cortas de Python; problemas de matemática de 2–4 pasos a veces (GSM8K ~10–25% después del SFT); MMLU cerca del azar |
| 100B | Python útil; GSM8K ~25–40%; MMLU ~30–35 |
| 250B | GSM8K ~35–55%; MATH500 ~15–30% |
| 1T | Entre MobileLLM-R1-360M y 950M: GSM8K ~50–65%; MATH500 ~30–50%; MMLU ~35–45 |

---

## 8. ¿Vale la pena que razone?

**Sí, con una definición realista de "razonar":** cadena de pensamiento corta en matemática escolar, Python como calculadora y el formato `<think>` desde el día uno. Casi todo el costo es elegir bien los datos, no cómputo extra.

**La receta de datos pesa más que el tamaño.** Modelos base de MobileLLM-R1 (Meta, ICLR 2026), preentrenados con 4.2T tokens sacados de ~2T de alta calidad:

| Modelo | Tokens de preentrenamiento | GSM8K base (8-shot) | MATH500 base | GSM8K post (0-shot) | MATH500 post |
|---|---|---|---|---|---|
| SmolLM2-360M | ~4T | 5.0 | 1.8 | 8.1 | 3.4 |
| MobileLLM-R1-360M | 4.2T | 39.4 | 13.4 | 24.5 | 28.4 |
| SmolLM2-1.7B | 11T | 31.8 | 11.6 | 41.8 | 19.2 |
| MobileLLM-R1-950M | 4.2T | 61.6 | 26.8 | 67.5 | 74.0 |
| Qwen3-0.6B | 36T | 60.9 | 29.8 | 79.2 | 73.0 |

Con el mismo tamaño y billones de tokens en los dos casos, el modelo de MobileLLM-R1 de 360M saca 8 veces más GSM8K base que SmolLM2-360M. Esa diferencia sale de los datos.

**¿Alcanza con 25B tokens?** Para matemática escolar de 2–4 pasos, sí, aunque de forma irregular. Para matemática de competencia o razonamiento largo estilo R1, no. Los modelos chicos que lo logran vieron billones de tokens: MobileLLM-R1-950M llega a 15.5 en AIME'24 con 4.2T.

**Cómo se busca:**
1. 10% de matemática en el tronco y 22% en las ramas (4.4 y 4.5).
2. Soluciones con Python (TIR): el modelo escribe el cálculo y Python lo ejecuta. Encaja con que la meta es Python.
3. SFT con trazas cortas (sección 9).
4. La idea de TinyGSM: problemas escolares con solución en Python. Con 12.3M problemas (1.8B tokens), Phi-1.5 de 1.3B pasó de 44.6% a 68.2% en GSM8K. Pero TinyGSM se generó con GPT-3.5 (🔴), así que la idea se regenera con un maestro 🟢 en la fábrica.
5. RL al final y opcional: en MobileLLM-R1, el SFT le ganó al RL en modelos chicos, y el RL extra casi no sumó.

---

## 9. ¿Vale la pena destilar un modelo grande?

### 9.1 De logits: no

- **El tokenizador es propio.** La destilación de logits necesita que maestro y alumno compartan tokenizador; entre tokenizadores distintos es otro proyecto.
- **En la T4 el maestro cuesta más que el alumno.** El forward de un maestro de 1.7B cuesta lo mismo que entrenar Ludwig 2 (3.4 contra 3.27 GFLOPs por token), y uno de 8B, ~5 veces más.
- **Así lo hizo Meta:** MobileLLM-R1 destiló logits de Llama-3.1-8B en su mid-training, pero con el mismo tokenizador y en sus GPUs.
- **Excepción a futuro:** si existe Ludwig 3 con el mismo tokenizador, podrá destilarle logits a Ludwig 2 en TPU.

### 9.2 De datos: sí

- **Es lo que mejor funciona en modelos chicos.** En DeepSeek-R1, destilar de R1 a Qwen2.5-32B le ganó a aplicarle RL directamente.
- **Pero los chicos aprenden mejor de cadenas cortas.** Los modelos de ≤3B no siempre ganan con cadenas largas ni con maestros grandes, y rinden mejor con razonamientos cortos y más simples. Mezclar cadenas largas y cortas ("destilación mixta") les sube el rendimiento.
- **Casi todo ya está generado y publicado:** OpenMathReasoning (🟢 CC-BY-4.0, generado con DeepSeek-R1 y QwQ-32B) se filtra por largo y dificultad. MobileLLM-R1 usó 3.2M de sus soluciones en su SFT de razonamiento.

### 9.3 La fábrica (lo que falta, con un maestro 🟢)

- **Qué genera:** soluciones cortas para el train de GSM8K y MATH; problemas escolares con solución en Python (la idea de TinyGSM); identidad; español; reescrituras cortas de trazas largas.
- **Maestros 🟢:** Qwen3 (Apache 2.0, modo sin pensar para cadenas cortas), DeepSeek-R1/V3 (MIT), gpt-oss (Apache 2.0).
- **Costo:** con el ritmo que estimamos en el plan grande (~2–3B tokens en 300–600 h de T4), los ~0.1–0.3B tokens que faltan son ~10–90 h. Corre con la cuota sobrante, sin frenar el relevo.

---

## 10. Publicación abierta

**Licencia:** Apache 2.0 para el código desde ya. Para los pesos, Apache 2.0 ❓ **solo después de auditar la ficha de datos fuente por fuente**: es requisito para publicar cualquier versión. Entrenar con fuentes CC-BY o CC-BY-SA (Wikipedia, StackExchange) y publicar pesos con licencia abierta es práctica común (OLMo, SmolLM), pero no está zanjado legalmente; la ficha debe atribuir cada fuente. (No soy abogado.)

**Qué se publica en cada versión:**
- Base, instruct y GGUF (Q8_0 y Q4_K_M) con el MTP.
- Tokenizador y `config.json`.
- Código de entrenamiento, del lector de datos y del relevo.
- Manifiesto de datos: fuentes, versiones, hashes y el orden exacto, para que cualquiera reproduzca el lote N.
- Listas de descontaminación.
- Logs completos: pérdida por dominio, router, alarmas, tokens/s y producción de entropía.
- Checkpoints intermedios: los hitos públicos.
- Evaluación: el harness y las salidas crudas.
- Model card: límites, cómputo usado (horas de GPU) y chequeos de contaminación.

**Estructura del repositorio** (`ludwig-lm`; el paquete se llama `ludwig_lm` porque `ludwig` ya existe en PyPI):

```
ludwig-lm/
├── ludwig_lm/      # modelo (Qwen3-Next en atención completa + MTP + pérdidas) y Muon
├── configs/        # ludwig2.json (el modelo real) y tiny.json (pruebas)
├── scripts/        # smoke_format.py (T0.1), bench_train.py (T0.2), convert_gguf.py, make_notebooks.py
├── notebooks/      # notebooks de Kaggle, generados desde scripts/make_notebooks.py
├── patches/        # parche de llama.cpp + borrador del PR
├── results/        # resultados crudos de cada prueba (JSON), la evidencia del diseño
└── docs/           # DECISIONS.md (inglés), es/DISENO.md (este documento)
```

Se agregan a medida que existan: `tokenizer/`, `data/` (recetas, lector, descontaminación, manifiestos), `train/` (bucle y relevo), `post/` (SFT, MTP, DPO, GRPO), `eval/`, `logs/`, y `docs/DATA_CARD.md` y `MODEL_CARD.md`.

**Reproducibilidad:** semillas fijas, versiones exactas de las librerías, un YAML por fase y el manifiesto de datos.

**Al cerrar la v1:** reporte técnico corto (sirve para el portafolio).

---

## 11. Riesgos y planes B

| Riesgo | Señal | Plan B |
|---|---|---|
| Inestabilidad en FP16 | El escalador cae; NaN | Ya hay QK-norm y compuerta; bajar el LR; z-loss; volver al checkpoint |
| Falta de memoria | Pico > 14.3 GiB | Momento en bf16; micro-batch menor; checkpointing selectivo |
| Router desbalanceado o expertos muertos | Máx/media > 3 | Subir el coeficiente de balanceo |
| Muy lento | < ~2,500 tokens/s | Probar el otro paralelismo, otro kernel de MoE y torch.compile. Si no basta: 16 capas × d1280 |
| El formato no funciona | Falla la prueba de logits | **Pasó:** con llama.cpp sin parche se cae; con el parche, todo pasa (2.5). Si el PR no entra o el A/B no favorece a A: GLM-4.5 sin compuerta |
| El MTP no acelera | Aceptación baja | Se publica sin MTP; el modelo no cambia |
| Kaggle cambia las cuotas, o ve el relevo coordinado como una forma de saltarse la cuota | Avisos o bloqueos | Cada amigo corre su turno en su propia cuenta, sin compartir credenciales. Salida: terminar la fase en una H100 alquilada |
| Licencias | Un dataset de origen dudoso | Ante la duda, fuera |
| Contaminación | Saltos raros en un benchmark | Filtro de Bloom de 13-gramas + evaluaciones propias |
| TRC no llega | — | El programa se detiene en 100B, o en ~180–300B con ~USD 1,000 |
| Cansancio | — | Fases cortas; cada una se puede publicar |

---

## 12. Próximos pasos (v1)

| # | Tarea | Dónde | Horas | Termina cuando |
|---|---|---|---|---|
| T0.1 | ✅ **Prueba de formato** (`notebooks/t0_1_formato_mtp_tokenizador.ipynb`): con parche, todo pasa (2.5). Falta correrla con el tokenizador real de 64K | CPU | ~0.5 | Hecho el 11 nov |
| T0.2 | ✅ **Velocidad y memoria** (T0.2 v2, `notebooks/t0_2v2_afmoe_2xT4.ipynb`): piezas de velocidad, micro-lote, 2K/4K/16K, guardado y carga, balanceo | 2×T4 | ~1 | Hecho el 12 nov |
| T0.3 + T2 | ✅ **LR y estabilidad en FP16** (`notebooks/t0_3_t2_lr_2xT4.ipynb`): dos rondas, todas estables. El tronco va a ×1, la receta del Ludwig 1 (P25) | 2×T4 | ~3 × 9.5 + 4 × 11 | Hecho el 15 nov |
| T1 | **Datos:** fuentes cerradas (4.2, mezcla v3); pre-tokenizado con descontaminación y manifiestos. El código está listo y probado (P18–P24). La sesión 1 hizo 28 shards en 10.4 h; faltan unas 5 sesiones, que pueden correr mientras corre el tronco (`notebooks/t1_shards_cpu.ipynb`) | CPU | ~10 min + ~6 × 11 | Los 168 shards del tronco, con sus manifiestos |
| G0 | ✅ **Compuerta G0** (5.7, P27): G0-0 (ensayo del relevo) y G0-1 (16K) el 17 nov, las condiciones 1–4 en el turno 1 del tronco y la reanudación en otra cuenta en el turno 5 | 2×T4 | ~1 + los dos primeros turnos | Hecho en el primer tronco; el `tronco2` usa el mismo relevo y no la repite |
| T3 | **Tronco** hasta 22.5B. El primer intento llegó al paso 3348 con el balanceo de los expertos roto desde el ~1250 (P29); el 25 nov, `tronco2` desde cero (P30), y desde su segundo turno `tronco3`, con 3 capas densas (P32) | Relevo | ~1,550–2,450 | — |
| T4 | **Rama de la v1** (2.5B, con 16K y MTP) | Relevo | ~225–355 | Base lista |
| T5 | **SFT, MTP, GGUF y evaluación** | 2×T4 | ~60–150 | Ludwig 2 · 25B publicado |

**Total de la v1:** ~1,920–3,090 h de 2×T4, de las cuales el relevo son ~1,750–2,800. Las tareas T0, T2, G0 y T5 corren con la cuota sobrante, en paralelo al relevo.

---

## 13. Fuentes

1. MobileMoE: Scaling On-Device Mixture of Experts (Meta, 2026). https://arxiv.org/pdf/2605.27358
2. MobileLLM-R1: Exploring the Limits of Sub-Billion Language Model Reasoners with Open Training Recipes (Meta, ICLR 2026). https://arxiv.org/pdf/2509.24945
3. Gated Attention for Large Language Models (Qwen, NeurIPS 2025). https://ar5iv.labs.arxiv.org/html/2505.06708
4. Muon is Scalable for LLM Training (Moonlight, Kimi). https://arxiv.org/abs/2502.16982v1
5. Fantastic Pretraining Optimizers and Where to Find Them (ICLR 2026). https://arxiv.org/abs/2509.02046v1
6. DeepSeek-V3 Technical Report (MTP). https://arxiv.org/pdf/2412.19437
7. FastMTP. https://arxiv.org/html/2509.18362v1
8. Pre-Training Curriculum for Multi-Token Prediction. https://arxiv.org/html/2505.22757v1
9. Model Merging in Pre-training of LLMs (ByteDance, NeurIPS 2025). https://neurips.cc/virtual/2025/poster/118889
10. SmolLM2: When Smol Goes Big. https://arxiv.org/pdf/2502.02737
11. The FineWeb Datasets. https://arxiv.org/pdf/2406.17557
12. OLMoE: Open Mixture-of-Experts Language Models. https://arxiv.org/html/2409.02060v2
13. Small Models Struggle to Learn from Strong Reasoners. https://arxiv.org/pdf/2502.12143
14. DeepSeek-R1. https://arxiv.org/html/2501.12948v1
15. TinyGSM. https://ar5iv.labs.arxiv.org/html/2312.09241
16. OpenMathReasoning (NVIDIA). https://huggingface.co/datasets/nvidia/OpenMathReasoning
17. Granite-3.0-1B-A400M (config). https://huggingface.co/ibm-granite/granite-3.0-1b-a400m-base
18. Precios de H100 (noviembre de 2026). https://akash.network/the-bid/h100-rental-price-2026-cost-per-hour/
19. Código revisado: transformers 5.22.2 (`models/qwen3_next`) y llama.cpp de noviembre de 2026 (`src/models/qwen3next.cpp`, `conversion/qwen.py`, `common/speculative.cpp`).
20. The Role of Language Imbalance in Cross-lingual Generalisation (2024). https://arxiv.org/abs/2404.07982v4
21. SmolLM3: smol, multilingual, long-context reasoner (Hugging Face). https://huggingface.co/blog/smollm3
22. Multilingual Pretraining Using a Large Corpus Machine-Translated from a Single Source Language (TransWeb-Edu). https://arxiv.org/pdf/2410.23956v1
23. Resultados de la T0.1 en este repo: `results/t0_1_stock.json` y `results/t0_1_patched.json`.
24. Estudio de FinePhrase (web reescrita por un modelo). https://arxiv.org/abs/2604.13977
25. Estudio de Meta sobre la dosis de texto reescrito. https://arxiv.org/abs/2510.01631
26. The Common Pile v0.1: An 8TB Dataset of Public Domain and Openly Licensed Text (EleutherAI, 2025). https://arxiv.org/abs/2506.05209 · dataset del código: https://huggingface.co/datasets/common-pile/stackv2_edu_filtered
27. CroissantLLM: A Truly Bilingual French-English Language Model (2024). https://arxiv.org/abs/2402.00786
28. DCLM sin duplicados (Zyphra, del trabajo de Zyda-2). https://huggingface.co/datasets/Zyphra/dclm-dedup
29. Revisión de las fuentes del tronco, en este repo: `results/datos_tronco_v3/`.
