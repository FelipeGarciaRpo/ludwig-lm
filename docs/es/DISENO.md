# Ludwig 2 — Documento de diseño

**Versión:** 1.1 · 11 de noviembre de 2026
**Estado:** diseño cerrado para la v1 (25B tokens), salvo una decisión que se toma con datos: el formato A o B (2.5). Fases hasta 1T planificadas, sin obligación de llegar.

> **Cambios de la v1.1** (tras la crítica externa y la primera prueba real):
> 1. La T0.1 encontró que llama.cpp **sin parche se cae** con el formato Qwen3-Next de atención completa; con un parche de 3 líneas pasan todas las pruebas, MTP incluido (2.5).
> 2. El formato se decide con un A/B en la T2: A = Qwen3-Next (compuerta + softmax) contra B = GLM-4.5 (sin compuerta, router sigmoide). El A/B también resuelve la duda del router.
> 3. Momento de Muon en bf16 por defecto; plan B de 32 expertos si no queda margen de memoria (5.2).
> 4. Compuerta G0: ninguna hora de tronco sin una prueba sostenida de varias horas con guardado y reanudación (5.7).
> 5. Español del 4% al 10% en el tronco y las ramas, y 25–30% en el SFT (4.9).
> 6. Topes y diversidad para los datos sintéticos (4.10); auditoría de licencias antes de publicar pesos (10); compuertas para seguir de una fase a la siguiente (7.1).
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

**Prioridades de capacidad:** inglés + Python que razone > JS/TS > español (un plus) > otros lenguajes.

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

- **El mismo de 64K del plan del modelo grande** (`tokenizador_64k_v1_1.py`): los 256 bytes siempre en el vocabulario, sin normalización, entrenado con documentos enteros, con val aparte y en modo datos (el texto crudo nunca produce tokens de control).
- **Tokens especiales desde el día uno:** plantilla de chat, `<think>` y `</think>` (id 4…), FIM (ids 10–13), llamadas a herramientas y fin de texto (id 0) 📏.
- **Ventaja de compartirlo con el modelo grande:** si algún día existe Ludwig 3 con el mismo tokenizador, podrá destilarle logits a Ludwig 2, y Ludwig 2 le servirá de borrador para decodificación especulativa.
- 📏 **Medir bytes por token** en inglés, Python, JS, matemática y español, y compararlo con el de Llama 3 (el que usa MobileMoE).
- ❗ **El tokenizador muestrea el código de la misma fuente que el tronco.** Si no, quedan tokens que casi no salen en el entrenamiento (glitch tokens).
- **Si el tokenizador del plan grande cambia, Ludwig 2 usa la versión final.** Después del primer paso del tronco ya no se puede cambiar.

---

## 4. Datos

### 4.1 Qué se reutiliza del plan del modelo grande y qué cambia

**Se reutiliza tal cual (PLAN_MODELO_NUEVO v0.5):**
- **Tronco + ramas.** El tronco lleva una mezcla estacionaria que aguanta hasta 1T sin pasar de ~4 pasadas por ninguna fuente. Lo escaso y lo mejor van solo en las ramas de enfriamiento.
- **Fuentes y procesamiento** (4.2 y 4.3).
- **Reglas de licencias** 🟢🟡🔴 (4.8).

**Cambia porque el modelo es ~4 veces más chico:**
1. **Menos dominios.** Con menos capacidad, cada dominio diluye a los demás. El código se concentra en Python y JS/TS.
2. **Más matemática:** 10% en el tronco (antes 6%) y 22% en las ramas (antes 18%), porque la meta es razonar. MobileLLM-R1 subió la matemática a ~35% en su fase 2 de preentrenamiento, y le funcionó.
3. **Español al 10%** en el tronco y las ramas, y 25–30% en el SFT (4.9).
4. **Trazas más cortas:** los modelos chicos aprenden mejor de cadenas cortas (sección 9).
5. **Contexto de 16K** en la v1 (antes 32K).

### 4.2 Fuentes

| Dominio | Fuente | Tamaño aprox. | Licencia | Dónde va |
|---|---|---|---|---|
| Web educativa | FineWeb-Edu (puntaje ≥3; ≥4 en las ramas) | ~1.3T | 🟢 ODC-By | Tronco y ramas |
| Web diversa | DCLM-baseline | ~3.8T | 📏 | Tronco y ramas |
| Libros de texto sintéticos | Cosmopedia v2 (generado con Mixtral) | ~25–28B | 📏 | Tronco y ramas |
| Conocimiento | Wikipedia (inglés), arXiv, StackExchange | 3.7B / 28B / 19.6B | 📏 | Tronco y ramas |
| Código | Stack-Edu (Python 21.8B, JS 11.1B, TS 3.0B…) o The Stack v3 filtrado con sus clasificadores ❓ | 125B | 📏 | Tronco y ramas |
| Código explicado | Notebooks de Jupyter y Kaggle, issues de GitHub | — | 📏 | Tronco y ramas |
| Matemática | FineMath 3+ y 4+, OpenWebMath, Algebraic Stack, MegaMath | 34B / 12.6B / 12.6B / — | 📏 | Tronco y ramas |
| Razonamiento | OpenMathReasoning: 3.2M soluciones con cadena de pensamiento + 1.7M con Python (TIR), generadas con DeepSeek-R1 y QwQ-32B | Millones de soluciones | 🟢 CC-BY-4.0 | Ramas y SFT |
| Matemática escolar | Train de GSM8K y MATH, con soluciones cortas generadas en la fábrica (9.3) | — | 🟢 | Ramas y SFT |
| Español | FineWeb2 (español) filtrado por calidad, Wikipedia en español, TransWeb-Edu en español (FineWeb-Edu traducido) y un poco de texto paralelo inglés–español | De sobra | 📏 | Tronco, ramas y SFT |

- **Del plan grande sigue pendiente:** Stack-Edu no trae el contenido de los archivos; hay que bajarlo de Software Heritage respetando sus términos. The Stack v3 lo trae inline y tiene licencia odc-by. ❓ Decidirlo antes del tronco.
- **Cada fuente lleva su ficha** con licencia, tamaño, versión y hash.

### 4.3 Procesamiento

1. **Lector propio**, determinista y con reanudación exacta: mezcla ponderada por fuente; streaming de parquet; EOS al final de cada documento; ventanas de largo fijo sin botar colas; guarda y recupera su estado; reanuda saltando por grupos de filas con pyarrow, no con `skip(n)`. En las T4 se reutiliza el de Ludwig 1, ampliado.
2. **Deduplicación:** cada fuente ya viene deduplicada por dentro. Entre FineWeb-Edu y DCLM se quitan las URL repetidas con un hash de 64 bits.
3. **Descontaminación** en el momento, con un filtro de Bloom de los 13-gramas de GSM8K, MATH, HumanEval, MBPP, MMLU, ARC, HellaSwag, PIQA y WinoGrande.
4. **Etiquetas de fuente** al estilo MeCo 🧪: la fuente antepuesta en el tronco y nada en las ramas, para que el modelo final funcione sin etiquetas.
5. **FIM** en el 10% del código. Los ids 10–13 los pone el lector, no el texto.
6. **Val:** el último archivo de cada fuente nunca se entrena. La pérdida se reporta en bits por byte, por dominio.
7. **Trazas y chat** con la plantilla de chat y `<think>` como token (id 4), nunca como texto plano. Si no, el modelo aprende dos formatos para lo mismo.

### 4.4 Mezcla del tronco (estacionaria, de 0 a 900B) ✅

| Dominio | % | Qué lleva | Tokens en la v1 (tronco de 22.5B) | Pasadas a 1T (tronco de 900B) |
|---|---|---|---|---|
| Web (inglés) | 50% | FineWeb-Edu 60% / DCLM 40% (como SmolLM2) | 11.25B | FineWeb-Edu ~0.21; DCLM ~0.05 |
| Código | 20% | Ver 4.6 | 4.5B | Python ~3.7 |
| Matemática | 10% | FineMath 3+, OpenWebMath, Algebraic Stack, MegaMath | 2.25B | ~1.5 sin contar MegaMath |
| Conocimiento | 5% | Wikipedia, arXiv, StackExchange | 1.1B | ~0.9 |
| Libros sintéticos | 3% | Cosmopedia v2 | 0.7B | ~1.1 |
| Español | 10% | FineWeb2, TransWeb-Edu y Wikipedia en español; ≤0.5% de texto paralelo | 2.25B | 📏 |
| Instrucción y razonamiento corto | 2% | Fuentes 🟢 de 4.2 | 0.45B | Bajo |

- **Contexto:** 2K hasta 50B; 4K desde 50B.
- **Por qué FineWeb-Edu pesa tanto:** en MobileLLM-R1, quitarlo fue lo que más degradó conocimiento, matemática y código a la vez; lo llaman el "pegamento" entre dominios.
- **Por qué el código ayuda a razonar:** en el mismo estudio, StarCoder subió más la matemática que lo que OpenWebMath subió el código.

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

| Lenguaje | % del código | Tronco v1 | Tronco a 1T | Pasadas por Stack-Edu a 1T |
|---|---|---|---|---|
| Python | 45% | 2.0B | 81B | 3.7 |
| JavaScript + TypeScript | 25% | 1.1B | 45B | 3.2 |
| C#, Go, Rust, Java, C/C++, SQL, Shell | 15% | 0.7B | 27B | — |
| Notebooks, issues y Markdown | 15% | 0.7B | 27B | — |

- Si la fuente termina siendo The Stack v3 filtrado, habrá más margen 📏.
- Más de ~4 pasadas por los datos escasos ya no valen como datos nuevos (Muennighoff). Pasado eso, se reformula al estilo Kimi en vez de repetir.

### 4.7 Lo que ve la v1 en total (25B)

| | Tronco (22.5B) | Rama (2.5B) | Total |
|---|---|---|---|
| Web (inglés) | 11.25B | 0.65B | ~11.9B |
| Código | 4.5B | 0.63B | ~5.1B |
| Matemática | 2.25B | 0.55B | ~2.8B |
| Razonamiento e instrucción | 0.45B | 0.33B | ~0.8B |
| Conocimiento y libros | 1.8B | 0.1B | ~1.9B |
| Español | 2.25B | 0.25B | ~2.5B |

### 4.8 Licencias 🟢🟡🔴

- 🟢 **Entra:** texto humano con licencia abierta, y datos sintéticos generados con modelos abiertos de licencia permisiva: DeepSeek-R1/V3 (MIT), Qwen3 y QwQ (Apache 2.0), gpt-oss (Apache 2.0), Mixtral (Apache 2.0). En cada dataset se revisa qué modelo generó cada subconjunto.
- 🟡 **Solo si aceptas la condición:** datos generados con modelos cuya licencia pone condiciones. Por ejemplo, Llama pide que el modelo entrenado con sus salidas lleve "Llama" en el nombre. Por defecto, fuera.
- 🔴 **No entra:** salidas de APIs cerradas, cuyos términos prohíben entrenar modelos competidores. Ejemplo: TinyGSM, generado con GPT-3.5. La regla es "ante la duda, fuera".

### 4.9 Español: por qué 10% y no 4% ni 50%

- **No 50/50.** En experimentos controlados con idiomas clonados, tener un idioma dominante mejora el rendimiento del idioma menos frecuente y alinea más las representaciones entre idiomas. Con modelos grandes o entrenamientos largos, una mezcla 90/10 rindió mejor en los dos idiomas que una 50/50. Con idiomas reales, el idioma menos frecuente igual se beneficia del dominante. Y el inglés tiene los mejores datos y casi todos los benchmarks.
- **No 4%.** En la v1 serían ~1B tokens de español: poco para que converse bien. Con 10% son ~2.5B en la v1 y ~90B a 1T.
- **Qué queda:** contando solo el lenguaje natural del tronco, ~86% inglés y ~14% español, cerca del 90/10.
- **Referencia:** SmolLM3 (3B) puso 12% de web multilingüe repartida entre 5 idiomas, y en HellaSwag en español le ganó a Qwen2.5-3B y Llama 3.2 3B.
- **Calidad sobre cantidad:** FineWeb-Edu traducido al español (TransWeb-Edu) aporta el mismo tipo de texto educativo que la base en inglés, y un poco de texto paralelo ayuda a alinear los dos idiomas.
- **Qué se mide:** bits por byte en español, compresión del tokenizador en español y MGSM, Belebele, Global-MMLU, HellaSwag y Flores (inglés↔español). Si el español se queda atrás, se sube en las ramas, que sí se pueden cambiar sin tocar el tronco.
- **Costo para el inglés:** 6 puntos menos de web en inglés en el tronco. Código y matemática no se tocan.

### 4.10 Datos sintéticos: topes y diversidad

- **Topes:** ≤10% del tronco y ≤35% de cada rama. Hoy el tronco tiene ~5% (Cosmopedia + instrucción) y las ramas ~25–30%.
- **Al menos 3 familias de maestros** (por ejemplo Qwen, DeepSeek y gpt-oss), para no copiar el estilo de uno solo.
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

### 5.4 Alarmas automáticas

| Señal | Umbral | Acción |
|---|---|---|
| Pico de pérdida | > media + 4σ de los últimos 200 pasos | Saltar el lote; si se repite 3 veces, volver al checkpoint anterior y bajar el LR un 20% |
| Gradientes no finitos | Más de 10 en 1,000 pasos | Alarma; revisar el escalador |
| Escalador de pérdida | Por debajo de 1 | Alarma (desborde de FP16) |
| Carga por experto | Máx/media > 3 durante 500 pasos | Subir el coeficiente de balanceo |
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
3. **Memoria pico ≤ 13.5 GiB** durante toda la corrida, evaluación y guardado incluidos.
4. **Al menos 2 guardados** subidos a Hugging Face y **una reanudación en otra cuenta**, con el hash del primer lote igual al esperado.
5. **16K probado:** 50 pasos a 16K de contexto sin falta de memoria.

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
| T0.2 | **Velocidad y memoria** (`notebooks/t0_2_velocidad_memoria_2xT4.ipynb`): kernels de MoE, micro-batch, 2K/4K/16K, 32 expertos, MTP, guardado y carga | 2×T4 | ~1 | Tokens/s y memoria pico medidos |
| T0.3 | **Estabilidad en FP16:** 2–3K pasos con el LR pico | 2×T4 | ~10–20 | Sin NaN y con el escalador estable |
| T1 | **Datos:** decidir la fuente de código; lector con descontaminación y manifiestos | CPU | — | Determinista y reanuda exacto |
| T2 | **LR y formato:** 3 corridas de LR (formato A) + 1 corrida del formato B, ~0.2–0.3B tokens cada una | 2×T4 | ~55–90 | LR y formato fijados (2.5) |
| G0 | **Compuerta G0** (5.7): ≥6 h sostenidas, guardado, reanudación en otra cuenta, 16K | 2×T4 | ~8–10 | Pasan las 5 condiciones |
| T3 | **Tronco** hasta 22.5B | Relevo | ~1,550–2,450 | — |
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
