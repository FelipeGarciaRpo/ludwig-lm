# Bitácora de Ludwig 2

Todo lo que se corre, se mide o se decide, en orden. Cada entrada dice qué se hizo, dónde corrió, qué archivos dejó y qué se concluyó. Las decisiones vigentes están en [`docs/DECISIONS.md`](../DECISIONS.md) y el diseño completo en [`DISENO.md`](DISENO.md).

---

## 2026-11-11 · Diseño v1.0 y v1.1

- **v1.0:** 26 capas × 1024, MoE de 40 expertos top-4 + experto compartido, formato `Qwen3NextForCausalLM` (formato A), MTP, Muon + AdamW, FP16, tronco con ramas. Receta de datos adaptada del plan del modelo grande.
- **Crítica externa (ChatGPT) → v1.1:**
  - Nombre más preciso: arquitectura propia guardada en el formato de Qwen3-Next.
  - Momento de Muon en bf16 y plan B de 32 expertos.
  - Compuerta G0 antes del tronco.
  - A/B de formato A contra B (GLM-4.5).
  - Español de 4% a 10%.
  - Topes de datos sintéticos.
  - Auditoría de licencias antes de publicar pesos.
  - Condiciones para pasar de una fase a la siguiente.
- **Dónde:** chat de diseño.

## 2026-11-11 · T0.1: formato A en CPU

- **Qué:** un modelo diminuto con la estructura de Ludwig 2 en formato A se entrena, se exporta, se convierte a GGUF y se compara contra llama.cpp (`scripts/smoke_format.py`).
- **Dónde:** sandbox del chat de diseño, llama.cpp `dd6b945`.
- **Resultado:** llama.cpp sin cambios **se cae** al cargar el modelo (`GGML_ASSERT(buffer)`): trata `qwen3next` como híbrido aunque ninguna capa sea recurrente. Con un parche de 3 líneas todo pasa: probabilidades a 1.9e-6, greedy 4/4 y MTP especulativo 4/4.
- **Archivos:** `results/t0_1_stock.json`, `results/t0_1_patched.json`, `patches/`.
- **Conclusión:** el formato A solo corre en un llama.cpp parchado. Eso es un riesgo para publicar.

## 2026-11-11 · Banco v1 en 2×T4 (Kaggle)

- **Qué:** banco de pruebas con un modelo propio en PyTorch, escrito por otro chat. Tiene router sigmoide con bias (sin pérdida auxiliar), compuerta, 1.ª capa densa, residuo en FP32, MoE con bmm y relleno, y Block AttnRes opcional. No usa `ludwig_lm/`.
- **Dónde:** Kaggle, 2× Tesla T4, torch 2.11 + CUDA 13.0, 31 GiB de RAM, 4 CPUs. Duró 43 min.
- **Archivos:** `results/banco_t4_v1/banco_t4_v1.py` (copia congelada) y `results/banco_t4_v1/`.

| Medición | Resultado |
|---|---|
| Matmul FP16 en una T4 | 21.8–27.0 TFLOPS de 65 teóricos. Las matrices grandes no van más rápido: la T4 está limitada por potencia |
| Comunicación entre las dos GPUs | 3.2 GB/s (all-gather, P2P activo) |
| 26 × d1024, 40 expertos, 8 secuencias por GPU | **3,856 tokens/s**, pico 13.2 GiB (reservado 13.8): **1,801 h para 25B** |
| 16 × d1280 / 11 × d1536 / 6 × d2048 | 4,267 / 4,287 / 3,866 tokens/s. La profunda es 1.11× más lenta que la más rápida |
| 48 expertos | Solo cabe con 4 secuencias: 2,926 tokens/s (−24%) |
| 60 expertos | Solo cabe con el momento en la RAM y 4 secuencias: 2,670 tokens/s (−31%) |
| MoE con un matmul por experto | 3,329 tokens/s, 14% menos que bmm con relleno |
| Block AttnRes | 2,389 tokens/s, pero con 4 secuencias contra 8: la comparación no es justa |
| 60 pasos en FP16 con datos sintéticos | Pérdida 11.27 → 1.79, 0 pasos saltados, 0 NaN, escala fija en 16,384 |
| Balance del router en esos 60 pasos | El experto más cargado llegó a 8.4× el promedio |
| KDA (Kimi Linear) | No se probó: faltó instalar `fla-core`, donde están los kernels |

- **Lo que queda decidido con esto:** la forma 26 × 1024 se queda, y 40 expertos también. FP16 es estable al menos en corridas cortas.
- **Lo que hay que corregir antes de creerle:**
  - **El bmm.** Sus pruebas de velocidad usaron tokens aleatorios, así que el router quedó balanceado: es el mejor caso del bmm. Con desbalance, el bmm rellena todos los expertos hasta la carga del más cargado, y eso explica parte de los 2,465 tokens/s del entrenamiento corto.
  - **El costo de partir el micro-lote** (~20–25%) se dedujo: no se midió 40 expertos con 4 secuencias.
  - **El bias de balanceo** se mueve 0.001 por paso con un signo fijo. En 60 pasos casi no alcanza a moverse.
  - **La memoria** está justa para 3 meses: 13.8 de ~14.5 GiB utilizables.
- **Tiempo realista:** 1,801 h es el ritmo puro. Con datos reales, evaluación, guardados, reinicios de sesión y la fase de 16K, planear **~2,000 h (~14–15 semanas de relevo)**.
- **Análisis del otro chat**, resumido: está de acuerdo con la forma y los 40 expertos. Atribuye la caída con 48 y 60 expertos a la comunicación por PCIe. Diagnosticó bien el error de instalación de KDA (lo verifiqué: `flash-linear-attention` 0.6.0 depende de `fla-core`, que trae `fla.ops`). Considera injusta la medición de AttnRes. Propone un banco v2 con KDA bien instalado, AttnRes optimizado, el momento en la RAM, gradientes en FP16 y arreglos del router.

## 2026-11-11 · T0.1-C: formato AFMoE en CPU

- **Por qué:** el modelo del banco v1 (compuerta + router sigmoide con bias + capa densa + experto compartido) no cabe en el formato A ni en el B. Buscando en transformers 5.22.2 y llama.cpp, **`AfmoeForCausalLM`** (Arcee Trinity) tiene todo eso y corre en llama.cpp sin parches. Sí cambia dos cosas: lleva normas a la entrada y a la salida de cada subcapa, y alterna 3 capas locales (ventana deslizante con RoPE) con 1 global (sin RoPE).
- **Qué:** `scripts/smoke_format_afmoe.py` con `configs/tiny_afmoe.json`. Los prompts son más largos que la ventana, y el bias de los expertos se puso grande a propósito para que cambie qué expertos se eligen.
- **Dónde:** sandbox del chat de diseño, llama.cpp `dd6b945` **sin cambios**.
- **Resultado:** todo pasa. Recarga idéntica, conversión a GGUF, probabilidades a 5.9e-5, greedy 4/4, y llama.cpp respeta el bias de los expertos.
- **Dos detalles obligatorios del formato:**
  - **`mup_enabled: true`:** el grafo de llama.cpp siempre multiplica los embeddings por √d.
  - **`route_norm: true`:** transformers siempre normaliza los pesos del top-k, y llama.cpp solo lo hace con esa bandera.
  - Sin ellos, las salidas no coinciden.
- **Archivos:** `results/t0_1c_afmoe.json`, `configs/ludwig2_afmoe.json` (el modelo real: **1.906B totales, 0.491B activos**, 6 capas globales de 26).

## 2026-11-11 · Propuesta v1.2 (pendiente de aprobación de Pipe)

1. **Formato AFMoE en lugar de Qwen3-Next.**
   - **Corre en llama.cpp sin parche** (T0.1-C).
   - **Trinity Nano es el precedente:** 6B totales y 1B activos, ancho 1024, 56 capas. Usa Muon, router sigmoide, compuerta, normas sándwich y 3 capas locales por 1 global. Se entrenó con 10T tokens sin picos de pérdida.
   - **Junta lo que querían los dos chats:** la compuerta de este y el router sigmoide con bias del otro.
   - **Pierde el MTP**, porque el grafo de AFMoE en llama.cpp no lo tiene.
   - **Gana caché KV:** a 16K son 136 MiB contra 416.
2. **Balanceo:** bias sin pérdida auxiliar actualizado con **SMEBU** (Trinity), en lugar del signo fijo, más una pérdida auxiliar por secuencia de 1e-4. El λ se calibra en la T0.3.
3. **AttnRes y KDA fuera de la v1.**
   - En llama.cpp solo existen dentro de la arquitectura Kimi-K3 (MLA + KDA + MoE latente).
   - transformers no tiene una clase nativa para ellos.
   - Los kernels de KDA en la T4 no están probados.
   - Quedan como línea de investigación, fuera del camino crítico.
4. **Entrenar con la clase de transformers más nuestras piezas de velocidad:**
   - MoE con bmm registrado en la interfaz de expertos de transformers.
   - Residuo en FP32.
   - Gradientes en FP16 entre GPUs.
   - Prefetch explícito de FSDP2.
   - Momento de Muon en la RAM como opción.
   - Lo mide la **T0.2 v2**.
5. **El notebook T0.2 del formato A ya no se corre.** Lo reemplaza la T0.2 v2. Además, su prueba de `batched_mm` se quedaría sin memoria: ese kernel de transformers copia los pesos de los expertos para cada token.

**Siguiente paso:** con el visto bueno de Pipe, pasar el modelo y el entrenamiento a AFMoE y armar la T0.2 v2 (~1 h de 2×T4).

## 2026-11-12 · Entran los archivos crudos del banco v1

- **Qué:** se copiaron sin editar `resultados.json`, `reporte.txt` y `logs/` de la salida del notebook en Kaggle (el zip que bajó Pipe). Quedan junto a la copia congelada del script y `salida_kaggle.txt`.
- **Dónde:** PC de Pipe (Windows), sin acelerador: solo se copiaron archivos.
- **Archivos:** `results/banco_t4_v1/resultados.json`, `results/banco_t4_v1/reporte.txt` y `results/banco_t4_v1/logs/` (22 logs; 10 salieron vacíos de Kaggle).
- **Conclusión:** la evidencia del banco v1 queda completa. Ojo con una línea: `reporte.txt` dice que los kernels de KDA no corren en FP16, pero el error real fue `ModuleNotFoundError: No module named 'fla.ops'` (`salida_kaggle.txt`, líneas 61–79). Vale lo de la entrada del banco v1: KDA no se probó.
- **Siguiente paso:** sigue pendiente el visto bueno de Pipe a la propuesta v1.2.

## 2026-11-12 · v1.2 aprobada

- Pipe aprobó la propuesta v1.2.
- **Formato:** AFMoE.
- **Balanceo:** SMEBU más una pérdida pequeña por secuencia.
- **Fuera de la v1:** el MTP, AttnRes y KDA.
- **Entrenamiento:** la clase de transformers más las piezas de velocidad.
- `docs/DECISIONS.md` marca con ⛔ las decisiones que quedan reemplazadas (D02, D03, la RoPE parcial de D04, el router de D05 y D07). El cuerpo de `DISENO.md` se reescribe con los números de la T0.2 v2.

## 2026-11-12 · T0.2 v2 lista para Kaggle

- **Qué se construyó:**
  - **`ludwig_lm/afmoe.py`.** Envuelve `AfmoeForCausalLM` sin cambiar el formato del checkpoint:
    - kernel de expertos con bmm y relleno, registrado en la interfaz de expertos de transformers;
    - router en FP32 que anota sus decisiones sin contarlas dos veces cuando el checkpointing recalcula;
    - `expert_bias` como buffer replicado;
    - balanceador SMEBU o de signo;
    - pérdidas: CE por bloques con la cabeza atada, z-loss de salida, pérdida por secuencia y z-loss del router;
    - inicialización con las normas de salida en 1/√26.
  - **`ludwig_lm/muon.py`.**
    - Distingue el `q_proj` de AFMoE, que no trae la compuerta adentro.
    - Manda el router a AdamW.
    - Puede guardar el momento en la RAM de la CPU como DTensor sobre una malla de CPU, para que el checkpoint distribuido lo guarde completo.
  - **`scripts/bench_afmoe.py`.** Banco del entrenamiento real:
    - modelo construido directo en cada GPU y luego fragmentado;
    - FSDP2 con gradientes en FP16 y prefetch explícito;
    - residuo en FP32 con autocast;
    - modo de 60 pasos con el texto de Markov del banco v1;
    - prueba del checkpoint completo, que incluye el bias y el momento del balanceador.
  - **`notebooks/t0_2v2_afmoe_2xT4.ipynb`**, generado como siempre desde `scripts/make_notebooks.py`.
- **Probado en CPU** (sandbox del chat de diseño, 2 procesos gloo, modelo diminuto):
  - El bmm con relleno da **exactamente** lo mismo que los expertos de transformers.
  - El router anota una vez por capa MoE aunque el checkpointing lo recalcule.
  - Con autocast, el residuo se queda en FP32.
  - SMEBU y el signo mueven el bias en la dirección correcta.
  - Las tres variantes de kernel y precisión corren.
  - El checkpoint se restaura exacto: pesos, bias y momento del balanceador.
  - En 40 pasos con texto de Markov, SMEBU (λ = 0.01) terminó con la carga máxima en 2.27× el promedio y 0 expertos sin tokens. El signo fijo (0.001) terminó en 3.63× con 1 experto sin tokens.
- **Lo que solo se puede probar en las T4:**
  - el momento en la CPU (memoria fijada y malla de CPU);
  - los gradientes en FP16 por NCCL;
  - la memoria real con 8 secuencias de 2K;
  - 16K de contexto.
- **Un cambio de memoria:** la pérdida va en bloques de 1,024 tokens, no de 4,096. Con 64K de vocabulario, un bloque de 4,096 pide ~3.8 GB temporales.
- **Siguiente:** correr el notebook en Kaggle (GPU T4 x2) y guardar `results/t0_2v2.jsonl`.

## 2026-11-12 · T0.2 v2, primer intento: sin memoria

- **Dónde:** Kaggle, 2× T4, torch 2.11 + CUDA 13.0, código `10a7d88` subido como dataset.
- **Resultado:** las mediciones de la A a la F se quedaron sin memoria en el primer micro-paso, antes de medir nada. Salida: `results/t0_2v2_intento1/`.
- **Causas** (dos errores del código, no de la T4):
  1. transformers inicia el router de AFMoE en cero. Todos los puntajes valen 0.5, así que los 16,384 tokens eligen los mismos 4 expertos, y el bmm con relleno reservó espacio para 16,384 tokens en los 40 expertos: 1.25 GiB por buffer (A, D, E, F) y 640 MiB (C).
  2. En la T4, SDPA con `enable_gqa` cae a un kernel que guarda los pesos de atención B×H×T×T en FP32: los 2.00 GiB de la medición B, en la primera capa global.
- **Arreglo** (patch 0004):
  - El router arranca con N(0, 0.02).
  - En GPUs anteriores a Ampere se repiten las cabezas KV en vez de usar `enable_gqa`.
  - Atención causal directa cuando la ventana cubre la secuencia.
  - Loop por experto si el más cargado pasa de 2× el promedio.
  - El notebook baja a 4 secuencias si 8 no caben, y los entrenamientos van con 4 × 2.
- **Probado en CPU:**
  - El bmm y el loop dan exactamente lo mismo que transformers, en salidas y gradientes.
  - Al inicio, la carga máxima quedó en 1.5–1.8× el promedio, en vez de 4× (modelo diminuto).
  - La atención causal directa da los mismos logits.
- **Siguiente:** subir el código nuevo como versión del dataset y repetir la T0.2 v2.

## 2026-11-12 · T0.2 v2, primer intento: lo que agrega el log completo

- **Qué:** la entrada anterior se escribió antes de que terminara la corrida. El log completo (`results/t0_2v2_intento1/salida_kaggle.txt`) muestra que se quedaron sin memoria **las 13 mediciones**, no solo de la A a la F:
  - G (4 secuencias) completó el paso 0, con la carga máxima en 10.00× el promedio (el máximo posible con top-4 de 40), y se cayó en el paso 1 pidiendo 542 MiB.
  - J, K, L y M pidieron los mismos 1.25 GiB que la A.
  - H (4K) pidió 1 GiB, e I (16K), 16 GiB.
- **Dónde:** PC de Pipe, leyendo el log. No corrió nada.
- **Pregunta abierta:** los 16 GiB de la I son 1 × 16 × 16,384² × 4 bytes, una matriz de atención completa en FP32, y el 1 GiB de la H no cuadra con ninguna de las dos causas. El patch 0004 solo quita la máscara cuando la ventana cubre toda la secuencia (hasta 2K): a 4K y 16K las capas locales siguen recibiendo una máscara explícita, y su memoria en la T4 no se ha medido.
- **Siguiente:** en el segundo intento, mirar primero la H y la I.
