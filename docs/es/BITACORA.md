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

## 2026-11-12 · T0.2 v2, segundo intento: todo cabe

- **Qué:** la T0.2 v2 completa con el código `60a341f` (patch 0004). Un primer arranque se cayó a los 10 s porque el dataset no estaba agregado al notebook (`IndexError` al buscar `ludwig_lm/`) y no midió nada.
- **Dónde:** Kaggle, 2× T4, torch 2.11 + CUDA 13.0, código subido como dataset. Duró 68 min.
- **Archivos:** `results/t0_2v2_intento2/salida_kaggle.txt` (el log, copiado del visor de Kaggle). Falta agregar el `t0_2v2.jsonl` de la salida del notebook.
- **Resultado:** las 13 mediciones terminaron bien, con 0 pasos saltados en FP16. En los entrenamientos de 60 pasos, la escala de la pérdida se quedó en 32,768 y no hubo expertos muertos.

| Medición | tokens/s | Frente a A | Pico reservado |
|---|---|---|---|
| A · base: bmm con relleno, residuo FP32, gradientes FP16, prefetch, 8 × 2K por GPU | 2,930 | — | 14.33 GiB |
| B · loop por experto | 3,436 | +17.3% | 14.3 GiB |
| C · residuo FP16 | 2,918 | −0.4% | 13.67 GiB |
| D · gradientes FP32 | 2,762 | −5.7% | 14.33 GiB |
| E · sin prefetch | 2,734 | −6.7% | 14.33 GiB |
| F · momento en la RAM de la CPU | 2,831 | −3.4% | 12.76 GiB |
| G · 4 secuencias por GPU | 2,242 | −23.5% | 12.77 GiB |
| H · 4K, 4 secuencias | 2,240 | | 14.33 GiB |
| I · 16K, 1 secuencia | 1,228 | | 14.33 GiB |
| J · checkpoint completo | 2,754 | | 14.32 GiB |

- **Checkpoint (J):** 11.29 GiB; guardar tomó 53.6 s y cargar 47.8 s; la restauración fue exacta.
- **Balanceo** (60 pasos de texto de Markov, 4 secuencias × 2 micro-pasos):

| | Pérdida | Carga máx/media al final (pico) | tokens/s |
|---|---|---|---|
| K · signo fijo, 0.001 | 11.35 → 1.86 | 7.57 (7.97) | 1,995 |
| L · SMEBU, λ = 0.01 | 11.35 → 1.87 | 2.41 (6.47) | 2,318 |
| M · SMEBU, λ = 0.001 | 11.35 → 1.86 | 6.86 (7.64) | 1,991 |

- **Lo que muestran los números:**
  - La pregunta abierta de la entrada anterior queda respondida: 4K y 16K caben.
  - La base da ~2,370 h para 25B (16.9 semanas a 140 h/semana), dentro del rango del diseño (1,750–2,800 h).
  - El loop por experto le gana al bmm con relleno por 17.3%. En el banco v1, con otro modelo, había sido al revés.
  - A, B, D, E, H, I y J llegan a 14.3 GiB reservados o más: la alarma del diseño (5.4) y casi el techo de la T4 (14.56 GiB). La G0 pide un pico ≤ 13.5 GiB: en estas pruebas cortas solo lo cumplen F, G y los entrenamientos de 4 × 2. F también pasa los 2,600 tokens/s de la G0.
  - Solo SMEBU con λ = 0.01 mantiene el balanceo (2.41× al final), y es el más rápido entrenando. La pérdida es la misma en los tres.
- **Preguntas para el chat de diseño:** qué piezas van al entrenamiento real (B y F no se midieron juntas), con cuánto margen de memoria, y si SMEBU con λ = 0.01 queda fijo.
- **Siguiente:** el chat de diseño fija las piezas, el micro-lote y el balanceo. Después viene la T0.3 (estabilidad en FP16 con el LR pico).

## 2026-11-12 · Entra el `t0_2v2.jsonl` del segundo intento

- **Qué:** se agregó la salida del notebook, con una fila por medición. Las 13 filas coinciden con el log en tokens/s y en memoria reservada.
- **Dónde:** PC de Pipe, sin acelerador.
- **Archivos:** `results/t0_2v2_intento2/t0_2v2.jsonl`. Llegó copiado a mano, con fines de línea de Windows y sin el salto final; se guardó con los fines de línea del notebook. Pesa 18,807 bytes, lo mismo que muestra Kaggle (18.81 kB), y el texto no cambió.
- **Lo que agrega al log:**
  - **Memoria asignada** además de la reservada: la base asigna 12.17 GiB (reserva 14.33), la medición más alta es la I a 16K con 12.92 GiB, y con el momento en la CPU (F) son 10.45 GiB.
  - **MFU:** 7.37% en la base, 8.65% con el loop y 3.59% a 16K.
- **Pregunta abierta:** la G0 (5.7) no dice si su tope de 13.5 GiB es de memoria reservada o asignada. Con la asignada, todas las mediciones lo cumplen; con la reservada, solo F, G y los entrenamientos de 4 × 2. El banco v1 usaba la reservada.
- **Siguiente:** el mismo de la entrada anterior.

## 2026-11-13 · Respuestas a la T0.2 v2 y bucle de entrenamiento (patches 0005 y 0006)

- Qué: el chat de diseño respondió las preguntas de la T0.2 v2 y mandó el bucle de entrenamiento real.
- Respuestas:
  1. Piezas: loop por experto + residuo FP32 + gradientes FP16 + prefetch + momento de Muon en la RAM,
     8 × 2K por GPU. Sin el momento en la RAM el loop reserva 14.3 GiB y no pasa la G0. Juntas: ~3,300 tokens/s
     y ~12.7 GiB esperados; la T0.3 + T2 mide la combinación.
  2. SMEBU λ = 0.01 queda; se revisa con texto real (sigue si la carga máx/media < 3 y no muere ningún experto).
  3. El tope de 13.5 GiB de la G0 es memoria reservada, como el banco v1.
- Cambios: scripts/train.py (bucle real), ludwig_lm/data.py (lector provisional hasta la T1),
  ludwig_lm/checkpoint.py (reanudación verificada bit a bit + hash del primer lote), scripts/test_train_cpu.py,
  notebooks/t0_3_t2_lr_2xT4.ipynb; DECISIONS (P2a, P5a, D16a, P6–P8), DISENO y READMEs.
- Errores corregidos (P8): el embedding del EOS (id 0) quedaba fijo por padding_idx; set_state_dict pasaba el
  momento de Muon a FP32 al cargar (la prueba J de la T0.2 v2 no revisaba su tipo).
- Dónde: chat de diseño; probado en CPU, 2 procesos, modelo diminuto: test_train_cpu.py da PASS.
- Pendiente: el LR pico de Ludwig 1 (el barrido está centrado en 1e-3 / 3e-4, los de Trinity Nano).
- Siguiente: 3 corridas en paralelo (cuentas 1–3): lr-x0.5/0.5, lr-x1/1.0, lr-x2/2.0.

## 2026-11-13 · El barrido de la T2 usa el LR del Ludwig 1 (patch 0007)

- Qué: la receta del Ludwig 1 es el ×1: Muon 1.5e-3 (3× la base; el A/B del 31 de octubre eligió ×3),
  AdamW 5e-4, weight decay 0.1 aplicado con el LR de AdamW también a las matrices de Muon. Antes el decay de
  Muon usaba su propio LR: con estos picos habría decaído 3 veces más rápido que en el Ludwig 1.
- La escala de Muon ya era la misma en los dos (Moonlight, 0.2·√max(filas, columnas)): los LR se comparan directo.
- Corridas: lr-x0.5 = 7.5e-4 / 2.5e-4 · lr-x1 = 1.5e-3 / 5e-4 · lr-x2 = 3e-3 / 1e-3 (Muon / AdamW).
  Calentamiento de 100 pasos y enfriamiento en el último 20%, como en el Ludwig 1.
- Dónde: chat de diseño; test_train_cpu.py da PASS, con una prueba nueva del weight decay.
- Resuelto: el pendiente del LR pico del Ludwig 1 de la entrada anterior.

## 2026-11-13 · Pruebas en CPU de los patches 0005–0007 en el PC

- **Qué:** las pruebas que piden las reglas del repo antes del push: pyflakes, la prueba lógica de `bench_afmoe.py` con `--ckpt-test` y `scripts/test_train_cpu.py`.
- **Dónde:** PC de Pipe, Windows sin GPU: Python 3.12, torch 2.15.0+cpu, transformers 5.22.2, 2 procesos con gloo.
- **Resultado:** todo pasa. pyflakes, sin errores; `bench_afmoe.py`, `status=ok` y checkpoint restaurado exacto; `test_train_cpu.py`, **PASS** en todas sus comprobaciones, incluidas la del weight decay de Muon y la reanudación bit a bit.
- **Problema de Windows:** `torchrun` no arranca con torch 2.15. Sus dos formas de coordinar los procesos (`--standalone` y la estática) crean el `TCPStore` sin pasar `use_libuv`, y torch para Windows no trae libuv; `USE_LIBUV=0` solo lo lee `init_process_group`. Las pruebas se corrieron lanzando los 2 procesos a mano con las variables que pone `torchrun` (`RANK`, `WORLD_SIZE`, `MASTER_ADDR`, `MASTER_PORT`) y `PYTHONUTF8=1` (la consola de Windows no imprime el "≈" de la línea `[result]`), sin cambiar el código ni las comprobaciones. En Kaggle, que es Linux, no aplica.
- **Siguiente:** las tres corridas de la T0.3 + T2 en Kaggle.

## 2026-11-13 · El repo pasa a público

- **Qué:** el repo se publica en GitHub, en `FelipeGarciaRpo/ludwig-lm`. Antes se rehízo el repositorio de GitHub desde cero y se limpió el historial: salió un archivo de notas de trabajo, que queda solo en el PC de Pipe, y se ajustó la redacción de unas frases de la bitácora, de los notebooks y del encabezado de la copia del banco v1. No cambió ningún dato, resultado ni línea de código.
- **Dónde:** PC de Pipe.
- **Identificadores:** al limpiar el historial cambiaron los de todos los commits menos el primero. Los zips del código que se subieron a Kaggle se armaron con los identificadores viejos:

| Antes | Ahora | Qué es |
|---|---|---|
| `cc3dbd4` | `10a7d88` | código del primer intento de la T0.2 v2 |
| `5735d38` | `60a341f` | código del segundo intento de la T0.2 v2 |
| `51f037e` | `52f5b0e` | código de las tres corridas de la T0.3 + T2 |

- **Siguiente:** el chat de diseño trabaja sobre el repo público en vez de recibir bundles. Los resultados de la T0.3 + T2 se registran con el código `52f5b0e`.

## 2026-11-14 · T0.3 + T2: las tres corridas de LR

- **Qué:** tres corridas de 400 pasos (~105M tokens) con el LR del Ludwig 1 ×0.5, ×1 y ×2 (Muon 7.5e-4 / 1.5e-3 / 3e-3; AdamW 2.5e-4 / 5e-4 / 1e-3), con FineWeb-Edu y el tokenizador de SmolLM2 como provisionales.
- **Dónde:** Kaggle, 2× T4, en tres cuentas; torch 2.11 + CUDA 13.0; código `52f5b0e` subido como dataset. Duraron entre 8.1 y 8.9 h.
- **Archivos:** `results/t0_3_t2/`: un `.jsonl` por corrida, su prueba de guardado y reanudación y su gráfica. Los `.jsonl` llegaron copiados a mano, con fines de línea de Windows; se guardaron con los del notebook, y el de ×0.5 coincide con su log en los pasos y evaluaciones que se compararon. Los logs de Kaggle no se guardaron.

| | ×0.5 | ×1 (Ludwig 1) | ×2 |
|---|---|---|---|
| Val, bits por byte, antes del enfriamiento | 1.4013 | 1.2879 | 1.2119 |
| Val, bits por byte, al final | 1.3470 | 1.2223 | **1.1516** |
| tokens/s en la última hora | 3,601 | 3,428 | 3,293 |
| Carga máx/media desde el paso 100: media / máx / pasos sobre 3 | 1.64 / 2.99 / 0 | 1.79 / 3.15 / 1 | 2.70 / 7.27 / 65 |
| Ganancia máxima de qk | 1.14 | 1.29 | 1.59 |

- **Resultado:**
  - Las tres son estables: 0 NaN, 0 pasos saltados por FP16 o por picos, 0 alarmas y la escala de FP16 fija en 32,768.
  - Memoria pico: 12.92 GiB reservados (12.17 asignados) en las tres.
  - La prueba de guardado y reanudación salió exacta en las tres: el estado bit a bit, el primer lote y la validación.
  - Ninguna evaluación encontró expertos muertos. Hubo expertos sin tokens en algunos pasos solo durante el calentamiento.
- **Regla de decisión (P6, fijada antes de correr):** las tres cuentan. Gana ×2: ×1 queda 6.1% arriba y ×0.5, 17% arriba, así que no hay empate. Como ×2 es un extremo, la regla pide correr ×4 antes de decidir.
- **Preguntas abiertas para el chat de diseño:**
  - En ×2 el balanceo no cumple la condición de SMEBU (carga máx/media bajo 3): 65 de 300 pasos pasan de 3, con un pico de 7.27. En ×0.5 y ×1 sí la cumple.
  - La velocidad baja con el desbalance: ×2 va 8.6% más lento que ×0.5 (~2,109 h para 25B, contra ~1,929).
  - La ganancia de qk sube con el LR (1.59 en ×2). En un tronco 200 veces más largo hay que vigilarla.
- **G0 (5.7):** con datos provisionales ya se cumplen tres de las cinco condiciones: más de 6 h sin NaN y con la escala estable, más de 2,600 tokens/s y menos de 13.5 GiB. Faltan los guardados en Hugging Face con la reanudación en otra cuenta y los 50 pasos a 16K, y todo con el lector y el tokenizador reales.
- **Siguiente:** el chat de diseño decide si se corre ×4, como pide la regla, y qué hacer con el balanceo de ×2. Mientras, el tokenizador de 64K corre en Kaggle, en CPU.
