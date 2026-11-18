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

## 2026-11-14 · Segunda ronda de la T2 y regla del LR del tronco (patch 0008)

- Qué: el chat de diseño revisó la primera ronda y preparó la segunda.
- Lectura de la primera ronda:
  - La diferencia de velocidad entre corridas es de las máquinas: dentro de cada corrida la velocidad no
    cambia con la carga (correlación −0.07 / +0.17 / +0.01 desde el paso 100; ×2 va igual de rápido con
    la carga en 7.2× que en 2.1×).
  - En ×2 sí hubo un episodio de desbalance: las ventanas de 50 pasos que terminan en 300 y 320 llegaron
    a 3.6× y 4.2×; SMEBU lo corrigió en ~30 pasos y en el 400 todas las capas quedan bajo 1.11×. El balanceo
    se juzga por capa en esa ventana, y λ escala con el LR: 0.01 × el multiplicador (P2b).
  - La ganancia de qk crece casi lineal con el LR constante y se frena con el enfriamiento.
- Decisión (P9, fijada antes de la segunda ronda): el mejor LR baja con la duración del entrenamiento
  (exponente ~0.32 en la literatura, para AdamW y para Muon). Se mide aquí con lr-x4 (400 pasos, λ 0.04)
  y lr-x1-1200 / lr-x2-1200 (siguen ×1 y ×2 desde el paso 320 hasta 1,200, con λ 0.01).
  scripts/lr_horizon.py lo lleva a los 25B del tronco; con la primera ronda sola da ×1.
- Cambios: scripts/lr_horizon.py; notebook con LAMBDA, PRUEBA y REANUDAR = "auto"; DECISIONS (P6a, P2b, P9),
  recuadro del DISENO y READMEs con los resultados de la primera ronda.
- Dónde: chat de diseño; probado en CPU: test_train_cpu.py da PASS y la continuación en dos sesiones con
  REANUDAR = "auto" sale exacta.
- Siguiente: la segunda ronda en tres cuentas.

## 2026-11-14 · Tokenizador de 64K: corrida completa

- **Qué:** `tokenizador_64k_v1_2.py` en Kaggle, completo: muestra de 1.2 GB de las 23 fuentes de la mezcla v3, con barrido de 0.3, 0.6 y 1.2 GB. Antes pasó una prueba en modo rápido, cuya salida no se guardó.
- **Dónde:** Kaggle, CPU, sin GPU; tokenizers 0.23.3. Tardó 56 min, con 4.7 GB de RAM pico.
- **Archivos:** `results/tokenizador_64k/con_transweb/`: el tokenizador (`tokenizer.json`, sha1 `0b7da8fdb062`), su reporte y una copia del script.
- **Resultado:**
  - Pasan todas las pruebas del script: 65,536 tokens, los ids especiales en orden (EOS 0, pad 1, `<think>` 4, FIM 10–13), los 256 bytes, ida y vuelta exacta y ningún token de control desde texto crudo. En el PC se repitieron por separado, con el mismo resultado, y el sha1 coincide.
  - Bytes por token: inglés 4.496, español 4.598, código 3.708, matemática 3.413. En la mezcla del tronco, 4.159, contra 3.551 de SmolLM2, 3.719 de OLMoE y 4.034 de Qwen3 (151K).
  - Tokens por palabra: 1.37 en inglés y 1.33 en español (SmolLM2: 1.38 y 2.08).
  - En textos que no vio (los documentos y el código del repo), usa 20% menos tokens que SmolLM2 en español, 7% menos en inglés y 12% menos en Python.
  - 236 tokens no aparecen en 100 MB de texto (0.36%).
  - La muestra alcanza: de 0.3 a 1.2 GB, la compresión del tronco sube solo de 4.151 a 4.159. Las otras mezclas de la muestra quedan a menos de 0.2%.
- **Ojo:** la corrida usó TransWeb-Edu (9% de la muestra), aunque se había acordado `USAR_TRANSWEB = False` mientras no se confirme su traductor.
- **Siguiente:** decidir si se repite sin TransWeb-Edu (~1 h de CPU) antes de fijarlo. Después, el tokenizador entra a la T1 y a la G0.

## 2026-11-14 · Rutas de la T0.1 antes de publicar

- **Qué:** en los tres resultados crudos de la T0.1 (`results/t0_1_stock.json`, `t0_1_patched.json` y `t0_1c_afmoe.json`), la ruta de la máquina donde corrió la prueba se cambió por `/home/user/` en todo el historial, por decisión de Pipe antes de publicar el repo. Es la única excepción a la regla de no editar resultados crudos: no cambió ningún número ni ninguna otra línea, y los tres archivos se leen igual que antes.
- **Dónde:** PC de Pipe.
- **Identificadores:** al reescribir el historial cambiaron otra vez los de todos los commits menos el primero, y las referencias de los documentos ya apuntan a los nuevos. Para los zips que están en Kaggle:

| Zip en Kaggle | Ahora | Qué es |
|---|---|---|
| `51f037e` | `52f5b0e` | código de la primera ronda de la T0.3 + T2 |
| `292f882` | `3e4f66c` | código de la segunda ronda de la T2 |

- **Siguiente:** publicar el repo.

## 2026-11-14 · Tokenizador fijado, T0.1b y rutas sin usuario (patch 0009)

- **Qué:**
  - El tokenizador de 64K queda fijado tal como salió (P10) y vive en `tokenizer/`. El script que lo hizo pasa a `scripts/train_tokenizer.py`, sin cambios.
  - No se repite sin TransWeb-Edu: su traductor es Mistral-7B-Instruct-v0.1 (Apache 2.0) y el dataset es ODC-By, así que es 🟢 (D20).
  - La T0.1b, con `scripts/smoke_tokenizer.py`: el tokenizador final en llama.cpp, en el formato AFMoE.
  - `load_tokenizer` pasa al modo datos de verdad (P11). Con `encode_special_tokens` solo, un `<think>` escrito en una página web salía como el id 4.
  - `configs/ludwig2_afmoe.json` queda con padding 1 y BOS 0 (P12).
  - `train.py` y el notebook de la T0.3 + T2 escriben las rutas de Kaggle sin el usuario (`/kaggle/input/notebooks/<usuario>/…`): en los `.jsonl`, en lo que imprimen y en los metadatos del checkpoint.
  - Al reanudar, los argumentos se comparan sin el usuario por los dos lados. Así, un checkpoint viejo, con el usuario, no marca cambios falsos.
- **Dónde:** chat de diseño, en CPU. llama.cpp `61dbf49` (13 nov 2026), compilado solo `llama-tokenize`: 8 min con 2 núcleos.
- **Archivos:**
  - `tokenizer/`: los 4 archivos de `results/tokenizador_64k/con_transweb/` y un README.
  - `scripts/train_tokenizer.py`, `scripts/smoke_tokenizer.py` y `results/t0_1b_tokenizer.json`.
- **Resultado:**
  - La T0.1b pasa:
    - El pre-tokenizador es `qwen2`.
    - 17 de 17 textos tokenizan igual que en Hugging Face: prosa en inglés y español, código, LaTeX, chat, `<think>`, herramientas y FIM.
    - llama.cpp detecta BOS 0, EOS 0, fin de turno 3, padding 1 y los seis de FIM, y para de generar en 0, 3, 13, 14 y 15.
  - `test_train_cpu.py` da PASS, con dos pruebas nuevas: el tokenizador real en modo datos (ningún id de control desde texto, ida y vuelta exacta) y las rutas sin usuario.
- **Ojo:**
  - La primera corrida de la T0.1b falló por tres cosas. Dos eran del arnés:
    - `llama-tokenize` interpreta escapes por defecto: el `\to` de LaTeX se volvió tabulador + `o`. Ahora va con `--no-escape`.
    - En el llama.cpp actual, los tokens especiales solo salen en el log con `-lv 4`.
  - La tercera era real: sin `bos_token_id` en el config, llama.cpp toma como BOS el id 11, que aquí es `<|fim_middle|>`.
  - Esa primera corrida no se guardó: medía el arnés, no el tokenizador.
  - El hash del vocabulario (`b01d3ee0…`) es nuevo para llama.cpp. Hasta que se registre (un PR que escribe Pipe), los GGUF se convierten con `scripts/convert_gguf.py`.
  - La segunda ronda de la T2 sigue con su zip (`3e4f66c`): nada de esto la toca. Sus `.jsonl` que ya traen el usuario se enmascaran al subirlos, con su nota (regla 6).
- **Siguiente:** el relevo con los checkpoints en Hugging Face y respaldo en Kaggle, y el lector de la T1.

## 2026-11-14 · Verificación independiente del tokenizador, en `results/`

- **Qué:**
  - Los números de la verificación con textos que el tokenizador nunca vio ya estaban en la entrada del tokenizador y ahora en los README, pero faltaba su archivo crudo (regla 1).
  - Se repitió sobre los documentos y el código del repo en `3e4f66c`, el commit de cuando se corrió por primera vez, y dio lo mismo.
- **Dónde:** el PC de Pipe, en CPU, con tokenizers 0.23.3. El tokenizador de SmolLM2 se bajó de Hugging Face.
- **Archivos:** `results/tokenizador_64k/verificacion_textos_repo.txt`.
- **Resultado:**
  - El sha1 coincide con el reporte, los ids 0–255 están en orden y en modo datos ninguna cadena de control escrita en el texto sale como id de control. La ida y vuelta es exacta.
  - Frente a SmolLM2 usa 20.4% menos tokens en español (`DISENO.md` y `BITACORA.md`), 7.2% menos en inglés (`README.md` y `DECISIONS.md`) y 12.0% menos en Python (`ludwig_lm/` y `scripts/`).
- **Ojo:** el patch 0009 no traía `README.es.md` ni la fila de `results/t0_1b_tokenizer.json` en `results/README.md`. Se agregaron aparte, con el mismo contenido que `README.md`.
- **Siguiente:** nada; la entrada solo cierra la regla 1 para esos números.

## 2026-11-14 · Segunda ronda de la T2: ×4 a 400 pasos

- **Qué:** la corrida ×4 de la segunda ronda (P9): Muon 6e-3 y AdamW 2e-3, SMEBU con λ = 0.04 (P2b) y 400 pasos de 0.25M tokens, igual que la primera ronda, con FineWeb-Edu y el tokenizador de SmolLM2 como provisionales.
- **Dónde:** Kaggle, 2× T4, con el código del zip `292f882` (= `3e4f66c`). Duró 8.2 h.
- **Archivos:**
  - `results/t0_3_t2/lr-x4.jsonl`, `lr-x4-prueba.jsonl` y `lr-x4.png`.
  - Los `.jsonl` llegaron con CRLF y sin salto final, como en la primera ronda. Se restauraron (231,348 y 8,937 bytes) y se revisaron fila por fila: 413 y 14 filas, con los pasos 0–399 seguidos.
  - No traen rutas con usuario: la corrida empezó de cero.
- **Resultado:**

| | ×0.5 | ×1 | ×2 | ×4 |
|---|---|---|---|---|
| Val, bits por byte, antes del enfriamiento | 1.4013 | 1.2879 | 1.2119 | 1.1884 |
| Val, bits por byte, al final | 1.3470 | 1.2223 | 1.1516 | **1.1284** |
| tokens/s en la última hora | 3,601 | 3,428 | 3,293 | 3,588 |
| Ganancia máxima de qk | 1.14 | 1.29 | 1.59 | 2.14 |
| Entropía del router al final | 3.450 | 3.262 | 2.865 | 2.185 |

  - **Estable:**
    - Sin NaN y sin pasos saltados.
    - Escala de FP16 fija en 32,768 y 12.92 GiB reservados.
    - La prueba de reanudación es exacta: el estado y el primer lote coinciden y la val sale igual.
  - **×4 gana a 400 pasos,** 2.0% por debajo de ×2. La parábola en log2 del LR tiene su mínimo en ×3.81 a 105M tokens.
  - **`scripts/lr_horizon.py` con las cuatro da ×1 para el tronco:** ×0.94 con el exponente 0.32 de la literatura. El exponente propio sale de las corridas de 1,200 pasos.
- **Preguntas abiertas para el chat de diseño:**
  - **Balanceo por ventanas (P2b):** en ×4 ninguna ventana pasa de 3; la mayor es 2.30, en la capa 14 y el paso 320. Pero en las capas 4–15 la media sube durante toda la corrida (1.08 en el paso 100, 1.49 en el 200, 1.90 en el 400) y no baja en el enfriamiento. En ×1 y ×2, esas capas bajan a 1.02 en el paso 400.
  - **Balanceo por paso:** desde el paso 100, la carga máx/media de ×4 pasa de 3 en 289 de 300 pasos (media 6.40, máxima 8.32). Con la entropía del router más baja de las cuatro, el router se concentra más cuanto más alto es el LR.
  - **qk:** la ganancia sigue subiendo con el LR (2.14 en ×4).
- **Siguiente:** esperar lr-x2-1200 y lr-x1-1200 y volver a correr `scripts/lr_horizon.py` (P9). lr-x1-1200 corre desde cero: hasta el paso 320 tiene el mismo calendario y los mismos datos que el ×1 de la primera ronda, así que sus evaluaciones de los pasos 50 a 300 deberían coincidir.

## 2026-11-14 · Segunda ronda de la T2: ×2 de 320 a 800 pasos (se queda ahí)

- **Qué:**
  - lr-x2-1200 siguió desde el checkpoint del paso 320 del ×2 de la primera ronda, con λ = 0.01 y el LR de ×2 fijo. El enfriamiento empezaba en el paso 960.
  - La sesión llegó al paso 800 y paró por tiempo (pausa en el 801).
  - Pipe decidió no reanudarla, porque el notebook no dejaba retomarla.
- **Dónde:** Kaggle, 2× T4, con el código del zip `292f882` (= `3e4f66c`). Duró 11.2 h.
- **Archivos:**
  - `results/t0_3_t2/lr-x2-320-800.jsonl` (tag `lr-x2-1200`), `lr-x2-320-800-prueba.jsonl` y `lr-x2-320-800.png`.
  - Los `.jsonl` llegaron con CRLF y sin salto final. Se restauraron (277,108 y 8,945 bytes) y se revisaron fila por fila: 497 y 14 filas, con los pasos 320–800 seguidos.
  - El `.jsonl` traía dos veces el usuario de Kaggle en la ruta de reanudación (`args.resume` y la fila `resume`). Se cambió por `<usuario>` (regla 6), y es el único cambio.
- **Resultado:**
  - **La reanudación fue exacta:** el estado y el primer lote coinciden, y la val del paso 320 da 1.2119, igual que en la primera ronda.
  - **Estable:**
    - Sin NaN y sin pasos saltados.
    - Escala de FP16 fija en 32,768 y 12.9 GiB reservados.
    - 3,152 tokens/s, en una máquina más lenta.
  - **Val en bits por byte, con el LR fijo:** 1.1572 (paso 400), 1.1179 (500), 1.0933 (600), 1.0733 (700) y 1.0563 (800). En el paso 400, la corrida de la primera ronda, que ya había enfriado, daba 1.1516.
  - **qk y router:** la ganancia de qk sube sin parar, de 1.54 (paso 320) a 2.00 (600) y 2.48 (800). La entropía del router baja de 2.89 a 2.38.
  - **Balanceo (P2b):** las capas 4–15 quedan en ~1.04, pero la capa 0 pasa de 3 en tres ventanas (3.21 en el paso 400, 4.13 en el 450 y 3.35 en el 650), como en la primera ronda. Por paso, la carga máx/media pasa de 3 en 405 de 481 pasos (media 3.98).
- **Ojo (para el chat de diseño):**
  - Sin el enfriamiento, esta corrida no cuenta para `scripts/lr_horizon.py`, que solo usa corridas terminadas. Si lr-x1-1200 tampoco llega al paso 1,200, la regla P9 se queda con el exponente de la literatura, o sea ×1.
  - Propuesta: comparar ×1 y ×2 en el mismo paso, antes del enfriamiento, hasta el paso 800. En la primera ronda, la brecha antes del enfriamiento (6.3%) fue casi igual a la del final (6.1%).
- **Siguiente:** lr-x1-1200 sigue desde el paso 530 por ~7 h, hasta cerca del paso 850.

## 2026-11-15 · Datos del tronco: mezcla v3 y fuentes cerradas

- **Qué:**
  - Pipe eligió la mezcla v3 el 13 de noviembre. Esa decisión no había quedado registrada en la bitácora, y el diseño y los README seguían con la mezcla anterior, que daba 10% al español.
  - Una revisión independiente, hecha en otro chat, cerró los ocho puntos abiertos de la v3 el 14 de noviembre. El 15 midió cuánto del código permisivo de Stack-Edu trae Common Pile. Pipe aceptó sus recomendaciones.
  - Se reescribieron la sección 4 del diseño, su línea de objetivos (sección 1), D17–D19 y los dos README, y se agregaron las decisiones P13–P17.
- **Dónde:** en el chat de la revisión, leyendo shards de Hugging Face. Nada corrió en Kaggle.
- **Archivos:** `results/datos_tronco_v3/revision_fuentes.txt` y `common_pile.txt`. Son el texto recibido sin editar; solo se quitó la línea de destinatarios.
- **Resultado:**
  - **Mezcla del tronco:** inglés 36, español 24, código 28, matemática 10 e instrucción 2, dimensionada para 100B. La revisión no cambió el porcentaje de ningún bloque.
  - **Código:** `common-pile/stackv2_edu_filtered` trae ~90% de los archivos permisivos de Stack-Edu en Python, JS y TS: ≈5.30B / 2.39B / 1.39B tokens, que son 1.9 / 2.7 / 2.5 pasadas a 90B. Reemplaza la descarga de Software Heritage y el recorrido de The Stack v3, y el correo a Software Heritage pasa a ser opcional. HTML y CSS salen de The Stack v1, como en el tokenizador.
  - **DCLM se queda.** La nota de "solo para investigación" de su ficha se anota en la ficha de datos.
  - **Ya tienen fuente:** StackExchange, los notebooks, el texto paralelo (ONU y Europarl), la instrucción (OpenMathReasoning y OpenAssistant) y Stack Overflow.
  - **Texto escrito por modelos:** topes por tipo; la v3 queda en 18.5% del tronco.
  - **No hay prueba proxy de mezcla antes del tronco** (P17). En su lugar, bits por byte por dominio desde el paso 0, revisados a 1B y a 2.25B tokens.
- **Siguiente:** el patch de la T1 del chat de diseño: el pre-tokenizado con estas fuentes. De Common Pile basta bajar los shards de esos lenguajes, ~45 GB.

## 2026-11-15 · Segunda ronda de la T2: ×1 hasta el paso 1,110, y la comparación con ×2

- **Qué:**
  - lr-x1-1200 corrió desde cero con el LR de ×1: 1,200 pasos planeados, con el enfriamiento desde el paso 960.
  - Fueron dos sesiones, de los pasos 0–566 y 567–1,109. Paró por tiempo en el 1,110, 90 pasos antes de terminar el enfriamiento.
- **Dónde:** Kaggle, 2× T4, con el código del zip `292f882` (= `3e4f66c`). Las sesiones duraron 11.3 h y 11.2 h.
- **Archivos:**
  - `results/t0_3_t2/lr-x1-1200.jsonl` (trae las dos sesiones), `lr-x1-1200-prueba.jsonl` y `lr-x1-1200.png`.
  - Los `.jsonl` llegaron con CRLF y sin salto final. Se restauraron (642,509 y 8,939 bytes) y se revisaron fila por fila: 1,144 y 14 filas, con los pasos 0–1,109 seguidos.
  - La ruta de reanudación traía dos veces el usuario de Kaggle. Se cambió por `<usuario>` (regla 6).
- **Resultado:**
  - **Estable:**
    - Sin NaN, sin pasos saltados y sin alarmas. La escala de FP16 subió de 32,768 a 65,536.
    - 3,676 y 3,543 tokens/s, con 12.92 GiB reservados.
    - La reanudación en el paso 567 fue exacta, en el estado y en el primer lote.
  - **Val en bits por byte:** 1.0670 (paso 800), 1.0438 (960, antes del enfriamiento) y 1.0094 (1,100).
  - **×1 contra ×2 en el mismo paso, los dos con el LR fijo:** la ventaja de ×2 baja de 6.65% (paso 300) a 4.09% (400), 2.69% (500), 1.79% (600), 1.35% (700) y 1.01% (800).
  - **El método de P9, aplicado a los valores de antes del enfriamiento:** el mejor LR pasa de ×4.25 (paso 320) a ×1.67 (paso 800). El exponente entre esos dos puntos (~1.0) es mucho más alto que el de la literatura (0.32) y no sirve para extrapolar: llevado a 25B daría ×0.02.
  - **`scripts/lr_horizon.py`** no cuenta esta corrida ni la de ×2, porque ninguna terminó el enfriamiento. Así, la regla P9 se queda con el exponente de la literatura y da **×1**.
  - **Balanceo (P2b):** después del calentamiento ninguna ventana pasa de 3 (la mayor es 1.66, en el paso 850). Por paso, la carga máx/media pasa de 3 en 482 de 1,010 pasos (media 3.01).
  - **qk:** la ganancia sube más lento que en ×2: 1.69 en el paso 800 (en ×2, 2.48) y 1.84 en el 1,100.
  - **Repetición:**
    - Con el mismo código, torch, GPU y datos, lr-x1-1200 no repite bit a bit al ×1 de la primera ronda: los pasos difieren desde el 1 (ce 11.30633 contra 11.30634).
    - Las evaluaciones de los pasos 50 a 300 quedan a 0.11% o menos (paso 300: 1.31315 contra 1.31285).
    - Hay un no determinismo chico en GPU entre corridas. La reanudación sí es exacta.
- **Preguntas abiertas para el chat de diseño:**
  - Confirmar ×1 para el tronco: lo da la regla, y la brecha con ×2 se cierra rápido.
  - El no determinismo en GPU: de dónde sale (por ejemplo, operaciones con sumas atómicas en el backward) y si afecta a la G0, que al reanudar en otra cuenta compara el hash del primer lote.
- **Siguiente:** la T1 y el relevo (patch 0010), y después la G0.

## 2026-11-15 · T1: el constructor de shards y su lector (patch 0010)

- **Qué:**
  - El pre-tokenizado que aprobó Pipe, con las fuentes de la mezcla v3 (P18–P24):
    - `scripts/build_shards.py` (con `ludwig_lm/sources.py` y `ludwig_lm/build.py`) convierte las fuentes en shards de 256 MiB de ids uint16, en cuatro etapas: `plan`, `prepare`, `val` y `train`.
    - La receta, con los ids del Hub, las columnas, los pesos, los dominios de evaluación y los 16 benchmarks, es `configs/data_v3.json`.
  - El lector de shards (`ludwig_lm/shards.py`) entra en `train.py` con `--shards`:
    - la validación por dominio (inglés, español, Python, JS y matemática), con un bits por byte por dominio en cada evaluación;
    - `--meta` para las etiquetas de MeCo;
    - un paro ordenado si se acaban los shards adjuntos.
  - La pérdida acepta una máscara (`ludwig_lm/losses.py`), porque MeCo deja los metadatos fuera de la pérdida. Sin máscara, el cálculo es el mismo de siempre.
  - `notebooks/t1_shards_cpu.ipynb`: la construcción en la CPU de Kaggle, con `plan` primero y después 3 sesiones de 56 shards.
- **Dónde:** chat de diseño, en CPU, sin Hugging Face (no responde desde ahí). Todo se probó sobre un Hub sintético local.
- **Archivos:** los de arriba y `scripts/test_shards_cpu.py`. El tokenizador asigna sus ids reservados 16 y 17 a las etiquetas (`tokenizer/README.md`; `tokenizer.json` no cambia).
- **Resultado:**
  - `test_shards_cpu.py` da PASS (48 revisiones):
    - 10 shards construidos en una sesión y en dos (6 + 4) son idénticos byte a byte.
    - Cada fuente queda a 0.12 puntos o menos de su peso.
    - La página con la pregunta del benchmark sale, y la frase plantilla no cuenta.
    - Ninguna URL entra en dos fuentes, ni una página junto a su reescritura.
    - FIM, la plantilla de chat, la limpieza de código y de HTML, Stack Overflow y OpenAssistant salen como deben.
    - El lector reanuda exacto, también en el modo sin etiquetas.
    - Entrenar con shards y reanudar es bit a bit.
  - `test_train_cpu.py` sigue en PASS.
  - Velocidad del constructor: 460–760K tokens/s con 2 núcleos, con texto sintético.
- **Ojo:**
  - **Cambios al diseño:**
    - La validación son los primeros documentos del último archivo, y el resto de ese archivo sí entrena (P19). Con archivos enteros, el código habría perdido 8% del Python, 6% del JavaScript y 20% del TypeScript.
    - La descontaminación usa hashes exactos en vez de un filtro de Bloom (P20).
  - **Escrito de memoria:** varios nombres de archivos y columnas de la receta, sobre todo los de los benchmarks, y Stack Exchange, Stack Overflow, la ONU y Europarl. Para eso está `plan`: lo que salga ❌ se corrige en la receta, no en el código.
  - **El no determinismo de la segunda ronda nace en el primer backward:**
    - El paso 0 es idéntico en las dos corridas: ce 11.32779, norma del gradiente 3.4302, el mismo lote.
    - El paso 1 ya difiere en el quinto decimal.
- **Siguiente:**
  - El patch 0011: el relevo con Hugging Face, respaldo en Kaggle y candado.
  - En Kaggle: `plan` y después las 3 sesiones de construcción.

## 2026-11-15 · El relevo, el LR del tronco y el plan de la G0 (patch 0011)

- **Qué:**
  - El relevo que aprobó Pipe (P26), con Hugging Face como principal y Kaggle de respaldo: `ludwig_lm/relay.py` y `notebooks/tronco_relevo_2xT4.ipynb`.
    - **El candado** es un *lease* de 2 h en `lock.json`, que se renueva cada 30 min y con cada subida. La valla es el commit con `parent_commit`.
    - **Un checkpoint cada 3 h**, subido mientras entrena, en un solo commit con el índice, la rotación (quedan 2) y la renovación del candado.
    - **Al final de cada turno:** respaldo en el dataset de Kaggle que toque, aplastado del historial y entrega del candado.
  - **El LR del tronco: ×1 (P25).** Los argumentos del tronco quedan en `configs/tronco_v1.json`: 46,800 pasos, 22.536B tokens.
  - **El plan de la G0 (P27):** G0-0, el ensayo del relevo; G0-1, la prueba de 16K; G0-2 y G0-3, los dos primeros turnos del tronco.
  - **El no determinismo de la segunda ronda (P28):** empieza en el primer backward y no afecta la G0.
- **Dónde:** chat de diseño, en CPU, sin Hugging Face ni Kaggle. Un almacén local hizo de Hugging Face, y una carpeta, de Kaggle.
- **Archivos:**
  - `ludwig_lm/relay.py`, `configs/tronco_v1.json` y `scripts/test_relay_cpu.py`.
  - `notebooks/tronco_relevo_2xT4.ipynb`, generado con `make_notebooks.py`.
- **Resultado:**
  - `test_relay_cpu.py` da PASS (23):
    - El candado se niega mientras vive el *lease*. Vencido, otra sesión lo toma, y la primera falla en su próximo commit.
    - Un commit sobre una cabeza vieja se rechaza, y el aplastado deja solo el árbol nuevo.
    - Los respaldos se turnan.
    - Tres turnos en tres "cuentas" dan bit a bit la corrida sin cortes, con reanudación exacta desde el respaldo y desde el almacén.
    - El turno libera el disco: el checkpoint de partida y los locales viejos ya no están cuando se arma el respaldo.
  - Las celdas del notebook (un turno del tronco y el ensayo) corrieron completas contra el almacén local.
  - `test_train_cpu.py` y `test_shards_cpu.py` siguen en PASS.
- **Ojo:**
  - Al terminar sus pasos, `train.py` solo guarda con `--save-final`. La config del tronco lo lleva, porque de ese checkpoint sale la rama. Lo encontró la prueba.
  - **El disco:** en el peor momento, un turno habría tenido ~57 GiB en `/tmp`: el checkpoint de partida, 3 locales y el tar del respaldo, contra los ~73 GB de disco de Kaggle. Quedó en dos checkpoints como máximo (~23 GiB):
    - el de partida se borra después del primer guardado del turno;
    - `keep` 1 en local, porque Hugging Face guarda 2 y los respaldos otros 2;
    - antes del respaldo, solo queda el último.

    La celda de preparación avisa si hay menos de 30 GiB libres.
  - **Lo que solo se puede probar en Kaggle es la G0-0:**
    - la velocidad de subida de ~11 GB a Hugging Face y a Kaggle;
    - `kaggle datasets version` con el token de cada cuenta;
    - el montaje de los datasets de respaldo.
  - **Antes del primer turno, Pipe crea:**
    - la organización y el repo privado del relevo, y el de prueba;
    - los dos datasets de respaldo, vacíos y con los amigos como colaboradores.

    Y fija `COMMIT`.
- **Siguiente:** aplicar 0010 y 0011; el `plan` de la T1 y las 3 sesiones de construcción; después la G0.

## 2026-11-15 · Patches 0010 y 0011: las pruebas en el PC

- **Qué:** se aplicaron los patches 0010 (la T1) y 0011 (el relevo, el LR del tronco y el plan de la G0) sobre `0abcb97`, y se corrieron las pruebas en CPU.
- **Dónde:** el PC de Pipe, en Windows y en CPU. Para que torchrun arranque en Windows, los 2 procesos se lanzan a mano: el código y las pruebas del repo no cambian.
- **Resultado:**
  - pyflakes limpio y notebooks en sincronía.
  - `bench_afmoe.py` en CPU: restauración exacta.
  - `test_train_cpu.py` y `test_relay_cpu.py`: PASS.
  - `test_shards_cpu.py` falla en Windows: `_Files.find` (`ludwig_lm/shards.py`) compara `str(h).endswith(name)`, y en Windows las rutas usan `\` mientras el nombre del manifiesto usa `/`. Con `h.as_posix().endswith(name)`, probado solo en una copia local, da PASS completo. En Kaggle, que es Linux, no pasa.
- **Pregunta abierta para el chat de diseño:** cambiar esa comparación por `as_posix()` en un próximo patch, para que la prueba también pase en Windows.
- **Siguiente:** el `plan` de la T1 en Kaggle con este commit.

## 2026-11-15 · T1, sesión 0: el plan, y dos benchmarks corregidos en la receta

- **Qué:** la sesión 0 de la T1 (`FASE = "plan"`) en Kaggle, con el commit `f971c9b`. Lee el primer archivo de cada fuente y de cada benchmark, y no construye nada.
- **Dónde:** Kaggle, CPU (4 núcleos, 31 GB de RAM). Duró ~2.5 min.
- **Archivos:** `results/t1/plan_intento1_salida_kaggle.txt`, el log copiado del visor de Kaggle.
- **Resultado:**
  - 43 de 45 entradas salieron ✅: las 29 fuentes del tronco y 14 de los 16 benchmarks, con las columnas que espera la receta.
  - `hellaswag-es` y `flores` salieron ❌ ("no files under ''"). Sus archivos son `.jsonl`, y a su entrada en la receta le faltaba `"format": "jsonl"`, así que el constructor buscaba `.parquet`. HellaSwag en español está en `alexandrainst/m_hellaswag`, en `data/es/val.jsonl`. Flores+ está en `dev/` y `devtest/`, como `eng_Latn.jsonl` y `spa_Latn.jsonl`.
  - **Arreglo en `configs/data_v3.json`, antes del primer shard (regla de la T1):** `"format": "jsonl"` en las dos, y `"prefix": "data/es/"` en `hellaswag-es`.
  - Se comprobó contra los archivos reales de Hugging Face con `select_files`: la receta vieja no encuentra nada, y la nueva encuentra `data/es/val.jsonl` y los cuatro de Flores+. Las columnas (`ctx` y `endings`; `text`) ya estaban bien.
  - **Los ejemplos de código del plan muestran otros lenguajes** (PHP en `python`, Java en `javascript`, Swift en `typescript`, Rust en `sql` y `shell`). No es un error: el plan enseña la primera fila del primer archivo, que en Common Pile cae en el límite con el lenguaje anterior, y el constructor filtra cada lenguaje por la lista de archivos permisivos de Stack-Edu.
- **Siguiente:** repetir el `plan` con el commit nuevo. Si sale todo ✅, la sesión 1 de construcción con ese mismo commit.

## 2026-11-16 · T1: el notebook imprime su avance cada 5 minutos

- **Qué:** la sesión 1 de la T1 (commit `a52a76d`) lleva más de 2 h en `[prepare]` sin imprimir nada.
  - Esa etapa baja de Stack-Edu la lista de archivos permisivos de cada lenguaje, y solo escribe una línea al terminar cada uno.
  - Sus parquet vienen en grupos de 1,000 filas (SQL: 2,505 grupos y 10 columnas), así que leerlos con `HfFileSystem` son miles de pedidos chicos. Desde el PC, el de SQL (0.26 GB) tardó 328 s, 0.7 MB/s.
  - A ese ritmo, los seis lenguajes (~6.9 GB) toman ~2.5–3 h, una sola vez: las sesiones siguientes reutilizan `prep/`.
- **Cambio:**
  - En `notebooks/t1_shards_cpu.ipynb`, generado con `make_notebooks.py`, `correr()` imprime cada `LATIDO` = 300 s una línea ⏱ con lo que baja la red, la CPU, la RAM, el tamaño de la salida, los archivos de `prep/` y los shards.
  - Es código de la celda, no del repo: con `COMMIT = a52a76d` en la celda, el constructor y los shards no cambian, y el chequeo de `builder_commit` entre sesiones sigue igual.
- **Siguiente:**
  - Las sesiones 2 y 3 usan el notebook nuevo con el mismo `COMMIT`.
  - Bajar los parquet de Stack-Edu enteros sería mucho más rápido. Queda como pregunta para el chat de diseño, para después de la T1, porque el código no cambia entre sesiones.

## 2026-11-16 · T1: Stack-Edu se lee en orden y una sola vez (patch 0012)

- **Qué:** la sesión 1 de la T1 se quedaba horas en `[prepare]`, bajando ~12 MB/s sin avanzar.
  - La sesión A iba por el segundo archivo de Python a las 2.9 h, recibió un HTTP 503 y murió.
  - La B había bajado 23.7 GB a los 35 min, cuando los seis lenguajes de Stack-Edu suman 6.9 GB.
- **La causa:** `Permissive.build` leía cada archivo con `ParquetFile.read`.
  - Antes de pyarrow 26, eso lee cada columna, a lo largo de todos los grupos de filas, en su propio hilo.
  - Los hilos piden posiciones lejanas del mismo archivo remoto, y la única ventana de lectura anticipada del archivo (4 MiB) se rellena en la posición de cada uno. Cada pedazo de ~20 KB cuesta un bloque entero.
  - **Medido en local,** con un archivo como los de Stack-Edu (grupos de 1,000 filas) detrás de una ventana como la de `HfFileSystem`:

    | pyarrow | Bajado |
    |---|---|
    | 19 | ~163× el archivo |
    | 21 | 37–136×: varía de una corrida a otra, según cómo se turnen los hilos |
    | 26 | 1.0×, porque junta las columnas de cada grupo en una sola lectura |

  - En Kaggle, con 4 núcleos, el ritmo de la A da ~100–200×: uno o dos archivos de Python de ~0.5 GB en 2.8 h, a ~12 MB/s. A ese ritmo, los seis lenguajes eran más de un día de descarga, más de lo que dura una sesión.
- **Cambio** (`ludwig_lm/sources.py`):
  - **`read_columns`** lee un archivo grupo por grupo, en un solo hilo: 1.02× el archivo con pyarrow 19, 21 y 26. `Permissive.build` lo usa, e imprime una línea por archivo con su velocidad.
  - **Las filas de las fuentes** (`iter_rows`) también se leen en un hilo: estaban en ~1.2–1.3× y quedan en 1.02×.
  - **16 MiB por pedido** en vez de 4, así que hay 4 veces menos pedidos al Hub.
  - **Reintentos:**
    - Un error pasajero (conexión caída, tiempo agotado, 429 o 5xx) se reintenta hasta 10 veces, con esperas de 5 s a 2 min (~12 min en total), y el aviso queda en el log.
    - Un 404 o un 401 falla de una.
    - Antes, la lista de Stack-Edu no tenía reintentos, así que un 503 tumbaba la sesión (probablemente lo que mató a la A). Las filas esperaban ~75 s en total.
  - **`shards.py`:** `as_posix()` en `_Files.find`, para que `test_shards_cpu.py` pase en Windows.
  - **El notebook de la T1** imprime las versiones de pyarrow, fsspec y huggingface_hub.
- **Dónde:** chat de diseño, en CPU, sin Hugging Face. Las mediciones con pyarrow 19 y 21 corrieron en entornos aparte.
- **Resultado:**
  - `test_shards_cpu.py` suma 5 chequeos:
    - la lista de Stack-Edu y las filas de un archivo con grupos de 1,000 filas, leídas a través de una ventana como la de `HfFileSystem`, bajan ≤1.1× el archivo;
    - un 503 a mitad de archivo se reintenta y da la misma lista, o las mismas filas;
    - un 404 no se reintenta.
  - Los 5 pasan también con pyarrow 19 y 21.
- **Ojo:**
  - El código del constructor cambió, así que el `COMMIT` de la T1 pasa a ser el de este patch, para las 3 sesiones. Como todavía no hay shards, no se pierde nada.
  - La sesión B no puede terminar `[prepare]` con el código viejo: hay que detenerla.
- **Siguiente:** repetir la sesión 1 con el commit nuevo. `[prepare]` debería tardar ~15 min, y en el log debe salir una línea por archivo de Stack-Edu, a la velocidad de la red.

## 2026-11-16 · Patch 0012: las pruebas en el PC, y una corrección

- **Qué:** se aplicó el patch 0012 sobre `2f4c718` y se corrieron las pruebas en CPU.
- **Dónde:** el PC de Pipe, en Windows y en CPU, con los 2 procesos lanzados a mano.
- **Resultado:**
  - pyflakes limpio y notebooks en sincronía.
  - `bench_afmoe.py`: restauración exacta.
  - `test_shards_cpu.py` (53 revisiones), `test_train_cpu.py` (36) y `test_relay_cpu.py` (23): PASS. La de shards ya pasa en Windows, con el `as_posix()` del patch.
  - **La lectura nueva, contra Stack-Edu de verdad:** `read_columns` sobre `SQL/train-00000-of-00001.parquet` (261 MB) hizo 10 pedidos y bajó 264 MB, 1.01× el archivo, en 358 s.
- **Corrección a la entrada del latido (2026-11-16):**
  - Los 0.7 MB/s medidos desde el PC eran el ancho de banda del PC, no pedidos de más. Aquí pyarrow es 26.0.1, que ya junta las columnas de cada grupo, así que en el PC el código anterior tampoco multiplicaba la lectura.
  - En Kaggle, con un pyarrow anterior al 26, sí la multiplicaba: es la causa que encontró el 0012.
- **Siguiente:** la sesión 1 de la T1 otra vez, con el commit nuevo.

## 2026-11-16 · T1, sesión 1: `prepare` en 47 min, y SQL estaba en otro shard

- **Qué:** la sesión 1 de la T1 con el commit `3260960` (patch 0012).
- **Dónde:** Kaggle, CPU (4 núcleos, 31 GB), con pyarrow 23.0.1, fsspec 2025.12.0 y huggingface_hub 1.33.0.
- **Archivos:**
  - `results/t1/sesion1_intento1_salida_kaggle.txt`, el log.
  - `results/t1/common_pile_lenguajes_0085_0091.txt`, la medición de los shards hecha desde el PC.
- **Resultado:**
  - **pyarrow < 26 en Kaggle,** como dijo el diagnóstico del 0012.
  - **`[prepare]` terminó en 47 min (2,830 s).** Cada archivo de Stack-Edu bajó una vez, a 2–4 MB/s; dos timeouts se reintentaron solos. Quedaron los permisivos de Python (4,563,384), JavaScript (2,596,127), TypeScript (1,703,720), Markdown (7,176,540), SQL (397,805) y Shell (1,394,774), y la descontaminación: 7,048,101 13-gramas de 16 benchmarks, con 1,240 descartados como plantillas.
  - **`[val]`:** 22 fuentes bien, hasta CSS. Falló en SQL: "its last file gave no validation document".
  - **La causa:** la receta buscaba SQL en el shard `0086` de Common Pile, que es Rust. Leyendo el comienzo de los shards 0085–0091 y el 0088 entero, el orden es Ruby (0085), Rust (0086), Shell (0087–0088), SQL (dentro del 0088, de la fila 217,222 a la 605,833: 388,612 filas), Swift (desde el final del 0088 hasta el 0090) y TypeScript (0091–0094). El mapa de la revisión independiente ponía SQL en el 0086.
  - **Arreglo en `configs/data_v3.json`, antes del primer shard:** el rango de SQL pasa de `[86, 86]` a `[88, 88]`. Shell queda en `[86, 88]`. `select_files` sobre el Hub elige `stack-edu-0088.json.gz` para SQL.
- **Siguiente:** la sesión 1 otra vez, de cero, con el commit nuevo. `prepare` se rehace (~47 min) y después vienen la validación y los shards.

## 2026-11-16 · T1, sesión 1: 6 shards, y un error en el caché de etiquetas

- **Qué:** la sesión 1 de la T1 con el commit `2d79a37` (SQL en el shard 0088).
- **Dónde:** Kaggle, CPU (4 núcleos), con pyarrow 23.0.1.
- **Archivos:** `results/t1/sesion1_intento2_salida_kaggle.txt`, el log.
- **Resultado:**
  - **`[prepare]` en 22 min,** con la red de Kaggle a ~20 MB/s.
  - **`[val]` completa:** las 29 fuentes, con SQL y Shell desde el shard 0088.
  - **`[train]`: 6 shards (0–5).** El primero tardó 43.6 min y los otros 22–24 min cada uno, a 94–101K tokens/s, con la CPU al ~35% de 4 núcleos y la red a ~2 MB/s.
  - **Después del shard 5, `KeyError: 'northjersey.com'`.** El caché de etiquetas MeCo se vaciaba al pasar de 200,000 dominios justo después de guardar el nuevo, y la línea siguiente lo buscaba. Le tocó al dominio 200,001.
- **Arreglo (`ludwig_lm/build.py`):**
  - El caché se vacía antes de guardar la etiqueta nueva. Es solo un caché: cada etiqueta sale igual con o sin él, así que los shards no cambian, y `BUILDER_VERSION` y el `build_id` siguen iguales.
  - Se comprobó con una simulación: el bucle viejo falla en cuanto se llena el caché, y el nuevo da lo mismo que sin caché.
- **Para seguir desde los 6 shards:**
  - El notebook de la T1 suma `OTRO_COMMIT`, que pasa `--allow-code-change`.
  - La sesión siguiente usa el commit nuevo con `OTRO_COMMIT = True`, porque el manifiesto guarda `builder_commit = 2d79a37`. Las que vengan después también, mientras el manifiesto no cambie ese campo.
- **Ojo:**
  - A ~23 min por shard, una sesión de 10.7 h da ~27 shards, no 56, así que los 168 piden 6–7 sesiones.
  - El tronco gasta ~1 shard por turno (~137M tokens en 11 h a ~3,450 tokens/s), así que la T1 va muy por delante.
  - Con la CPU al ~35%, el constructor corre casi todo en un solo hilo. Queda como pregunta para el chat de diseño.
- **Siguiente:**
  - Con la salida de esta sesión (los 6 shards y su estado), el dataset `ludwig2-tronco-v3-1`.
  - La sesión siguiente con el commit nuevo.
  - En paralelo, la G0-0 y la G0-1.

## 2026-11-17 · T1: 20 shards, Stack Overflow colado en StackExchange, y se reconstruye

- **Qué:** la sesión 1 de la T1 con `454673c`, que empezó de cero, construyó 20 shards. Antes del primer turno del tronco se revisaron su reporte y sus muestras, como pide el notebook.
- **Dónde:** Kaggle, CPU. Duró 10.2 h, a 30.6 min por shard en promedio, contando `prepare` y la validación.
- **Archivos:**
  - `results/t1/build_367d680a/report_00020.json` y `muestras.txt`.
  - Llegaron copiados a mano, con CRLF y sin salto final. Se restauraron como los escribe el constructor: el JSON con su salto final y las muestras sin él.
- **Resultado:**
  - **La mezcla es exacta:** cada una de las 29 fuentes quedó a 0.001 puntos o menos de su peso.
  - **Los formatos salen bien en todas las fuentes:**
    - la plantilla de chat con `<think>` en las trazas de matemática y en OpenAssistant;
    - FIM en ~11% del código;
    - las etiquetas MeCo;
    - el texto paralelo frase por frase;
    - los notebooks con sus marcas.
  - **El problema:** las 3 muestras de `stackexchange` son de Stack Overflow.
    - `skip_sites` compara `stackoverflow.com` distinguiendo mayúsculas, y el repo escribe `data/Stackoverflow.com/`.
    - Ese sitio es el 69% de los bytes (335 de 758 archivos) y, en orden alfabético, va tercero.
    - El 1.5% de StackExchange del tronco habría sido casi todo Stack Overflow durante toda la v1, y además duplicado con la fuente de Stack Overflow (1.0%).
- **Arreglo en la receta (`configs/data_v3.json`):**
  - `contains` elige por archivo los sitios de la red salvo Stack Overflow y su meta: 422 archivos, 6.06 GB, 342 sitios. El archivo de validación sigue siendo el mismo.
  - `skip_sites` suma `Stackoverflow.com`, como segundo filtro por URL.
- **Por qué se reconstruye:**
  - La receta cambia, así que cambia el `build_id` y el build no puede seguir. Ningún turno del tronco usó esos shards.
  - Los 20 shards viejos sirven solo para la G0-0 y la G0-1, que prueban el relevo y la memoria a 16K, no los datos. Después se borran, para no adjuntarlos al tronco por error.
- **Preguntas abiertas para el chat de diseño:**
  - La descontaminación saca 12,743 documentos de Python (4.3%) y 15,076 de OpenMathReasoning (29% de los que no salen por largos). ¿Es lo esperable, o hay falsos positivos en código?
  - OpenMathReasoning descarta 708,197 trazas por largas (más de 2K), contra 36,356 que entran.
- **Siguiente:** la sesión 1 de cero, con el commit nuevo, `OTRO_COMMIT = False` y sin datasets de entrada. Su salida va a un dataset con otro nombre.

## 2026-11-17 · Los notebooks traen los nombres reales de Hugging Face y el commit vigente

- **Qué:**
  - La G0-0 falló con un 404 en `ludwig-lm/ludwig2-relevo-prueba`, el nombre de ejemplo del notebook. Al importarlo de nuevo, la primera celda había vuelto a sus valores por defecto.
  - Antes, el `plan` con `75cd3a0` confirmó el arreglo de StackExchange (422 archivos, 6.1 GB). Sus únicos ❌ fueron los datasets con términos (The Stack, StarCoderData y Flores+), con un 401: el secreto de Kaggle tenía el token viejo, ya borrado.
- **Cambio en `make_notebooks.py`:**
  - El relevo trae por defecto `HF_REPO = ludwig-v2-lm/ludwig-v2-lm-1` y `HF_REPO_PRUEBA = ludwig-v2-lm/ludwig-v2-lm-1-relevo-prueba`.
  - Los dos notebooks traen `COMMIT = 75cd3a0…` en vez de `main`.
  - `RESPALDO` sigue con `DUENO`, porque el usuario de Kaggle no entra al repo (regla 6). Es código de celda: el commit del tronco sigue siendo `75cd3a0`.
- **Siguiente:** la G0-0 con los nombres reales, y la T1 con el token nuevo en el secreto.

## 2026-11-17 · G0-0: el repo de prueba tenía restos, y la celda se arregla para poder repetirla

- **Qué:** la G0-0 (`MODO = "prueba-relevo"`) con los nombres reales de Hugging Face, sobre el código `75cd3a0`.
- **Dónde:** Kaggle, 2×T4, 2.3 min.
- **Archivo:** `results/g0/g0_0_con_restos_salida_kaggle.txt`, con el usuario de Kaggle enmascarado.
- **Resultado: ❌, pero no por el relevo.**
  - El repo de prueba ya tenía el paso 20 con la etiqueta `prueba-relevo`, de una corrida anterior cuyo log no llegó. La sesión A siguió desde ahí y, como tiene pedido parar en el 20, se detuvo sin dar un paso: no subió ni respaldó nada. La prueba pide 3 subidas y un respaldo de A.
  - Todo lo que corrió funcionó:
    - el candado, tomado y soltado dos veces;
    - la descarga del paso 20 desde Hugging Face, en 1–2 s;
    - la reanudación de B, exacta en el estado y en el primer lote;
    - 10 pasos a ~10K tokens/s;
    - la subida del paso 30 (0.13 GiB en 3 s), el respaldo en `ludwig2-respaldo-a` (5 s), el aplastado y la entrega.
  - Las líneas de `train.py` salen dos veces en el log. No son dos procesos: `train.py` solo imprime desde el rank 0, así que es el visor de Kaggle el que muestra dos veces la salida del subproceso.
- **Dos fallas de la celda, arregladas en `make_notebooks.py`.** Solo cambia la celda: el código del relevo es el mismo y el commit sigue siendo `75cd3a0`.
  - **No se podía repetir.** Usaba siempre la etiqueta `prueba-relevo`, así que cada corrida seguía desde lo que dejó la anterior, en Hugging Face y en los respaldos. Ahora cada una usa la suya, `prueba-relevo-<fecha y hora>`, y no ve las anteriores. Se comprobó en el PC con el relevo real (`LocalStore`) sobre un repo con los pasos 20 y 30 viejos: A empieza de cero, y el índice termina con el 20 y el 30 nuevos.
  - **La subida del paso 10 era cuestión de suerte.** La sesión miraba el disco cada 10 s, y el modelo diminuto va del paso 10 al 20 en ~3 s. Si no miraba justo en ese hueco, solo subía el 20, y la prueba fallaba con 2 subidas de 3. Ahora mira cada 0.5 s, como `scripts/test_relay_cpu.py`.
  - Además, la celda imprime cada condición con ✅ o ❌, y lo que Kaggle muestra montado de cada respaldo (si `checkpoint.tar` llega como un `.tar`).
  - La celda nueva se probó en el PC con un relevo simulado: da ✅ en una corrida limpia y ❌ en una como esta, con sus tres razones.
- **Preguntas abiertas para el chat de diseño:**
  - Revisar estos cambios de la celda.
  - La G0-0 nunca sigue desde un respaldo de Kaggle. B sigue desde Hugging Face, porque el respaldo de A es una versión nueva del dataset, y una sesión solo ve las versiones que estaban montadas al empezar. Seguir desde Kaggle solo se prueba en CPU, con un Kaggle simulado.
- **Siguiente:** repetir la G0-0 con el notebook regenerado, y después la G0-1.

## 2026-11-17 · G0-0 ✅, y Kaggle desempaca el `checkpoint.tar` de los respaldos

- **Qué:** la G0-0 con la celda de `186de96`, sobre el código `75cd3a0`.
- **Dónde:** Kaggle, 2×T4, 3.7 min.
- **Archivo:** `results/g0/g0_0_salida_kaggle.txt`, con el usuario de Kaggle enmascarado.
- **Resultado: ✅, las 8 condiciones.**
  - A empezó de cero. Subió el paso 10 mientras entrenaba y el 20 al pausar, respaldó el 20 en `ludwig2-respaldo-a`, aplastó y soltó.
  - Con el candado tomado, una segunda sesión recibió *ocupado*.
  - B siguió desde Hugging Face en el paso 20, exacta en el estado y en el primer lote. Subió el 30, lo respaldó, aplastó y soltó.
  - El índice de Hugging Face rotó como en la prueba del PC con restos: el 10 nuevo sacó al 20 viejo, y el 20 nuevo al 30 viejo.
  - Tiempos con 0.13 GiB: subidas de 4.5–7.8 s, salvo la del paso 10 (52.3 s, sin causa clara); respaldos de 15 s; descarga de 7.7 s. La velocidad con ~11 GB se verá en el primer turno del tronco.
  - Los pasos 20–29 de B y sus evaluaciones dan los mismos números que en la corrida con restos, que siguió desde el paso 20 de otra corrida: el modelo diminuto da lo mismo en dos sesiones de Kaggle, a la precisión del log.
- **Lo que mostró el respaldo montado (línea 15 del log):**
  - Kaggle desempacó `checkpoint.tar`. El dataset tiene `checkpoint.json` y una carpeta `checkpoint/step_000030/` con el checkpoint, y ya no tiene el `.tar`.
  - `KaggleBackup.copies()` busca `checkpoint.tar` junto a `checkpoint.json`, así que nunca encuentra un respaldo:
    - un turno nunca sigue desde Kaggle. Si la subida final a Hugging Face falla, el siguiente sigue desde la subida anterior, de hasta 3 h antes;
    - con `SIN_CANDADO = True`, el turno no encuentra nada, empieza de cero, y su respaldo reemplaza al de `ludwig2-respaldo-a`;
    - `target()` elige siempre `ludwig2-respaldo-a`, así que los respaldos no se turnan.
  - La prueba del relevo en CPU no lo veía: su Kaggle simulado guarda el `.tar` tal cual.
  - Falta saber si `.metadata`, el archivo oculto que necesita `dcp.load`, sobrevivió al desempacado. La lista de esta corrida no muestra archivos ocultos.
- **Cambio en `make_notebooks.py`:** la lista de lo montado en cada respaldo pasa a la celda de preparación, y ahora incluye los archivos ocultos. Sale en todos los modos, así que la G0-1 dirá si está `.metadata`, si se le adjunta `ludwig2-respaldo-a`. Es solo una impresión: si falla, no detiene el turno.
- **Error del relevo, para el chat de diseño.** Hay que arreglarlo antes del primer turno del tronco, porque después el commit queda fijo. Opciones:
  - que `copies()` acepte también la carpeta desempacada, y que el turno la lea donde está, sin copiarla a `/tmp`;
  - si `.metadata` no sobrevive, subir también una copia con otro nombre y rehacer la carpeta en `/tmp` con enlaces;
  - que el Kaggle simulado de `scripts/test_relay_cpu.py` desempaque como el real;
  - una prueba en Kaggle que siga desde un respaldo montado, porque la G0-0 no lo hace.
- **Siguiente:** la G0-1 con `ludwig2-respaldo-a` adjunto, el arreglo del respaldo, y la T1 con el token nuevo.

## 2026-11-17 · G0-1 ✅: 16K cabe, y el `.metadata` sobrevive en el respaldo

- **Qué:** la G0-1 (`MODO = "g0-16k"`): 50 pasos del modelo real a 16K de contexto, con `configs/tronco_v1.json` y sin guardar. Es la condición 5 de la G0.
- **Dónde:** Kaggle, 2×T4, 26 min, con el notebook de `62c4210` sobre el código `75cd3a0`.
- **Archivo:** `results/g0/g0_1_salida_kaggle.txt`, con el usuario de Kaggle enmascarado.
- **Resultado: ✅, sin falta de memoria.**
  - Pico de 13.40 GiB reservada, bajo los 13.5 de la G0, con 0.1 GiB de margen.
  - 1,227 tokens/s. La T0.2 v2 dio 1,228 tokens/s a 16K, con 14.33 GiB reservada. La alarma de menos de 2,500 tokens/s es esperable a 16K.
  - La validación bajó de 4.1131 a 2.5028 bits por byte en 50 pasos (1.6M tokens), sin saltos de FP16.
- **Para mirar en el primer turno del tronco:** la carga máx/media de los expertos subió de 4.89 a 10.00 en el paso 15, y bajó a 3.5 hacia los pasos 45–49. Con 40 expertos y 4 por token, 10 es el techo: en alguna capa, todos los tokens eligieron al mismo experto. Parece el transitorio del arranque, con el sesgo de SMEBU corrigiendo. La alarma del diseño salta solo si la carga pasa de 3 durante 500 pasos seguidos.
- **El respaldo (línea 12 del log):** el `.metadata` de DCP sobrevivió al desempacado de Kaggle, en `checkpoint/step_000030/`. Así que el arreglo del relevo es el simple: que `copies()` acepte la carpeta desempacada y que el turno la lea donde está, sin copiarla a `/tmp`. No hace falta subir una copia con otro nombre.
- **Estado de la G0:** G0-0 ✅ y G0-1 ✅. Faltan los dos primeros turnos del tronco (G0-2 y G0-3).
- **Antes del primer turno:**
  - el arreglo del respaldo, con un commit nuevo para el tronco. La T1 sigue con `75cd3a0`, el mismo en sus tres sesiones;
  - los shards de la T1 nueva;
  - no adjuntar los shards viejos (build `367d680a`). El notebook elige el manifiesto con más shards, así que si se adjuntan junto con un build nuevo más corto, el turno entrenaría con los viejos.

## 2026-11-18 · T1, reconstrucción: la sesión 1 hizo 28 shards

- **Qué:** la sesión 1 de la T1 reconstruida (`FASE = "construir"`, de cero), con el arreglo de StackExchange de `75cd3a0`.
- **Dónde:** Kaggle, CPU. Duró 10.39 h, a 22.25 min por shard contando `prepare` y la validación. Paró por el tope de horas, no por `SHARDS = 56`.
- **Archivo:** `results/t1/sesion1_reconstruccion_resumen_salida_kaggle.txt`, el final del log con la tabla por fuente. El resto del log, el reporte, las muestras y el manifiesto llegan aparte.
- **Resultado:**
  - 28 shards (3.76B tokens) y 8.7 GB de salida.
  - Las 29 fuentes quedaron en su peso exacto, a dos decimales.
  - Cuadra con el build viejo de 20 shards multiplicado por 1.4:
    - los documentos, la descontaminación y el FIM (~11% del código) crecen ×1.4;
    - los repetidos crecen ×1.9–2.25, lo esperable cuando cada documento nuevo se compara con todos los ya vistos;
    - OpenMathReasoning sigue descartando ~19 trazas largas por cada una que entra (971,437 contra 50,098). Ya está anotado como pregunta abierta.
  - **StackExchange:** los documentos que se excluyen por estar en la validación (`writers.stackexchange.com`) suben de 119 a 1,290. En el build viejo, el 69% de los bytes era Stack Overflow; ahora el constructor lee los 342 sitios de la red, en orden barajado. Cuadra con el arreglo, pero la prueba son las muestras.
- **La cuenta de sesiones:** a 28 shards por sesión, los 168 del tronco son unas 6 sesiones, no 3 de 56. No apura, porque el tronco gasta ~1 shard por turno.
  - Se corrigen DISENO, los README y el notebook.
  - P18 en `DECISIONS.md` (3 sesiones de 56) queda para el chat de diseño.
- **Cambio en `make_notebooks.py`:**
  - El notebook de la T1 dice ~28 shards por sesión.
  - Los datasets nuevos se llaman `ludwig2-shards-1`, `-2`, …, para no confundirlos con los del build viejo (`ludwig2-tronco-v3-*`).
  - El notebook del tronco pide todos los `ludwig2-shards-*` y ninguno viejo.
- **Siguiente:**
  - el dataset `ludwig2-shards-1` con la salida de esta sesión;
  - las muestras y el reporte, para revisarlos antes del primer turno;
  - la sesión 2 de la T1, con el mismo `COMMIT` y `ludwig2-shards-1` como entrada.
