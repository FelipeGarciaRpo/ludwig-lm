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

## 2026-11-18 · El relevo lee los respaldos como los deja Kaggle

- **Qué:** el arreglo del error de los respaldos que encontró la G0-0 (entradas del 2026-11-17). Pipe lo aprobó. Commit `3b9c114`.
- **Cambio en `ludwig_lm/relay.py`:**
  - `KaggleBackup.copies()` reconoce lo que deja Kaggle: `checkpoint.json` en la raíz del dataset y la carpeta `checkpoint/step_XXXXXX/`. La cuenta solo si su `meta.json` es del mismo paso y está el `.metadata` de DCP. Una copia que todavía traiga `checkpoint.tar` también cuenta.
  - Un turno que sigue desde Kaggle lee esa carpeta donde está montada, sin copiar ~11 GB a `/tmp` (evento `in_place`). `train.py` solo la lee.
  - Un turno solo borra su punto de partida si es una copia dentro de su carpeta de trabajo.
- **Cambio en `scripts/test_relay_cpu.py`:** el Kaggle simulado hace lo que hace el real: desempaca, no deja el `.tar` y deja todo de solo lectura, como `/kaggle/input`.
  - La prueba revisa la copia desempacada y la empacada, y que no cuente una copia sin `.metadata`.
  - La sesión 2 sigue desde el respaldo leyéndolo donde está, y los pasos 0–11 siguen idénticos bit a bit a la corrida sin cortes.
  - Con el `relay.py` anterior, la parte del respaldo falla en 3 condiciones: no encuentra ninguna copia.
- **Cambio en `make_notebooks.py`:**
  - Un modo nuevo, `prueba-respaldo`. Sigue desde el respaldo de la última `prueba-relevo` sin Hugging Face ni candado, como un turno con `SIN_CANDADO`. Entrena 10 pasos del modelo diminuto y respalda en el otro dataset. Es lo único que prueba en Kaggle real que un turno puede seguir desde un respaldo.
  - Con `SIN_CANDADO = True`, si no hay ningún respaldo del tronco montado, el turno se niega a entrenar desde cero.
  - El `COMMIT` del tronco pasa a `3b9c114` y es el de todos sus turnos. La T1 sigue con `75cd3a0`, el mismo en todas sus sesiones.
- **Pruebas en el PC:**
  - pyflakes;
  - la prueba lógica de `bench_afmoe.py`, con el checkpoint exacto;
  - `test_relay_cpu.py`: PASS, 27 condiciones;
  - `test_train_cpu.py`: PASS, 36 condiciones;
  - la celda de `prueba-respaldo` con el `Session` real, un `/kaggle/input` falso con el respaldo desempacado (paso 30) y un `train.py` simulado: ✅ en sus 6 condiciones, y respalda en el dataset B.
- **Para el chat de diseño:** revisar el arreglo, la prueba nueva y el freno de `SIN_CANDADO`.
- **Siguiente:** la `prueba-respaldo` en Kaggle. El respaldo de la última G0-0 (paso 30) se entrenó con el build viejo, así que lleva como entradas solo `ludwig2-tronco-v3-2` y los dos respaldos. Después se pueden borrar los datasets viejos.

## 2026-11-18 · T1: las muestras del build nuevo, y StackExchange ya no es Stack Overflow

- **Qué:** la revisión de lo que dejó la sesión 1 de la reconstrucción, antes del primer turno del tronco, como pide el notebook.
- **Dónde:** el PC.
- **Archivos:** `results/t1/build_386755d6/`, con `manifest.json`, `report_00028.json`, `prepare.json` y `muestras.txt`.
  - Llegaron copiados a mano, con CRLF y sin salto final. A `prepare.json` le faltaba su primera línea (`{`).
  - Se restauraron como los escribe el constructor. Cada JSON coincide byte a byte con lo que el constructor escribe para su contenido, así que llegaron completos. Las muestras no se pueden comprobar así.
- **Resultado:**
  - Build `386755d60af32507`, con el constructor de `75cd3a0` y el tokenizador `0b7da8fdb062`.
  - 28 shards de 134,217,728 tokens, cada uno con su sha256, y la validación de las 29 fuentes (31.8M tokens).
  - Cada fuente quedó en su peso exacto.
  - La descontaminación son los mismos 7,048,101 13-gramas de 16 benchmarks. `prepare` tardó 16 min.
  - **StackExchange:** sus 3 muestras son de `rpg.stackexchange.com`, con la etiqueta MeCo del sitio, la pregunta y sus respuestas. Ya no es Stack Overflow.
    - Que las tres sean del mismo sitio es lo esperable: el constructor baraja los archivos de la fuente, cada archivo es de un solo sitio, y un archivo da para más de un shard.
  - Los formatos salen bien:
    - la plantilla de chat con `<think>` en OpenMathReasoning y en OASST, cerrada con `<|im_end|>`;
    - FIM en orden prefijo-sufijo-medio;
    - las etiquetas MeCo;
    - Europarl y la ONU frase por frase, en las dos direcciones;
    - los notebooks con sus marcas.
- **Corrección:** el constructor sí termina `muestras.txt` con un salto de línea, porque cada muestra termina en `\n`. La entrada del 2026-11-17 dijo lo contrario. Se agregó ese salto al `muestras.txt` de `build_367d680a`.
- **Conclusión:** el build sirve para el tronco.
- **Cambio en `make_notebooks.py`:** la G0-0 y `prueba-respaldo` escriben en los respaldos reales, y cada versión de un dataset de respaldo guarda un solo checkpoint. Una vez que empiece el tronco, una de esas pruebas reemplazaría su respaldo. El notebook ahora advierte que son solo para antes del primer turno.
- **Siguiente:** la `prueba-respaldo`, y después el primer turno del tronco (G0-2) con `ludwig2-shards-1`.

## 2026-11-18 · G0-0 ✅ con el commit del tronco, y la prueba del respaldo sin su dataset

- **Qué:** la G0-0 otra vez, ahora con `3b9c114` (el relevo que lee los respaldos desempacados) y los shards del build nuevo (`ludwig2-shards-1`). Después, un primer intento de `prueba-respaldo`.
- **Dónde:** Kaggle, 2×T4. La G0-0 tardó 2 min.
- **Archivo:** `results/g0/g0_0_3b9c114_salida_kaggle.txt`, con el usuario de Kaggle enmascarado.
- **Resultado de la G0-0: ✅, las 8 condiciones,** sobre el commit con el que va a correr el tronco.
  - Subidas de 0.13 GiB en 4.1–5.8 s y respaldos en 7–8 s. A y B respaldaron en `ludwig2-respaldo-a`, porque una sesión no ve las versiones que crea ella misma.
  - Al empezar, `copies()` ya veía en ese dataset el paso 30 de la G0-0 anterior, desempacado y con su `.metadata`. Era de otra etiqueta, así que no contaba.
- **`prueba-respaldo`, primer intento:** no llegó a entrenar. El notebook no tenía adjunto `ludwig2-respaldo-a`, que es donde quedó el respaldo de la G0-0, y la celda se detuvo con su aviso, como debe. El log no se guardó.
- **Siguiente:** repetir `prueba-respaldo` con los dos respaldos adjuntos. Después, el primer turno del tronco.

## 2026-11-18 · Los respaldos borran sus versiones viejas: cada turno en un notebook nuevo

- **Qué:** Pipe repitió la `prueba-relevo` con un dataset de respaldo nuevo, `ludwig2-respaldo-a2`, y pasó. Al editar el mismo notebook para la `prueba-respaldo`, Kaggle dijo que `ludwig2-respaldo-a2` ya no estaba disponible.
  - Antes, otro intento de `prueba-respaldo` se había detenido con su aviso: `RESPALDO` apuntaba a `ludwig2-respaldo-a2`, que solo tenía el `.txt` del principio, y no a `ludwig2-respaldo-a`, donde estaba el respaldo de la G0-0.
- **Por qué:** cada respaldo sube una versión nueva del dataset con `kaggle datasets version -d`, y `-d` borra las versiones viejas. Así el espacio de Kaggle no se llena con checkpoints de 11 GB. El notebook tenía adjunta la versión del `.txt`, que la prueba borró.
- **Consecuencia para el tronco:** cada turno va en un notebook nuevo, con las entradas recién agregadas. Si un turno no ve la última versión de los dos respaldos, no puede seguir desde Kaggle, y `target()` elige mal dónde respaldar.
- **Cambio en `make_notebooks.py`:** las instrucciones del turno y de `prueba-respaldo` lo dicen.
- **Siguiente:** `prueba-respaldo` en un notebook nuevo, con `ludwig2-shards-1`, `ludwig2-respaldo-a2` y `ludwig2-respaldo-b`. Esos dos son los respaldos del tronco desde ahora.

## 2026-11-18 · `prueba-respaldo`: el arreglo funciona en Kaggle; la corrida fue sin GPU y con `DUENO`

- **Qué:** antes, Pipe repitió la `prueba-relevo` con `ludwig2-respaldo-a2`, y salió todo ✅ (ese log no llegó). Después corrió la `prueba-respaldo` con `3b9c114`, desde el respaldo que esa prueba dejó en `ludwig2-respaldo-a2`: paso 30, shards del build nuevo.
- **Dónde:** Kaggle, pero sin acelerador. `nvidia-smi` no estaba, y `train.py` corrió en CPU: 2 procesos, ~35 tokens/s, 12 min.
- **Archivo:** `results/g0/prueba_respaldo_cpu_salida_kaggle.txt`, con el usuario de Kaggle enmascarado.
- **Resultado: 4 ✅ y 2 ❌. Los dos ❌ vienen de cómo se lanzó la prueba.**
  - ✅ **El arreglo funciona en Kaggle real.** Encontró el respaldo desempacado y siguió desde él en el paso 30, leyéndolo donde está montado (`in_place`), sin copiarlo. El primer lote fue idéntico al guardado, y entrenó hasta el paso 40.
  - ❌ **Estado exacto.** La huella del estado son sumas en float64 de cada tensor, hechas en el dispositivo del tensor. Al cargar se calcularon en CPU y al guardar en GPU, y la misma suma redondea distinto en cada una. El modelo se cargó bien: la validación en el paso 30 da 3.3214 bits por byte, igual que en GPU.
  - ❌ **El respaldo.** Eligió bien el otro dataset (`ludwig2-respaldo-b`), pero `RESPALDO` decía `DUENO/ludwig2-respaldo-b`, el ejemplo del notebook, y la subida falló. El evento guarda solo el principio del error, que son las barras de progreso de `kaggle`, así que la causa no se ve en el log.
- **Cambio en `make_notebooks.py`:** la celda de preparación se detiene si no hay GPU, y también si `RESPALDO` todavía dice `DUENO` en los modos que respaldan (`tronco`, `prueba-relevo`, `prueba-respaldo`). En un turno del tronco, cualquiera de los dos errores habría costado el turno o su respaldo.
- **Para el chat de diseño:** `backup_failed` guarda los primeros 200 caracteres del error, que con `kaggle` son barras de progreso. Convendría guardar el final, pero cambiarlo mueve el commit del tronco. Queda como pregunta.
- **Siguiente:** repetir la `prueba-respaldo` con GPU y el usuario de Pipe en `RESPALDO`. Después, el primer turno del tronco.

## 2026-11-18 · `prueba-respaldo` ✅: un turno puede seguir desde Kaggle, y arranca el tronco

- **Qué:** la `prueba-respaldo` repetida con GPU y con `RESPALDO` apuntando a los datasets de Pipe (`ludwig2-respaldo-a2` y `ludwig2-respaldo-b`), sobre `3b9c114`.
- **Dónde:** Kaggle, 2×T4, ~2.5 min.
- **Archivo:** `results/g0/prueba_respaldo_salida_kaggle.txt`, el final del log (líneas 60–76).
- **Resultado: ✅, las 6 condiciones.**
  - Siguió desde el respaldo de Kaggle en el paso 30, leyéndolo donde está montado, sin copiarlo.
  - El estado y el primer lote fueron exactos.
  - Entrenó hasta el paso 40 y respaldó en el otro dataset, `ludwig2-respaldo-b`.
  - Queda probado en Kaggle real el arreglo de `3b9c114`, y que los respaldos se turnan.
- **Estado de la G0:** pasaron la G0-0 (sobre `3b9c114`), la G0-1 y la prueba del respaldo. Faltan los dos primeros turnos del tronco, la G0-2 y la G0-3.
- **Cómo arranca el tronco:**
  - con el código `3b9c114`, el mismo en todos los turnos;
  - con los shards de `ludwig2-shards-1` (build `386755d60af32507`);
  - con los respaldos `ludwig2-respaldo-a2` y `ludwig2-respaldo-b`;
  - cada turno en un notebook nuevo, con las entradas recién agregadas;
  - desde ahora no se corren `prueba-relevo` ni `prueba-respaldo`, porque escribirían sobre los respaldos del tronco.
- **Siguiente:** el primer turno del tronco (G0-2), en la cuenta de Pipe. El segundo (G0-3), en la cuenta de un amigo, para probar la reanudación entre cuentas.

## 2026-11-19 · Tronco, turno 1 (G0-2): 549 pasos y las condiciones de la G0; el candado no se soltó

- **Qué:** el primer turno del tronco, desde cero, en la cuenta 1. Es la G0-2.
- **Dónde:** Kaggle, 2×T4, del 2026-11-18 a las 23:04 al 2026-11-19 a las 10:10 UTC (11.1 h de sesión).
- **Archivo:** `results/tronco/turno01_cuenta-1_salida_kaggle.txt`, con el usuario de Kaggle enmascarado. Faltan el `.jsonl` y el `.relevo.json` del turno (regla 10).
- **Resultado:**
  - Pasos 0–549 (143.9M tokens) en 10.97 h de entrenamiento.
  - 3,656 tokens/s en la última hora y un pico de 12.91 GiB reservada.
  - Ningún salto de FP16: la escala se quedó en 32,768 todo el turno.
  - La entropía cruzada bajó de 11.33 a 5.00. El LR sube hasta el paso 2,000, así que el turno terminó en ×0.27 del pico.
  - La carga de los expertos quedó entre 1.3 y 1.9 veces la media, sin el pico que tuvo la G0-1.
  - 4 subidas de 11.29 GiB (pasos 151, 303, 455 y 549), de 74 a 115 s cada una.
  - El respaldo en `ludwig2-respaldo-a2` tardó 265 s. El cierre entero tomó ~7 min de los 45 que reserva el turno.
  - **Condiciones de la G0 en este turno: 1–4 ✅.** La reanudación en otra cuenta le toca al turno 2.
- **El candado no se soltó:**
  - Justo después del aplastado, la entrega falló con `Conflict: … moved from 0ad1034e`. El checkpoint 549 quedó completo en Hugging Face y en el respaldo; solo el candado siguió tomado.
  - **Causa probable:** `super_squash_history` no devuelve el commit nuevo, así que el relevo lee la cabeza con `repo_info` justo después. Con dos checkpoints de 11 GB en el repo, esa lectura habría llegado desactualizada, y la entrega con ese padre se rechazó. En las G0-0, con el repo de prueba chico, no pasó.
  - El candado vence solo a las 2 h de su última renovación (la subida del paso 549): el 2026-11-19 a las 12:05 UTC. Un turno que se lance antes se niega a entrenar, como debe.
- **Cambio en `make_notebooks.py`:**
  - Si la entrega falla, la celda del turno espera 30 s, vuelve a leer la cabeza y suelta el candado, pero solo si sigue siendo de su sesión. Lo intenta hasta 6 veces.
  - Usa el `Lock` del relevo tal cual, así que el `COMMIT` del tronco no cambia.
  - Se probó con el `LocalStore`: suelta el candado propio, no toca uno ajeno y sobrevive a un intento fallido.
- **Para el chat de diseño:** el arreglo de fondo está en `relay.py`: reintentar la entrega con la cabeza nueva. Pero eso cambia el `COMMIT` del tronco, así que primero se decide en `DECISIONS.md`.
- **Siguiente:**
  - el `.jsonl` y el `.relevo.json` del turno;
  - el turno 2 (G0-3) en la cuenta de un amigo, después de las 12:05 UTC;
  - la sesión 2 de la T1.

## 2026-11-21 · Tronco, turno 2: del paso 549 al 1083, y el candado se soltó al reintentar

- **Qué:** el segundo turno del tronco, otra vez en la cuenta 1. Además llegaron los `.jsonl` y `.relevo.json` de los turnos 1 y 2.
- **Dónde:** Kaggle, 2×T4, 10.78 h de entrenamiento.
- **Archivos:** en `results/tronco/`:
  - `turno01_cuenta-1.jsonl` y `.relevo.json`;
  - `turno02_cuenta-1.jsonl`, `.relevo.json` y `_final_salida_kaggle.txt` (el final del log).

  Llegaron con CRLF y se restauraron; cada fila se reconstruye byte a byte, así que llegaron completos. El usuario de Kaggle, en el respaldo de los `.relevo.json`, quedó enmascarado.
- **Resultado:**
  - **Siguió desde el respaldo de Kaggle** (`ludwig2-respaldo-a2`, paso 549), leyéndolo donde está montado: estado exacto y primer lote idéntico. Es la primera reanudación del tronco, y salió desde Kaggle porque Hugging Face y el respaldo empataban en el paso 549.
  - Cargar el checkpoint tomó 688.5 s, o sea 11.3 GB a ~17 MB/s. El montaje de Kaggle es lento para archivos grandes.
  - Pasos 549–1083 (140M tokens), a 3,642 tokens/s en la última hora, con un pico de 12.90 GiB.
  - Ningún salto de FP16. La escala de pérdida subió de 32,768 a 65,536 en el paso 1000, tras un tramo largo sin desbordes.
  - **Un pico de pérdida en el paso 731:** entropía cruzada de 5.94, contra una media de 4.62 (unas 4.9 desviaciones). El freno del diseño se saltó ese paso sin actualizar, y en el 732 ya estaba en 4.39.
  - **Validación:** 1.7153 bits por byte en el paso 549 y 1.1534 en el 1000 (inglés 1.44, español 1.34, Python 0.96, JS 0.82, matemática 1.15). Ningún experto muerto, y la entropía del router baja de 3.65 a 3.36.
  - 4 subidas, y el respaldo en `ludwig2-respaldo-b`: los dos respaldos se turnan.
  - La entrega del candado volvió a fallar después del aplastado. La celda la reintentó a los 30 s y lo soltó al primer intento, así que el arreglo del notebook funciona.
- **Para el chat de diseño:**
  - Cuando Hugging Face y el respaldo empatan, que es siempre, el relevo lee de Kaggle. Este turno tardó 11.5 min en cargar. Preferir Hugging Face en el empate se puede hacer en el notebook, sin tocar el `COMMIT`, pero falta medir una descarga de 11 GB.
  - El ✅ de "reanudación en otra cuenta" de la celda solo mira que la reanudación sea exacta, no que la cuenta sea otra. Este turno fue de la misma cuenta, así que la G0 sigue esperando un turno de otra cuenta.
- **Siguiente:** el turno 3, que ya arrancó; un turno en la cuenta de un amigo; la sesión 2 de la T1.

## 2026-11-25 · Tronco: el turno que siguió del paso 3131 se detuvo por picos de pérdida

- **Qué:** un turno del tronco que siguió desde el paso 3131. Entre el turno 2 y este corrieron otros turnos que todavía no tienen sus archivos en el repo. Al empezar, Hugging Face tenía el paso 3131, `ludwig2-respaldo-a2` el 1608 y `ludwig2-respaldo-b` el 2151.
- **Dónde:** Kaggle, 2×T4. Entrenó 4.74 h.
- **Archivo:** `results/tronco/pasos_3131-3348_salida_kaggle.txt`, el log, con el usuario de Kaggle enmascarado.
- **Resultado:**
  - **Siguió desde Hugging Face,** porque los respaldos de Kaggle eran más viejos. Bajó el checkpoint de 11.29 GiB en 55.4 s y lo cargó en 28 s, con el estado y el primer lote exactos. Son 83 s en total, contra los 688 s que tardó el turno 2 leyendo el respaldo desde el montaje de Kaggle.
  - Validación en el paso 3131: 0.8853 bits por byte (inglés 1.19, español 1.08, Python 0.67, JS 0.57, matemática 0.85).
  - **Se detuvo solo en el paso 3348,** con el estado `stopped_spikes`. Saltó tres picos de pérdida en 37 pasos: 3.04 en el 3311, 3.72 en el 3333 y 3.04 en el 3347, con la pérdida alrededor de 2.6. La regla del diseño (5.4) salta un pico de más de 4σ y detiene la corrida con 3 picos en 200 pasos.
  - Al detenerse no guardó: el último checkpoint es el 3270, subido a Hugging Face a las 3.1 h.
  - **La carga de los expertos está muy desbalanceada.** Estuvo entre 7.3 y 8.3 veces la media durante todo el turno, contra 1.3–1.9 en los turnos 1 y 2. El techo, con 40 expertos y 4 por token, es 10.
    - La entropía del router bajó a 2.60: era 3.36 en el paso 1000, y el máximo es 3.69.
    - La ganancia qk subió a 2.25: era 1.25 en el paso 1000.
    - La velocidad bajó de ~3,600 a ~3,300 tokens/s.
  - Ningún salto de FP16. La escala está en 262,144 y la norma del gradiente en ~0.13.
  - **El respaldo falló.** La subida a `ludwig2-respaldo-a2` llegó al 99% y `kaggle` devolvió un error. El evento guarda solo el principio del mensaje, que son barras de progreso, así que la causa no se ve. Los respaldos de Kaggle siguen en los pasos 1608 y 2151.
  - La entrega del candado volvió a fallar, y el reintento la soltó al primer intento.
- **Por qué no basta con lanzar el siguiente turno:** seguiría desde el 3270 con el mismo estado y los mismos datos, así que lo más probable es que vuelva a encontrar los mismos picos y se detenga en el mismo lugar.
- **Para el chat de diseño, antes del siguiente turno:**
  - el desbalance de los expertos, que habría empezado después del paso 1083, quizás cuando el LR llegó al pico en el paso 2000;
  - los picos y la regla que detiene la corrida;
  - qué cambiar en `configs/tronco_v1.json`, que se decide primero en `DECISIONS.md`.
- **Siguiente:**
  - los `.jsonl` y `.relevo.json` de los turnos desde el 3, para ver cuándo empezó el desbalance;
  - revisar el espacio de Kaggle, porque los respaldos dejaron de actualizarse;
  - la decisión del chat de diseño.

## 2026-11-25 · Tronco, turnos 3 a 7: el balanceo de los expertos se rompió desde el paso ~1200

- **Qué:** los registros de los turnos 3 a 7, y el análisis de la serie completa del tronco (pasos 0–3348).
- **Dónde:** el PC. Los turnos corrieron en Kaggle, 2×T4.
- **Archivos:**
  - `results/tronco/turno03_cuenta-1` a `turno07_cuenta-1`, cada uno con su `.jsonl` y su `.relevo.json`. Se restauraron como los anteriores (cada fila se reconstruye byte a byte), con los usuarios de Kaggle enmascarados.
  - `scripts/resumen_tronco.py`, que resume todos los turnos de `results/tronco/`, y su salida, `results/tronco/resumen_t01-t07.txt`.
  - El log del turno 7 es `results/tronco/pasos_3131-3348_salida_kaggle.txt`.
- **Los turnos:**
  - Turno 3: pasos 1083–1608. Turno 4: 1608–2151. Turno 5: 2151–2633, detenido por picos. Turno 6: del 2601 (el último checkpoint del 5) al 3131. Turno 7: 3131–3348, detenido por picos.
  - El turno 6 volvió a encontrar en el paso 2632 el pico que detuvo al 5: con el mismo estado y los mismos datos, el pico se repite.
  - El turno 5 corrió en otra cuenta de Kaggle aunque dice "cuenta 1", porque su respaldo apuntó al usuario de esa cuenta. Su reanudación desde el respaldo de Kaggle fue exacta.
  - Todas las reanudaciones fueron exactas.
  - Los respaldos de los turnos 5, 6 y 7 fallaron. El del 5, por el usuario equivocado en `RESPALDO`. Los del 6 y el 7, al 99%, con la causa cortada en el mensaje.
- **Cuándo se rompió el balanceo** (`resumen_t01-t07.txt`):
  - Hasta el paso ~1150, la carga máx/media por paso estuvo en ~1.5 y la mínima en ~0.6.
  - Desde el paso ~1200 (turno 3, con el LR en ×0.6), la carga mínima cae a 0: desde el 1400, en cada paso hay algún experto sin tokens. La máxima pasa de 2 desde el paso 1249, de 3 desde el 1335 y de 5 desde el 1767, y llega a ~7.8.
  - Está en las primeras 5 capas MoE. En la ventana de 1000 pasos, su experto más cargado tiene 1.5 veces la media (paso 2000) y 1.7 (paso 3000), y las otras 20 capas ~1.00. Que la ventana se vea casi pareja y cada paso no quiere decir que los expertos que acaparan van rotando.
  - La entropía del router baja de 3.36 (paso 1000) a 2.60, y la velocidad cae de ~3,650 a ~3,300 tokens/s.
  - No empezó con el LR en el pico (paso 2000), sino en ×0.6. Después siguió creciendo.
  - Hubo 16 picos de pérdida en los turnos 2–7, casi todos chicos: ~0.35–1 por encima de la media, con σ ~0.08.
- **Explicación más probable** (falta confirmarla en un checkpoint):
  - El router usa scores sigmoide, y elige el top-k de score + el sesgo de SMEBU.
  - El z-loss del router, pensado para routers softmax, penaliza logsumexp(logits)² y empuja todos los logits hacia abajo. `z_router` bajó de 15 a ~0.8, y el quiebre empezó cuando pasó de ~0.9.
  - Con los logits en la cola de la sigmoide, los pesos de la compuerta, que están normalizados, casi no dependen de su nivel, así que la entropía cruzada no frena la deriva. AdamW, que es el que actualiza el router, convierte ese gradiente chico pero constante en pasos del tamaño del LR.
  - Con los scores cada vez más chicos, las diferencias entre expertos se achican, y el sesgo de SMEBU, que es igual para todos los tokens, termina decidiendo. Casi todos los tokens van a los mismos expertos; SMEBU los castiga, y en el paso siguiente gana otro grupo.
  - Los picos de pérdida serían una consecuencia de esos cambios bruscos de expertos.
- **Propuesta para el chat de diseño:**
  1. **Confirmar antes de cambiar.** Cargar el checkpoint 3270 en Kaggle y medir, por capa, el nivel de los logits del router, la diferencia de score entre el 4.º y el 5.º experto, y el rango del sesgo. Si en las capas 0–4 el sesgo es mucho mayor que esa diferencia, la explicación queda confirmada.
  2. **Seguir desde el 3270 con:**
     - el z-loss del router en 0 (`router_z`); DeepSeek-V3 no lo usa con scores sigmoide;
     - el sesgo de los expertos y su momento en cero al reanudar, con un flag nuevo en `train.py`, para que vuelvan a decidir los scores;
     - SMEBU con `lam` 0.001 en vez de 0.01, para que el sesgo no vuelva a dominar;
     - el `seq_aux` en 1e-3 en vez de 1e-4, una fuerza de balanceo que sí depende de cada token.
  3. **El LR, sin cambios por ahora.** El problema empezó en ×0.6 y la validación siguió bajando (0.9604 en el paso 2000, 0.8853 en el 3131). Si el balanceo no se recupera con lo anterior, se revisa.
  4. **La regla de los picos, opcional:** detener con 5 picos en 200 pasos en vez de 3, mientras el router se recupera.
  - No hay checkpoint de antes del quiebre: Hugging Face tiene el 3131 y el 3270, y los respaldos, el 1608 y el 2151. Volver al 1608 perdería ~1,660 pasos, y ahí la carga ya estaba en ~4.5.
  - Todo esto cambia `configs/tronco_v1.json` y `train.py`, y con eso el `COMMIT` del tronco. Primero se decide en `DECISIONS.md`.
- **Siguiente:** que el chat de diseño decida, y no lanzar turnos hasta entonces. Revisar quién corrió los turnos 5 a 7, que cada cuenta tenga su `CUENTA` y permiso de edición en los dos respaldos, y que `RESPALDO` lleve siempre el usuario de Pipe.

## 2026-11-25 · El router rompió el primer tronco, y el tronco vuelve a empezar (patch 0013)

- **Qué:** el diagnóstico del desbalance de los expertos del primer tronco (P29), la decisión (P30) y la sonda que lo confirma (P31). También lo que hace falta para el nuevo arranque.
- **Dónde:** chat de diseño, en CPU, con los `.jsonl` de `results/tronco/` y de `results/t0_3_t2/`.
- **El diagnóstico (P29):**
  - **El quiebre depende del nivel del router, no del paso ni del LR.** Empieza con el mismo `z_router` (~0.85, con λ 0.01) en dos corridas con calendarios muy distintos:
    - el tronco, en el paso 1249;
    - `lr-x1-1200` de la T2, con 100 pasos de calentamiento, en el 544.

    `lr-x2` se rompió con z ≈ 1.0–1.3, y `lr-x4`, con λ 0.04, con z ≈ 2.2. Las corridas que nunca bajaron de z ≈ 1.5 nunca se rompieron.
  - **El mecanismo:**
    1. El z-loss del router (1e-4 × logsumexp²) empuja todos los logits hacia abajo.
    2. Los pesos de la compuerta están normalizados entre los 4 expertos elegidos, así que la entropía cruzada no ve el nivel de los logits.
    3. AdamW convierte ese gradiente chico pero constante en pasos del tamaño del LR.
    4. Los scores sigmoide se hunden en la cola, y la diferencia entre el 4.º y el 5.º score de cada token se achica con ellos.
    5. SMEBU mueve el sesgo hasta λ por paso. Cuando λ es varias veces esa diferencia, un paso del sesgo re-enruta a casi todos los tokens a la vez. Los acaparadores se castigan y gana el grupo siguiente: el lazo da vueltas.
  - **Un modelo de juguete del lazo lo reproduce** (`scripts/toy_router_loop.py`, `results/tronco/toy_router_loop.txt`): scores sigmoide, top-4 de score + sesgo, y SMEBU.
    - Con λ 0.01, el experto más cargado queda en 1.1× mientras λ/brecha ≤ 3.9, tambalea con 10 (2.8× de media, 6.8× de máximo) y colapsa con 27 (8.6× y 9.8×, de un techo de 10, turnándose).
    - Con λ 0.001 aguanta una brecha 10 veces más chica.
  - **La T2 comparó corridas rotas:** `lr-x1-1200` se rompió en el paso 544 y `lr-x2-320-800` en el 470, antes del paso 800 que compara la P25. ×1 se queda, porque es la opción segura.
- **La decisión (P30):** el tronco vuelve a empezar desde el paso 0 como `tronco2` (`configs/tronco_v2.json`).
  - Sin z-loss del router (`router_z` 0, ahora también por defecto). El z-loss de salida se queda, porque la cabeza sí es una softmax.
  - SMEBU con λ 0.01 hasta el paso 500 (arranque rápido) y 0.001 después (el γ de DeepSeek-V3).
  - `seq_aux` 1e-3 en vez de 1e-4.
  - Evalúa cada 250 pasos.
  - El LR, los datos y el resto, igual.
- **Por qué de cero y no desde el 3270:**
  - No queda ningún checkpoint de antes del quiebre: Hugging Face tiene el 3131 y el 3270, y los respaldos, el 1608 y el 2151.
  - En el 3270, el sesgo llevaba ~2,000 pasos eligiendo los expertos de todos los tokens en las primeras 5 capas MoE.
  - Empezar de cero cuesta ~3,350 pasos: ~3 días de relevo, 3.7% del tronco. A cambio, la corrida tiene una sola configuración desde el paso 0, y prueba el arreglo justo donde falló. Pipe prefería reentrenar para que los expertos queden limpios.
- **Archivos:**
  - `ludwig_lm/afmoe.py`: la salud del router por capa: la brecha mediana y cuántas veces decide el sesgo.
  - `scripts/train.py`:
    - el cambio de λ (`--lam-final`, `--lam-steps`);
    - la salud del router en cada paso y, por capa, en cada evaluación;
    - la alarma de brecha;
    - la parada por balanceo (`--balance-stop`), que guarda, igual que ahora la parada por picos;
    - `--eval-only`;
    - `router_z` 0 por defecto.
  - `ludwig_lm/relay.py`: cada corrida rota solo sus checkpoints, bajo `checkpoints/<etiqueta>/`.
  - `configs/tronco_v2.json`.
  - `scripts/toy_router_loop.py` y su salida.
  - El notebook del tronco: `tronco2` y el `MODO` `sonda`.
- **Resultado:**
  - `test_train_cpu.py` PASS (41): el cambio de λ en el paso 3; la parada por balanceo, que guarda donde paró; y `--eval-only`, que da la evaluación de B en el paso 3 bit a bit, router incluido, sin dar un paso.
  - `test_relay_cpu.py` PASS (29): una corrida con etiqueta nueva empieza de cero y deja los checkpoints de la otra.
  - `test_shards_cpu.py` PASS (53).
  - La celda de la sonda corrió completa contra el almacén local del test del relevo.
- **Ojo:**
  - **El `COMMIT` del `tronco2` es el de este patch,** en todos sus turnos.
  - **No borrar el 3131 ni el 3270** de Hugging Face antes de que corra la sonda. Los respaldos de Kaggle del primer tronco se reemplazan con los del `tronco2`.
  - **Qué mirar en el turno 1 del `tronco2`:**
    - `z_router` (ahora solo se registra) no debería bajar como en el primer tronco;
    - el `λ/gap` del log, por debajo de ~2;
    - el experto más cargado, en ~1.5× la media.
  - En el primer tronco, la ganancia qk subió a 2.25 en 3,000 pasos. No cambia nada todavía, pero hay que vigilarla.
- **Siguiente:** aplicar 0013, la sonda del 3270 en cualquier cuenta, y el turno 1 del `tronco2`.

## 2026-11-25 · Patch 0013: el tronco empieza de nuevo como tronco2; las pruebas y su COMMIT

- **Qué:**
  - Se aplicó el patch 0013 (`35b1816`). El tronco vuelve a empezar desde el paso 0 como `tronco2` (P29–P31), con el router corregido.
  - Encima va un cambio en el relevo (`a9286c4`): cuando un respaldo falla, el error muestra lo que dijo `kaggle` y no sus barras de progreso. Pipe lo aprobó.
- **Dónde:** el PC, en CPU.
- **Pruebas con el 0013:**
  - pyflakes y los notebooks en sincronía;
  - la prueba lógica de `bench_afmoe.py`, con el checkpoint exacto;
  - `test_train_cpu.py` (41), `test_relay_cpu.py` (29) y `test_shards_cpu.py` (53), todas PASS;
  - `scripts/toy_router_loop.py` reproduce byte a byte `results/tronco/toy_router_loop.txt`.
- **Pruebas con el cambio del respaldo:** `test_relay_cpu.py` (30, con un respaldo rechazado cuyo error sí se ve), `test_train_cpu.py` (41) y la prueba lógica de `bench_afmoe.py`.
- **La evidencia del P29, comprobada con los `.jsonl` del repo.** El criterio es la carga máx/media por encima de 3 en el 80% de 50 pasos, desde el paso 150.
  - Con λ 0.01, se rompieron las corridas cuyo `z_router` bajó de ~1.3: el ×2 de 400 pasos en el paso 266 (1.31), el ×1 de 1,200 pasos en el 618 (0.77) y el tronco en el 1335 (0.79).
  - El ×0.5 y el ×1 de 400 pasos no se rompieron. Su `z_router` no bajó de 3.31 y 1.46.
  - Con λ 0.04 (×4), se rompió en el paso 150, con 2.18.
  - Depende del nivel del router frente a λ, no del paso ni del LR.
- **El `COMMIT` del tronco2 es `a9286c4`**, el mismo en todos sus turnos.
  - El notebook ya lo trae y usa `configs/tronco_v2.json`.
  - La T1 sigue con `75cd3a0`.
  - Los checkpoints 3131 y 3270 del primer tronco quedan en Hugging Face, bajo su etiqueta.
- **Siguiente:**
  - antes, revisar el espacio de Kaggle y que cada cuenta del relevo tenga permiso de edición en los dos respaldos;
  - la sonda sobre el checkpoint 3270 (P31);
  - el turno 1 del tronco2.

## 2026-11-25 · La sonda del checkpoint 3270 (P31): λ/brecha no separa las capas rotas, el rango del sesgo sí

- **Qué:** la sonda del router (`MODO = "sonda"`) sobre el checkpoint 3270 del primer tronco, con `a9286c4`.
- **Dónde:** Kaggle, 2×T4, 4 min.
- **Archivo:** `results/tronco/sonda-tronco-3270_salida_kaggle.txt`, con el usuario de Kaggle enmascarado. Falta su `.jsonl`.
- **Resultado:**
  - El checkpoint bajó en 52 s y cargó exacto. La validación da 0.8812 bits por byte.
  - **λ/brecha es bajo en todas las capas:** 0.6–1.6 en las capas MoE 0–4 y 0.5–2.5 en las otras 20. La brecha mediana entre el score del 4.º y el del 5.º experto tampoco es menor en las capas rotas: 0.0096, contra 0.011.
  - **Lo que separa a las capas 0–4:**
    - el rango del sesgo: 0.23–0.42, contra 0.03–0.14 en las demás. Es 22–60 veces la brecha, contra 4–17;
    - el sesgo cambia los expertos del 92–100% de los tokens, contra el 37–66%;
    - el experto más cargado recibe 4.5–7.6 veces la media, contra 1.4–1.85.
- **Conclusión:**
  - El P31 decía que el P29 queda confirmado si λ/brecha sale alto (≳5) en las capas 0–4 y bajo en las demás. No sale.
  - Las capas 0–4 sí están dominadas por el sesgo, pero por su tamaño acumulado, no por el paso λ frente a la brecha.
  - Es el estado del paso 3270, ~2,000 pasos después del quiebre; el comienzo pudo ser distinto.
- **Para el chat de diseño** (el P31 dice que, si la sonda contradice al P29, se decide de nuevo antes del paso ~1,000 del tronco2):
  - ¿Alcanza el arreglo del tronco2? El `seq_aux` 1e-3 actúa sobre el router mismo, que es lo que haría falta si el sesgo está compensando una preferencia del router por ciertos expertos, igual para todos los tokens.
  - La alarma de λ/brecha no habría avisado en este estado; la parada por balanceo (3×) sí. Las evaluaciones del tronco2 registran por capa el rango del sesgo y la brecha, así que su cociente se puede vigilar.
  - Los checkpoints 1608 (`ludwig2-respaldo-a2`) y 2151 (`ludwig2-respaldo-b`) están más cerca del comienzo del quiebre, pero el tronco2 los reemplaza en sus dos primeros respaldos. Para sondearlos, la sonda tendría que leerlos desde Kaggle.
- **Siguiente:** el turno 1 del tronco2, según el plan. Sus evaluaciones muestran el sesgo por capa cada 250 pasos. La decisión del chat va antes del paso ~1,000.

## 2026-11-25 · El sesgo no causó el quiebre: era más chico justo cuando empezó

- **Qué:** el tamaño del sesgo de SMEBU (`bias_abs_max`, el máximo sobre todas las capas, registrado en cada paso) y `z_router`, por tramos del primer tronco. Sale de `results/tronco/turno01` a `turno07_cuenta-1.jsonl`.
- **Dónde:** el PC.
- **Resultado** (medias por tramo):

  | Pasos | Sesgo (máx) | `z_router` | Carga máx/media |
  |---|---|---|---|
  | 0–550 (sano) | 0.317 (0.416) | 10.24 | 1.50 |
  | 550–1100 (sano) | 0.226 (0.350) | 2.50 | 1.51 |
  | 1150–1300 (el quiebre) | 0.146 (0.179) | 0.88 | 1.85 |
  | 1300–2000 | 0.222 (0.443) | 0.71 | 4.63 |
  | 2000–3348 | 0.392 (0.563) | 0.86 | 7.02 |

- **Conclusión:**
  - Con un sesgo de 0.3–0.4, los expertos estaban balanceados mientras `z_router` estaba alto (turno 1). El quiebre empezó con el sesgo en su punto más chico y `z_router` en ~0.88.
  - El tamaño del sesgo no lo causó: creció después del quiebre, persiguiendo el desbalance. Es lo que la sonda vio en el paso 3270, donde el rango del sesgo separa a las capas rotas.
  - Lo que cambió en el quiebre fue el nivel del router, como dice el P29. El test del P31 (λ/brecha en el 3270) medía el estado final, no la causa.
  - `z_router` bajó de 1.5 hacia el paso ~1000, unos 250 pasos antes del quiebre. En todas las corridas fue la señal previa: con λ 0.01, ninguna que se quedó por encima de ~1.5 se rompió.
- **Siguiente:** proponer al chat de diseño mostrar `z_router` en cada línea del turno, con una alarma si baja de 1.5, antes del turno 1 del tronco2.

## 2026-11-25 · Vigilar el nivel del router: el retoque aprobado, y el COMMIT del tronco2 pasa a 048efa5

- **Qué:** el chat de diseño aprobó vigilar el nivel del router en vez de λ/brecha, con un agregado. La receta del tronco2 no cambia. Commit `048efa5`.
  - **Cada paso:** la línea `[train]` muestra `z_router`. Con `--z-router-alarm` (1.5 en `configs/tronco_v2.json`), salta una alarma si se queda debajo durante 50 pasos. El primer tronco lo cruzó ~250 pasos antes de su quiebre.
  - **Cada evaluación** registra por capa MoE el nivel como `z` y como `lse` con signo. Un promedio de 25 capas puede esconder 5 que se hunden, y `z`, que es un cuadrado, volvería a subir si el router cayera por debajo de 0.
  - **Aviso por capa:** cada evaluación marca la capa donde el sesgo decide más del 80% de los tokens, el rango del sesgo pasa de 20× la brecha o el experto más cargado pasa de 2×. Son los umbrales de la sonda: las capas sanas llegaron a 66%, 17× y 1.85×, y las rotas empezaron en 92%, 22× y 4.49×.
  - λ/brecha se sigue registrando, pero no es una señal de salud.
- **`DECISIONS.md`:**
  - P29: se corrige el mecanismo. Los routers de las capas rotas tenían preferencias comunes a todos los tokens, y el sesgo necesario se achicó con el nivel del router. Lleva la tabla del sesgo, con λ en el 6.9% del sesgo justo antes del quiebre del tronco y en el 5.2% en `lr-x1-1200`, contra ~3% en las corridas ×2.
  - P30: la vigilancia pasa a ser el nivel del router.
  - P31: ❌, con los números de la sonda.
- **Notebook:** el `COMMIT` del tronco2 pasa a `048efa5`, y las instrucciones del turno dicen qué mirar mientras corre.
- **Pruebas en el PC:** pyflakes, notebooks en sincronía, la prueba lógica de `bench_afmoe.py` (checkpoint exacto), `test_train_cpu.py` (43, con la alarma de `z_router` y el nivel por capa) y `test_relay_cpu.py` (30).
- **Siguiente:** el turno 1 del tronco2. En sus evaluaciones (pasos 250 y 500) se mira sobre todo las capas MoE 0–4. Si una capa se acerca a los umbrales, o `z_router` baja de forma sostenida, se avisa al chat de diseño.

## 2026-11-25 · Pregunta abierta: qué hace la literatura con el router sigmoide y las primeras capas MoE

- **Qué:** una revisión de papers durante el turno 1 del tronco2, porque `z_router` baja de forma sostenida, aunque más lento que en el primer tronco. La pregunta: cómo mantienen otros el nivel del router, y si el tronco2 hace lo suficiente.
- **Dónde:** el PC (lectura de papers; no corrió nada).
- **Lo que dicen:**
  - **ST-MoE** ([2202.08906](https://arxiv.org/abs/2202.08906)) hizo el router z-loss para routers softmax, para que los logits no crezcan. Ninguno de los papers leídos lo usa para frenar una caída del nivel.
  - **DeepSeek-V3** ([2412.19437](https://arxiv.org/abs/2412.19437)): sigmoide normalizada entre los expertos elegidos, el sesgo solo para elegir, γ 0.001 (0 al final), pérdida por secuencia α 1e-4, sin router z-loss, 3 capas densas de 61.
  - **Trinity Large** ([2602.17004](https://arxiv.org/abs/2602.17004), de donde salen AFMoE y SMEBU): sus primeras corridas fallaron como el primer tronco ("routing behavior drifted and expert load became increasingly uneven, eventually resulting in collapsed experts"; MaxVio estable y luego una subida de golpe). Lo arreglaron con seis cambios a la vez, sin ablaciones: SMEBU (λ 5e-4, β 0.5, κ 2), BF16 en vez de MXFP8, z-loss 1e-6 (no dicen si del router o de la salida; por un bug entrenaron con 0, y valores mayores desestabilizaban), pérdida por secuencia 1e-4, capas densas de 3 a 6 (de 60) y máscara dentro de cada documento.
  - **Sigma-MoE-Tiny** ([2512.16248](https://arxiv.org/abs/2512.16248), 1 experto de 96): con el sesgo de DeepSeek, el sesgo de las capas bajas crece hasta dominar los scores y un experto se lleva casi todos los tokens. Lo arreglaron activando más expertos en las 8 primeras capas (8, 8, 6, 6, 4, 4, 2, 2) y con una pérdida de balanceo global de 1e-3.
- **Comparación con el tronco2** (`configs/tronco_v2.json`, `configs/ludwig2_afmoe.json`):

  | | DeepSeek-V3 | Trinity Large | Tronco2 |
  |---|---|---|---|
  | Router z-loss | no | 0 por el bug; luego 1e-6 | 0 |
  | Paso del sesgo | 0.001 | 5e-4 | 0.01 hasta el paso 500, luego 0.001 |
  | Pérdida por secuencia | 1e-4 | 1e-4 | 1e-3 |
  | Capas densas | 3 de 61 | 6 de 60 | 1 de 26 |

- **Pregunta abierta para el chat de diseño:** en Trinity, en Sigma-MoE y en el primer tronco (capas MoE 0–4), el punto débil son las primeras capas MoE. Si el tronco2 se rompe y hay que reiniciar, ¿se agregan capas densas o se activan más expertos en las primeras capas? Eso solo cambia con un reinicio. El paso del sesgo, la pérdida por secuencia y un piso para el nivel del router sí pueden cambiar entre turnos.
- **Siguiente:** las evaluaciones de los pasos 250 y 500 del turno 1.

## 2026-11-25 · Tres capas densas: el tronco vuelve a empezar como tronco3 (patch 0014)

- **Qué:** la respuesta del chat de diseño a la pregunta abierta de `f515922`, que era cómo evitar que los routers vuelvan a colapsar (P32–P33).
- **Dónde:** chat de diseño, en CPU. Leí las fuentes: el informe de Trinity Large (§6 y tabla 2) y el de Sigma-MoE-Tiny (§2.2.2 y §3.5.2).
- **Lo que confirman las fuentes:**
  - **Trinity Large:** su enrutado derivó hasta colapsar los expertos. Lo arreglaron con seis cambios a la vez y sin ablaciones; uno fue pasar de 3 a 6 capas densas (de 60). Del z-loss de 1e-6 que metieron a mitad de corrida no dice si era del router o de la salida, pero lo calculan con el kernel fusionado de la entropía cruzada (§3.3) y hablan del logit máximo, lo que apunta a la salida.
  - **Sigma-MoE-Tiny:** con el sesgo de DeepSeek, el sesgo de las capas bajas crece hasta que los scores del router dejan de importar, y el experto con más sesgo se lleva casi todos los tokens: lo mismo que la sonda vio en las capas MoE 0–4 del primer tronco. También vieron que la pérdida de balanceo, en esas capas, se satisface volviendo uniformes las probabilidades sin balancear los tokens. Es una razón para vigilar que el `seq_aux` 1e-3 no aplane los scores de las primeras capas.
- **Lo que vimos en el tronco2:** sano en el paso 220 (carga 1.4–1.6), pero el nivel del router baja aun sin z-loss: `z_router` 13.31 en el paso 220, contra 11.01 del primer tronco, ~45% del ritmo. El z-loss aceleraba una caída que no causaba él solo. Lo que se ve encaja con que la entropía cruzada afila la compuerta bajando el nivel, pero eso no está medido.
- **La decisión (P32):** el tronco vuelve a empezar como `tronco3`, con 3 capas densas en vez de 1 (`configs/ludwig2_afmoe_d3.json`).
  - 1.793B parámetros totales, contra 1.906B; los mismos 0.491B activos, porque el FFN denso (4,096) es tan ancho como los 4 expertos compartidos y los 4 elegidos de 512.
  - El MoE empieza en la capa 3, después de las dos capas MoE que peor se rompieron.
  - El resto de la receta es la del `tronco2`.
  - Es el momento más barato, porque solo se pierde el primer turno del `tronco2`.
- **El piso (P33):** `--router-floor` y `--router-floor-lse`, una pena (τ − lse medio)² por capa por debajo de τ = 1.2. Está apagado en `tronco_v3.json`; encenderlo es un cambio de config y un commit nuevo.
- **Archivos:**
  - `ludwig_lm/afmoe.py` y `scripts/train.py`: el piso, y su término en cada paso.
  - `configs/ludwig2_afmoe_d3.json` y `configs/tronco_v3.json`.
  - El notebook del tronco: `tronco3`, y la sonda con la config de cada etiqueta.
  - `scripts/test_train_cpu.py`: los dos modelos con sus capas y sus conteos, y la corrida F con el piso encendido.
  - `configs/tiny_afmoe_d3.json` y `results/t0_1c_afmoe_d3.json`: la prueba del formato con 3 capas densas.
- **Resultado:**
  - `test_train_cpu.py` PASS: los dos modelos dan 1.906B y 1.793B; sin piso, su término es 0; con τ encima del nivel, el lse medio en el paso 8 sube de 2.077 a 2.665.
  - `test_relay_cpu.py` y `test_shards_cpu.py` PASS.
  - `smoke_format_afmoe.py --config configs/tiny_afmoe_d3.json`, contra llama.cpp `dd6b945` sin cambios: C1–C5 PASS, probabilidades a 4.0e-6, greedy 4/4. El formato sigue sirviendo con 3 capas densas.
- **Ojo:**
  - **El turno 1 del `tronco2` termina normal.** Sus evaluaciones de los pasos 250 y 500 muestran cómo van sus primeras capas con 1 capa densa.
  - **El `COMMIT` del `tronco3` es el de este patch,** en todos sus turnos.
  - **Hugging Face guarda 2 checkpoints por etiqueta.** Para no acercarse a los 100 GB:
    - borrar el 3131 del `tronco` (el 3270 queda para el registro);
    - del `tronco2`, borrar el más viejo cuando termine su turno 1, y el otro cuando el `tronco3` pase el paso ~1,500 sano.
  - **En cada evaluación del `tronco3`:**
    - que el lse de sus primeras capas MoE no baje más rápido que el de las demás;
    - que el sesgo no decida más del ~80% de sus tokens.
- **Siguiente:** aplicar 0014, que no toca el turno que corre porque ese usa su propio `COMMIT`. Cuando el turno 1 del `tronco2` suelte el candado, el turno 1 del `tronco3`.

## 2026-11-25 · El 0014 aplicado con 44 expertos, la condición de Pipe

- **Qué:** se aplicó el patch 0014 (P32–P33). Pipe aprobó reiniciar con una condición: que el modelo no tenga menos parámetros totales que el `tronco2` (1.906B). Con 3 capas densas y 40 expertos quedaba en 1.793B. Con 44 expertos por capa MoE queda en **1.938B**, con los mismos 0.491B activos (P32).
- **Dónde:** el PC.
- **Las opciones,** contadas con `LudwigAfmoe` en `meta`, todas con 3 capas densas salvo donde se dice:

  | Opción | Totales | Activos |
  |---|---|---|
  | 40 expertos (el 0014) | 1.793B | 0.491B |
  | 43 expertos | 1.902B | 0.491B |
  | **44 expertos** | **1.938B** | 0.491B |
  | 46 expertos | 2.010B | 0.491B |
  | 2 capas densas y 42 expertos | 1.925B | 0.491B |
  | 40 expertos y 28 capas | 1.939B | 0.523B |

- **Por qué 44:**
  - Es el número más chico que pasa de 1.906B.
  - El cálculo por token es el mismo: cada token sigue pasando por 4 expertos. En el banco v1, los matmul de expertos rindieron 27.0 TFLOPS con 40 y 26.7 con 60 (`results/banco_t4_v1/reporte.txt`).
  - La memoria sube ~0.15 GiB por GPU (estimado con 5.26 bytes por parámetro y por GPU): ~13.1 GiB reservados a 2K, donde el primer tronco llegó a 12.91, y ~13.5 a 16K, donde la G0-1 midió 13.40 (`results/g0/g0_1_salida_kaggle.txt`). 13.5 es el límite de la G0. Con 46 expertos, 16K iría a ~13.9.
- **Pruebas en el PC:**
  - pyflakes y notebooks en sincronía;
  - `test_train_cpu.py`: PASS, 47 checks, con los conteos de 1.906B y 1.938B y el piso, que sube el lse medio de 2.077 a 2.665;
  - `test_relay_cpu.py`: PASS, 30 checks, con una etiqueta nueva que empieza de cero;
  - la prueba lógica de `bench_afmoe.py`, con `tiny_afmoe.json` y `tiny_afmoe_d3.json`: checkpoint exacto.
- **Además:**
  - Las filas "Forma" y "FFN" de los README todavía decían 1 capa densa y 25 MoE.
  - **El turno 1 del `tronco2` no llega al paso 500.** A ~80 s por paso, sus 11 h terminan cerca del paso 490, y al final de un turno no se evalúa. Su control son las evaluaciones de los pasos 0 y 250 y las líneas `[train]`.
  - **En Hugging Face no se toca nada mientras corre un turno.** Un commit externo, como borrar un archivo desde la web, hace fallar la renovación del candado (`parent_commit`): el relevo lo toma como candado perdido y detiene el entrenamiento sin respaldo final.
  - **El espacio en Hugging Face:** cada checkpoint pesa ~11.5 GB, y un turno sube 4 (cada 3 h y al final), que quedan en el historial hasta el aplastado del final del turno. El turno 1 del `tronco3` empieza con 4 (2 del `tronco` y 2 del `tronco2`) y llega a ~91 GB. Lo que se borra entre turnos sigue ocupando espacio hasta el aplastado del turno siguiente. Por eso el 3131 del `tronco` y el más viejo del `tronco2` se borran antes del turno 1 del `tronco3`: si no, su turno 2 empezaría con 6 checkpoints y pasaría de 100 GB.
- **El `COMMIT` del `tronco3`** es el de este commit, en todos sus turnos.
- **Siguiente:** cuando el turno 1 del `tronco2` suelte el candado, guardar sus archivos, borrar en Hugging Face el 3131 del `tronco` y el más viejo del `tronco2`, y lanzar el turno 1 del `tronco3`.

## 2026-11-26 · El piso del router se enciende en el tronco3 (patch 0015)

- **Qué:** la respuesta del chat de diseño al turno 1 del `tronco3` y a su pregunta: el piso del P33 se enciende antes de que haga falta (P34).
- **Dónde:** chat de diseño, en CPU.
- **Lo que mostró el turno 1,** según el relevo:
  - 498 pasos y 130.5M tokens en 10.96 h, a 3,330 tokens/s, con pico de 13.11 GiB y 0 saltos de FP16;
  - val bpb 2.3009 en el paso 250, contra 2.3015 del `tronco2`;
  - carga de 1.47–1.79× en los últimos 80 pasos.
- **El nivel del router sigue bajando:**
  - `z_router` 15.84 en el paso 0, 15.41 en el 120, 11.71 en el 420 y 10.78 en el 480;
  - en lse, −0.70 desde el paso 0, contra −1.36 del primer tronco en el mismo paso;
  - en los últimos 60 pasos, ~0.05 cada 20 pasos, cerca del ~0.06 del primer tronco entre los pasos 250 y 650.
  - El LR sigue en calentamiento (×0.24 en el paso 480, ×0.50 en el ~990). Si la caída crece con él, la media llega a lse ~1.5 (z ~2.2) al final del turno 2 y a ~1.2 hacia el paso 1,050.
- **La decisión (P34):** el piso se enciende con τ = 1.5 y peso 0.01, desde el primer turno que corra este commit.
  - Vale cero mientras el lse medio de cada capa MoE esté sobre τ: la corrida F0 da la misma pérdida, la misma norma del gradiente y los mismos niveles del router que sin piso.
  - τ = 1.5 y no 1.2: con el piso, cada capa queda en z ≥ ~2.1. Eso está sobre todos los quiebres de la P29, menos el de `lr-x4` (2.2, con un λ 40 veces el del tronco después del paso 500). Y la alarma de `z_router` (1.5) queda como aviso de que el piso falló.
  - Peso 0.01: a 0.05 bajo τ, empuja hacia arriba más o menos lo que el z-loss del primer tronco empujaba hacia abajo al principio.
- **Archivos:**
  - `configs/tronco_v3.json`: `router_floor` 0.01 y `router_floor_lse` 1.5;
  - el notebook del tronco: qué mirar con el piso encendido;
  - `scripts/test_train_cpu.py`: la corrida F0.
- **Resultado:** `test_train_cpu.py` PASS, 48 checks, con F0; `test_relay_cpu.py` y `test_shards_cpu.py` PASS.
- **Ojo:**
  - **Si el turno 2 todavía no arrancó,** arranca con este commit. El 272 del `tronco2` se borra antes, entre turnos.
  - **Si ya arrancó,** en Hugging Face no se toca nada mientras corre, y el piso entra en el turno 3. La excepción: que la evaluación del paso 498 muestre una capa con lse por debajo de 2.8, porque a ese ritmo llegaría a ~1.0 antes del final del turno. Entonces se corta el turno apenas termina esa evaluación, se suelta el candado a mano y se relanza con este commit.
  - **Desde ahora, en cada evaluación:**
    - las capas que llegan al piso deben quedar a menos de 0.1 bajo τ; si una queda más abajo, el peso sube ×3;
    - si una capa sale marcada en dos evaluaciones seguidas (el sesgo decide más del 80% de sus tokens, o su experto más cargado pasa de 2× la media), λ después del paso 500 baja de 0.001 a 5e-4, el de Trinity Large.
  - En cada paso, `router_floor` vale 0 hasta que una capa llega al piso.
- **Siguiente:** el turno 2 del `tronco3` con este commit, o el 3 según lo de arriba.

## 2026-11-25 · tronco2: el turno 1 se cancela, y Hugging Face se limpia

- **Qué:** Pipe canceló el turno 1 del `tronco2` cuando se decidió el `tronco3` (P32), para no gastar más GPU. Después, una celda en un notebook de Kaggle limpió Hugging Face entre turnos.
- **Dónde:** Kaggle (el turno, 2×T4; la limpieza, CPU).
- **El turno cancelado:** su última subida a Hugging Face fue el checkpoint 272. Cada subida lleva también el `.jsonl` del entrenamiento hasta ese momento, así que el log del turno está en `logs/20261126-cuenta-1-d73cbd62.jsonl` de Hugging Face, con la evaluación del paso 250. Todavía no está en `results/`.
- **La limpieza:** la celda revisa el candado, borra los checkpoints y pone el índice al día en un solo commit, y aplasta el historial.
  - Primero corrió en seco, para mostrar qué haría.
  - El candado seguía tomado por la sesión cancelada, así que se soltó a mano en el mismo commit.
  - Se borraron el 3131 del `tronco` y el 136 del `tronco2`; quedaron el 3270 y el 272.
  - Con el historial aplastado, ese espacio se liberó en el momento.
- **Siguiente:** bajar el log del `tronco2` a `results/tronco/tronco2/`.

## 2026-11-26 · tronco3: turno 1, pasos 0–498

- **Qué:** el primer turno del `tronco3` (P32), desde el paso 0.
- **Dónde:** Kaggle 2×T4, cuenta 1, código `020ba9c`, shards `ludwig2-shards-1` (build `386755d6`), respaldos `ludwig2-respaldo-a2` y `-b`.
- **Archivos:** `results/tronco/tronco3/turno01_cuenta-1.jsonl` (una fila por evento) y `.relevo.json` (los eventos del relevo). Llegaron con CRLF y sin el último salto de línea, y se restauraron; cada fila se re-serializa byte por byte. El usuario de Kaggle del error del respaldo queda como `<usuario>`.
- **Resultado:**
  - 498 pasos y 130.5M tokens en 10.96 h, a 3,330 tokens/s en la última hora, con pico de 13.11 GiB reservados y 0 saltos de FP16. Cuatro subidas a Hugging Face: 137, 274, 412 y 498.
  - Val bpb 4.0500 en el paso 0 y 2.3009 en el 250.
  - **El router en el paso 250, por capa MoE:**
    - lse de 3.59 a 3.85, el más bajo en la capa MoE 7. Las capas MoE 0–2, que son las 3–5 del modelo, están en 3.77, 3.68 y 3.67;
    - el sesgo decide el 96–99% de los tokens en todas las capas, y su rango es 21–32× la brecha;
    - el experto más cargado va de 1.16 a 1.56× (capas MoE 0–4: 1.20–1.27×).
    - El aviso por capa marcó las 23, por los dos primeros criterios.
  - **`z_router`:** 15.84 en el paso 0, 15.41 en el 120, 13.86 en el 240, 11.94 en el 400 y 10.50 en el 497.
  - **El sesgo más grande:** 0.30 en el paso 240, 0.41 en el 400 y 0.54 en el 497. El primer tronco tenía 0.39 en el 400 y 0.40 en el 500 (`results/tronco/turno01_cuenta-1.jsonl`).
  - **La carga máx/media por paso**, entre los pasos 400 y 497: de 1.36 a 2.19, con media 1.54.
  - **El respaldo falló.** `kaggle datasets version` subió los archivos, y Kaggle respondió 403 Forbidden al crear la versión nueva de `ludwig2-respaldo-a2`. Los respaldos de los turnos 6 y 7 del primer tronco ya habían fallado al 99% en ese mismo dataset, con la causa cortada; probablemente es lo mismo. Hugging Face tiene el 412 y el 498, y los dos respaldos siguen con el 1608 y el 2151 del primer tronco.
  - El candado se soltó y el historial se aplastó. Las condiciones 1–4 de la G0 se cumplieron.
- **Conclusión:** el turno está sano. El nivel del router sigue bajando, a ~la mitad del ritmo del primer tronco, y el chat de diseño encendió el piso desde el turno 2 (P34, patch 0015).
- **Pregunta abierta para el chat de diseño:** el aviso por capa marca las 23 capas desde el paso 250 por el criterio del 80% (el sesgo decide el 96–99%), con el experto más cargado bajo 1.6×. La regla de la P34 que baja λ a 5e-4 tras dos evaluaciones marcadas se cumpliría en la del paso 498 sin que ninguna capa esté desbalanceada.
- **Siguiente:** borrar el 272 del `tronco2` en Hugging Face y lanzar el turno 2 con el `COMMIT` `e074712`. Revisar por qué Kaggle rechaza el respaldo.

## 2026-11-26 · tronco3: turno 2, pasos 498–944, detenido por balanceo roto

- **Qué:** el segundo turno del `tronco3`, con el piso del router encendido (P34).
- **Dónde:** Kaggle 2×T4, cuenta 1, código `e074712`. Siguió desde el 498 de Hugging Face, con el estado y el primer batch idénticos.
- **Archivos:** `results/tronco/tronco3/turno02_cuenta-1.jsonl` y `.relevo.json`. Se restauraron de CRLF, y el usuario de Kaggle del error del respaldo queda como `<usuario>`.
- **Resultado:**
  - 447 pasos en 9.26 h, a 3,554 tokens/s de media, con pico de 13.13 GiB y 0 saltos de FP16. Un pico de pérdida saltado, en el paso 731.
  - λ pasó de 0.01 a 0.001 en el paso 500.
  - **Desde el paso 500 subió el experto más cargado por paso.** Su media en tramos de 50 pasos fue 1.87 (pasos 498–547), 2.16, 2.37, 2.66, 2.54, 2.62, 2.84, 3.08 y 3.33 (898–944). La del menos cargado bajó de 0.42 a 0.15.
  - **Vista en una ventana de pasos, la carga parece pareja.** En la evaluación del paso 750, la ventana da de 1.17 a 1.38× por capa, y el batch de validación, hasta 2.61× (capa MoE 0). Los expertos que acaparan se turnan, como en el primer tronco (P29).
  - **El router en el paso 750:** lse de 1.84 a 3.02 por capa, con el más alto en la MoE 0 y el más bajo en la MoE 16. El sesgo decide el 94–100% de los tokens.
  - **El sesgo más grande siguió creciendo:** 0.54 en el paso 500, 0.62 en el 750 y 0.66 en el 944.
  - **El piso empezó a actuar hacia el paso 850** (`router_floor` mayor que 0).
  - Val bpb 1.7911 en el paso 498, 1.7878 en el 500 y 1.3715 en el 750. 0 expertos muertos.
  - **La parada:** en el paso 944, el experto más cargado pasaba de 3× en 40 de los últimos 50 pasos (alarma `expert_balance_broke`). El turno guardó el 945 y se detuvo.
  - **Hugging Face se llenó.** Subieron el 644 y el 790. Las subidas siguientes fallaron 6 veces: "Private storage limit reached for org ludwig-v2-lm". El respaldo de Kaggle volvió a fallar con 403, así que el 945 se perdió. En Hugging Face quedan el 644 y el 790 del `tronco3`.
  - El candado se soltó al primer reintento.
- **Pregunta abierta:** el balanceo se rompió con dos recetas distintas.
  - El primer tronco, con λ 0.01, se rompió cuando bajó el nivel del router.
  - El `tronco3`, con λ 0.001, se rompió con el nivel todavía alto.
  - En los dos, el sesgo decide casi todos los tokens de todas las capas desde los primeros cientos de pasos. Pipe pide una revisión independiente del código y de los datos.
- **Siguiente:** no se lanza el turno 3 hasta tener el diagnóstico. Antes hay que liberar espacio en Hugging Face y arreglar el respaldo de Kaggle.

## 2026-11-27 · Por qué se rompe el balanceo: una revisión independiente de los dos troncos

- **Qué:** la revisión independiente que pidió Pipe después del turno 2 del `tronco3` (la pregunta abierta de la entrada anterior): el código del router y del balanceador, los logs de los dos troncos y de la T2, y un banco de pruebas en CPU con un modelo chico.
- **Dónde:** el PC (CPU de 6 núcleos, torch 2.15.0, transformers 5.22.2), con el código `015d2c0`. Nada corrió en Kaggle.
- **Archivos:** `results/diagnostico_moe/`, con su fila en `results/README.md`:
  - `analisis_logs.py` y `analisis_logs.txt`: cada número que el diagnóstico saca de `results/tronco/` y de `results/t0_3_t2/`;
  - `calib_init.py`, `calib_real.py` y sus `.txt`: qué parte común de los logits del router reproduce los números de las evaluaciones del `tronco3`;
  - `potencia_elegidos.py` y su `.txt`: cuánto pesan los expertos elegidos frente a los compartidos en un bloque MoE del tamaño real;
  - `moe_lab.py`, con `prep_data.py`, `lab_report.py`, `lab_table.py` y `lab_extra.py`: el banco. Entrena un AFMoE de 7M parámetros (ancho 128, 1 capa densa y 5 MoE, 44 expertos, top-4 y 4 compartidos) con el router, las pérdidas, Muon/AdamW y el `ExpertBalancer` del repo, sobre texto local, y mide el router por experto;
  - `runs/`, un `.jsonl` por corrida del banco, y `banco_resumen.txt`, sus tablas;
  - `referencia_arreglo.py`: el arreglo como se escribió primero, antes de implementarlo.
  - **Rutas enmascaradas:** la fila `config` de cada corrida traía dos rutas locales del PC, la de los datos y la del repo. Se guardan como `<datos>` y `<repo en 015d2c0>`, y es lo único que cambia en esos archivos. El corpus no entra al repo: es texto y código de terceros que estaba en el PC.
- **El código:** no hay bug en el balanceador. Se revisó el signo de la actualización, el conteo entre micro-pasos y entre las dos GPUs, los tokens enmascarados (cuentan para la carga, y está bien: ocupan expertos), el centrado, el momentum, el momento en que se aplica, la precisión (FP32) y la reanudación.
- **Lo que dicen los logs** (`analisis_logs.txt`):
  - **El `tronco3` se rompió con el cambio de λ.** En el paso 500, λ pasó de 0.01 a 0.001. El piso valió cero hasta el paso 813, así que fue lo único que cambió. El experto menos cargado de cada paso bajó de 0.60× la media (pasos 495–504) a 0.46× (505–514).
  - **La carga dejó de ser ruido.** La autocorrelación de `load_max_over_mean` entre un paso y el siguiente pasó de −0.01 (pasos 300–497) a 0.85 (500–699) y 0.93 (700–944). El 0.93 es sin la fila del paso 731, un pico de pérdida saltado que `train.py` registra con carga 0.0; con esa fila da 0.84.
  - **El sesgo iba hacia donde debía, pero lento.** Entre las evaluaciones de los pasos 250 y 498, con λ 0.01, su rango creció en las 23 capas MoE (0.14 en promedio); entre el 498 y el 750 se achicó en las 23 (0.11 en promedio). En la ventana de los pasos 500–749, los expertos extremos de cada capa (1.17–1.38× y 0.64–0.83×) recorrieron 0.08–0.16 según la propia regla de SMEBU, y λ 0.001 da como máximo 0.25.
  - **El primer tronco se rompió por el otro lado.** En los pasos 1200–1229, 27 de 28 cambios seguidos de `load_min_over_mean` invierten el signo; hasta el 1230 inclusive, 27 de 29. En los pasos 600–1149 lo hace el 65%, y pasos independientes dan 67%. `lr-x1-1200` hace lo mismo en los pasos 520–559 (34 de 38). Es un lazo que se pasa de largo en cada paso: λ 0.01 quedó grueso cuando los scores se achicaron.
  - **El nivel baja sin z-loss.** El lse medio del `tronco3` va de 3.98 a 3.23 en el paso 498 y a 2.29 en el 750. Hasta el paso 944 bajó ~2.1, contra ~2.7 del primer tronco, que sí tenía z-loss del router.
- **El mecanismo:**
  - El router es lineal y sin bias: el logit del experto i para el token t es w_i · x_t. La parte de x que es igual para todos los tokens (su media) le da a cada experto un puntaje extra fijo, w_i · x_media: su *offset*. Son las "preferencias comunes a todos los tokens" de la P31, y no se aprenden.
  - El sesgo existe para cancelar las diferencias entre los offsets. Por eso decide el 94–100% de los tokens también en una corrida sana.
  - Los offsets se mueven mientras la red aprende, y se achican cuando baja el nivel. SMEBU los sigue con pasos de tamaño fijo en unidades de score: hace falta un paso grande mientras los scores son grandes y los offsets se mueven, y uno chico cuando los scores son chicos. Ningún λ fijo sirve todo el camino.
  - **En el `tronco3`** (`calib_init.txt`, `calib_real.txt`): la carga sin sesgo del paso 0 (1.83–3.88× por capa) sale con ~5–10% de parte común. La brecha, el rango del sesgo y la fracción de tokens que decide el sesgo salen, en el paso 250, con ~41% de la varianza de los logits, y en el 498, con ~50–64%. Es un ajuste, no una medición.
- **El banco** (`banco_resumen.txt`). Las cargas son la del experto más cargado y la del menos cargado de cada paso, en veces la media, promediadas sobre los pasos 300–500. Su piso es ~1.6× porque sus pasos son de 4,096 tokens.
  - Arranca como el `tronco3`: lse 3.95–4.05, brecha 0.013 y carga inicial 2.6–3.6×.
  - La parte común de la entrada del router pasa de 11% a 68% en 100 pasos y baja a 17% en el paso 500. La dispersión de los offsets va de 0.23 a 0.50 logits.
  - En los primeros 50 pasos, a los offsets los mueve la entrada: 0.0085 logits por paso, contra 0.0008 de las filas del router. Después pesan parecido. El nivel lo bajan las filas del router, y el gradiente de la entropía cruzada sola ya empuja hacia abajo.
  - El sesgo que balancearía el lote se mueve ~0.001 por paso en promedio, y 0.002–0.004 en los expertos más rápidos.
  - SMEBU con λ 0.01: 1.62× / 0.53×. Con λ 0.001 desde el paso 150, la receta del `tronco3`: 2.25× / 0.24×.
  - Con los offsets iguales y ese mismo calendario: 1.66× / 0.52×. Es la prueba causal: sin offsets, el paso lento alcanza.
  - La regla por cuantiles con los offsets iguales: 1.60× / 0.54×. El rango del sesgo baja de 0.38 a 0.13, y el sesgo decide el 53% de los tokens en vez del 94%. Aplicada en el paso 250 de una corrida con SMEBU: 1.59× / 0.55×, con un paso de transición (2.54× / 0.06×) y la pérdida igual.
  - Forzando la caída del nivel (z-loss del router 0.003: el lse va de 3.1 a 0.8 entre los pasos 100 y 300), en esos 200 pasos: SMEBU 0.01 da 1.84× / 0.46×; SMEBU 0.04 se rompe (2.95× / 0.25×, con picos de 4.9×); la regla por cuantiles con los offsets iguales, 1.75× / 0.49×.
  - La pérdida de validación en el paso 400 está entre 3.51 y 3.53 en todas esas corridas.
- **Las hipótesis del pedido:**
  - (a) λ 0.001 es lento: confirmada. Es lo que detuvo al `tronco3`.
  - (b) El router aprende preferencias comunes: las tiene, pero no las aprende. Salen de la parte común de su entrada.
  - (c) `route_scale` 1.0 deja a los elegidos con poco peso: cierto, pero no causa la rotura. Con 4 compartidos y 1.0, los 4 elegidos salen de un bloque con el 6% de la potencia de los compartidos (`potencia_elegidos.txt`); la configuración de DeepSeek-V3 da 78%. En el banco, 4.0 da el mismo balance y una pérdida de validación 0.6% menor en el paso 400 (3.499 contra 3.520). Subirlo en el paso 250 cuesta +0.085 de pérdida durante ~50 pasos si se hace de golpe, y nada visible con una rampa de 100 pasos.
  - (d) La caída del nivel: real, y no la causa el z-loss, que solo la aceleraba.
  - (e) Un bug en el balanceador: descartada.
- **Conclusión:** las dos roturas son el mismo lazo por lados opuestos, y la raíz está en lo que el sesgo tiene que cancelar. El arreglo tiene dos piezas: que el paso del sesgo salga en las unidades de los propios scores (la regla por cuantiles), y que el router no tenga offsets distintos por experto.
- **Lo que no se pudo comprobar:**
  - La parte común del modelo real es un ajuste a sus logs. La mide la sonda del checkpoint 790.
  - El banco es un modelo de 7M parámetros con texto local. No reproduce el colapso total del primer tronco con λ 0.01, solo con 0.04, porque sus scores no se achican tanto. Esa parte se apoya en los logs reales.
  - Las corridas del banco no son reproducibles bit a bit (hilos): dos corridas iguales difieren en ~0.005 de pérdida.
- **Siguiente:** el patch con las dos piezas, sus pruebas y los dos modos del notebook, en la entrada que sigue.

## 2026-11-27 · Patch de la P35: la regla por cuantiles y los offsets iguales; el tronco vuelve a empezar como `tronco4`

- **Qué:** el arreglo que salió de la revisión de la entrada anterior, y la decisión P35.
  - **`--balance quantile`** (`bias_targets` y `ExpertBalancer` en `ludwig_lm/afmoe.py`). En cada paso y capa MoE calcula, para cada experto, el sesgo con el que recibiría justo su parte de una muestra de los tokens del paso, y mueve el sesgo la mitad del camino (`--balance-damp` 0.5). SMEBU y el signo siguen disponibles y dan los mismos números que antes.
    - La muestra son 512 tokens por GPU y micro-paso (`--balance-sample`; 8,192 por paso), con un salto fijo y un desfase que depende del paso y del micro-paso. No usa números al azar, así que una corrida reanudada toma los mismos.
    - Las dos GPUs reciben las dos muestras con un all-reduce y ordenan el mismo tensor: llegan al mismo sesgo bit a bit. Cada paso lo comprueba (`bias_ranks_equal`), con la alarma `expert_bias_differs_between_gpus`.
    - No guarda nada entre pasos: el checkpoint tiene las mismas claves que con SMEBU.
  - **`--router-offsets equal`** (`equalise_offsets`). Después de cada paso del optimizador, las filas del router de cada capa MoE pierden un décimo (`--router-offsets-rate` 0.1) de las diferencias entre los offsets de los expertos, a lo largo de la media de la entrada del router en ese paso. Las filas están repartidas entre las GPUs por FSDP2: cada una edita las suyas, con la media de la entrada y la de las filas contra ella reducidas entre GPUs y micro-pasos. Tampoco guarda estado.
  - **`--bias-reset-step N`:** al final del paso N iguala los offsets del todo y pone el sesgo en cero en vez de actualizarlo. Es para aplicar los offsets iguales a una corrida que ya los tiene, como hace la prueba sobre el 790. Va por número de paso, no "al reanudar": una corrida reanudada sigue igual bit a bit a una sin cortes, y pasado ese paso no hace nada.
  - **Registro nuevo.** Cada paso: `bias_step_max`, el mayor movimiento del sesgo. Cada evaluación, por capa MoE: `router_common` (la parte de la entrada del router común a todos los tokens), `router_offset_std` y `router_specific_std` (cuánto difieren los offsets entre expertos, y cuánto varía un logit entre tokens), `score_mean_min`/`max`, `score_sd_min`/`max` y `bias_score_corr`.
  - **`configs/tronco_v4.json`:** la corrida nueva, `tronco4`. Es el modelo y la receta del `tronco3` con `balance` quantile y `router_offsets` equal desde el paso 0, sin el calendario de λ. `configs/tronco_v3.json` no cambia: queda como el registro del `tronco3`.
  - **El notebook del tronco:** el turno pasa a `tronco4`. La `sonda` apunta al 790 del `tronco3` e imprime la tabla nueva. El modo nuevo `prueba-arreglo` entrena 26 pasos desde ese checkpoint con la receta del `tronco3` más el arreglo y el reinicio del sesgo en el paso 790, sin guardar ni subir nada y sin candado. Evalúa cada 5 pasos, compara cada paso con el mismo paso del turno original e imprime cada condición con ✅ o ❌.
  - **Documentos:** la P35 en `DECISIONS.md`, con notas en la P29, la P30, la P32 y la P34; el recuadro nuevo, la tabla de alarmas y la fila T3 de `DISENO.md`; el estado, la fila del router y la T3 de los dos README.
- **Dónde:** el PC (CPU de 6 núcleos, torch 2.15.0, transformers 5.22.2). Nada corrió en Kaggle.
- **Archivos:** `ludwig_lm/afmoe.py`, `scripts/train.py`, `scripts/test_train_cpu.py`, `scripts/make_notebooks.py` y `notebooks/tronco_relevo_2xT4.ipynb`, `configs/tronco_v4.json`, `docs/DECISIONS.md`, `docs/es/DISENO.md`, los dos README; en `results/diagnostico_moe/`, `banco_patch.txt` y las corridas `P35_from0`, `P35b_from0` y `P35b_midrun_250` de `runs/`.
- **Un cambio sobre la marcha:** la primera versión de los offsets iguales los quitaba completos en cada paso, con la media de ese solo paso. En el banco, con la implementación del repo, eso balanceó peor que SMEBU: en los pasos 100–300, 1.99× / 0.40× contra 1.79× / 0.45× de SMEBU (`P35_from0` en `banco_patch.txt`). La media de un paso es la parte común más lo que es propio del texto de ese paso, y al quitarla entera las filas y el sesgo siguen al texto de cada paso. Con un décimo por paso: 1.73× / 0.49× en esos mismos pasos y 1.62× / 0.54× en los pasos 300–500, como el prototipo del diagnóstico (`P35b_from0`).
- **Pruebas en el PC:**
  - `python -m pyflakes ludwig_lm scripts`: sin errores. Los notebooks están en sincronía con `scripts/make_notebooks.py`.
  - `scripts/test_train_cpu.py`: PASS, 69 comprobaciones. Las nuevas:
    - la regla por cuantiles balancea un lote fijo con scores cerca de 0.5 y cerca de 0.002 con el mismo ajuste (de 3.59× a 1.00×);
    - Q (cuantiles), O (offsets iguales) y X (los dos, con el sesgo en cero en el paso 4) corren con FSDP2 en dos procesos, con el mismo sesgo en los dos en todos los pasos;
    - QR, OR y XR reanudan bit a bit, XR pasando por el paso del reinicio; XS salta ese paso como pico de pérdida, y los offsets y el sesgo se reinician igual;
    - O4: con un cuarto por paso, los offsets quedan entre los de la corrida sin igualar y los de la corrida que los quita completos;
    - un paso con un score o una entrada del router no finitos deja el sesgo y las filas del router como estaban;
    - T hace en CPU lo que la `prueba-arreglo` hace en Kaggle: reanuda un checkpoint guardado con SMEBU y offsets libres, con la regla por cuantiles, los offsets iguales y el reinicio en ese mismo paso, y no guarda nada. El estado carga exacto; el primer paso da la misma pérdida que en la corrida original, y al final de ese paso el sesgo queda en cero y los offsets igualados.
  - **Las corridas viejas no cambian:** las 11 corridas de `test_train_cpu.py` que ya existían (SMEBU, 118 filas) dan, valor por valor, los mismos números antes y después del patch en todo lo que ya se registraba.
  - La prueba lógica de `bench_afmoe.py` con `tiny_afmoe.json` y `tiny_afmoe_d3.json`: checkpoint exacto.
  - El reporte de la `prueba-arreglo` (la tabla por paso y las condiciones) se corrió sobre los `.jsonl` de una corrida reanudada de `test_train_cpu.py`: lee todas las columnas que usa.
  - **El banco con la implementación del repo** (`results/diagnostico_moe/banco_patch.txt` y las corridas `P35b_*`; pasos 300–500): desde el paso 0, 1.62× / 0.54×, con el rango del sesgo en 0.14 (0.38 con SMEBU) y el sesgo decidiendo el 55% de los tokens (94% con SMEBU); encendida en el paso 250 de una corrida con SMEBU, con el reinicio, 1.62× / 0.53×, después de un solo paso en 2.41× / 0.26×, y con la pérdida de entrenamiento a menos de 0.01 de la de la corrida con SMEBU en todos los pasos alrededor del cambio.
  - **No corrieron:**
    - `test_relay_cpu.py` y `test_shards_cpu.py`: el patch no toca `ludwig_lm/relay.py` ni el lector de shards.
    - Los dos modos del notebook, `sonda` y `prueba-arreglo`: necesitan las dos T4, los shards y el checkpoint de Hugging Face. Su primera corrida en Kaggle es también su prueba.
    - `torchrun`: en Windows no arranca con torch 2.15, y los dos procesos de cada prueba se lanzaron a mano, como en las entradas anteriores.
- **Ojo:**
  - **El `COMMIT` del notebook** sigue siendo el del `tronco3`. Hay que poner el hash con el que quede este patch al aplicarlo, antes de correr la sonda: los modos nuevos necesitan este código.
  - **La P35 está escrita para el reinicio como `tronco4`.** Si al final se sigue el `tronco3` desde el 790, el código ya lo soporta: `configs/tronco_v3.json` llevaría `balance` quantile, `balance_damp` 0.5, `router_offsets` equal, `router_offsets_rate` 0.1 y `bias_reset_step` 790, sin `lam_final` ni `lam_steps`, y el notebook volvería a leer esa config. Es lo que hace la `prueba-arreglo` durante 26 pasos.
  - **El espacio de Hugging Face y el respaldo de Kaggle** (turno 2 del `tronco3`) se resuelven aparte. La sonda y la prueba no guardan ni suben nada, así que no dependen de eso; el primer turno del `tronco4` sí.
- **Pregunta abierta para el chat de diseño: `route_scale`.** Con `route_scale` 1.0, los 4 expertos elegidos salen de cada bloque MoE con el 6% de la potencia del experto compartido, que tiene el ancho de 4 expertos (`potencia_elegidos.txt`); con 4.0, con la misma. No causó la rotura, y este patch no lo cambia. Es lo único de la receta que solo un paso 0 cambia gratis: en el banco (`banco_resumen.txt`), 4.0 desde el paso 0 da el mismo balanceo y 3.499 contra 3.520 de pérdida de validación en el paso 400, y subirlo de golpe en el paso 250 cuesta +0.085 de pérdida durante ~50 pasos. Antes de usarlo, la T0.1-C tiene que pasar con un `route_scale` distinto de 1: nunca corrió así.
- **Siguiente:**
  1. Aplicar el patch y fijar el `COMMIT` del notebook.
  2. La `sonda` del 790 (~10 min) y la `prueba-arreglo` (~40 min), cada una en su notebook. Sus `.jsonl` y lo que imprimen van a `results/tronco/`.
  3. Si pasan, el turno 1 del `tronco4`.

## 2026-11-27 · `route_scale`: más corridas del banco, y lo que falta para decidirlo

- **Qué:** la pregunta que la P35 dejó abierta. Cuatro corridas más del banco de `results/diagnostico_moe/`, todas desde el paso 0, y la configuración para probar el formato con un `route_scale` distinto de 1. No cambia el modelo ni la receta del `tronco4`: eso lo decide el chat de diseño.
- **Dónde:** el PC (CPU de 6 núcleos, torch 2.15.0, transformers 5.22.2). Nada corrió en Kaggle.
- **Archivos:**
  - en `results/diagnostico_moe/`, `banco_route_scale.txt` y cuatro corridas en `runs/`: `R25_routescale25` (2.5 con SMEBU), `P35b_from0_rs4` (4.0 con la regla por cuantiles y los offsets iguales, como el `tronco4`), y otra semilla con SMEBU, `A_base_seed1` (1.0) y `R_routescale4_seed1` (4.0). Las rutas locales de su fila `config` van enmascaradas, como en las demás;
  - `configs/tiny_afmoe_d3_rs4.json`: `tiny_afmoe_d3.json` con `route_scale` 4.0, para la T0.1-C.
- **Por qué importa:** los pesos de los 4 expertos elegidos suman `route_scale`. Con 1.0 salen de cada bloque MoE con el 6% de la potencia del experto compartido, que tiene el ancho de 4 expertos; con 2.5, con el 39%; con 4.0, con la misma (`potencia_elegidos.txt`, al inicio). La norma de salida del bloque no lo compensa: escala la suma de los dos.
- **Resultado** (`banco_route_scale.txt`): la pérdida de validación en los pasos 100, 200, 300 y 400.

| Corrida | `route_scale` | Paso 100 | Paso 200 | Paso 300 | Paso 400 |
|---|---|---|---|---|---|
| SMEBU 0.01 (`A_base`) | 1.0 | 7.077 | 5.322 | 4.162 | 3.520 |
| SMEBU 0.01 (`R25_routescale25`) | 2.5 | 7.063 | 5.306 | 4.143 | 3.506 |
| SMEBU 0.01 (`R_routescale4`) | 4.0 | 7.063 | 5.308 | 4.137 | 3.499 |
| El arreglo de la P35 (`P35b_from0`) | 1.0 | 7.076 | 5.321 | 4.150 | 3.515 |
| El arreglo de la P35 (`P35b_from0_rs4`) | 4.0 | 7.063 | 5.304 | 4.132 | 3.496 |
| SMEBU 0.01, semilla 1 (`A_base_seed1`) | 1.0 | 7.094 | 5.327 | 4.139 | 3.544 |
| SMEBU 0.01, semilla 1 (`R_routescale4_seed1`) | 4.0 | 7.082 | 5.313 | 4.112 | 3.527 |

- **Lo que se ve:**
  - En las tres parejas, 4.0 queda por debajo de 1.0 en las cuatro evaluaciones. En el paso 400 la diferencia es de 0.017 a 0.021 (0.5–0.6%).
  - 2.5 queda entre los dos en los pasos 300 y 400.
  - El balanceo no cambia: el experto más cargado y el menos cargado de cada paso dan 1.62–1.63× y 0.53–0.54× la media en los pasos 300–500 con la semilla 0, y 1.66× / 0.50–0.51× con la semilla 1.
- **Cuánto vale ese número:**
  - Dos corridas iguales del banco difieren en ~0.005 de pérdida, así que cada pareja está fuera de ese ruido.
  - Entre las dos semillas, la misma receta difiere en 0.024 (con 1.0) y 0.028 (con 4.0) en el paso 400, más que el efecto: lo que se sostiene es la comparación dentro de cada pareja, que comparte la inicialización y el orden de los datos.
  - Es un modelo de 7M parámetros en 500 pasos. No dice cuánto vale en el modelo real ni a 25B tokens. Sí dice que no se ve un costo, ni en la pérdida ni en el balanceo.
- **Por qué decidirlo ahora:** es lo único de la receta que solo un paso 0 cambia gratis. Subirlo a mitad de una corrida cambia la función de un paso al siguiente: en el banco, de golpe en el paso 250 costó +0.085 de pérdida durante ~50 pasos; con una rampa de 100 pasos no se vio (`banco_resumen.txt`).
- **El formato:**
  - Leído en el código de llama.cpp (rama `master`, 27 de noviembre): `conversion/afmoe.py` escribe `route_scale` en el GGUF como `expert_weights_scale`, y `build_moe_ffn` (`src/llama-graph.cpp`) multiplica por ese valor los pesos de los expertos elegidos después de normalizarlos. Es el mismo orden que usa transformers 5.22.2 (`AfmoeTokenChoiceRouter.forward`) y el router de `ludwig_lm/afmoe.py`.
  - No está probado: la T0.1-C siempre corrió con 1.0. La prueba es `python scripts/smoke_format_afmoe.py --config configs/tiny_afmoe_d3_rs4.json --out results/t0_1c_afmoe_rs4.json`, donde corre la T0.1: necesita llama.cpp compilado, y eso corre en Kaggle.
- **Pruebas en el PC:** `python -m pyflakes ludwig_lm scripts` sin errores. La prueba lógica de `bench_afmoe.py` con `configs/tiny_afmoe_d3_rs4.json` da el checkpoint exacto, y en ese modelo los pesos de los expertos elegidos suman 4.0. El patch no toca `ludwig_lm/` ni `scripts/train.py`.
- **Siguiente:** lo decide el chat de diseño, antes del primer turno del `tronco4`. Si es que sí:
  1. la T0.1-C de arriba, que tiene que pasar;
  2. una copia de `configs/ludwig2_afmoe_d3.json` con `route_scale` 4.0, con su línea en `test_trunk_model`, y `configs/tronco_v4.json` apuntando a ella;
  3. la decisión en `DECISIONS.md`.
