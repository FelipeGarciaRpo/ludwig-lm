# Copia congelada del banco de pruebas v1 en 2×T4, tal como se corrió en Kaggle el 2026-11-11.
# Lo escribió otro chat con un modelo propio en PyTorch (no el de ludwig_lm/).
# Sus resultados están junto a este archivo. No se edita: las versiones nuevas van en otro archivo.

# =============================================================================
# Ludwig 2 — banco de pruebas en 2×T4 (Kaggle) · TODO EN UNA SOLA CELDA
# 1) Settings → Accelerator → GPU T4 x2   2) Settings → Internet → On (solo para la prueba de KDA)
# 3) Pega esta celda completa y córrela, o usa Save Version → Save & Run All para que corra sola.
# 4) Al terminar, mándame /kaggle/working/ludwig2_t4_banco/reporte.txt
# =============================================================================
MODO = "completo"            # "completo" (~1–1.5 h) o "rapido" (~30–40 min)
PROBAR_KDA = True            # instala flash-linear-attention y prueba si sus kernels corren en T4
OBJETIVO_TOKENS = 25e9       # tokens de Ludwig 2
TOKENS_POR_PASO = 524288     # lote del entrenamiento real (~0.5M tokens por paso del optimizador)
HORAS_RELEVO_SEMANA = 140
PICO_T4_TFLOPS = 65.0        # FP16 con tensor cores, por GPU
MEMORIA_SEGURA_GIB = 13.5    # por encima de esto (reservado) un entrenamiento largo queda justo (NCCL y CUDA usan ~1 GiB aparte)

import os as _os
_CODIGO = _os.environ.get("LUDWIG_CODIGO", "/kaggle/working/ludwig2_t4_banco/codigo")
_os.makedirs(_CODIGO, exist_ok=True)
_ARCHIVOS = {
"modelo.py": r'''# Ludwig 2 — modelo MoE para 2×T4 (PyTorch). Lo usa el banco de pruebas y después el entrenamiento.
# Diseño: GQA + QK-norm + compuerta sigmoide a la salida de la atención (Qwen), MoE con router sigmoide
# y bias de balanceo sin pérdida auxiliar (DeepSeek-V3), experto compartido, 1.ª capa densa,
# embeddings atados, Block Attention Residuals opcional (Kimi) y optimizador Muon + AdamW.
import math
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.checkpoint import checkpoint

try:
    from torch.distributed.tensor import DTensor
except ImportError:  # torch < 2.5
    from torch.distributed._tensor import DTensor


@dataclass
class Config:
    vocab: int = 64000
    d: int = 1024
    capas: int = 26
    capas_densas: int = 1
    cabezas: int = 16
    cabezas_kv: int = 4
    dim_cabeza: int = 64
    expertos: int = 40
    top_k: int = 4
    h_experto: int = 512
    unidades_compartido: int = 4      # el experto compartido mide 4 expertos normales
    theta_rope: float = 500000.0
    seq: int = 2048
    compuerta: bool = True
    qk_norm: bool = True
    attnres: bool = False
    attnres_s: int = 6                # subcapas por bloque de AttnRes (atención y MoE cuentan por separado)
    moe_impl: str = "bmm"             # "bmm" (relleno + bmm) o "loop" (un matmul por experto)
    tasa_bias: float = 1e-3
    coef_z: float = 1e-4
    coef_aux: float = 1e-4
    init_std: float = 0.02
    checkpointing: bool = True
    trozo_ce: int = 1024

    @property
    def h_compartido(self):
        return self.unidades_compartido * self.h_experto

    @property
    def h_denso(self):
        return (self.top_k + self.unidades_compartido) * self.h_experto

    def params(self):
        d, H, KV, D, E, h = self.d, self.cabezas, self.cabezas_kv, self.dim_cabeza, self.expertos, self.h_experto
        atn = d * (H + 2 * KV + (H if self.compuerta else 0)) * D + H * D * d + (2 * D if self.qk_norm else 0)
        comun = atn + 2 * d + (4 * d if self.attnres else 0)
        denso = 3 * d * self.h_denso
        experto = 3 * d * h
        compartido = 3 * d * self.h_compartido
        n_moe = self.capas - self.capas_densas
        moe_total = E * experto + compartido + d * E
        moe_activo = self.top_k * experto + compartido + d * E
        emb = self.vocab * d
        cuerpo_total = self.capas * comun + self.capas_densas * denso + n_moe * moe_total
        cuerpo_activo = self.capas * comun + self.capas_densas * denso + n_moe * moe_activo
        extra = 2 * d if self.attnres else 0
        return {"total": cuerpo_total + emb + d + extra, "activo": cuerpo_activo + emb, "activo_sin_emb": cuerpo_activo}

    def flops_token(self, seq=None):
        seq = seq or self.seq
        return 6 * self.params()["activo"] + 3 * seq * self.cabezas * 2 * self.dim_cabeza * self.capas

    def dict(self):
        return asdict(self)


def local(t):
    return t.to_local() if isinstance(t, DTensor) else t


def dtype_autocast(tipo):
    try:
        if torch.is_autocast_enabled(tipo):
            return torch.get_autocast_dtype(tipo)
    except TypeError:
        if tipo == "cuda" and torch.is_autocast_enabled():
            return torch.get_autocast_gpu_dtype()
    return None


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        xf = x.float()
        y = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return (y * self.weight.float()).to(x.dtype)


_ROPE = {}


def rope(T, D, theta, device):
    clave = (T, D, theta, str(device))
    if clave not in _ROPE:
        inv = 1.0 / (theta ** (torch.arange(0, D, 2, device=device, dtype=torch.float32) / D))
        f = torch.outer(torch.arange(T, device=device, dtype=torch.float32), inv)
        _ROPE[clave] = (f.cos(), f.sin())
    return _ROPE[clave]


def aplicar_rope(x, cos, sin):  # x [B, T, H, D]
    x1, x2 = x.float().chunk(2, -1)
    c, s = cos[None, : x.shape[1], None, :], sin[None, : x.shape[1], None, :]
    return torch.cat([x1 * c - x2 * s, x1 * s + x2 * c], -1).to(x.dtype)


class Atencion(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c = c
        H, KV, D = c.cabezas, c.cabezas_kv, c.dim_cabeza
        self.cortes = [H * D, KV * D, KV * D] + ([H * D] if c.compuerta else [])
        self.qkvg = nn.Linear(c.d, sum(self.cortes), bias=False)
        self.o = nn.Linear(H * D, c.d, bias=False)
        if c.qk_norm:
            self.qn = RMSNorm(D)
            self.kn = RMSNorm(D)

    def forward(self, x):
        c = self.c
        B, T, _ = x.shape
        H, KV, D = c.cabezas, c.cabezas_kv, c.dim_cabeza
        partes = self.qkvg(x).split(self.cortes, dim=-1)
        q = partes[0].view(B, T, H, D)
        k = partes[1].view(B, T, KV, D)
        v = partes[2].view(B, T, KV, D)
        if c.qk_norm:
            q, k = self.qn(q), self.kn(k)
        cos, sin = rope(T, D, c.theta_rope, x.device)
        q, k = aplicar_rope(q, cos, sin), aplicar_rope(k, cos, sin)
        if KV != H:  # se expanden las cabezas de KV para usar el kernel eficiente de SDPA en T4
            k = k.repeat_interleave(H // KV, dim=2)
            v = v.repeat_interleave(H // KV, dim=2)
        o = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True)
        o = o.transpose(1, 2)
        if c.compuerta:
            o = o * torch.sigmoid(partes[3].view(B, T, H, D))
        return self.o(o.reshape(B, T, H * D))


class Denso(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.w13 = nn.Linear(c.d, 2 * c.h_denso, bias=False)
        self.w2 = nn.Linear(c.h_denso, c.d, bias=False)

    def forward(self, x):
        a, b = self.w13(x).chunk(2, -1)
        return self.w2(F.silu(a) * b), x.new_zeros((), dtype=torch.float32), None


class Router(nn.Module):
    def __init__(self, d, E):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(E, d))

    def forward(self, x):
        return F.linear(x, self.weight)


def expertos_bmm(x, idx, w, w13, w2):
    # x [N, d], idx/w [N, k]. Ordena las asignaciones por experto, rellena hasta la carga máxima
    # y calcula todos los expertos con dos bmm.
    N, d = x.shape
    E = w13.shape[0]
    k = idx.shape[1]
    plano = idx.reshape(-1)
    orden = torch.argsort(plano, stable=True)
    exp_ord = plano[orden]
    cuentas = torch.bincount(plano, minlength=E)
    C = int(cuentas.max().item())
    C = max(8, ((C + 7) // 8) * 8)
    inicio = torch.cumsum(cuentas, 0) - cuentas
    pos = torch.arange(plano.numel(), device=x.device) - inicio[exp_ord]
    tok = orden // k
    xpad = x.new_zeros(E, C, d)
    xpad[exp_ord, pos] = x[tok]
    a, b = torch.bmm(xpad, w13.to(x.dtype)).chunk(2, -1)
    ypad = torch.bmm(F.silu(a) * b, w2.to(x.dtype))
    y = ypad[exp_ord, pos] * w.reshape(-1)[orden].unsqueeze(-1).to(x.dtype)
    return torch.zeros_like(x).index_add_(0, tok, y)


def expertos_loop(x, idx, w, w13, w2):
    N, d = x.shape
    E = w13.shape[0]
    k = idx.shape[1]
    plano = idx.reshape(-1)
    orden = torch.argsort(plano, stable=True)
    cuentas = torch.bincount(plano, minlength=E).tolist()
    tok = orden // k
    partes = torch.split(x[tok], cuentas)
    ys = []
    for e, xe in enumerate(partes):
        if xe.shape[0] == 0:
            ys.append(xe)
            continue
        a, b = (xe @ w13[e].to(x.dtype)).chunk(2, -1)
        ys.append((F.silu(a) * b) @ w2[e].to(x.dtype))
    y = torch.cat(ys) * w.reshape(-1)[orden].unsqueeze(-1).to(x.dtype)
    return torch.zeros_like(x).index_add_(0, tok, y)


class MoE(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c = c
        E, d, h, hs = c.expertos, c.d, c.h_experto, c.h_compartido
        self.router = Router(d, E)
        self.register_buffer("bias_ruteo", torch.zeros(E))
        self.w13 = nn.Parameter(torch.empty(E, d, 2 * h))
        self.w2 = nn.Parameter(torch.empty(E, h, d))
        self.c13 = nn.Linear(d, 2 * hs, bias=False)
        self.c2 = nn.Linear(hs, d, bias=False)

    def forward(self, x):
        c = self.c
        B, T, d = x.shape
        xt = x.reshape(-1, d)
        tipo = xt.device.type
        with torch.autocast(device_type=tipo, enabled=False):
            logits = self.router(xt.float())                                   # [N, E] en FP32
            s = torch.sigmoid(logits)
            idx = torch.topk(s + self.bias_ruteo.float(), c.top_k, dim=-1).indices  # el bias solo elige
            w = s.gather(-1, idx)
            w = w / w.sum(-1, keepdim=True).clamp_min(1e-20)
            aux = c.coef_z * torch.logsumexp(logits, -1).pow(2).mean()
            if c.coef_aux > 0:  # pérdida complementaria por secuencia (DeepSeek-V3)
                cuentas = torch.zeros(B, c.expertos, device=xt.device).scatter_add_(
                    1, idx.view(B, -1), torch.ones(B, T * c.top_k, device=xt.device))
                f = cuentas * (c.expertos / (c.top_k * T))
                P = (s / s.sum(-1, keepdim=True)).view(B, T, -1).mean(1)
                aux = aux + c.coef_aux * (f * P).sum(-1).mean()
        dt = dtype_autocast(tipo) or xt.dtype
        xh = xt.to(dt)
        fn = expertos_bmm if c.moe_impl == "bmm" else expertos_loop
        y = fn(xh, idx, w, self.w13, self.w2)
        a, b = self.c13(xh).chunk(2, -1)
        y = y + self.c2(F.silu(a) * b).to(y.dtype)
        carga = torch.bincount(idx.reshape(-1), minlength=c.expertos).detach()
        return y.view(B, T, d), aux, carga


def agregar(fuentes, q, norma):
    # Block AttnRes (Kimi, 2026): softmax sobre la profundidad con una pseudo-consulta aprendida por subcapa;
    # las llaves son las fuentes normalizadas con RMSNorm. Con q = 0 empieza como promedio de las fuentes.
    if len(fuentes) == 1:
        return fuentes[0]
    qw = q.float() * norma.weight.float()
    puntajes = []
    for v in fuentes:
        vf = v.float()
        puntajes.append((vf @ qw) * torch.rsqrt(vf.pow(2).mean(-1) + norma.eps))
    a = torch.softmax(torch.stack(puntajes, 0), dim=0)
    h = a[0].unsqueeze(-1) * fuentes[0].float()
    for i in range(1, len(fuentes)):
        h = h + a[i].unsqueeze(-1) * fuentes[i].float()
    return h


class Bloque(nn.Module):
    def __init__(self, c, i):
        super().__init__()
        self.c = c
        self.i = i
        self.es_moe = i >= c.capas_densas
        self.n1 = RMSNorm(c.d)
        self.atn = Atencion(c)
        self.n2 = RMSNorm(c.d)
        self.ffn = MoE(c) if self.es_moe else Denso(c)
        if c.attnres:
            self.q_res = nn.Parameter(torch.zeros(2, c.d))
            self.kn_res = nn.ModuleList([RMSNorm(c.d), RMSNorm(c.d)])

    def forward(self, *args):
        if self.c.checkpointing and torch.is_grad_enabled():
            return checkpoint(self._fwd, *args, use_reentrant=False)
        return self._fwd(*args)

    def _fwd(self, *args):
        if not self.c.attnres:
            (x,) = args
            x = x + self.atn(self.n1(x)).float()
            y, aux, carga = self.ffn(self.n2(x))
            return x + y.float(), aux, carga
        j0, fuentes, parcial = args
        S = self.c.attnres_s
        completados = []
        aux, carga = None, None
        for sub in range(2):
            disponibles = list(fuentes) + completados + ([parcial] if parcial is not None else [])
            h = agregar(disponibles, self.q_res[sub], self.kn_res[sub])
            if sub == 0:
                out = self.atn(self.n1(h)).float()
            else:
                out, aux, carga = self.ffn(self.n2(h))
                out = out.float()
            parcial = out if parcial is None else parcial + out
            if (j0 + sub + 1) % S == 0:
                completados.append(parcial)
                parcial = None
        return completados, parcial, aux, carga


class Ludwig(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c = c
        self.emb = nn.Embedding(c.vocab, c.d)
        self.bloques = nn.ModuleList([Bloque(c, i) for i in range(c.capas)])
        self.nf = RMSNorm(c.d)
        if c.attnres:
            self.q_fin = nn.Parameter(torch.zeros(c.d))
            self.kn_fin = RMSNorm(c.d)
        self.max_h = None

    def forward(self, ids, objetivos):
        c = self.c
        x = self.emb(ids).float()
        aux_total = x.new_zeros(())
        cargas = []
        if not c.attnres:
            for b in self.bloques:
                x, aux, carga = b(x)
                aux_total = aux_total + aux
                if carga is not None:
                    cargas.append(carga)
            h = x
        else:
            fuentes, parcial = [x], None
            for i, b in enumerate(self.bloques):
                completados, parcial, aux, carga = b(2 * i, fuentes, parcial)
                fuentes = fuentes + list(completados)
                aux_total = aux_total + aux
                if carga is not None:
                    cargas.append(carga)
            if parcial is not None:
                fuentes = fuentes + [parcial]
            h = agregar(fuentes, self.q_fin, self.kn_fin)
        self.max_h = h.detach().abs().max()
        perdida = ce_trozos(self.nf(h), self.emb.weight, objetivos, c.trozo_ce)
        return perdida, aux_total, cargas


def _ce(h, W, y):
    return F.cross_entropy(F.linear(h, W).float(), y, reduction="sum")


def ce_trozos(h, W, y, trozo):
    h = h.reshape(-1, h.shape[-1])
    y = y.reshape(-1)
    total = h.new_zeros((), dtype=torch.float32)
    for i in range(0, h.shape[0], trozo):
        if torch.is_grad_enabled():
            total = total + checkpoint(_ce, h[i:i + trozo], W, y[i:i + trozo], use_reentrant=False)
        else:
            total = total + _ce(h[i:i + trozo], W, y[i:i + trozo])
    return total / h.shape[0]


def inicializar(modelo, c):
    std, std_salida = c.init_std, c.init_std / math.sqrt(2 * c.capas)
    with torch.no_grad():
        for nombre, p in modelo.named_parameters():
            t = local(p)
            if nombre.endswith("q_res") or nombre.endswith("q_fin"):
                t.zero_()
            elif isinstance(_buscar(modelo, nombre), RMSNorm):
                t.fill_(1.0)
            elif nombre.endswith(("atn.o.weight", "ffn.w2", "ffn.w2.weight", "ffn.c2.weight")):
                t.normal_(0.0, std_salida)
            else:
                t.normal_(0.0, std)
        for nombre, b in modelo.named_buffers():
            if nombre.endswith("bias_ruteo"):
                b.zero_()


def _buscar(modelo, nombre_param):
    partes = nombre_param.split(".")[:-1]
    m = modelo
    for p in partes:
        m = getattr(m, p)
    return m


def envolver_fsdp(modelo, mesh, reshard=True, dtype_param=torch.float16):
    try:
        from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
    except ImportError:  # torch < 2.6
        from torch.distributed._composable.fsdp import fully_shard, MixedPrecisionPolicy
    mp = MixedPrecisionPolicy(param_dtype=dtype_param, reduce_dtype=torch.float32, cast_forward_inputs=False)
    mp32 = MixedPrecisionPolicy(param_dtype=torch.float32, reduce_dtype=torch.float32, cast_forward_inputs=False)
    for b in modelo.bloques:
        if b.es_moe:
            fully_shard(b.ffn.router, mesh=mesh, mp_policy=mp32, reshard_after_forward=reshard)
        fully_shard(b, mesh=mesh, mp_policy=mp, reshard_after_forward=reshard)
    fully_shard(modelo, mesh=mesh, mp_policy=mp, reshard_after_forward=reshard)
    return modelo


def construir(c, dispositivo, mesh=None, reshard=True, dtype_param=torch.float16):
    with torch.device("meta"):
        modelo = Ludwig(c)
    if mesh is not None:
        envolver_fsdp(modelo, mesh, reshard, dtype_param)
    modelo.to_empty(device=dispositivo)
    inicializar(modelo, c)
    return modelo


# ----------------------------------------------------------------------------- optimizador

def ns5(G, pasos=5, dtype=torch.float32):
    # Newton–Schulz quíntico de Muon (Keller Jordan). Trabaja sobre las dos últimas dimensiones.
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.float()
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    trans = X.shape[-2] > X.shape[-1]
    if trans:
        X = X.mT
    X = X.to(dtype)
    for _ in range(pasos):
        A = X @ X.mT
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if trans:
        X = X.mT
    return X.float()


def clasificar(nombre, p, c):
    # Devuelve None (AdamW) o (eje, cortes) para Muon: cada corte es una matriz lógica que se ortogonaliza aparte.
    if nombre.endswith("atn.qkvg.weight"):
        H, KV, D = c.cabezas, c.cabezas_kv, c.dim_cabeza
        return 0, [H * D, KV * D, KV * D] + ([H * D] if c.compuerta else [])
    if nombre.endswith("ffn.w13.weight"):
        return 0, [c.h_denso, c.h_denso]
    if nombre.endswith("ffn.c13.weight"):
        return 0, [c.h_compartido, c.h_compartido]
    if nombre.endswith("ffn.w13"):
        return -1, [c.h_experto, c.h_experto]
    if nombre.endswith(("atn.o.weight", "ffn.w2.weight", "ffn.c2.weight", "ffn.w2")):
        return 0, None
    return None


class MuonAdamW:
    def __init__(self, modelo, c, lr=1e-3, wd=0.1, beta=0.95, betas=(0.9, 0.95), eps=1e-8,
                 momento_dtype=torch.float16, momento_cpu=False, ns_dtype=torch.float32):
        self.lr, self.wd, self.beta, self.betas, self.eps = lr, wd, beta, betas, eps
        self.momento_dtype, self.momento_cpu, self.ns_dtype = momento_dtype, momento_cpu, ns_dtype
        self.ws = dist.get_world_size() if dist.is_initialized() else 1
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.muon, self.adam = [], []
        for nombre, p in modelo.named_parameters():
            if not p.requires_grad:
                continue
            spec = clasificar(nombre, p, c)
            if spec is None:
                sin_wd = p.ndim < 2 or "router" in nombre or "q_res" in nombre or "q_fin" in nombre
                self.adam.append({"p": p, "nombre": nombre, "wd": 0.0 if sin_wd else wd})
            else:
                self.muon.append({"p": p, "nombre": nombre, "eje": spec[0], "cortes": spec[1]})
        self.estado = {}
        self.t = 0

    def parametros(self):
        return [g["p"] for g in self.muon + self.adam]

    def zero_grad(self):
        for p in self.parametros():
            p.grad = None

    def _orto(self, U, g):
        partes = torch.split(U, g["cortes"], dim=g["eje"]) if g["cortes"] else [U]
        salida = []
        for P in partes:
            m, n = P.shape[-2], P.shape[-1]
            salida.append(ns5(P, dtype=self.ns_dtype) * (0.2 * math.sqrt(max(m, n))))
        return torch.cat(salida, dim=g["eje"]) if len(salida) > 1 else salida[0]

    @torch.no_grad()
    def step(self, lr_mult=1.0):
        self.t += 1
        lr = self.lr * lr_mult
        for g in self.muon:
            p = g["p"]
            if p.grad is None:
                continue
            G, W = local(p.grad).float(), local(p)
            st = self.estado.get(id(p))
            if st is None:
                M = torch.zeros(W.shape, dtype=self.momento_dtype, device="cpu" if self.momento_cpu else W.device)
                if self.momento_cpu and W.is_cuda:
                    M = M.pin_memory()
                # En FP16 el momento se guarda escalado por matriz (máximo ≈ 2^15) para no perder gradientes chicos.
                esc = (torch.ones(W.shape[:-2] + (1, 1), dtype=torch.float32, device=W.device)
                       if self.momento_dtype == torch.float16 else None)
                st = self.estado[id(p)] = (M, esc)
            M, esc = st
            Mg = M.to(W.device, non_blocking=True).float()
            if esc is not None:
                Mg.mul_(esc)
            Mg.mul_(self.beta).add_(G)
            U = G.add(Mg, alpha=self.beta)                  # Nesterov
            if esc is not None:
                esc.copy_(Mg.abs().amax(dim=(-2, -1), keepdim=True).clamp_min(1e-30) / 32768.0)
                M.copy_((Mg / esc).to(M.dtype), non_blocking=True)
            else:
                M.copy_(Mg.to(M.dtype), non_blocking=True)
            if isinstance(p, DTensor) and self.ws > 1 and W.ndim == 2:
                completa = DTensor.from_local(U, p.device_mesh, p.placements, shape=p.shape, stride=p.stride()).full_tensor()
                O = torch.chunk(self._orto(completa, g), self.ws, dim=0)[self.rank]
            else:                                            # expertos: cada GPU tiene matrices completas
                O = self._orto(U, g)
            W.mul_(1 - lr * self.wd).add_(O.to(W.dtype), alpha=-lr)
        b1, b2 = self.betas
        for g in self.adam:
            p = g["p"]
            if p.grad is None:
                continue
            G, W = local(p.grad).float(), local(p)
            st = self.estado.get(id(p))
            if st is None:
                st = self.estado[id(p)] = (torch.zeros_like(W, dtype=torch.float32), torch.zeros_like(W, dtype=torch.float32))
            m, v = st
            m.mul_(b1).add_(G, alpha=1 - b1)
            v.mul_(b2).addcmul_(G, G, value=1 - b2)
            mhat = m / (1 - b1 ** self.t)
            vhat = v / (1 - b2 ** self.t)
            W.mul_(1 - lr * g["wd"]).addcdiv_(mhat.to(W.dtype), (vhat.sqrt() + self.eps).to(W.dtype), value=-lr)


class Escalador:
    # Escalado dinámico de la pérdida para FP16: si aparece inf/NaN se salta el paso y se baja la escala.
    def __init__(self, activo, inicial=2.0 ** 14, intervalo=500):
        self.activo, self.s, self.intervalo = activo, (inicial if activo else 1.0), intervalo
        self.buenos, self.saltos = 0, 0

    def escalar(self, perdida):
        return perdida * self.s if self.activo else perdida

    def revisar(self, norma_escalada):
        if not torch.isfinite(norma_escalada).item():
            if self.activo:
                self.s = max(self.s / 2, 1.0)
            self.buenos = 0
            self.saltos += 1
            return False
        self.buenos += 1
        if self.activo and self.buenos % self.intervalo == 0:
            self.s *= 2
        return True


def norma_grad(params):
    dev = local(params[0]).device
    s = torch.zeros((), device=dev, dtype=torch.float32)
    for p in params:
        if p.grad is not None:
            s += local(p.grad).float().pow(2).sum()
    if dist.is_initialized():
        dist.all_reduce(s)
    return s.sqrt()


def escalar_grads(params, factor):
    for p in params:
        if p.grad is not None:
            local(p.grad).mul_(factor)


@torch.no_grad()
def actualizar_bias(modelo, cargas, tasa):
    # cargas: tensor [n_moe, E] acumulado en el lote. Sube el bias de los expertos con poca carga y baja el de los sobrecargados.
    if dist.is_initialized():
        dist.all_reduce(cargas)
    cargas = cargas.float()
    i = 0
    for b in modelo.bloques:
        if b.es_moe:
            err = cargas[i].mean() - cargas[i]
            b.ffn.bias_ruteo.add_(tasa * torch.sign(err))
            i += 1
    media = cargas.mean(1, keepdim=True).clamp_min(1e-9)
    return {"carga_max_sobre_media": float((cargas / media).max()),
            "expertos_muertos": int((cargas == 0).sum()),
            "carga_min_sobre_media": float((cargas / media).min())}


def lr_wsd(paso, total, calentamiento, frac_decay=0.1, lr_final=0.0):
    if paso < calentamiento:
        return (paso + 1) / calentamiento
    inicio_decay = int(total * (1 - frac_decay))
    if paso < inicio_decay:
        return 1.0
    f = (paso - inicio_decay) / max(1, total - inicio_decay)
    return 1.0 - (1.0 - lr_final) * f


class DatosMarkov:
    # Texto sintético con estructura aprendible: cadena de Markov sobre 4,096 tokens (8 sucesores por token)
    # y la segunda mitad de cada secuencia copia un trozo de la primera (obliga a la atención a trabajar).
    def __init__(self, vocab, seq, semilla, estados=4096, ramas=8):
        g = torch.Generator().manual_seed(1234)
        self.estados = min(estados, vocab)
        self.suc = torch.randint(0, self.estados, (self.estados, ramas), generator=g)
        self.prob = torch.softmax(torch.randn(self.estados, ramas, generator=g) * 1.5, -1)
        self.seq = seq
        self.rng = torch.Generator().manual_seed(semilla)

    def lote(self, B):
        T = self.seq + 1
        x = torch.empty(B, T, dtype=torch.long)
        cur = torch.randint(0, self.estados, (B,), generator=self.rng)
        for t in range(T):
            x[:, t] = cur
            elec = torch.multinomial(self.prob[cur], 1, generator=self.rng).squeeze(1)
            cur = self.suc[cur, elec]
        mitad = T // 2
        x[:, mitad:] = x[:, : T - mitad].clone()
        return x[:, :-1], x[:, 1:]

    def entropia_bigrama(self):
        p = self.prob
        return float(-(p * p.log()).sum(-1).mean())
''',
"worker.py": r'''# Ludwig 2 — trabajador del banco de pruebas. Se lanza con torchrun: un proceso por GPU.
# Modos: "nccl" (ancho de banda entre GPUs), "velocidad" (tokens/s y memoria) y "entrenar" (estabilidad).
import json
import os
import sys
import time
import traceback
from datetime import timedelta

import torch
import torch.distributed as dist

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import modelo as L  # noqa: E402


def iniciar():
    gpu = torch.cuda.is_available()
    if "RANK" in os.environ:
        if gpu:
            torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        espera = int(os.environ.get("LUDWIG_TIMEOUT_NCCL", "300"))
        dist.init_process_group("nccl" if gpu else "gloo", timeout=timedelta(seconds=espera))
    rank = dist.get_rank() if dist.is_initialized() else 0
    ws = dist.get_world_size() if dist.is_initialized() else 1
    dev = torch.device("cuda", torch.cuda.current_device()) if gpu else torch.device("cpu")
    mesh = None
    if ws > 1:
        from torch.distributed.device_mesh import init_device_mesh
        mesh = init_device_mesh(dev.type, (ws,))
    return rank, ws, dev, mesh


def sync(dev):
    if dev.type == "cuda":
        torch.cuda.synchronize()


def max_ranks(valores, dev):
    t = torch.tensor(valores, dtype=torch.float64, device=dev)
    if dist.is_initialized():
        dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return t.tolist()


def media_ranks(valor, dev):
    t = torch.tensor([valor], dtype=torch.float64, device=dev)
    if dist.is_initialized():
        dist.all_reduce(t)
        t /= dist.get_world_size()
    return float(t.item())


def preparar(a, dev, mesh):
    c = L.Config(**a["config"])
    gpu = dev.type == "cuda"
    m = L.construir(c, dev, mesh=mesh, reshard=a.get("reshard", True),
                    dtype_param=torch.float16 if gpu else torch.float32)
    opt = L.MuonAdamW(m, c, lr=a.get("lr", 1e-3), momento_cpu=a.get("momento_cpu", False),
                      momento_dtype=torch.float16 if gpu else torch.float32,
                      ns_dtype=torch.float16 if gpu else torch.float32)
    esc = L.Escalador(activo=gpu)
    return c, m, opt, esc


class Paso:
    # Acumula micro-lotes y aplica un paso del optimizador con escalado FP16, recorte de norma y ajuste del bias.
    def __init__(self, c, m, opt, esc, dev, ga, recorte=1.0):
        self.c, self.m, self.opt, self.esc, self.dev, self.ga, self.recorte = c, m, opt, esc, dev, ga, recorte
        self.params = opt.parametros()
        n_moe = sum(1 for b in m.bloques if b.es_moe)
        self.cargas = torch.zeros(max(n_moe, 1), c.expertos, device=dev, dtype=torch.float32)
        self.gpu = dev.type == "cuda"

    def micro(self, x, y):
        with torch.autocast(device_type=self.dev.type, dtype=torch.float16, enabled=self.gpu):
            perdida, aux, cargas = self.m(x, y)
        self.esc.escalar((perdida + aux) / self.ga).backward()
        if cargas:
            self.cargas.add_(torch.stack(cargas).float())
        return perdida.detach()

    def paso(self, lr_mult=1.0):
        n = L.norma_grad(self.params)
        ok = self.esc.revisar(n)
        norma = float("nan")
        if ok:
            norma = float(n.item()) / self.esc.s
            L.escalar_grads(self.params, (1.0 / self.esc.s) * min(1.0, self.recorte / max(norma, 1e-12)))
            self.opt.step(lr_mult)
        self.opt.zero_grad()
        stats = L.actualizar_bias(self.m, self.cargas, self.c.tasa_bias)
        self.cargas.zero_()
        return ok, norma, stats


def nccl(a, rank, ws, dev, mesh):
    n = int(a.get("mb", 128)) * 2 ** 20 // 2
    x = torch.randn(n, device=dev, dtype=torch.float16 if dev.type == "cuda" else torch.float32)
    partes = [torch.empty_like(x) for _ in range(ws)]
    for _ in range(2):
        dist.all_gather(partes, x)
    sync(dev)
    t0 = time.time()
    rep = 5
    for _ in range(rep):
        dist.all_gather(partes, x)
    sync(dev)
    dt = (time.time() - t0) / rep
    p2p = None
    if dev.type == "cuda" and torch.cuda.device_count() > 1:
        p2p = bool(torch.cuda.can_device_access_peer(0, 1))
    return {"gb_s": n * x.element_size() * (ws - 1) / dt / 1e9, "p2p": p2p, "mb": a.get("mb", 128)}


def velocidad(a, rank, ws, dev, mesh):
    c, m, opt, esc = preparar(a, dev, mesh)
    mb, ga, rondas = int(a["mb"]), int(a.get("ga", 4)), int(a.get("rondas", 2))
    P = Paso(c, m, opt, esc, dev, ga)
    gen = torch.Generator().manual_seed(1000 + rank)

    def lote():
        ids = torch.randint(0, c.vocab, (mb, c.seq + 1), generator=gen).to(dev, non_blocking=True)
        return ids[:, :-1], ids[:, 1:]

    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t_ini = time.time()
    perdida = None
    for _ in range(2):  # calentamiento; el primer paso crea los estados del optimizador
        perdida = P.micro(*lote())
    ok, norma, stats = P.paso()
    sync(dev)
    t_cal = time.time() - t_ini
    t_micro, t_paso = [], []
    for _ in range(rondas):
        t0 = time.time()
        for _ in range(ga):
            perdida = P.micro(*lote())
        sync(dev)
        t1 = time.time()
        ok, norma, stats = P.paso()
        sync(dev)
        t2 = time.time()
        t_micro.append((t1 - t0) / ga)
        t_paso.append(t2 - t1)
    pico = torch.cuda.max_memory_allocated() / 2 ** 30 if dev.type == "cuda" else 0.0
    res_mem = torch.cuda.max_memory_reserved() / 2 ** 30 if dev.type == "cuda" else 0.0
    pico, res_mem, tm, tp = max_ranks([pico, res_mem, sum(t_micro) / len(t_micro), sum(t_paso) / len(t_paso)], dev)
    tokens_micro = mb * c.seq * ws
    pr = c.params()
    return {"config": c.dict(), "mb": mb, "ga": ga, "ws": ws, "tokens_micro": tokens_micro,
            "t_micro_s": tm, "t_paso_s": tp, "t_calentamiento_s": t_cal,
            "tokens_s": tokens_micro / tm, "pico_gib": pico, "reservado_gib": res_mem,
            "params_total": pr["total"], "params_activos": pr["activo"], "flops_token": c.flops_token(),
            "perdida": media_ranks(float(perdida), dev), "escala": esc.s, "saltos": esc.saltos,
            "momento_cpu": bool(a.get("momento_cpu", False)), **stats}


def entrenar(a, rank, ws, dev, mesh):
    c, m, opt, esc = preparar(a, dev, mesh)
    pasos, ga, mb = int(a["pasos"]), int(a.get("ga", 2)), int(a["mb"])
    P = Paso(c, m, opt, esc, dev, ga)
    datos = L.DatosMarkov(c.vocab, c.seq, semilla=5000 + rank)
    cal = int(a.get("calentamiento", max(5, pasos // 10)))
    cada = max(1, int(a.get("cada", 5)))
    curva = []
    t0 = time.time()
    for s in range(pasos):
        mult = L.lr_wsd(s, pasos, cal, frac_decay=0.0)
        perdidas = []
        for _ in range(ga):
            x, y = datos.lote(mb)
            perdidas.append(P.micro(x.to(dev), y.to(dev)))
        ok, norma, stats = P.paso(mult)
        if s % cada == 0 or s == pasos - 1:
            perdida = media_ranks(float(torch.stack(perdidas).mean()), dev)
            max_h = max_ranks([float(m.max_h) if m.max_h is not None else 0.0], dev)[0]
            curva.append({"paso": s, "perdida": perdida, "norma": norma, "escala": esc.s, "ok": ok,
                          "max_h": max_h, **stats})
    sync(dev)
    dt = time.time() - t0
    finitas = [p["perdida"] for p in curva if p["perdida"] == p["perdida"]]
    return {"config": c.dict(), "mb": mb, "ga": ga, "pasos": pasos, "curva": curva,
            "perdida_inicial": curva[0]["perdida"], "perdida_final": curva[-1]["perdida"],
            "perdida_min": min(finitas) if finitas else float("nan"),
            "saltos": esc.saltos, "escala_final": esc.s, "tiempo_s": dt,
            "tokens_s": pasos * ga * mb * c.seq * ws / dt,
            "entropia_bigrama": datos.entropia_bigrama(), "nan": len(curva) - len(finitas)}


def main():
    a = json.loads(sys.argv[1])
    rank, ws, dev, mesh = iniciar()
    try:
        res = {"nccl": nccl, "velocidad": velocidad, "entrenar": entrenar}[a["modo"]](a, rank, ws, dev, mesh)
        res["ok"] = True
    except Exception as e:  # incluye torch.cuda.OutOfMemoryError
        texto = traceback.format_exc()
        bajo = texto.lower()
        res = {"ok": False, "oom": "out of memory" in bajo or "outofmemory" in bajo or "can't allocate memory" in bajo,
               "error": texto[-3000:]}
    with open(f'{a["salida"]}.rank{rank}', "w") as f:
        json.dump(res, f)
    if dist.is_initialized() and res.get("ok"):
        try:
            dist.barrier()
            dist.destroy_process_group()
        except Exception:
            pass


if __name__ == "__main__":
    main()
''',
"micro.py": r'''# Ludwig 2 — pruebas de un solo proceso en la GPU 0: info, eficiencia de matmul y kernels de KDA.
# Se corre en un subproceso aparte para que el notebook nunca abra CUDA (así las GPUs quedan libres).
import importlib
import inspect
import json
import os
import sys
import time
import traceback

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def info():
    r = {"torch": torch.__version__, "cuda": torch.version.cuda, "gpus": []}
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            r["gpus"].append({"nombre": p.name, "memoria_gib": p.total_memory / 2 ** 30,
                              "capacidad": f"{p.major}.{p.minor}"})
        if torch.cuda.device_count() > 1:
            r["p2p"] = bool(torch.cuda.can_device_access_peer(0, 1))
        libre, total = torch.cuda.mem_get_info(0)
        r["libre_gib"] = libre / 2 ** 30
    return r


def medir(fn, reps=10):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(reps):
        fn()
    torch.cuda.synchronize()
    return (time.time() - t0) / reps


def matmul():
    dev, dt = "cuda", torch.float16
    casos = [
        ("grande 8192x4096 @ 4096x4096", (8192, 4096), (4096, 4096), None),
        ("atención d1024: 16384x1024 @ 1024x2048", (16384, 1024), (1024, 2048), None),
        ("atención d2048: 16384x2048 @ 2048x4096", (16384, 2048), (2048, 4096), None),
        ("expertos d1024, 40 exp (1640 tok c/u)", (1640, 1024), (1024, 1024), 40),
        ("expertos d1024, 60 exp (1092 tok c/u)", (1092, 1024), (1024, 1024), 60),
        ("expertos d1280, 40 exp (1640 tok c/u)", (1640, 1280), (1280, 1280), 40),
        ("expertos d2048, 40 exp (1640 tok c/u)", (1640, 2048), (2048, 2048), 40),
    ]
    r = []
    for nombre, (m, k), (k2, n), e in casos:
        if e is None:
            a, b = torch.randn(m, k, device=dev, dtype=dt), torch.randn(k2, n, device=dev, dtype=dt)
            t = medir(lambda: a @ b)
            fl = 2 * m * k * n
        else:
            a, b = torch.randn(e, m, k, device=dev, dtype=dt), torch.randn(e, k2, n, device=dev, dtype=dt)
            t = medir(lambda: torch.bmm(a, b))
            fl = 2 * e * m * k * n
        r.append({"caso": nombre, "tflops": fl / t / 1e12})
        del a, b
        torch.cuda.empty_cache()
    return {"matmul": r}


def tiempo_capa(capa, B, T, d, dtype):
    x = torch.randn(B, T, d, device="cuda", requires_grad=True)

    def fn():
        with torch.autocast(device_type="cuda", dtype=dtype, enabled=dtype != torch.float32):
            out = capa(x)
        o = out[0] if isinstance(out, (tuple, list)) else out
        o.float().pow(2).mean().backward()
    return medir(fn, reps=5)


def kda():
    import modelo as L
    B, T, d = 4, 2048, 1024
    r = {"forma": [B, T, d], "capas": []}
    c = L.Config(d=d, cabezas=16, cabezas_kv=4, dim_cabeza=64, seq=T)
    atn = L.Atencion(c).cuda()
    for p in atn.parameters():
        torch.nn.init.normal_(p, 0, 0.02)
    try:
        r["atencion_ms"] = 1000 * tiempo_capa(atn, B, T, d, torch.float16)
    except Exception:
        r["atencion_error"] = traceback.format_exc()[-1500:]
    try:
        import fla
        r["fla"] = getattr(fla, "__version__", "?")
    except Exception:
        r["error"] = "no se pudo importar fla: " + traceback.format_exc()[-800:]
        return r
    candidatos = [("fla.layers", "KimiDeltaAttention"), ("fla.layers.kda", "KimiDeltaAttention"),
                  ("fla.layers", "GatedDeltaNet")]
    vistos = set()
    for modulo, clase in candidatos:
        info = {"capa": f"{modulo}.{clase}"}
        try:
            C = getattr(importlib.import_module(modulo), clase)
            if C in vistos:
                continue
            vistos.add(C)
            sig = inspect.signature(C.__init__).parameters
            kw = {k: v for k, v in {"hidden_size": d, "num_heads": 16, "head_dim": 64, "mode": "chunk"}.items()
                  if k in sig}
            info["kwargs"] = {k: v for k, v in kw.items()}
            capa = C(**kw).cuda()
            for nombre, dt in (("fp16", torch.float16), ("fp32", torch.float32)):
                try:
                    info[f"{nombre}_ms"] = 1000 * tiempo_capa(capa, B, T, d, dt)
                except Exception:
                    info[f"{nombre}_error"] = traceback.format_exc()[-1200:]
        except Exception:
            info["error"] = traceback.format_exc()[-1200:]
        r["capas"].append(info)
        torch.cuda.empty_cache()
    return r


if __name__ == "__main__":
    modo, salida = sys.argv[1], sys.argv[2]
    try:
        res = {"info": info, "matmul": matmul, "kda": kda}[modo]()
        res["ok"] = True
    except Exception:
        res = {"ok": False, "error": traceback.format_exc()[-3000:]}
    with open(salida, "w") as f:
        json.dump(res, f)
''',
}
for _nombre, _texto in _ARCHIVOS.items():
    with open(_os.path.join(_CODIGO, _nombre), "w") as _f:
        _f.write(_texto)
print("Módulos escritos en", _CODIGO)

import datetime
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import time

LOCAL = os.environ.get("LUDWIG_LOCAL") == "1"
DIR = os.environ.get("LUDWIG_DIR", "/tmp/ludwig2_banco" if LOCAL else "/kaggle/working/ludwig2_t4_banco")
CODIGO = os.environ.get("LUDWIG_CODIGO", os.path.join(DIR, "codigo"))
WORKER, MICRO = os.path.join(CODIGO, "worker.py"), os.path.join(CODIGO, "micro.py")
os.makedirs(os.path.join(DIR, "logs"), exist_ok=True)
os.makedirs(os.path.join(DIR, "tmp"), exist_ok=True)
NPROC = int(os.environ.get("LUDWIG_NPROC", "2"))
T0 = time.time()

if LOCAL:  # configuraciones diminutas para probar la lógica en CPU
    BASE = dict(vocab=512, d=64, capas=4, cabezas=4, cabezas_kv=2, dim_cabeza=16, expertos=8, top_k=2,
                h_experto=32, unidades_compartido=2, seq=64, trozo_ce=64, attnres_s=3)
    FORMAS = [("4 × d64", dict(d=64, capas=4, cabezas=4, h_experto=32)),
              ("2 × d96", dict(d=96, capas=2, cabezas=6, h_experto=48))]
    EXPERTOS_EXTRA = [12, 16]
    MBS, GA_VEL, RONDAS, PASOS, PASOS_ATTNRES = [2, 1], 2, 1, 30, 12
else:
    BASE = dict(vocab=64000, d=1024, capas=26, cabezas=16, cabezas_kv=4, dim_cabeza=64, expertos=40, top_k=4,
                h_experto=512, unidades_compartido=4, seq=2048)
    FORMAS = [("26 × d1024", dict(d=1024, capas=26, cabezas=16, h_experto=512)),
              ("16 × d1280", dict(d=1280, capas=16, cabezas=20, h_experto=640)),
              ("11 × d1536", dict(d=1536, capas=11, cabezas=24, h_experto=768)),
              ("6 × d2048 (≈ diseño anterior)", dict(d=2048, capas=6, cabezas=32, h_experto=1024))]
    EXPERTOS_EXTRA = [48, 60]
    if MODO == "rapido":
        MBS, GA_VEL, RONDAS, PASOS, PASOS_ATTNRES = [4, 2, 1], 2, 2, 30, 0
    else:
        MBS, GA_VEL, RONDAS, PASOS, PASOS_ATTNRES = [8, 4, 2, 1], 4, 2, 60, 30

R = {"inicio": datetime.datetime.now().isoformat(timespec="seconds"), "modo": "local" if LOCAL else MODO,
     "formas": [], "expertos": [], "moe_loop": None, "attnres": None, "kda": None, "estabilidad": None}
ENV = os.environ.copy()
ENV.update({"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True", "OMP_NUM_THREADS": "1",
            "LUDWIG_TIMEOUT_NCCL": "300", "TOKENIZERS_PARALLELISM": "false"})


def reloj():
    s = int(time.time() - T0)
    return f"[{s // 3600:d}:{s % 3600 // 60:02d}:{s % 60:02d}]"


def slug(t):
    return "".join(ch if ch.isalnum() else "_" for ch in t.replace("×", "x")).strip("_")[:40]


def say(*a):
    print(reloj(), *a, flush=True)


def guardar():
    with open(os.path.join(DIR, "resultados.json"), "w") as f:
        json.dump(R, f, indent=1, default=str)


def correr(cmd, log, timeout, env=None):
    with open(log, "w") as f:
        p = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, env=env or ENV, start_new_session=True)
        try:
            p.wait(timeout=timeout)
            return p.returncode
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, signal.SIGKILL)
            p.wait()
            return "timeout"


def micro(modo, timeout=900):
    salida = os.path.join(DIR, "tmp", f"micro_{modo}.json")
    if os.path.exists(salida):
        os.remove(salida)
    rc = correr([sys.executable, MICRO, modo, salida], os.path.join(DIR, "logs", f"micro_{modo}.log"), timeout)
    try:
        with open(salida) as f:
            return json.load(f)
    except Exception:
        return {"ok": False, "error": f"código {rc}: " + open(os.path.join(DIR, "logs", f"micro_{modo}.log")).read()[-1500:]}


def lanzar(nombre, args, timeout=1200):
    salida = os.path.join(DIR, "tmp", nombre)
    for r in range(NPROC):
        if os.path.exists(f"{salida}.rank{r}"):
            os.remove(f"{salida}.rank{r}")
    args = dict(args, salida=salida)
    cmd = [sys.executable, "-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={NPROC}",
           WORKER, json.dumps(args)]
    log = os.path.join(DIR, "logs", f"{nombre}.log")
    rc = correr(cmd, log, timeout)
    res = []
    for r in range(NPROC):
        try:
            with open(f"{salida}.rank{r}") as f:
                res.append(json.load(f))
        except Exception:
            pass
    ok = next((x for x in res if x.get("ok")), None)
    if ok is not None:
        return ok
    texto = open(log).read()
    oom = any(x.get("oom") for x in res) or "out of memory" in texto.lower() or "can't allocate memory" in texto.lower()
    err = next((x.get("error") for x in res if x.get("error")), None) or texto[-1500:]
    return {"ok": False, "oom": oom, "timeout": rc == "timeout", "rc": rc, "error": err}


def derivar(r):
    tm, tp, tok = r["t_micro_s"], r["t_paso_s"], r["tokens_micro"]
    ga_real = max(1, math.ceil(TOKENS_POR_PASO / tok))
    tps = tok / (tm + tp / ga_real)
    r.update(ga_real=ga_real, tokens_s_real=tps, horas_25b=OBJETIVO_TOKENS / tps / 3600,
             semanas=OBJETIVO_TOKENS / tps / 3600 / HORAS_RELEVO_SEMANA,
             mfu=r["flops_token"] * tps / (NPROC * PICO_T4_TFLOPS * 1e12))
    return r


def velocidad(nombre, config, mbs=None, extra=None):
    for mb in mbs or MBS:
        say(f"  {nombre}: probando {mb} secuencia(s) por GPU…")
        r = lanzar(f"vel_{slug(nombre)}_mb{mb}",
                   {"modo": "velocidad", "config": config, "mb": mb, "ga": GA_VEL, "rondas": RONDAS, **(extra or {})})
        if r.get("ok"):
            r["nombre"] = nombre
            derivar(r)
            say(f"  ✅ {nombre}: {r['tokens_s_real']:,.0f} tokens/s · pico {r['pico_gib']:.1f} GiB "
                f"(reservado {r['reservado_gib']:.1f}) · {r['horas_25b']:,.0f} h para 25B")
            return r
        motivo = "sin memoria" if r.get("oom") else ("se colgó" if r.get("timeout") else "error")
        say(f"  ❌ {nombre} con {mb}: {motivo}")
        if not (r.get("oom") or r.get("timeout")):
            r["nombre"] = nombre
            return r
    return {"ok": False, "nombre": nombre, "oom": True, "error": "no cupo ni con 1 secuencia por GPU"}


def tabla(filas, cols):
    anchos = [max(len(str(c)), *(len(str(f.get(c, ""))) for f in filas)) for c in cols]
    linea = "  ".join(str(c).ljust(a) for c, a in zip(cols, anchos))
    return "\n".join([linea, "-" * len(linea)] + ["  ".join(str(f.get(c, "")).ljust(a) for c, a in zip(cols, anchos)) for f in filas])


def fila_vel(r):
    if not r or not r.get("ok"):
        return {"prueba": (r or {}).get("nombre", "?"), "tokens/s": "no cupo" if (r or {}).get("oom") else "error"}
    c = r["config"]
    return {"prueba": r["nombre"], "params": f'{r["params_total"] / 1e9:.2f}B / {r["params_activos"] / 1e9:.2f}B',
            "sec/GPU": r["mb"], "tokens/s": f'{r["tokens_s_real"]:,.0f}', "MFU": f'{100 * r["mfu"]:.1f}%',
            "pico GiB": f'{r["pico_gib"]:.1f}', "reserv. GiB": f'{r["reservado_gib"]:.1f}',
            "horas 25B": f'{r["horas_25b"]:,.0f}', "semanas": f'{r["semanas"]:.1f}'}


# ---------------------------------------------------------------- 1. sistema
say("1/8 Sistema")
S = {"python": sys.version.split()[0], "cpus": os.cpu_count(),
     "disco_libre_gib": round(shutil.disk_usage(DIR).free / 2 ** 30, 1)}
try:
    with open("/proc/meminfo") as f:
        S["ram_gib"] = round(int(f.readline().split()[1]) / 2 ** 20, 1)
except Exception:
    pass
try:
    S["nvidia_smi"] = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                                      "--format=csv,noheader"], capture_output=True, text=True, timeout=30).stdout.strip()
except Exception:
    S["nvidia_smi"] = "no disponible"
S.update(micro("info"))
R["sistema"] = S
say(f"  torch {S.get('torch')} · CUDA {S.get('cuda')} · GPUs: {[g['nombre'] for g in S.get('gpus', [])]} · "
    f"RAM {S.get('ram_gib')} GiB · CPUs {S['cpus']}")
if not LOCAL and len(S.get("gpus", [])) < 2:
    raise SystemExit("Este notebook necesita 2 GPUs: en Settings → Accelerator elige 'GPU T4 x2'.")
guardar()

# ---------------------------------------------------------------- 2. matmul y comunicación
say("2/8 Eficiencia de matmul en una T4 y ancho de banda entre las dos GPUs")
if not LOCAL:
    R["matmul"] = micro("matmul")
    for x in R["matmul"].get("matmul", []):
        say(f"  {x['caso']}: {x['tflops']:.1f} TFLOPS")
r = lanzar("nccl", {"modo": "nccl", "mb": 128 if not LOCAL else 4}, timeout=240)
if not r.get("ok") and not LOCAL:
    say("  La comunicación falló con la configuración por defecto; reintento con NCCL_P2P_DISABLE=1")
    ENV["NCCL_P2P_DISABLE"] = "1"
    r = lanzar("nccl_sin_p2p", {"modo": "nccl", "mb": 128}, timeout=240)
    r["nccl_p2p_disable"] = True
R["nccl"] = r
say(f"  all-gather: {r.get('gb_s', float('nan')):.1f} GB/s · P2P: {r.get('p2p')}" if r.get("ok") else f"  ❌ NCCL: {r.get('error', '')[-300:]}")
if not r.get("ok"):
    guardar()
    raise SystemExit("Las dos GPUs no se pueden comunicar; revisa logs/nccl*.log")
guardar()

# ---------------------------------------------------------------- 3. formas
say("3/8 Velocidad y memoria por forma (todas con ~0.49B activos y 40 expertos)")
for nombre, cambios in FORMAS:
    R["formas"].append(velocidad(nombre, dict(BASE, **cambios)))
    guardar()
base = R["formas"][0]
EXTRA = {}
if not base.get("ok") and base.get("oom"):
    say("  La forma base no cupo; pruebo con el momento de Muon en la RAM de la CPU")
    r = velocidad(FORMAS[0][0] + " (momento en CPU)", dict(BASE, **FORMAS[0][1]), extra={"momento_cpu": True})
    R["formas"].append(r)
    if r.get("ok"):
        base, EXTRA = r, {"momento_cpu": True}
BASE_OK = bool(base.get("ok"))
mb_base = base["mb"] if BASE_OK else None
guardar()

if BASE_OK:
    # ------------------------------------------------------------ 4. expertos
    say("4/8 ¿Cuántos expertos caben? (forma base)")
    for E in EXPERTOS_EXTRA:
        cfg = dict(BASE, expertos=E)
        mbs = [m for m in MBS if m <= mb_base]
        r = velocidad(f"{E} expertos", cfg, mbs=mbs, extra=EXTRA)
        if not r.get("ok") and r.get("oom") and not EXTRA:
            say(f"  {E} expertos no cupo; pruebo con el momento de Muon en la RAM de la CPU")
            R["expertos"].append(r)
            r = velocidad(f"{E} expertos (momento en CPU)", cfg, mbs=mbs, extra={"momento_cpu": True})
        R["expertos"].append(r)
        guardar()

    # ------------------------------------------------------------ 5. implementación del MoE
    say("5/8 MoE: bmm con relleno vs. un matmul por experto")
    R["moe_loop"] = velocidad("MoE en loop", dict(BASE, moe_impl="loop"), mbs=[mb_base], extra=EXTRA)
    guardar()

    # ------------------------------------------------------------ 6. AttnRes
    say("6/8 Costo de Attention Residuals (Block AttnRes)")
    R["attnres"] = velocidad("base + AttnRes", dict(BASE, attnres=True), mbs=[m for m in MBS if m <= mb_base], extra=EXTRA)
    guardar()

    # ------------------------------------------------------------ 7. estabilidad
    say(f"7/8 Entrenamiento corto en FP16 con datos sintéticos ({PASOS} pasos)")
    mb_ent = min(mb_base, 4) if not LOCAL else 1
    R["estabilidad"] = lanzar("entrenar_base", {"modo": "entrenar", "config": BASE, "mb": mb_ent, "ga": 2,
                                                "pasos": PASOS, "lr": 1e-3, "cada": 5, **EXTRA}, timeout=5400)
    e = R["estabilidad"]
    if e.get("ok"):
        say(f"  pérdida {e['perdida_inicial']:.2f} → {e['perdida_final']:.2f} (piso teórico ≈ {e['entropia_bigrama'] / 2:.2f}) · "
            f"pasos saltados {e['saltos']} · NaN {e['nan']} · {e['tokens_s']:,.0f} tokens/s")
    else:
        say(f"  ❌ {e.get('error', '')[-300:]}")
    if PASOS_ATTNRES and R["attnres"] and R["attnres"].get("ok"):
        R["estabilidad_attnres"] = lanzar("entrenar_attnres", {
            "modo": "entrenar", "config": dict(BASE, attnres=True), "mb": min(R["attnres"]["mb"], mb_ent), "ga": 2,
            "pasos": PASOS_ATTNRES, "lr": 1e-3, "cada": 5, **EXTRA}, timeout=3600)
        ea = R["estabilidad_attnres"]
        say(f"  con AttnRes: pérdida {ea['perdida_inicial']:.2f} → {ea['perdida_final']:.2f} · NaN {ea['nan']}"
            if ea.get("ok") else f"  ❌ AttnRes: {ea.get('error', '')[-300:]}")
    guardar()
else:
    say("4–7/8 Omitidas: la forma base no corrió (el error está en resultados.json y logs/).")

# ---------------------------------------------------------------- 8. KDA
say("8/8 ¿Corren los kernels de KDA (Kimi Linear) en T4?")
if LOCAL or not PROBAR_KDA:
    R["kda"] = {"ok": False, "omitido": True}
    say("  omitido")
else:
    pip = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "flash-linear-attention"],
                         capture_output=True, text=True, timeout=900)
    if pip.returncode != 0:
        R["kda"] = {"ok": False, "error": "no se pudo instalar flash-linear-attention (¿Internet apagado?): "
                    + pip.stderr[-600:]}
        say("  ❌ no se pudo instalar flash-linear-attention")
    else:
        R["kda"] = micro("kda", timeout=1500)
        k = R["kda"]
        say(f"  atención con compuerta (referencia): {k.get('atencion_ms', float('nan')):.1f} ms ida+vuelta")
        for capa in k.get("capas", []):
            estado = "; ".join(f"{t}: {capa[t + '_ms']:.1f} ms" if t + "_ms" in capa else f"{t}: falló"
                               for t in ("fp16", "fp32"))
            say(f"  {capa['capa']}: {capa.get('error', '')[-120:] or estado}")
guardar()

# ---------------------------------------------------------------- reporte y decisiones
D = {}
formas_ok = [f for f in R["formas"] if f.get("ok")]
if not BASE_OK:
    D["base"] = ("la forma base (26 × d1024, 40 expertos) no corrió: "
                 + ("no cupo en memoria ni con el momento en CPU" if base.get("oom") else "error, ver resultados.json")
                 + ". Hay que bajar expertos o revisar logs/.")
if formas_ok and BASE_OK:
    mejor = max(formas_ok, key=lambda f: f["tokens_s_real"])
    lenta = mejor["tokens_s_real"] / base["tokens_s_real"]
    if lenta <= 1.4:
        D["forma"] = (f"{base['nombre']}: es {lenta:.2f}× más lenta que la más rápida ({mejor['nombre']}), "
                      f"dentro del límite de 1.4× que acordamos. La profundidad se queda.")
    else:
        cand = [f for f in formas_ok if f["tokens_s_real"] >= mejor["tokens_s_real"] / 1.4]
        elegida = max(cand, key=lambda f: f["config"]["capas"])
        D["forma"] = (f"{elegida['nombre']}: la de 26 capas es {lenta:.2f}× más lenta que la más rápida, más que el "
                      f"límite de 1.4×; esta es la más profunda dentro del límite.")
grande = None
if BASE_OK:
    opciones = [base] + [x for x in R["expertos"] if x.get("ok")]
    validas = [x for x in opciones if x["tokens_s_real"] >= 0.85 * base["tokens_s_real"]] or [base]
    grande = max(validas, key=lambda x: (x["config"]["expertos"], -int(bool(x.get("momento_cpu")))))
    justo = grande["reservado_gib"] > MEMORIA_SEGURA_GIB
    D["expertos"] = (f"{grande['config']['expertos']} expertos ({grande['params_total'] / 1e9:.2f}B totales)"
                     + (" con el momento de Muon en la CPU" if grande.get("momento_cpu") else "")
                     + f": {grande['tokens_s_real']:,.0f} tokens/s, reservado {grande['reservado_gib']:.1f} GiB"
                     + (" — JUSTO de memoria; para un entrenamiento largo conviene bajar uno." if justo else "."))
    loop = R["moe_loop"]
    if loop and loop.get("ok"):
        gana_loop = loop["tokens_s_real"] > base["tokens_s_real"] * 1.03
        D["moe"] = (f"{'loop' if gana_loop else 'bmm'}: loop {loop['tokens_s_real']:,.0f} vs bmm "
                    f"{base['tokens_s_real']:,.0f} tokens/s.")
    ar = R["attnres"]
    if ar and ar.get("ok"):
        sobre = 1 - ar["tokens_s_real"] / base["tokens_s_real"]
        D["attnres"] = (f"cuesta {100 * sobre:.1f}% de velocidad y {ar['reservado_gib'] - base['reservado_gib']:+.1f} GiB. "
                        + ("Listo para el duelo." if sobre <= 0.10 else "Más caro de lo esperado (~4%); revisar antes del duelo."))
    elif ar:
        D["attnres"] = "no corrió: " + ("sin memoria" if ar.get("oom") else "error (ver resultados.json)")
k = R["kda"] or {}
if k.get("omitido"):
    D["kda"] = "no se probó."
elif not k.get("ok"):
    D["kda"] = "no se pudo probar: " + str(k.get("error", ""))[-200:]
else:
    corren = [c for c in k.get("capas", []) if "fp16_ms" in c]
    if corren and k.get("atencion_ms"):
        c0 = corren[0]
        D["kda"] = (f"{c0['capa']} corre en FP16: {c0['fp16_ms']:.0f} ms vs {k['atencion_ms']:.0f} ms de la atención "
                    f"({c0['fp16_ms'] / k['atencion_ms']:.2f}×). Se puede considerar el duelo con KDA.")
    else:
        D["kda"] = "los kernels no corren en FP16 en T4 → KDA queda descartada para Ludwig 2."
e = R["estabilidad"] or {}
if e.get("ok"):
    sano = e["nan"] == 0 and e["perdida_final"] < e["perdida_inicial"] - 0.5 and e["saltos"] <= max(3, e["pasos"] // 10)
    ultimo = e["curva"][-1]
    D["estabilidad"] = (("sana" if sano else "REVISAR") + f": pérdida {e['perdida_inicial']:.2f} → {e['perdida_final']:.2f}, "
                        f"{e['saltos']} pasos saltados por FP16, carga máx/media {ultimo['carga_max_sobre_media']:.2f}, "
                        f"expertos sin tokens {ultimo['expertos_muertos']}, |h| máx {ultimo['max_h']:.3g}.")
elif BASE_OK:
    D["estabilidad"] = "no corrió: " + str(e.get("error", ""))[-300:]
if grande:
    D["horas"] = (f"Con {grande['config']['expertos']} expertos: {grande['horas_25b']:,.0f} h de 2×T4 para 25B tokens "
                  f"≈ {grande['semanas']:.1f} semanas de relevo a {HORAS_RELEVO_SEMANA} h/semana (MFU {100 * grande['mfu']:.1f}%).")
R["decisiones"] = D
R["fin"] = datetime.datetime.now().isoformat(timespec="seconds")
R["duracion_min"] = round((time.time() - T0) / 60, 1)
guardar()

lineas = ["LUDWIG 2 — BANCO DE PRUEBAS EN 2×T4", "=" * 40,
          f"Modo: {R['modo']} · duración {R['duracion_min']} min · torch {S.get('torch')} · CUDA {S.get('cuda')}",
          f"GPUs: {S.get('nvidia_smi')}", f"Comunicación: {R['nccl'].get('gb_s', 0):.1f} GB/s (P2P {R['nccl'].get('p2p')})", ""]
if R.get("matmul"):
    lineas += ["EFICIENCIA DE MATMUL EN UNA T4 (pico teórico 65 TFLOPS)",
               tabla([{"caso": x["caso"], "TFLOPS": f"{x['tflops']:.1f}"} for x in R["matmul"].get("matmul", [])],
                     ["caso", "TFLOPS"]), ""]
cols = ["prueba", "params", "sec/GPU", "tokens/s", "MFU", "pico GiB", "reserv. GiB", "horas 25B", "semanas"]
lineas += ["FORMAS (40 expertos)", tabla([fila_vel(r) for r in R["formas"]], cols), ""]
if BASE_OK:
    fila_base = dict(fila_vel(base), prueba=f"{base['config']['expertos']} expertos (base)")
    lineas += ["EXPERTOS (forma base)", tabla([fila_base] + [fila_vel(r) for r in R["expertos"]], cols), "",
               "OTRAS PRUEBAS", tabla([fila_vel(R["moe_loop"]), fila_vel(R["attnres"])], cols), ""]
if e.get("ok"):
    lineas += ["ENTRENAMIENTO CORTO (datos sintéticos)",
               tabla([{"paso": p["paso"], "pérdida": f"{p['perdida']:.3f}", "norma": f"{p['norma']:.3f}",
                       "escala FP16": f"{p['escala']:.0f}", "carga máx/media": f"{p['carga_max_sobre_media']:.2f}",
                       "muertos": p["expertos_muertos"], "|h| máx": f"{p['max_h']:.3g}"} for p in e["curva"]],
                     ["paso", "pérdida", "norma", "escala FP16", "carga máx/media", "muertos", "|h| máx"]), ""]
lineas += ["DECISIONES", *[f"- {k.upper()}: {v}" for k, v in D.items()], "",
           "Archivos: resultados.json (todo), logs/ (salida de cada prueba)."]
reporte = "\n".join(lineas)
with open(os.path.join(DIR, "reporte.txt"), "w") as f:
    f.write(reporte)
print("\n" + reporte)