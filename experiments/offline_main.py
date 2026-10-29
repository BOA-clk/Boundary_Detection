# -*- coding: utf-8 -*-
"""MAMBOA Adam/50 — standalone Colab / Python, baseline-training-recipe control.

GTEA -> 50Salads -> Breakfast, all official folds x training seeds0/1 (26 runs).
Architecture / weighted context loss / 15Hz / DP / refinement unchanged:
G32 x width64, state16, conv4, expand1, D=0 FROZEN, nonlinear MLP, original TCN.

MATCHED TO SUPPLIED OLD baseline_common.py / run_protocol.py:
Adam lr5e-4, wd0, default betas/eps; FP32; NO scheduler/warmup/gradient clip;
max50 epochs, patience10; val20% split with EACH TRAINING SEED (not fixed0).
Checkpoint uses FIXED first native decoder-grid entry (lambda0); only after
loading BEST checkpoint is lambda tuned on validation. Training order seed+epoch.
Validation median nearest-prediction residual is subtracted from test times,
without post-calibration clipping or snapping, as in the old baseline runner.
Native losses/decoder parameter values need not equal those of other methods.

Primary metric defaults to the GREEDY one-to-one scorer from supplied bnd_seg.py,
matching old baseline reports. If a baseline boundary_prf is available in this
same notebook runtime, it must be identified as greedy or optimal BEFORE training.
Optimal Hungarian scores + matched-pair calibration are ALSO saved from the same
trained model as a SEPARATE diagnostic; they do not select the primary checkpoint.
No reported value is taken from the paper. Preflight + exact recipe tests run
before training; real dataset scores require CUDA and an installed Mamba.

Input paths inherit previous notebook configs. Outputs use a NEW adam50 subfolder.
All split IDs / model and source hashes / raw logs / every-epoch validation,
best+last checkpoints / val+test logits / GT+timestamps / alternate scores saved.
Interrupted runs resume from LAST completed epoch; no old AdamW checkpoints used.

Colab: mount Drive; run this ENTIRE file in one cell. No model imports required.
Python: python mamboa_adam50_colab.py --dataset all
Single dataset: --dataset gtea / 50salads / breakfast. Import never starts training.
"""

# Snapshot notebook paths/metric BEFORE embedded reference defaults are defined.
_M5_RUNTIME_ROOTS = dict(globals().get('MB_ROOTS', {}))
_M5_RUNTIME_ROOTS.update(globals().get('MB_REVIEW_ROOTS', {}))
_M5_RUNTIME_OUTPUT = globals().get('MB_REVIEW_OUTPUT', globals().get('MB_OUTPUT_DIR', './Mamboa_Comparison'))
_M5_RUNTIME_METRIC = globals().get('boundary_prf', getattr(globals().get('BC'), 'boundary_prf', None))

import copy
import csv
import gc
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import random
import shutil
import sys
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint as mb_checkpoint

MB_VERSION = "mamboa-final-singleloss-fixedsplit-2026-09-15.1"


@dataclass(frozen=True)
class MBDatasetSpec:
    name: str
    folds: int
    expected_videos: int
    label_hz: float
    sigma_sec: float
    tolerances: tuple
    min_gap_sec: float
    old_decision_hz: float
    feature_hz: object = None  # None: yalniz bilinen hizlardan uzunluk denetimi
    feature_time_offset_sec: float = 0.0


MB_DATASETS = {
    "gtea": MBDatasetSpec("gtea", 4, 28, 15., .3, (.2, .4, .8), .8, 5.),
    "50salads": MBDatasetSpec("50salads", 5, 50, 30., 1., (.5, 1., 2.), 4., 2.),
    "breakfast": MBDatasetSpec("breakfast", 4, 1712, 15., .8, (.5, 1., 2.), 2., 3.),
}


@dataclass(frozen=True)
class MBTrainConfig:
    target_hz: float = 15.0
    seed: int = 0
    val_fraction: float = .2
    epochs: int = 60
    patience: int = 10
    learning_rate: float = 5e-4
    weight_decay: float = .01
    warmup_fraction: float = .1
    grad_clip: float = 1.
    positive_alpha: float = 4.
    direct_weight: float = 1.
    n_groups: int = 32
    input_dim: int = 2048
    context_dim: int = 64
    sigma_override_sec: object = None
    min_gap_override_sec: object = None
    context_dilations: object = None
    context_causal: bool = False
    checkpoint_scans: bool = True
    precision: str = "auto"  # BF16 destekleniyorsa BF16, aksi halde FP32
    lambda_reference: float = 2.
    lambda_grid: tuple = tuple(i / 5 for i in range(21))
    refine: bool = True
    shuffle_repeats: int = 3  # inference tekraridir, ek training seed DEGIL
    stage_features: bool = True
    feature_cache_dir: str = "/content/mamboa_v2_feature_cache"
    resume: bool = True
    log_every: int = 1


def mb_jsonable(value):
    if isinstance(value, dict):
        return {str(k): mb_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [mb_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return mb_jsonable(value.tolist())
    if isinstance(value, np.generic):
        return mb_jsonable(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def mb_write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(mb_jsonable(value), ensure_ascii=False,
                                     indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def mb_save_torch(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def mb_load_own_checkpoint(path):
    # Yalniz bu programin kendi output klasorundeki checkpoint'ler icin.
    return torch.load(path, map_location="cpu", weights_only=False)


def mb_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # Custom CUDA scan icin bit-exact determinism iddiasi yoktur.


def mb_stable_seed(*items):
    return int.from_bytes(hashlib.sha256("|".join(map(str, items)).encode()).digest()[:4], "little")


def mb_environment():
    versions = {}
    for name in ("torch", "numpy", "scipy", "mamba-ssm", "causal-conv1d"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return dict(python=sys.version, platform=platform.platform(), packages=versions,
                cuda=torch.version.cuda,
                gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                version=MB_VERSION)


def mb_resolve_root(dataset, explicit=None):
    """Once acik yol, sonra eski dosyalardaki root adlari / bilinen Colab konumlari."""
    def valid(p):
        return all((p / d).is_dir() for d in ("features", "groundTruth", "splits"))

    if explicit:
        p = Path(explicit).expanduser().resolve()
        if not valid(p):
            raise FileNotFoundError(f"{dataset}: {p} altinda features/groundTruth/splits yok")
        return p
    # Kullanicinin Colab duzeni: drive.mount('/gdrive'), cwd=salads_50/data.
    known_base = Path('/gdrive/MyDrive/bora_Deneme/MAMBOA_seg/salads_50/data')
    known = known_base if dataset == '50salads' else known_base / dataset
    if valid(known):
        return known.resolve()
    bases = [Path.cwd(), Path("/content"), Path("/content/data"), Path("/content/datasets"),
             Path('/gdrive/MyDrive/bora_Deneme/MAMBOA_seg'), known_base,
             Path("/content/drive/MyDrive"), Path("/content/drive/MyDrive/data"),
             Path("/content/drive/MyDrive/datasets"),
             Path("/content/drive/MyDrive/bora_Deneme/MAMBOA_VLM")]
    candidates = []
    for base in bases:
        for p in (base / dataset, base / "data" / dataset, base / "datasets" / dataset):
            if valid(p) and p.resolve() not in candidates:
                candidates.append(p.resolve())
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        raise ValueError(f"{dataset}: birden fazla root bulundu {candidates}; MB_ROOTS'ta secin")
    raise FileNotFoundError(f"{dataset}: veri bulunamadi. MB_ROOTS['{dataset}'] = "
                            "'/.../features-groundTruth-splits-iceren-klasor' yazin.")


def mb_read_bundle(root, fold, kind):
    path = Path(root) / "splits" / f"{kind}.split{fold}.bundle"
    lines = [s.strip() for s in path.read_text(encoding="utf-8-sig").splitlines() if s.strip()]
    ids = [str(Path(s).with_suffix("")) for s in lines]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError(f"Bos veya yinelenen bundle: {path}")
    if any(Path(s).is_absolute() or ".." in Path(s).parts for s in ids):
        raise ValueError(f"Bundle video id'leri root'a gore relative olmali: {path}")
    return ids


def mb_fold_ids(root, spec, fold, config):
    official_train = mb_read_bundle(root, fold, "train")
    test = mb_read_bundle(root, fold, "test")
    if set(official_train) & set(test):
        raise ValueError(f"{spec.name}/fold{fold}: resmi train/test cakismasi")
    # Onceki train_split ile ayni siralama ve RandomState(seed) ayirimi.
    order = np.random.RandomState(config.seed).permutation(len(official_train))
    nval = max(1, int(round(config.val_fraction * len(official_train))))
    if nval >= len(official_train):
        raise ValueError("Validation ayirimi egitim videosu birakmiyor")
    val = [official_train[i] for i in order[:nval]]
    train = [official_train[i] for i in order[nval:]]
    return dict(train=train, validation=val, test=test)


def mb_feature_hz(n_features, n_labels, spec):
    candidates = ([float(spec.feature_hz)] if spec.feature_hz is not None else
                  sorted(set((15., spec.label_hz))))
    # Yuvarlama nedeniyle en cok 1 kaynak feature adimi farki kabul edilir.
    matches = [hz for hz in candidates
               if abs(n_features - n_labels * hz / spec.label_hz) <= 1.01]
    if len(matches) != 1:
        raise ValueError(f"{spec.name}: feature={n_features}, labels={n_labels}, "
                         f"label_hz={spec.label_hz}; feature Hz belirsiz/uyumsuz: {matches}. "
                         "Extraction stride ve feature_hz ayarini dogrulayin; kirpma yapilmadi.")
    return matches[0]


class MBVideo:
    def __init__(self, vid, root, spec, config):
        self.vid, self.spec, self.config = vid, spec, config
        self.feature_path = Path(root) / "features" / f"{vid}.npy"
        label_path = Path(root) / "groundTruth" / f"{vid}.txt"
        label_bytes = label_path.read_bytes()
        labels = [s.strip() for s in label_bytes.decode("utf-8-sig").splitlines() if s.strip()]
        raw = np.load(self.feature_path, mmap_mode="r", allow_pickle=False)
        if raw.ndim != 2 or not np.issubdtype(raw.dtype, np.floating):
            raise ValueError(f"{vid}: feature 2D float olmali, {raw.shape}/{raw.dtype}")
        if raw.shape == (config.input_dim, config.input_dim):
            raise ValueError(f"{vid}: kare matrisin kanal ekseni belirsiz; feature layoutunu dogrulayin")
        if raw.shape[0] == config.input_dim:
            self.channels_first, n_features = True, raw.shape[1]
        elif raw.shape[1] == config.input_dim:
            self.channels_first, n_features = False, raw.shape[0]
        else:
            raise ValueError(f"{vid}: kanal={config.input_dim} bekleniyor, gelen={raw.shape}")
        if not labels or n_features < 2:
            raise ValueError(f"{vid}: bos/cok kisa video")
        self.native_hz = mb_feature_hz(n_features, len(labels), spec)
        ratio = self.native_hz / config.target_hz
        self.stride = int(round(ratio))
        if self.stride < 1 or not math.isclose(ratio, self.stride, abs_tol=1e-6):
            raise ValueError(f"{vid}: {self.native_hz}->{config.target_hz} Hz tam sayi stride degil. "
                             "Upsampling/ortalama/interpolasyon desteklenmez.")
        self.duration = len(labels) / spec.label_hz
        self.source_indices = np.arange(0, n_features, self.stride, dtype=np.int64)
        self.times = self.source_indices / self.native_hz + spec.feature_time_offset_sec
        if self.times[0] < 0 or self.times[-1] > self.duration + 1.01 / self.native_hz:
            raise ValueError(f"{vid}: feature timestamp offset/video suresi uyumsuz")
        self.gt = np.array([(i - .5) / spec.label_hz for i in range(1, len(labels))
                            if labels[i] != labels[i-1]], dtype=np.float64)
        st = self.feature_path.stat()
        self.signature = dict(vid=vid, shape=list(raw.shape), dtype=str(raw.dtype),
                              feature_bytes=st.st_size, feature_mtime_ns=st.st_mtime_ns,
                              labels_sha256=hashlib.sha256(label_bytes).hexdigest(),
                              n_labels=len(labels), native_hz=self.native_hz,
                              stride=self.stride, n_steps=len(self.times))
        self._staged_path = None
        del raw

    def features(self):
        path = self.feature_path
        if self.config.stage_features:
            if self._staged_path is None:
                token = hashlib.sha256(json.dumps([str(path.resolve()), self.signature],
                                                  sort_keys=True).encode()).hexdigest()
                staged = Path(self.config.feature_cache_dir) / (token + ".npy")
                staged.parent.mkdir(parents=True, exist_ok=True)
                if not staged.exists() or staged.stat().st_size != self.signature["feature_bytes"]:
                    temporary = staged.with_suffix(".tmp")
                    shutil.copyfile(path, temporary)
                    os.replace(temporary, staged)
                self._staged_path = staged
            path = self._staged_path
        raw = np.load(path, mmap_mode="r", allow_pickle=False)
        temporal = raw.T if self.channels_first else raw
        sampled = np.array(temporal[::self.stride], dtype=np.float32, order="C", copy=True)
        if sampled.shape != (len(self.times), self.config.input_dim):
            raise RuntimeError(f"{self.vid}: feature dosyasi denetimden sonra degisti")
        if not np.isfinite(sampled).all():
            raise ValueError(f"{self.vid}: NaN/Inf feature")
        return torch.from_numpy(sampled)


def mb_dataset_preflight(root, spec, config):
    folds, all_ids, tests = {}, set(), []
    for fold in range(1, spec.folds + 1):
        ids = mb_fold_ids(root, spec, fold, config)
        universe = set(ids["train"] + ids["validation"] + ids["test"])
        if all_ids and universe != all_ids:
            raise ValueError(f"{spec.name}: fold video evrenleri farkli")
        all_ids = universe
        tests.extend(ids["test"])
        folds[fold] = ids
    if len(all_ids) != spec.expected_videos:
        raise ValueError(f"{spec.name}: {len(all_ids)} video var, tam protokol {spec.expected_videos} bekliyor")
    if len(tests) != len(set(tests)) or set(tests) != all_ids:
        raise ValueError(f"{spec.name}: her video resmi test fold'larinda tam bir kere olmali")
    videos = {vid: MBVideo(vid, root, spec, config) for vid in sorted(all_ids)}
    rates = sorted(set(v.native_hz for v in videos.values()))
    if len(rates) != 1:
        raise ValueError(f"{spec.name}: veri setinde karisik feature hizlari {rates}; extraction'i dogrulayin")
    print(f"[veri] {spec.name}: {len(videos)} video, {spec.folds} fold | "
          f"native={rates[0]:g} Hz -> {config.target_hz:g} Hz SECIM | averaging=YOK", flush=True)
    return folds, videos


def mb_get_mamba_class():
    try:
        from mamba_ssm import Mamba
        return Mamba
    except Exception as exc:
        raise RuntimeError("Gercek mamba_ssm Mamba-1 yuklenemedi. Dosya basindaki CUDA kurulumuna bakin.") from exc


def mb_runtime_preflight(config):
    if not config.lambda_grid or any(not math.isfinite(lam) or lam < 0 for lam in config.lambda_grid):
        raise ValueError("lambda_grid bos olamaz ve sonlu, negatif olmayan sayilar icermeli")
    if not torch.cuda.is_available():
        raise RuntimeError("Colab Runtime > Change runtime type > GPU secin. CPU fallback YOK.")
    scan = mb_get_mamba_class()(d_model=64, d_state=16, d_conv=4, expand=1).cuda()
    try:
        x = torch.randn(1, 32, 64, device="cuda", requires_grad=True)
        y = scan(x)
        y.square().mean().backward()
        torch.cuda.synchronize()
        if not torch.isfinite(y).all() or x.grad is None or not torch.isfinite(x.grad).all():
            raise RuntimeError("Mamba forward/backward sonlu degil")
    except Exception as exc:
        raise RuntimeError("Mamba CUDA forward/backward basarisiz; sessiz CPU/TCN yedegi kullanilmaz.") from exc
    del scan, x, y
    torch.cuda.empty_cache()
    return torch.device("cuda")


def mb_context_dilations(spec, config):
    if config.context_dilations is not None:
        values = tuple(int(x) for x in config.context_dilations)
    else:
        # Onceki TCN'nin nominal zaman yaricapini koru; ilk filtre d=1 kalsin.
        # 15 Hz: GTEA=(1,7,13), 50Salads=(1,17,35), Breakfast=(1,11,23).
        total = max(3, int(math.floor(7 * config.target_hz / spec.old_decision_hz + .5)))
        second = max(1, int(round((total - 1) / 3)))
        values = (1, second, total - 1 - second)
    if len(values) != 3 or any(d < 1 for d in values):
        raise ValueError(f"3 pozitif context dilation bekleniyor: {values}")
    return values


class MBIsolatedScan(nn.Module):
    def __init__(self, config):
        super().__init__()
        dim = config.input_dim // config.n_groups
        self.norm = nn.LayerNorm(dim)
        self.scans = nn.ModuleList([
            mb_get_mamba_class()(d_model=dim, d_state=16, d_conv=4, expand=1)
            for _ in range(config.n_groups)])
        self.checkpoint_scans = config.checkpoint_scans
        for scan in self.scans:
            with torch.no_grad():
                scan.D.zero_()
            scan.D.requires_grad_(False)

    def forward(self, groups):
        normalized = self.norm(groups)  # (G,T,d), sadece son eksen
        outputs = []
        for i, scan in enumerate(self.scans):
            x = normalized[i:i+1].contiguous()
            if self.training and self.checkpoint_scans and torch.is_grad_enabled():
                y = mb_checkpoint(scan, x, use_reentrant=False)
            else:
                y = scan(x)
            outputs.append(y)
        return torch.cat(outputs, dim=0)


class MBTemporalContext(nn.Module):
    def __init__(self, dim, dilations, causal):
        super().__init__()
        self.dilations, self.causal = tuple(dilations), causal
        self.norms = nn.ModuleList([nn.LayerNorm(dim) for _ in dilations])
        self.convs = nn.ModuleList([nn.Conv1d(dim, dim, 3, dilation=d) for d in dilations])

    def forward(self, z):
        for norm, conv, d in zip(self.norms, self.convs, self.dilations):
            x = norm(z).T.unsqueeze(0)
            x = F.pad(x, (2*d, 0) if self.causal else (d, d))
            z = z + F.silu(conv(x)[0].T)
        return z


class MamboaV2(nn.Module):
    def __init__(self, spec, config, mode="full"):
        super().__init__()
        if mode not in ("no_tcn", "full"):
            raise ValueError(mode)
        if config.input_dim % config.n_groups:
            raise ValueError("input_dim n_groups'a tam bolunmeli; 2048 icin 36 kullanilamaz")
        self.mode, self.config = mode, config
        self.group_dim = config.input_dim // config.n_groups
        self.scan = MBIsolatedScan(config)
        self.fuse = nn.Linear(config.input_dim, config.context_dim)
        self.direct_head = nn.Linear(config.context_dim, 1)
        self.direct_head.requires_grad_(False)
        if mode == "full":
            self.context = MBTemporalContext(config.context_dim, mb_context_dilations(spec, config),
                                             config.context_causal)
            self.context_head = nn.Sequential(nn.Linear(config.context_dim, config.context_dim),
                                               nn.SiLU(), nn.Linear(config.context_dim, 1))
        else:
            self.context = None
            self.context_head = None

    def forward(self, features):
        if features.ndim != 2 or features.shape[1] != self.config.input_dim:
            raise ValueError(f"Girdi (T,{self.config.input_dim}) olmali: {tuple(features.shape)}")
        t = features.shape[0]
        groups = torch.stack([features[:, g*self.group_dim:(g+1)*self.group_dim]
                              for g in range(self.config.n_groups)], dim=0)
        scanned = self.scan(groups)
        joined = scanned.permute(1, 0, 2).contiguous().reshape(t, self.config.input_dim)
        z = self.fuse(joined)  # L2 norm veya noise-cleaner yok
        direct = self.direct_head(z).squeeze(-1)
        context = (self.context_head(self.context(z)).squeeze(-1)
                   if self.context is not None else None)
        return {"direct": direct, "context": context,
                "logit": direct if context is None else context}

    def init_prior(self, gaussian_mass):
        # Gaussian hedef kutlesine gore baslat; olay/adim yogunlugu ile karistirma.
        p = float(np.clip(gaussian_mass, 1e-4, 1-1e-4))
        with torch.no_grad():
            self.direct_head.bias.fill_(math.log(p / (1-p)))
            if self.context_head is not None:
                self.context_head[-1].bias.fill_(math.log(p / (1-p)))

    def trainable_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def mb_gaussian_target(times, gt, sigma_sec):
    """En yakin GT mesafesiyle max Gaussian; T x N veya T x T matris olusturmaz."""
    times, gt = np.asarray(times, dtype=np.float64), np.sort(np.asarray(gt, dtype=np.float64))
    if sigma_sec <= 0:
        raise ValueError("sigma_sec > 0 olmali")
    if not len(gt):
        return np.zeros(len(times), dtype=np.float32)
    right = np.searchsorted(gt, times)
    dleft = np.abs(times - gt[np.clip(right-1, 0, len(gt)-1)])
    dright = np.abs(times - gt[np.clip(right, 0, len(gt)-1)])
    return np.exp(-.5 * (np.minimum(dleft, dright) / sigma_sec)**2).astype(np.float32)


def mb_loss(output, target, config, mode):
    if mode != 'full': raise ValueError('Nihai model sadece FULL single context loss kullanir')
    weights = 1. + config.positive_alpha * target
    value = (F.binary_cross_entropy_with_logits(output['context'].float(), target, reduction='none') * weights).sum()/weights.sum()
    return value, dict(direct=0., context=float(value.detach()))


def mb_match(pred, gt, tolerance):
    """Max cardinality, sonra min toplam |hata|. Eslesen (pred_i,gt_i) listesi."""
    pred, gt = np.asarray(pred, float), np.asarray(gt, float)
    if not len(pred) or not len(gt):
        return []
    distances = np.abs(gt[:, None] - pred[None, :])
    # Bir eksik eslesmenin cezasi tum gecerli hatalarin toplamindan buyuk.
    penalty = (min(len(pred), len(gt)) + 1) * (tolerance + 1.)
    costs = np.full((len(gt), len(pred) + len(gt)), penalty, dtype=np.float64)
    costs[:, :len(pred)] = np.where(distances <= tolerance + 1e-9, distances, 2*penalty)
    rows, cols = linear_sum_assignment(costs)
    return [(int(c), int(r)) for r, c in zip(rows, cols)
            if c < len(pred) and distances[r, c] <= tolerance + 1e-9]


def mb_normalize_score(logits):
    x = np.asarray(logits, dtype=np.float64)
    if x.ndim != 1 or not np.isfinite(x).all():
        raise ValueError("Decoder skoru sonlu 1D olmali")
    sd = float(np.std(x, ddof=1)) if len(x) > 1 else 0.
    return (x - x.mean()) / max(sd, .5) if len(x) else x.copy()


def mb_dp_indices(scores, lam, min_gap_steps):
    q = np.asarray(scores, dtype=np.float64)
    gap = max(1, int(min_gap_steps))
    values = np.zeros(len(q)+1, dtype=np.float64)
    take = np.zeros(len(q), dtype=bool)
    for i in range(len(q)):
        cut = values[max(0, i+1-gap)] + q[i] - lam
        take[i] = cut > values[i]
        values[i+1] = max(values[i], cut)
    selected, i = [], len(q)-1
    while i >= 0:
        if take[i]:
            selected.append(i)
            i -= gap
        else:
            i -= 1
    return list(reversed(selected))


def mb_decode(logits, times, duration, spec, config, lam, offset=0.):
    q = mb_normalize_score(logits)
    gap_sec = spec.min_gap_sec if config.min_gap_override_sec is None else config.min_gap_override_sec
    gap = max(1, int(math.ceil(gap_sec * config.target_hz - 1e-9)))
    indices = mb_dp_indices(q, lam, gap)
    predictions = []
    for i in indices:
        shift = 0.
        if config.refine and 0 < i < len(q)-1 and q[i] >= q[i-1] and q[i] >= q[i+1]:
            den = q[i-1] - 2*q[i] + q[i+1]
            if den < -1e-12:
                shift = float(np.clip(.5*(q[i-1]-q[i+1])/den, -.5, .5))
        predictions.append(float(np.clip(times[i] + shift/config.target_hz - offset, 0., duration)))
    return predictions


def mb_score_predictions(predictions, videos, spec):
    rows = []
    for vid, pred in predictions.items():
        v = videos[vid]
        row = {"vid": vid, "n_pred": len(pred), "n_gt": len(v.gt)}
        for tol in spec.tolerances:
            pairs = mb_match(pred, v.gt, tol)
            tp = len(pairs)
            p = tp / len(pred) if len(pred) else 0.
            r = tp / len(v.gt) if len(v.gt) else 0.
            f1 = 2*tp/(len(pred)+len(v.gt)) if len(pred)+len(v.gt) else 0.
            key = f"{tol:g}"
            row[f"F1@{key}"], row[f"P@{key}"], row[f"R@{key}"] = f1, p, r
            if tol == spec.tolerances[1]:
                row["signed_errors"] = [float(pred[i]-v.gt[j]) for i, j in pairs]
        rows.append(row)
    if not rows:
        raise ValueError("Bos degerlendirme")
    metrics = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]
               if k.startswith(("F1@", "P@", "R@"))}
    npred, ngt = sum(r["n_pred"] for r in rows), sum(r["n_gt"] for r in rows)
    errors = [e for r in rows for e in r["signed_errors"]]
    metrics.update(n_videos=len(rows), n_pred=npred, n_gt=ngt,
                   pred_gt=npred/ngt if ngt else float("nan"),
                   bias_sec=float(np.mean(errors)) if errors else float("nan"),
                   median_bias_sec=float(np.median(errors)) if errors else float("nan"))
    return {"metrics": metrics, "videos": rows}


def mb_evaluate_logits(logits, videos, spec, config, lam, offset=0.):
    pred = {vid: mb_decode(x, videos[vid].times, videos[vid].duration,
                           spec, config, lam, offset) for vid, x in logits.items()}
    result = mb_score_predictions(pred, videos, spec)
    result["predictions"] = pred
    return result


def mb_calibrate(logits, videos, spec, config):
    key = f"F1@{spec.tolerances[1]:g}"
    trials = []
    for lam in config.lambda_grid:
        report = mb_evaluate_logits(logits, videos, spec, config, lam)
        trials.append((report["metrics"][key], -abs(lam-config.lambda_reference), -lam, report))
    score, _, neg_lam, selected = max(trials, key=lambda x: x[:3])
    lam = -neg_lam
    if score <= 0:
        lam = config.lambda_reference
        selected = mb_evaluate_logits(logits, videos, spec, config, lam)
    signed = [e for r in selected["videos"] for e in r["signed_errors"]]
    offset = float(np.median(signed)) if signed else 0.
    # Sadece VAL matched pair medyani; test verisi bu fonksiyona girmemeli.
    return dict(lam=float(lam), offset_sec=offset, val_uncalibrated=selected["metrics"],
                lambda_trials=[{"lambda": -t[2], key: t[0]} for t in trials])


def mb_amp_context(config, device):
    if device.type != "cuda" or config.precision == "fp32":
        return nullcontext()
    if config.precision not in ("auto", "bf16"):
        raise ValueError("precision: auto/fp32/bf16")
    if torch.cuda.is_bf16_supported():
        return torch.autocast("cuda", dtype=torch.bfloat16)
    if config.precision == "bf16":
        raise RuntimeError("GPU BF16 desteklemiyor; precision='fp32' secin")
    return nullcontext()


@torch.inference_mode()
def mb_predict(model, ids, videos, config, device, shuffle_repeat=None):
    model.eval()
    logits, direct, restored = {}, {}, {}
    for vid in ids:
        x = videos[vid].features().to(device)
        perm = None
        if shuffle_repeat is not None:
            rng = np.random.default_rng(mb_stable_seed("shuffle", config.seed, videos[vid].spec.name,
                                                      vid, shuffle_repeat))
            perm = rng.permutation(len(x))
            x = x[torch.as_tensor(perm, device=device)]
        with mb_amp_context(config, device):
            output = model(x)
        main = output["logit"].float().cpu().numpy()
        if not np.isfinite(main).all():
            raise FloatingPointError(f"Inference NaN/Inf: {vid}")
        logits[vid] = main
        direct[vid] = output["direct"].float().cpu().numpy()
        if perm is not None:
            back = np.empty_like(main)
            back[perm] = main
            restored[vid] = back
    return logits, direct, restored


def mb_sigma(spec, config):
    return spec.sigma_sec if config.sigma_override_sec is None else config.sigma_override_sec


def mb_fold_fingerprint(spec, config, mode, ids, videos):
    value = dict(version=MB_VERSION, spec=asdict(spec), config=asdict(config), mode=mode,
                 ids=ids, video_signatures=[videos[v].signature for v in sorted(sum(ids.values(), []))])
    # Resume/log/cache ayarlari model/protokolu degistirmez.
    for k in ("resume", "log_every", "feature_cache_dir", "stage_features"):
        value["config"].pop(k)
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def mb_rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)


def mb_restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None:
        torch.cuda.set_rng_state_all(state["cuda"])




# Old baseline scorer: verbatim source bnd_seg.py implementation.
def boundary_prf(pred, gt, tol):
    """Tolerans penceresi içinde BİRE BİR eşleme: Precision, Recall, F1."""
    matched, tp = set(), 0
    for p in pred:
        best, best_d = None, tol
        for i, g in enumerate(gt):
            if i not in matched and abs(p - g) <= best_d:
                best, best_d = i, abs(p - g)
        if best is not None:
            matched.add(best)
            tp += 1
    prec = tp / len(pred) if pred else 0.0
    rec = tp / len(gt) if gt else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec > 0) else 0.0
    return prec, rec, f1

# ======================================================================
# ADAM50 USER SETTINGS — GTEA -> 50SALADS -> BREAKFAST, 26 training jobs
# ======================================================================
M5_VERSION = 'mamboa-adam50-baseline-recipe-v1'
M5_ROOTS = {'gtea':'./gtea','50salads':'./50salads','breakfast':'./breakfast'}
M5_ROOTS.update(_M5_RUNTIME_ROOTS)
M5_OUTPUT = str(Path(_M5_RUNTIME_OUTPUT)/'mamboa_adam50')
M5_ORDER = ('gtea','50salads','breakfast')
M5_SEEDS = (0,1)
M5_CONFIG = replace(MBTrainConfig(),epochs=50,weight_decay=0.,warmup_fraction=0.,
                    grad_clip=None,precision='fp32')


def m5_metric_policy(runtime_metric=None):
    """Detect actual old notebook scorer, or use supplied legacy implementation."""
    if runtime_metric is None:
        return 'legacy_greedy'
    probes=[([1.,2.],[0.,1.1],1.),([.1,.3],[0.,.2],.2),([],[],.5),
            ([0.,0.],[0.],.5),([.5,1.5],[0.,1.],.5)]
    matches={'legacy_greedy':True,'optimal_hungarian':True}
    for pred,gt,tol in probes:
        actual=tuple(runtime_metric(pred,gt,tol))
        pairs=mb_match(pred,gt,tol)
        tp=len(pairs)
        precision=tp/len(pred) if pred else 0.
        recall=tp/len(gt) if gt else 0.
        f1=2*tp/(len(pred)+len(gt)) if len(pred)+len(gt) else 0.
        for name,expected in [('legacy_greedy',boundary_prf(pred,gt,tol)),
                              ('optimal_hungarian',(precision,recall,f1))]:
            matches[name] &= bool(np.allclose(actual,expected,rtol=0.,atol=1e-12))
    accepted=[name for name,ok in matches.items() if ok]
    if len(accepted)!=1:
        raise RuntimeError('Runtime boundary_prf does not match supplied greedy or optimal scorer. No training started.')
    return accepted[0]


M5_PRIMARY_METRIC = m5_metric_policy(_M5_RUNTIME_METRIC)


def m5_score(predictions,videos,spec,policy='legacy_greedy'):
    if policy=='optimal_hungarian':
        return mb_score_predictions(predictions,videos,spec)
    if policy!='legacy_greedy':
        raise ValueError(policy)
    rows=[]
    nearest_errors=[]
    for vid,pred in predictions.items():
        gt=videos[vid].gt.tolist()
        row=dict(vid=vid,n_pred=len(pred),n_gt=len(gt))
        for tol in spec.tolerances:
            precision,recall,f1=boundary_prf(pred,gt,tol)
            row.update({f'P@{tol:g}':precision,f'R@{tol:g}':recall,f'F1@{tol:g}':f1})
        rows.append(row)
        nearest_errors.extend(min(pred,key=lambda x:abs(x-g))-g for g in gt if pred)
    if not rows:
        raise ValueError('Empty evaluation')
    metrics={key:float(np.mean([row[key] for row in rows])) for key in rows[0]
             if key.startswith(('P@','R@','F1@'))}
    npred=sum(row['n_pred'] for row in rows); ngt=sum(row['n_gt'] for row in rows)
    metrics.update(n_videos=len(rows),n_pred=npred,n_gt=ngt,pred_gt=npred/ngt if ngt else float('nan'),
        mean_n_pred=npred/len(rows),mean_n_gt=ngt/len(rows),
        signed_offset=float(np.median(nearest_errors)) if nearest_errors else float('nan'))
    return dict(metrics=metrics,videos=rows,predictions=predictions)


def m5_decode_logits(logits,videos,spec,config,lam):
    # Original full decoder unchanged: no cross-model decoder substitution.
    return {vid:mb_decode(value,videos[vid].times,videos[vid].duration,spec,config,lam,0.)
            for vid,value in logits.items()}


def m5_calibrate(logits,videos,spec,config,policy):
    """Exactly old baseline max(F1,param) tuning + nearest-GT residual median."""
    key=f'F1@{spec.tolerances[1]:g}'
    trials=[]
    for lam in config.lambda_grid:
        pred=m5_decode_logits(logits,videos,spec,config,lam)
        score=m5_score(pred,videos,spec,policy)['metrics'][key]
        trials.append((score,lam))
    _,lam=max(trials)  # Old baseline tie rule: larger parameter wins a tie.
    chosen=m5_decode_logits(logits,videos,spec,config,lam)
    errors=[min(pred,key=lambda x:abs(x-g))-g for vid,pred in chosen.items()
            for g in videos[vid].gt if pred]
    return dict(lam=float(lam),offset_sec=float(np.median(errors)) if errors else 0.,
                metric_policy=policy,offset_rule='median nearest-prediction residual for every validation GT; no tolerance gate',
                trials=[dict(lam=lam,F1=score) for score,lam in trials])


def m5_subtract_offset(predictions,offset):
    # Old run_protocol.py does NOT clip/snap after subtracting the offset.
    return {vid:[float(t-offset) for t in values] for vid,values in predictions.items()}


def m5_resolve_root(dataset):
    requested=M5_ROOTS[dataset]
    path=Path(requested).expanduser()
    for base in (Path.cwd(),*Path.cwd().parents):
        for candidate in (base/path,base/path/'data'):
            if all((candidate/d).is_dir() for d in ('features','groundTruth','splits')):
                return candidate.resolve()
        if path.is_absolute():
            break
    if path.is_absolute():
        raise FileNotFoundError(f'{dataset}: explicit root {path} not found')
    return mb_resolve_root(dataset)


def m5_data_preflight(root,spec,config):
    _,videos=mb_dataset_preflight(root,spec,replace(config,seed=0))
    by_seed={seed:{fold:mb_fold_ids(root,spec,fold,replace(config,seed=seed))
                    for fold in range(1,spec.folds+1)} for seed in M5_SEEDS}
    sample_rate=2 if spec.name=='50salads' else 1
    checked=[]
    for vid,v in videos.items():
        shape=v.signature['shape']
        nf=shape[1] if v.channels_first else shape[0]
        nl=v.signature['n_labels']
        old_n=min(nf,nl)
        old_indices=np.arange(0,old_n,sample_rate)
        if (not np.array_equal(v.source_indices,old_indices) or
                not math.isclose(v.native_hz,spec.label_hz,abs_tol=1e-9)):
            raise RuntimeError(f'{vid}: supplied old baseline loader and full.py do not sample the same input. '
                               f'features={nf}, labels={nl}, native_hz={v.native_hz}; no training started.')
        # Verify that old min(feature,label)-truncation did not lose any GT.
        labels=[s.strip() for s in (Path(root)/'groundTruth'/f'{vid}.txt').read_text(encoding='utf-8-sig').splitlines() if s.strip()]
        old_gt=np.array([(i-.5)/spec.label_hz for i in range(1,old_n) if labels[i]!=labels[i-1]],dtype=float)
        if old_gt.shape!=v.gt.shape or not np.allclose(old_gt,v.gt,rtol=0.,atol=1e-9):
            raise RuntimeError(f'{vid}: old baseline GT differs; no training started')
        checked.append(vid)
    return by_seed,videos,dict(all_videos_checked=len(checked),same_source_sample_indices=True,
        same_ground_truth=True,baseline_sample_rate=sample_rate,model_hz=config.target_hz)


def m5_source_hash():
    # Colab pasted cells have no __file__. Normalize authored code objects so
    # identical code also hashes identically after a cell is rerun/re-numbered.
    import types
    def normalize(value):
        if isinstance(value,types.CodeType):
            return dict(bytecode=value.co_code.hex(),constants=[normalize(x) for x in value.co_consts],
                        names=value.co_names,varnames=value.co_varnames,argcount=value.co_argcount,
                        positional_only=value.co_posonlyargcount,kwonly=value.co_kwonlyargcount,
                        freevars=value.co_freevars,cellvars=value.co_cellvars,flags=value.co_flags)
        if isinstance(value,(tuple,list)):
            return [normalize(x) for x in value]
        if isinstance(value,bytes):
            return dict(bytes=value.hex())
        if isinstance(value,frozenset):
            return sorted([normalize(x) for x in value],key=repr)
        if value is None or isinstance(value,(str,int,float,bool)):
            return value
        return repr(value)
    code={name:normalize(fn.__code__) for name,fn in globals().items()
          if isinstance(fn,types.FunctionType) and (name.startswith(('mb_','m5_')) or name=='boundary_prf')}
    for cls in (MBVideo,MBIsolatedScan,MBTemporalContext,MamboaV2):
        code[cls.__name__]={name:normalize(fn.__code__) for name,fn in cls.__dict__.items()
                            if isinstance(fn,types.FunctionType)}
    return hashlib.sha256(json.dumps(code,sort_keys=True).encode()).hexdigest()


def m5_recipe(config,policy):
    return dict(optimizer='Adam',lr=config.learning_rate,weight_decay=0.,betas=[.9,.999],eps=1e-8,
        scheduler=None,warmup=False,gradient_clip=None,precision='fp32',epochs=config.epochs,patience=config.patience,
        val_fraction=config.val_fraction,split_seed='same as training seed',training_seeds=M5_SEEDS,
        checkpoint_rule='fixed FIRST native grid entry; tune ONLY after BEST checkpoint',
        checkpoint_lambda=config.lambda_grid[0],primary_metric=policy,
        calibration='validation median nearest-prediction signed residual; no post-offset clip/snap',
        architecture='unchanged original full G32/width64/D0 frozen',loss='original weighted Gaussian context BCE')


def m5_preflight(config,policy,device):
    state=mb_rng_state()
    try:
        model=MamboaV2(MB_DATASETS['50salads'],config,'full').to(device)
        if any(p.is_floating_point() and p.dtype!=torch.float32 for p in model.parameters()):
            raise RuntimeError('FP32 recipe requires float32 model parameters; reset the default torch dtype')
        total=sum(p.numel() for p in model.parameters())
        if (total,model.trainable_parameters())!=(695234,693121):
            raise RuntimeError('Original parameter count changed')
        if any(scan.D.requires_grad or torch.count_nonzero(scan.D) for scan in model.scan.scans):
            raise RuntimeError('Original frozen D=0 policy changed')
        optimizer=torch.optim.Adam(model.parameters(),lr=config.learning_rate)
        if type(optimizer) is not torch.optim.Adam or optimizer.defaults['weight_decay']!=0:
            raise RuntimeError('Optimizer does not match old baselines')
        x=torch.randn(13,2048,device=device)
        output=model(x)
        loss,_=mb_loss(output,torch.zeros(13,device=device),config,'full')
        loss.backward()
        bad=[name for name,p in model.named_parameters() if p.requires_grad and
             (p.grad is None or not torch.isfinite(p.grad).all())]
        if bad:
            raise RuntimeError(f'Model gradient failure: {bad}')
        report=dict(total_params=total,trainable_params=model.trainable_parameters(),
            all_trainable_gradients_finite=True,recipe=m5_recipe(config,policy),
            runtime_baseline_metric_detected=_M5_RUNTIME_METRIC is not None)
        print('[ADAM50 PREFLIGHT PASS]',json.dumps(report),flush=True)
        return report
    finally:
        mb_restore_rng(state)
        gc.collect()
        if device.type=='cuda':
            torch.cuda.empty_cache()


def m5_save_logits(path,val_logits,test_logits,videos):
    payload={}
    for stage,logits in [('val',val_logits),('test',test_logits)]:
        payload[stage+'_ids']=np.asarray(list(logits),dtype=str)
        for i,(vid,values) in enumerate(logits.items()):
            payload[f'{stage}_logits_{i}']=np.asarray(values,dtype=np.float32)
            payload[f'{stage}_times_{i}']=videos[vid].times
            payload[f'{stage}_gt_{i}']=videos[vid].gt
            payload[f'{stage}_duration_{i}']=np.asarray(videos[vid].duration)
    path=Path(path)
    temporary=path.with_name(path.name+'.tmp')
    with temporary.open('wb') as f:
        np.savez_compressed(f,**payload)
    os.replace(temporary,path)


def m5_train_fold(spec,config,fold,ids,videos,out,device,policy='legacy_greedy'):
    run=Path(out)/'full'/spec.name/f'seed{config.seed}_fold{fold}'
    run.mkdir(parents=True,exist_ok=True)
    payload=dict(version=M5_VERSION,source_sha256=m5_source_hash(),
        code_hash_method='normalized Python bytecode, excludes notebook filename/cell numbering',
        packages=mb_environment()['packages'],spec=asdict(spec),config=asdict(config),
        recipe=m5_recipe(config,policy),ids=ids,
        signatures={vid:videos[vid].signature for vid in sum(ids.values(),[])})
    fingerprint=hashlib.sha256(json.dumps(mb_jsonable(payload),sort_keys=True).encode()).hexdigest()
    if (run/'result.json').exists():
        result=json.loads((run/'result.json').read_text())
        if result['fingerprint']!=fingerprint:
            raise RuntimeError('Existing result protocol mismatch; choose a new Adam50 output folder')
        if config.resume:
            print('[completed; reused]',run,flush=True)
            return result
        raise FileExistsError(run/'result.json')
    mb_seed(config.seed)
    model=MamboaV2(spec,config,'full').to(device)
    target_cache={vid:mb_gaussian_target(videos[vid].times,videos[vid].gt,mb_sigma(spec,config)) for vid in ids['train']}
    target_mass=sum(float(t.sum()) for t in target_cache.values())/sum(map(len,target_cache.values()))
    model.init_prior(target_mass)
    # Exactly the old baseline optimizer API: one Adam parameter group,
    # no decay/scheduler/warmup/clipping. Frozen parameters remain frozen.
    optimizer=torch.optim.Adam(model.parameters(),lr=config.learning_rate)
    start,best_f1,best_epoch,bad,history=0,-np.inf,-1,0,[]
    if (run/'last.pt').exists():
        if not config.resume:
            raise FileExistsError(run/'last.pt')
        state=mb_load_own_checkpoint(run/'last.pt')
        if state['fingerprint']!=fingerprint:
            raise RuntimeError('Checkpoint is not this exact Adam50 run')
        model.load_state_dict(state['model']); optimizer.load_state_dict(state['optimizer'])
        start,best_f1,best_epoch,bad,history=(state['next_epoch'],state['best_f1'],state['best_epoch'],state['bad'],state['history'])
        mb_restore_rng(state['rng']); del state
    mb_write_json(run/'manifest.json',dict(**payload,fingerprint=fingerprint,environment=mb_environment(),
        actual_optimizer_defaults=optimizer.defaults,
        n_params=model.trainable_parameters(),n_params_total=sum(p.numel() for p in model.parameters()),
        gaussian_prior_mass=target_mass,split_seed=config.seed,context_dilations=mb_context_dilations(spec,config)))
    key=f'F1@{spec.tolerances[1]:g}'
    fixed_lambda=config.lambda_grid[0]
    print(f'[training Adam50] {spec.name} fold{fold} seed{config.seed}; '
          f'train/val/test={len(ids["train"])}/{len(ids["validation"])}/{len(ids["test"])}; '
          f'checkpoint_lambda={fixed_lambda}; metric={policy}',flush=True)
    for epoch in range(start,config.epochs):
        if bad>=config.patience:
            break
        model.train(); total=0.; begin=time.perf_counter()
        order=np.random.RandomState(config.seed+epoch).permutation(len(ids['train']))
        for i in order:
            vid=ids['train'][i]
            features=videos[vid].features().to(device)
            target=torch.as_tensor(target_cache[vid],device=device)
            optimizer.zero_grad(set_to_none=True)
            try:
                output=model(features)  # FP32; no autocast.
                loss,_=mb_loss(output,target,config,'full')
                if not torch.isfinite(loss):
                    raise FloatingPointError(f'Nonfinite loss: {vid}')
                loss.backward()
                optimizer.step()
            except torch.cuda.OutOfMemoryError as exc:
                raise RuntimeError(f'FP32 full-video OOM: {vid}, T={len(features)}. No crop/pool/mixed-precision '
                                   'fallback applied. last.pt resumes from the last complete epoch on a larger GPU.') from exc
            total+=float(loss.detach())
            del output,loss,features,target
        val_logits,_,_=mb_predict(model,ids['validation'],videos,config,device)
        pred=m5_decode_logits(val_logits,videos,spec,config,fixed_lambda)
        validation=m5_score(pred,videos,spec,policy)
        current=validation['metrics'][key]
        record=dict(epoch=epoch+1,seconds=time.perf_counter()-begin,loss=total/len(order),
            validation=validation['metrics'],validation_lambda=fixed_lambda,lr=optimizer.param_groups[0]['lr'])
        history.append(record)
        if current>best_f1:
            best_f1,best_epoch,bad=current,epoch+1,0
            mb_save_torch(run/'best.pt',dict(fingerprint=fingerprint,epoch=best_epoch,val_f1=best_f1,
                model={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}))
        else:
            bad+=1
        mb_save_torch(run/'last.pt',dict(fingerprint=fingerprint,model=model.state_dict(),optimizer=optimizer.state_dict(),
            rng=mb_rng_state(),next_epoch=epoch+1,best_f1=best_f1,best_epoch=best_epoch,bad=bad,history=history))
        mb_write_json(run/'history.json',history)
        print(f'  e{epoch+1:02d} loss={record["loss"]:.5f} val {key}={current:.5f} '
              f'best={best_f1:.5f} bad={bad}/{config.patience} lr={record["lr"]:.6g}',flush=True)
    best=mb_load_own_checkpoint(run/'best.pt')
    if best['fingerprint']!=fingerprint:
        raise RuntimeError('Best checkpoint mismatch')
    model.load_state_dict(best['model'])
    val_logits,_,_=mb_predict(model,ids['validation'],videos,config,device)
    calibration=m5_calibrate(val_logits,videos,spec,config,policy)
    mb_write_json(run/'calibration.json',calibration)
    native_cal=mb_calibrate(val_logits,videos,spec,config)  # validation-only diagnostic
    # Test is never read during training, checkpointing, or decoder selection.
    test_logits,_,_=mb_predict(model,ids['test'],videos,config,device)
    m5_save_logits(run/'best_logits.npz',val_logits,test_logits,videos)
    raw_pred=m5_decode_logits(test_logits,videos,spec,config,config.lambda_reference)
    tuned_pred=m5_decode_logits(test_logits,videos,spec,config,calibration['lam'])
    clean=m5_subtract_offset(tuned_pred,calibration['offset_sec'])
    raw=m5_score(raw_pred,videos,spec,policy)
    tuned=m5_score(tuned_pred,videos,spec,policy)
    calibrated=m5_score(clean,videos,spec,policy)
    # Independent metric diagnostic, SAME checkpoint and SAME predictions.
    hungarian_same_predictions=m5_score(clean,videos,spec,'optimal_hungarian')
    # Also preserve the original full.py calibration rule as an audit only.
    native_pred={vid:mb_decode(value,videos[vid].times,videos[vid].duration,spec,config,
                              native_cal['lam'],native_cal['offset_sec']) for vid,value in test_logits.items()}
    native=m5_score(native_pred,videos,spec,'optimal_hungarian')
    mb_write_json(run/'predictions.json',dict(raw=raw_pred,tuned=tuned_pred,calibrated=clean,
        native_full_diagnostic=native_pred,ground_truth={vid:videos[vid].gt for vid in ids['test']},
        validation_tuned=m5_decode_logits(val_logits,videos,spec,config,calibration['lam']),
        validation_gt={vid:videos[vid].gt for vid in ids['validation']},
        calibration=calibration,native_calibration=native_cal))
    shuffles=[]
    for repeat in range(config.shuffle_repeats):
        shuffled,_,restored=mb_predict(model,ids['test'],videos,config,device,repeat)
        reports={}
        for label,values in [('timeline',shuffled),('restored',restored)]:
            decoded=m5_decode_logits(values,videos,spec,config,calibration['lam'])
            reports[label]=m5_score(m5_subtract_offset(decoded,calibration['offset_sec']),videos,spec,policy)
        chance={}
        for vid in ids['test']:
            rng=np.random.default_rng(mb_stable_seed('chance',config.seed,spec.name,vid,repeat))
            chance[vid]=np.sort(rng.uniform(0.,videos[vid].duration,len(clean[vid]))).tolist()
        reports['count_matched_random']=m5_score(chance,videos,spec,policy)
        reports['repeat']=repeat
        shuffles.append(reports)
    result=dict(dataset=spec.name,mode='full',seed=config.seed,fold=fold,fingerprint=fingerprint,
        recipe=m5_recipe(config,policy),primary_metric_policy=policy,
        n_params=model.trainable_parameters(),n_params_total=sum(p.numel() for p in model.parameters()),
        best_epoch=best['epoch'],stopping_epoch=history[-1]['epoch'],
        reached_epoch_limit=history[-1]['epoch']==config.epochs,stopped_by_patience=bad>=config.patience,
        checkpoint_f1=best_f1,calibration=calibration,raw=raw,tuned_uncalibrated=tuned,
        calibrated=calibrated,tuned_calib=calibrated,
        hungarian_same_predictions=hungarian_same_predictions,
        native_full_diagnostic=dict(calibration=native_cal,report=native),shuffle=shuffles,
        manifest=str(run/'manifest.json'),logits=str(run/'best_logits.npz'),predictions=str(run/'predictions.json'))
    mb_write_json(run/'result.json',result)
    print(f'[TEST Adam50 PRIMARY] {spec.name} fold{fold} seed{config.seed} {key}={calibrated["metrics"][key]:.5f}; '
          f'best_epoch={best["epoch"]}, stop={history[-1]["epoch"]}',flush=True)
    del model,optimizer,best,target_cache
    gc.collect()
    if device.type=='cuda':
        torch.cuda.empty_cache()
    return mb_jsonable(result)


def m5_rescore_saved(run_dir,policy='legacy_greedy'):
    """Audit/recompute from saved logits, without a model, CUDA or retraining.

    Writes a SEPARATE diagnostic file; never overwrites the primary result.
    Do not choose the paper's primary policy by looking at test scores.
    """
    from types import SimpleNamespace
    run=Path(run_dir)
    manifest=json.loads((run/'manifest.json').read_text())
    spec=MBDatasetSpec(**manifest['spec'])
    config=MBTrainConfig(**manifest['config'])
    videos={}; values={}
    with np.load(run/'best_logits.npz',allow_pickle=False) as saved:
        for stage in ('val','test'):
            logits={}
            for i,vid in enumerate(saved[stage+'_ids'].tolist()):
                videos[vid]=SimpleNamespace(times=saved[f'{stage}_times_{i}'].copy(),
                    gt=saved[f'{stage}_gt_{i}'].copy(),duration=float(saved[f'{stage}_duration_{i}']))
                logits[vid]=saved[f'{stage}_logits_{i}'].copy()
            values[stage]=logits
    calibration=m5_calibrate(values['val'],videos,spec,config,policy)
    predictions=m5_subtract_offset(m5_decode_logits(values['test'],videos,spec,config,calibration['lam']),calibration['offset_sec'])
    report=dict(audit_only=True,policy=policy,calibration=calibration,
        report=m5_score(predictions,videos,spec,policy),predictions=predictions,
        source_logits=str(run/'best_logits.npz'),training_performed=False)
    mb_write_json(run/('audit_rescore_'+policy+'.json'),report)
    return report


class M5Tee:
    def __init__(self,console,log):
        self.console,self.log=console,log
    def write(self,text):
        self.console.write(text); self.log.write(text); self.log.flush()
        return len(text)
    def flush(self):
        self.console.flush(); self.log.flush()


def m5_longest_shape_check(spec,config,videos,device):
    """Full FP32 forward/backward/Adam at max T, using zeros, NO test labels."""
    state=mb_rng_state()
    try:
        steps=max(len(v.times) for v in videos.values())
        model=MamboaV2(spec,config,'full').to(device)
        optimizer=torch.optim.Adam(model.parameters(),lr=config.learning_rate)
        x=torch.zeros(steps,config.input_dim,device=device)
        target=torch.zeros(steps,device=device)
        output=model(x)
        loss,_=mb_loss(output,target,config,'full')
        loss.backward(); optimizer.step()
        if not torch.isfinite(loss) or any(not torch.isfinite(p).all() for p in model.parameters()):
            raise RuntimeError('Nonfinite longest-sequence check')
        report=dict(dataset=spec.name,max_steps=steps,fp32_full_video_adam_step=True,
                    test_labels_used=False,checkpoint_scans=config.checkpoint_scans)
        print('[MAX LENGTH FP32 CHECK PASS]',report,flush=True)
        return report
    finally:
        mb_restore_rng(state)
        gc.collect()
        if device.type=='cuda':
            torch.cuda.empty_cache()


def m5_summary(rows,spec):
    expected={(seed,fold) for seed in M5_SEEDS for fold in range(1,spec.folds+1)}
    if len(rows)!=len(expected) or {(r['seed'],r['fold']) for r in rows}!=expected:
        raise RuntimeError('Incomplete official seed/fold matrix')
    summary=dict(dataset=spec.name,training_seeds=M5_SEEDS,split_rule='training seed also sets validation split',
        primary_metric_policy=rows[0]['primary_metric_policy'],std_ddof=0,
        std_note='ddof0 matches old run_protocol.py; ddof1 also provided; overlapping CV runs are not independent')
    for stage in ('raw','tuned_uncalibrated','calibrated','hungarian_same_predictions'):
        stats={}
        for key in rows[0][stage]['metrics']:
            if not key.startswith(('F1@','P@','R@')):
                continue
            values=np.array([r[stage]['metrics'][key] for r in rows])
            stats[key]=dict(mean=float(values.mean()),run_sd=float(values.std(ddof=0)),
                sample_sd_ddof1=float(values.std(ddof=1)),
                seed_means={str(seed):float(np.mean([r[stage]['metrics'][key] for r in rows if r['seed']==seed])) for seed in M5_SEEDS})
        summary[stage]=stats
    summary['convergence']=dict(epoch_cap=50,patience=10,stopping_epochs=[r['stopping_epoch'] for r in rows],
        best_epochs=[r['best_epoch'] for r in rows],reached_cap=sum(r['reached_epoch_limit'] for r in rows),
        stopped_by_patience=sum(r['stopped_by_patience'] for r in rows))
    print(f'\nMAMBOA ADAM50 — {spec.name.upper()} | {spec.folds} fold x 2 seed; metric={summary["primary_metric_policy"]}',flush=True)
    for stage,label in [('raw','raw'),('tuned_uncalibrated','validation tuned'),('calibrated','KIYAS NOKTASI: baseline-compatible calibrated, NO snapping')]:
        print(label+': '+'  '.join(f'F1@{tol:g}={summary[stage][f"F1@{tol:g}"]["mean"]:.4f}'
              f'+-{summary[stage][f"F1@{tol:g}"]["run_sd"]:.4f}' for tol in spec.tolerances),flush=True)
    print('[convergence]',summary['convergence'],flush=True)
    return summary


def m5_guard():
    if torch.is_autocast_enabled():
        raise RuntimeError('Leave the enclosing autocast context before running the FP32 Adam50 protocol')
    expected=replace(MBTrainConfig(),epochs=50,weight_decay=0.,warmup_fraction=0.,grad_clip=None,precision='fp32')
    ignored=('resume','log_every','feature_cache_dir','stage_features')
    actual=asdict(M5_CONFIG); ref=asdict(expected)
    for key in ignored:
        actual.pop(key); ref.pop(key)
    if actual!=ref or M5_SEEDS!=(0,1):
        raise ValueError('Adam50 matched recipe changed; training has not started')


def m5_run(selected='all'):
    from contextlib import redirect_stdout,redirect_stderr
    import traceback
    if selected not in ('all',*M5_ORDER):
        raise ValueError(selected)
    m5_guard()
    requested=[name for name in M5_ORDER if selected in ('all',name)]
    out=Path(M5_OUTPUT).expanduser()
    for mount in ('/gdrive','/content/drive'):
        if str(out).startswith(mount+'/') and not (Path(mount)/'MyDrive').is_dir():
            raise RuntimeError(f'Mount Drive first at {mount}')
    out.mkdir(parents=True,exist_ok=True)
    completed=[]; results={}
    expected=[dict(dataset=name,seed=seed,fold=fold) for name in requested for seed in M5_SEEDS
              for fold in range(1,MB_DATASETS[name].folds+1)]
    with (out/'session.log').open('a',encoding='utf-8',buffering=1) as log:
        with redirect_stdout(M5Tee(sys.stdout,log)),redirect_stderr(M5Tee(sys.stderr,log)):
            try:
                prepared={}; protocol=dict(version=M5_VERSION,source_sha256=m5_source_hash(),
                    recipe=m5_recipe(M5_CONFIG,M5_PRIMARY_METRIC),datasets={})
                # Validate ALL requested data/splits before the first training.
                for name in requested:
                    root=m5_resolve_root(name)
                    by_seed,videos,audit=m5_data_preflight(root,MB_DATASETS[name],M5_CONFIG)
                    prepared[name]=(by_seed,videos)
                    protocol['datasets'][name]=dict(root=str(root),ids_by_seed=by_seed,alignment=audit)
                path=out/('protocol_'+selected+'.json')
                json_protocol=mb_jsonable(protocol)
                if path.exists() and json.loads(path.read_text())!=json_protocol:
                    raise RuntimeError('Existing Adam50 protocol differs; use a fresh output folder')
                mb_write_json(path,json_protocol)
                device=mb_runtime_preflight(M5_CONFIG)
                checks=m5_preflight(M5_CONFIG,M5_PRIMARY_METRIC,device)
                checks['longest_sequences']=[m5_longest_shape_check(MB_DATASETS[name],M5_CONFIG,prepared[name][1],device) for name in requested]
                mb_write_json(out/'preflight_checks.json',checks)
                print('[plan]',len(expected),'training jobs; recipe=',m5_recipe(M5_CONFIG,M5_PRIMARY_METRIC),flush=True)
                for name in requested:
                    spec=MB_DATASETS[name]; by_seed,videos=prepared[name]
                    rows=[]
                    for seed in M5_SEEDS:
                        config=replace(M5_CONFIG,seed=seed)
                        for fold in range(1,spec.folds+1):
                            run=out/'full'/name/f'seed{seed}_fold{fold}'
                            run.mkdir(parents=True,exist_ok=True)
                            with (run/'raw.log').open('a',encoding='utf-8',buffering=1) as rawlog:
                                with redirect_stdout(M5Tee(sys.stdout,rawlog)),redirect_stderr(M5Tee(sys.stderr,rawlog)):
                                    try:
                                        row=m5_train_fold(spec,config,fold,by_seed[seed][fold],videos,out,device,M5_PRIMARY_METRIC)
                                    except BaseException:
                                        traceback.print_exc()
                                        raise
                            rows.append(row)
                            completed.append(dict(dataset=name,seed=seed,fold=fold))
                            mb_write_json(out/'full'/name/'partial_results.json',rows)
                    summary=m5_summary(rows,spec)
                    results[name]=dict(summary=summary,rows=rows)
                    mb_write_json(out/'full'/name/'summary.json',summary)
                    flat=[]; epochs=[]
                    for row in rows:
                        flat.append(dict(dataset=name,seed=row['seed'],fold=row['fold'],best_epoch=row['best_epoch'],
                            stopping_epoch=row['stopping_epoch'],reached_cap=row['reached_epoch_limit'],
                            n_params=row['n_params'],n_params_total=row['n_params_total'],
                            **row['calibrated']['metrics']))
                        run=out/'full'/name/f'seed{row["seed"]}_fold{row["fold"]}'
                        for record in json.loads((run/'history.json').read_text()):
                            epochs.append(dict(dataset=name,seed=row['seed'],fold=row['fold'],
                                epoch=record['epoch'],loss=record['loss'],seconds=record['seconds'],lr=record['lr'],
                                **record['validation']))
                    for filename,data in [('fold_results.csv',flat),('epoch_history.csv',epochs)]:
                        with (out/'full'/name/filename).open('w',newline='',encoding='utf-8') as f:
                            writer=csv.DictWriter(f,fieldnames=sorted({k for r in data for k in r}))
                            writer.writeheader(); writer.writerows(data)
                    mb_write_json(out/'all_results.json',results)
            except BaseException:
                traceback.print_exc()
                raise
            finally:
                status=dict(requested=requested,planned=len(expected),completed=completed,
                    not_completed=[job for job in expected if job not in completed],
                    output_files=sorted(str(p) for p in out.rglob('*') if p.is_file()))
                mb_write_json(out/'execution_status.json',status)
                print(f'[execution status] completed={len(completed)}/{len(expected)}; '
                      f'details={out/"execution_status.json"}',flush=True)
    return results


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description='MAMBOA original architecture, matched old baseline Adam/50 recipe')
    parser.add_argument('--dataset',choices=('all',*M5_ORDER),default='all')
    args=parser.parse_args([] if 'ipykernel' in sys.modules else None)
    M5_RESULTS=m5_run(args.dataset)
