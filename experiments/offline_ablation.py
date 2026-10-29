# -*- coding: utf-8 -*-
# STANDALONE: run this file only. 50Salads component ablations.
"""MAMBOA FINAL — TEK DOSYA / COLAB
Bu dosyanin TAMAMINI tek hucrede calistirin. Eski sistem/ablation importu veya
checkpoint gerekmez. Drive'i once drive.mount('/gdrive') ile baglayin.
Calisan Mamba CUDA kurulumunuz korunur. Gerekenler: torch, numpy, scipy, mamba_ssm.

MIMARI: 2048 I3D -> acik indekslerle 32x64 kanal grubu -> 32 bagimsiz temporal
Mamba-1 (LayerNorm, state16, conv4, expand1, D=0) -> concat -> Linear(2048,64)
-> mevcut TCN -> boundary head. Yalniz weighted Gaussian context BCE egitilir.
Eski deneyle baslangic esdegerligi icin 65-parametre direct head tutulur ama
dondurulur, kayba veya nihai karara katilmaz. Beklenen trainable=693121,total=695234.
Temporal averaging/pooling yok; 15Hz feature secimi. Kanallar spatial patch degildir.

SABIT PROTOKOL: her resmi fold icin split seed0 ile %%20 validation bir kere
ayrilir. AYNI train/validation/test kimlikleri ile training seed0 ve seed1.
Yalniz 50Salads: 7 mimari x 5 fold x 2 seed = 70 egitim.
50Salads sigma=1s/min-gap=4s; TCN dilation=(1,17,35), 15Hz.
Ek decoder kontrolleri ayni 10 FULL checkpoint'i kullanir; ek egitim yok.
Checkpoint: val-tuned uncalibrated primary F1. Sonra val median offset.
Decoder: video zscore floor=.5, lambda grid 0:.2:4, min-gap DP, parabolic refine.
Raw = fixed lambda2 + zscore. Main = val lambda + calibration. 1:1 matching.
Her kosuda 3 shuffle; video makro F1, run SD ve seed fold ortalamalari raporlanir.

YOLLAR ve MB_ABL_AUTORUN asagidaki 50SALADS COMPONENT ABLATIONS bolumunde.
MB_ABL_ROOT='./50salads'; features/groundTruth/splits beklenir.
protocol.json split listelerini kaydeder. best.pt/last.pt ve epoch resume vardir.
Tum videolar kullanilir, uzun videolar sessizce kirpilmaz. CPU fallback yok.
Bu offline modeldir; simetrik TCN ve video-global decoder streaming ispati degildir.
FLOPs teorik tahmindir; GPU hizi gercek video uzunluklarinda olculur.
"""

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


def mb_training_fold(spec, config, mode, fold, ids, videos, output_dir, device):
    """Test forward yalniz best checkpoint ve val kalibrasyonu tamamlandiktan sonra."""
    run_dir = Path(output_dir) / mode / spec.name / f"seed{config.seed}_fold{fold}"
    run_dir.mkdir(parents=True, exist_ok=True)
    fingerprint = mb_fold_fingerprint(spec, config, mode, ids, videos)
    result_path = run_dir / "result.json"
    if result_path.exists():
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result.get("fingerprint") != fingerprint:
            raise RuntimeError(f"{run_dir}: ayarlar/veri degismis. Yeni MB_OUTPUT_DIR kullanin.")
        if config.resume:
            print(f"[tamamlanmis] {mode} {spec.name} fold{fold}: tekrar kullaniliyor", flush=True)
            return result
        raise FileExistsError(f"{run_dir}: sonuc var; uzerine yazilmaz, yeni output_dir secin")
    mb_seed(config.seed)
    model = MamboaV2(spec, config, mode).to(device)
    target_cache = {vid: mb_gaussian_target(videos[vid].times, videos[vid].gt, mb_sigma(spec, config))
                    for vid in ids["train"]}
    target_mass = sum(float(t.sum()) for t in target_cache.values()) / sum(map(len, target_cache.values()))
    model.init_prior(target_mass)
    # Ortak scan/fuse/direct, iki modda ayni seed ve kurulus sirasiyla baslar.
    decay, no_decay = [], []
    for p in model.parameters():
        if p.requires_grad:
            (no_decay if getattr(p, "_no_weight_decay", False) else decay).append(p)
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": config.weight_decay},
                            {"params": no_decay, "weight_decay": 0.}], lr=config.learning_rate)
    total_steps = config.epochs * len(ids["train"])
    warm = max(1, round(config.warmup_fraction * total_steps))
    def schedule(step):
        if step < warm:
            return (step+1)/warm
        progress = min(1., (step-warm)/max(1, total_steps-warm))
        return .5*(1+math.cos(math.pi*progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, schedule)
    best_path, last_path = run_dir / "best.pt", run_dir / "last.pt"
    start_epoch, best_f1, best_epoch, bad, history = 0, -1., -1, 0, []
    if last_path.exists():
        if not config.resume:
            raise FileExistsError(f"{last_path}: yeni output_dir kullanin")
        state = mb_load_own_checkpoint(last_path)
        if state["fingerprint"] != fingerprint:
            raise RuntimeError(f"{last_path}: checkpoint/veri/ayar uyusmazligi")
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start_epoch, best_f1, best_epoch, bad, history = (
            state["next_epoch"], state["best_f1"], state["best_epoch"], state["bad"], state["history"])
        mb_restore_rng(state["rng"])
        del state
        print(f"[devam] {mode} {spec.name} fold{fold}: epoch {start_epoch+1}", flush=True)
    manifest = dict(fingerprint=fingerprint, version=MB_VERSION, mode=mode, spec=asdict(spec),
                    config=asdict(config), fold=fold, ids=ids, gaussian_prior_mass=target_mass,
                    n_params=model.trainable_parameters(), n_params_total=sum(p.numel() for p in model.parameters()),
                    objective='weighted_context_BCE_only', split_seed=0, environment=mb_environment(),
                    context_dilations=mb_context_dilations(spec, config) if mode == "full" else None,
                    timing={v: videos[v].signature for v in sum(ids.values(), [])})
    mb_write_json(run_dir / "manifest.json", manifest)
    key = f"F1@{spec.tolerances[1]:g}"
    print(f"[egitim] {mode} {spec.name} fold{fold} seed{config.seed} | "
          f"train/val/test={len(ids['train'])}/{len(ids['validation'])}/{len(ids['test'])} | "
          f"params={model.trainable_parameters():,}", flush=True)
    for epoch in range(start_epoch, config.epochs):
        if bad >= config.patience:
            break
        model.train()
        order = np.random.RandomState(config.seed+epoch).permutation(len(ids["train"]))
        totals = {"loss": 0., "direct": 0., "context": 0.}
        begin = time.perf_counter()
        for i in order:
            vid = ids["train"][i]
            features = videos[vid].features().to(device)
            target = torch.as_tensor(target_cache[vid], device=device)
            opt.zero_grad(set_to_none=True)
            try:
                with mb_amp_context(config, device):
                    output = model(features)
                    loss, pieces = mb_loss(output, target, config, mode)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Loss NaN/Inf: {vid}")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip, error_if_nonfinite=True)
                opt.step()
                scheduler.step()
            except torch.cuda.OutOfMemoryError as exc:
                raise RuntimeError(f"GPU bellegi yetmedi: {vid}, T={len(features)}. "
                                   "Video kirpilmadi/pooling uygulanmadi. Daha buyuk GPU veya "
                                   "checkpoint_scans=True kullanin; son tamamlanan epoch last.pt'de.") from exc
            totals["loss"] += float(loss.detach())
            for k, v in pieces.items():
                totals[k] += v
            del features, target, output, loss
        val_logits, _, _ = mb_predict(model, ids["validation"], videos, config, device)
        # Ham logit olcegi kucukken sabit lambda=2 tum epoch'lari F1=0 yapabilir.
        # Iki modda da ayni onceden tanimli grid, yalniz VALIDATION secimi.
        epoch_calibration = mb_calibrate(val_logits, videos, spec, config)
        val = mb_evaluate_logits(val_logits, videos, spec, config, epoch_calibration["lam"])
        current = val["metrics"][key]
        record = dict(epoch=epoch+1, seconds=time.perf_counter()-begin,
                      **{k: v/len(order) for k, v in totals.items()}, validation=val["metrics"],
                      validation_lambda=epoch_calibration["lam"])
        history.append(record)
        if current > best_f1:
            best_f1, best_epoch, bad = current, epoch+1, 0
            mb_save_torch(best_path, dict(fingerprint=fingerprint,
                          model={k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                          epoch=best_epoch, val_f1=best_f1))
        else:
            bad += 1
        mb_save_torch(last_path, dict(fingerprint=fingerprint, model=model.state_dict(),
                      optimizer=opt.state_dict(), scheduler=scheduler.state_dict(), rng=mb_rng_state(),
                      next_epoch=epoch+1, best_f1=best_f1, best_epoch=best_epoch, bad=bad, history=history))
        mb_write_json(run_dir / "history.json", history)
        if (epoch+1) % config.log_every == 0:
            print(f"  e{epoch+1:02d} loss={record['loss']:.4f} direct={record['direct']:.4f} "
                  f"context={record['context']:.4f} val {key}={current:.4f} "
                  f"best={best_f1:.4f} ({record['seconds']:.1f}s)", flush=True)
    if not best_path.exists():
        raise RuntimeError("Best checkpoint yok; epochs/patience ayarlarini kontrol edin")
    best = mb_load_own_checkpoint(best_path)
    model.load_state_dict(best["model"])
    if best["fingerprint"] != fingerprint:
        raise RuntimeError("Best checkpoint fingerprint uyusmuyor")
    val_logits, val_direct, _ = mb_predict(model, ids["validation"], videos, config, device)
    calibration = mb_calibrate(val_logits, videos, spec, config)
    # FULL yardimci dalin kendi validation kalibrasyonu; ana sonuc secimini etkilemez.
    direct_calibration = None
    mb_write_json(run_dir / "calibration.json", dict(main=calibration, direct=direct_calibration))
    test_logits, test_direct, _ = mb_predict(model, ids["test"], videos, config, device)
    raw = mb_evaluate_logits(test_logits, videos, spec, config, config.lambda_reference)
    tuned = mb_evaluate_logits(test_logits, videos, spec, config, calibration["lam"])
    clean = mb_evaluate_logits(test_logits, videos, spec, config, calibration["lam"], calibration["offset_sec"])
    direct_report = None
    clean_f1 = clean["metrics"][key]
    print(f"[test] {mode} {spec.name} fold{fold}: {key}={clean_f1:.4f} | "
          f"lambda={calibration['lam']:.2f}, offset={calibration['offset_sec']:+.4f}s", flush=True)
    shuffle_reports = []
    for repeat in range(config.shuffle_repeats):
        shuffled, _, restored = mb_predict(model, ids["test"], videos, config, device, repeat)
        timeline = mb_evaluate_logits(shuffled, videos, spec, config,
                                     calibration["lam"], calibration["offset_sec"])
        alignment = mb_evaluate_logits(restored, videos, spec, config,
                                      calibration["lam"], calibration["offset_sec"])
        # Sifir beklentisine karsi, clean ile ayni sayida uniform rastgele tahmin.
        # Min-gap zorlanmaz; bu sadece count-matched sans referansidir, model degil.
        random_pred = {}
        for vid in ids["test"]:
            rng = np.random.default_rng(mb_stable_seed("chance", config.seed, spec.name, vid, repeat))
            random_pred[vid] = np.sort(rng.uniform(0., videos[vid].duration,
                                                   len(clean["predictions"][vid]))).tolist()
        chance = mb_score_predictions(random_pred, videos, spec)
        shuffle_reports.append(dict(repeat=repeat, timeline=timeline, restored=alignment,
                                    count_matched_random=chance,
                                    delta_timeline=timeline["metrics"][key]-clean_f1,
                                    delta_restored=alignment["metrics"][key]-clean_f1))
        print(f"  [shuffle {repeat+1}] timeline={timeline['metrics'][key]:.4f} "
              f"restore-alignment={alignment['metrics'][key]:.4f} "
              f"random-count={chance['metrics'][key]:.4f}", flush=True)
    result = dict(fingerprint=fingerprint, mode=mode, dataset=spec.name, fold=fold,
                  seed=config.seed, n_params=model.trainable_parameters(),
                  n_params_total=sum(p.numel() for p in model.parameters()), best_epoch=best["epoch"],
                  calibration=calibration, direct_calibration=direct_calibration,
                  raw=raw, tuned_uncalibrated=tuned, calibrated=clean,
                  direct_only=direct_report, shuffle=shuffle_reports, manifest=str(run_dir / "manifest.json"))
    mb_write_json(result_path, result)
    del model, opt, scheduler, best, target_cache
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return mb_jsonable(result)



# ================================================================
# KULLANICI AYARLARI — TEK DOSYA, ONCEKI HUCRE/CHECKPOINT GEREKMEZ
# ================================================================
MB_ROOTS = {'gtea':'./gtea', '50salads':'./50salads', 'breakfast':'./breakfast'}
MB_OUTPUT_DIR = './MamboaV2_Final_ThreeDatasets'
MB_DATASET_ORDER = ('gtea','50salads','breakfast')
MB_SEEDS = (0,1)
MB_SPLIT_SEED = 0
MB_AUTORUN = False
MB_CONFIG = MBTrainConfig(target_hz=15., seed=0, epochs=60, patience=10,
    direct_weight=1., shuffle_repeats=3, checkpoint_scans=True, precision='auto', resume=True)


def mb_final_root(name, requested):
    path=Path(requested).expanduser()
    required=('features','groundTruth','splits')
    if all((path/d).is_dir() for d in required): return path.resolve()
    if path.is_absolute(): raise FileNotFoundError(f'{name}: {path}: features/groundTruth/splits yok')
    checked=[]
    for base in (Path.cwd(),*Path.cwd().parents):
        for candidate in (base/path,base/path/'data'):
            if candidate in checked: continue
            checked.append(candidate)
            if all((candidate/d).is_dir() for d in required): return candidate.resolve()
    raise FileNotFoundError(f'{name}: cwd={Path.cwd()}; aranan yollar={checked}. MB_ROOTS duzeltin.')


def mb_final_model_check(spec,config,device):
    rng=mb_rng_state()
    try:
        model=MamboaV2(spec,config,'full').to(device)
        x=torch.randn(9,config.input_dim,device=device)
        target=torch.zeros(9,device=device); target[4]=1.
        output=model(x)
        loss,_=mb_loss(output,target,config,'full'); loss.backward()
        if output['logit'].shape!=(9,): raise RuntimeError('Logit boyutu hatali')
        for module in (model.scan,model.fuse,model.context,model.context_head):
            grads=[p.grad for p in module.parameters() if p.requires_grad and p.grad is not None]
            if not grads or not all(torch.isfinite(g).all() for g in grads):
                raise RuntimeError('Ana dal gradient kontrolu basarisiz')
        if any(p.grad is not None for p in model.direct_head.parameters()):
            raise RuntimeError('Kullanilmayan direct head gradient almamali')
        n_total=sum(p.numel() for p in model.parameters())
        print(f'[model] {spec.name}: trainable={model.trainable_parameters():,}, total={n_total:,}',flush=True)
        del model,x,target,loss,output
    finally:
        mb_restore_rng(rng)
        gc.collect()
        if device.type=='cuda': torch.cuda.empty_cache()


def mb_final_summary(rows,spec):
    expected={(seed,fold) for seed in MB_SEEDS for fold in range(1,spec.folds+1)}
    if len(rows)!=len(expected) or {(r['seed'],r['fold']) for r in rows}!=expected:
        raise RuntimeError('Tam seed/fold sayisi tamamlanmamis')
    summary=dict(dataset=spec.name,seeds=list(MB_SEEDS),folds=spec.folds,split_seed=MB_SPLIT_SEED,
        uncertainty='run_sd is across seed/fold runs; seed_means are fold averages, no significance claim')
    print(f'\nMAMBOA FINAL — {spec.name.upper()} | {spec.folds} fold x {len(MB_SEEDS)} seed',flush=True)
    for stage in ('raw','tuned_uncalibrated','calibrated'):
        stats={}
        for key in rows[0][stage]['metrics']:
            if not key.startswith(('F1@','P@','R@')): continue
            vals=np.array([r[stage]['metrics'][key] for r in rows],float)
            stats[key]=dict(mean=float(vals.mean()),run_sd=float(vals.std(ddof=1)),
                seed_means={str(seed):float(np.mean([r[stage]['metrics'][key] for r in rows if r['seed']==seed])) for seed in MB_SEEDS})
        summary[stage]=stats
        print(stage+': '+'  '.join(f"F1@{tol:g}={stats[f'F1@{tol:g}']['mean']:.3f}+-{stats[f'F1@{tol:g}']['run_sd']:.3f}" for tol in spec.tolerances),flush=True)
    primary=f'F1@{spec.tolerances[1]:g}'
    summary['shuffle']={}
    for kind in ('timeline','restored','count_matched_random'):
        vals=[float(np.mean([s[kind]['metrics'][primary] for s in r['shuffle']])) for r in rows]
        summary['shuffle'][kind]=dict(mean=float(np.mean(vals)),run_sd=float(np.std(vals,ddof=1)))
        print(f'shuffle/{kind}: {np.mean(vals):.3f}+-{np.std(vals,ddof=1):.3f}',flush=True)
    summary['trainable_params']=sorted({r['n_params'] for r in rows})
    summary['total_params']=sorted({r['n_params_total'] for r in rows})
    print('Seed fold ortalamalari:',summary['calibrated'][primary]['seed_means'],flush=True)
    return summary


def mb_run_final():
    if MB_DATASET_ORDER!=('gtea','50salads','breakfast') or MB_SEEDS!=(0,1) or MB_SPLIT_SEED!=0:
        raise ValueError('Nihai protokol: gtea/50salads/breakfast, egitim seed0/1, split seed0')
    if MB_CONFIG.target_hz!=15. or MB_CONFIG.sigma_override_sec is not None or MB_CONFIG.min_gap_override_sec is not None:
        raise ValueError('Nihai protokol 15Hz ve asagida kayitli dataset sigma/min-gap degerlerini kullanir')
    out=Path(MB_OUTPUT_DIR).expanduser()
    for mount in ('/gdrive','/content/drive'):
        if str(out).startswith(mount+'/') and not (Path(mount)/'MyDrive').is_dir():
            raise RuntimeError(f'Drive bagli degil: drive.mount({mount!r})')
    configs={seed:replace(MB_CONFIG,seed=seed) for seed in MB_SEEDS}
    split_config=replace(MB_CONFIG,seed=MB_SPLIT_SEED)
    prepared={}; protocol=dict(version=MB_VERSION,split_seed=MB_SPLIT_SEED,training_seeds=list(MB_SEEDS),datasets={})
    for name in MB_DATASET_ORDER:
        spec=MB_DATASETS[name]; root=mb_final_root(name,MB_ROOTS[name])
        print(f'[dataset] {name}: {root}',flush=True)
        prepared[name]=mb_dataset_preflight(root,spec,split_config)
        folds,videos=prepared[name]
        protocol['datasets'][name]=dict(spec=asdict(spec),root=str(root),ids=folds,
            context_dilations=mb_context_dilations(spec,MB_CONFIG),
            configs={str(seed):asdict(config) for seed,config in configs.items()})
    out.mkdir(parents=True,exist_ok=True)
    protocol_path=out/'protocol.json'
    if protocol_path.exists():
        old=json.loads(protocol_path.read_text())
        # Source identity is additionally verified by per-fold fingerprints.
        for name in MB_DATASET_ORDER:
            if old.get('version')!=MB_VERSION or old['datasets'][name]['ids']!=mb_jsonable(protocol['datasets'][name]['ids']):
                raise RuntimeError('Kayitli protokol/split degismis; yeni output klasoru kullanin')
    mb_write_json(protocol_path,protocol)
    device=mb_runtime_preflight(MB_CONFIG)
    for name in MB_DATASET_ORDER: mb_final_model_check(MB_DATASETS[name],MB_CONFIG,device)
    print('3 dataset: (4+5+4) fold x 2 seed = 26 egitim. Her foldun split listesi iki seed icin ayni.',flush=True)
    results={}; flat=[]
    for name in MB_DATASET_ORDER:
        spec=MB_DATASETS[name]; folds,videos=prepared[name]; rows=[]
        for seed,config in configs.items():
            for fold in range(1,spec.folds+1):
                # Split is never regenerated with training seed1.
                result=mb_training_fold(spec,config,'full',fold,folds[fold],videos,out,device)
                rows.append(result)
                flat.append(dict(dataset=name,seed=seed,fold=fold,n_params=result['n_params'],
                    n_params_total=result['n_params_total'],**result['calibrated']['metrics']))
                mb_write_json(out/'partial_results.json',dict(completed=flat))
        summary=mb_final_summary(rows,spec)
        results[name]=dict(summary=summary,folds=rows)
        mb_write_json(out/'full'/name/'summary.json',summary)
        mb_write_json(out/'final_results.json',results)
        with (out/'final_fold_results.csv').open('w',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=sorted({k for r in flat for k in r}))
            writer.writeheader(); writer.writerows(flat)
    print(f'[bitti] {out}',flush=True)
    return results





# ======================================================================
# 50SALADS COMPONENT ABLATIONS — STANDALONE COLAB CELL
# The preceding code embeds the final training protocol, not an import.
# One component changes at a time. No test-based configuration selection.
# FLOPs: analytical estimates, NOT measured CUDA instruction counts.
# ======================================================================
MB_ABL_VERSION = "50salads-components-v1"
MB_ABL_ROOT = "./50salads"
MB_ABL_OUTPUT = "./Mamboa_Final_Component_Ablations"
MB_ABL_VARIANTS = (
    "full", "no_input_norm", "no_mamba", "no_tcn",
    "mean_fusion", "linear_head", "group_width_32",
)
MB_ABL_SEEDS = (0, 1)
MB_ABL_AUTORUN = False
MB_ABL_PROFILE_REPEATS = 3
MB_ABL_PROFILE_WARMUP = 5
MB_ABL_CONFIG = MBTrainConfig(
    target_hz=15., epochs=60, patience=10, precision="auto",
    checkpoint_scans=True, shuffle_repeats=3, resume=True,
)

MB_ABL_DESCRIPTIONS = {
    "full": "Final reference: 32 groups x 64 channels, single loss",
    "no_input_norm": "Remove only LayerNorm before isolated scans",
    "no_mamba": "Keep input LayerNorm; replace selective scans by identity",
    "no_tcn": "Replace temporal context with identity; keep original MLP and loss",
    "mean_fusion": "Replace concat + learned projection by group mean (64 channels)",
    "linear_head": "Replace 64->64->1 SiLU head with Linear(64,1)",
    "group_width_32": "64 isolated groups x 32 channels; same 2048 input and TCN",
}


class MBAblNoScan(nn.Module):
    def __init__(self, norm):
        super().__init__()
        self.norm = norm
    def forward(self, groups):
        return self.norm(groups)


class MBAblMean(nn.Module):
    def __init__(self, groups, width):
        super().__init__()
        self.groups, self.width = groups, width
    def forward(self, joined):
        return joined.reshape(len(joined), self.groups, self.width).mean(1)


_MBAblOriginalModel = MamboaV2
_MBAblOriginalFingerprint = mb_fold_fingerprint


class MBAblModel(_MBAblOriginalModel):
    def __init__(self, spec, config, mode="full", variant="full"):
        # Construct first, then remove modules: unchanged modules keep the
        # reference's initialization for the same seed (except group_width_32).
        super().__init__(spec, config, mode)
        self.variant = variant
        if variant == "no_input_norm":
            self.scan.norm = nn.Identity()
        elif variant == "no_mamba":
            self.scan = MBAblNoScan(self.scan.norm)
        elif variant == "no_tcn":
            self.context = nn.Identity()
        elif variant == "mean_fusion":
            if self.group_dim != config.context_dim:
                raise ValueError("mean_fusion requires group width == context_dim")
            self.fuse = MBAblMean(config.n_groups, self.group_dim)
        elif variant == "linear_head":
            self.context_head = nn.Sequential(nn.Linear(config.context_dim, 1))
        elif variant not in ("full", "group_width_32"):
            raise ValueError(variant)

    def forward(self, features):
        if features.ndim != 2 or features.shape[1] != self.config.input_dim:
            raise ValueError(f"Expected (T,{self.config.input_dim}), got {features.shape}")
        groups = torch.stack([
            features[:, g*self.group_dim:(g+1)*self.group_dim]
            for g in range(self.config.n_groups)], 0)
        scanned = self.scan(groups)
        joined = scanned.permute(1, 0, 2).contiguous().reshape(len(features), -1)
        z = self.fuse(joined)
        score = self.context_head(self.context(z)).squeeze(-1)
        # Frozen legacy direct head is kept only for reference initialization/
        # parameter accounting. It is neither evaluated nor optimized.
        return dict(logit=score, context=score, direct=score)


def mba_trainer(variant):
    import types
    namespace = dict(mb_training_fold.__globals__)
    namespace["MamboaV2"] = lambda spec, config, mode: MBAblModel(spec, config, mode, variant)
    def fingerprint(spec, config, mode, ids, videos):
        payload = dict(version=MB_ABL_VERSION, variant=variant,
            original=_MBAblOriginalFingerprint(spec, config, mode, ids, videos))
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    namespace["mb_fold_fingerprint"] = fingerprint
    namespace["MB_VERSION"] = MB_ABL_VERSION + "/" + variant
    return types.FunctionType(mb_training_fold.__code__, namespace,
                              mb_training_fold.__name__, mb_training_fold.__defaults__)


def mba_config(variant, seed):
    return replace(MB_ABL_CONFIG, seed=seed,
                   n_groups=64 if variant == "group_width_32" else 32)


def mba_check(variant, spec, config, device):
    state = mb_rng_state()
    try:
        mb_seed(0)
        model = MBAblModel(spec, config, variant=variant).to(device)
        x = torch.randn(19, config.input_dim, device=device)
        loss, _ = mb_loss(model(x), torch.zeros(19, device=device), config, "full")
        loss.backward()
        missing = [name for name, p in model.named_parameters()
                   if p.requires_grad and (p.grad is None or not torch.isfinite(p.grad).all())]
        if missing:
            raise RuntimeError(f"{variant}: invalid/missing gradients: {missing}")
        print(f"[shape/gradient OK] {variant}: {config.n_groups} x {model.group_dim}", flush=True)
    finally:
        mb_restore_rng(state)
        if device.type == "cuda":
            torch.cuda.empty_cache()


@torch.inference_mode()
def mba_flops(model, steps, config, device):
    # Explicit Mamba-1 analytical proxy:
    # projections 2*T*(d*2E + E*(r+2N) + r*E + E*d)
    # depthwise convolution 2*T*E*K
    # selective scan proxy 9*T*E*N, gate 1*T*E.
    # E=expanded width, N=state size, r=dt rank.
    # The 9 coefficient is a declared approximation, not a kernel measurement.
    # LayerNorm, SiLU/softplus/exp, bias, residual additions, memory traffic,
    # DP and feature extraction are omitted. Never label the sum "exact FLOPs".
    excluded = set()
    mamba_flops = 0
    formulas = []
    for scan in getattr(model.scan, "scans", []):
        excluded.update(id(m) for m in scan.modules())
        d, e, n, r, k = scan.d_model, scan.d_inner, scan.d_state, scan.dt_rank, scan.d_conv
        count = steps * (2*(d*2*e + e*(r+2*n) + r*e + e*d + e*k) + 9*e*n + e)
        mamba_flops += count
        formulas.append(dict(d=d, expanded=e, state=n, dt_rank=r, kernel=k))
    dense = [0]
    handles = []
    def linear_hook(module, inputs, output):
        dense[0] += 2 * output.numel() * module.in_features
    def conv_hook(module, inputs, output):
        dense[0] += 2 * output.numel() * (module.in_channels // module.groups) * module.kernel_size[0]
    for module in model.modules():
        if id(module) in excluded:
            continue
        if isinstance(module, nn.Linear):
            handles.append(module.register_forward_hook(linear_hook))
        elif isinstance(module, nn.Conv1d):
            handles.append(module.register_forward_hook(conv_hook))
    try:
        with mb_amp_context(config, device):
            model(torch.zeros(steps, config.input_dim, device=device))
    finally:
        for handle in handles:
            handle.remove()
    return dict(steps=steps, dense_conv_gflops=dense[0]/1e9,
        mamba_theoretical_proxy_gflops=mamba_flops/1e9,
        total_estimated_gflops=(dense[0]+mamba_flops)/1e9,
        status="ANALYTICAL_ESTIMATE_NOT_MEASURED",
        convention="multiply+add=2 FLOPs; selective scan proxy=9*T*E*N",
        exclusions="norm, nonlinearities except gate product, bias, residual, decoder, I/O, backbone",
        mamba_shapes=formulas)


@torch.inference_mode()
def mba_profile(variant, spec, config, videos, out, device):
    # Same checkpoint position for every variant, never best test-F1 selection.
    run = Path(out)/variant/"full"/spec.name/"seed0_fold1"
    checkpoint = mb_load_own_checkpoint(run/"best.pt")
    calibration = json.loads((run/"calibration.json").read_text())["main"]
    model = MBAblModel(spec, config, variant=variant).to(device).eval()
    model.load_state_dict(checkpoint["model"])
    del checkpoint
    records = []
    video_ids = sorted(videos)
    for j, vid in enumerate(video_ids):
        video = videos[vid]
        x = video.features().to(device)
        for _ in range(MB_ABL_PROFILE_WARMUP):
            with mb_amp_context(config, device):
                model(x)
        torch.cuda.synchronize(device)
        neural, pipeline = [], []
        for _ in range(MB_ABL_PROFILE_REPEATS):
            # Synchronized wall-clock includes Python launch overhead for 32/64
            # independently invoked scans. Features already resident on GPU.
            torch.cuda.synchronize(device)
            start = time.perf_counter()
            with mb_amp_context(config, device):
                output = model(x)["logit"]
            torch.cuda.synchronize(device)
            neural.append(1000*(time.perf_counter()-start))
            logits = output.float().cpu().numpy()
            mb_decode(logits, video.times, video.duration, spec, config,
                      calibration["lam"], calibration["offset_sec"])
            pipeline.append(1000*(time.perf_counter()-start))
        records.append(dict(video=vid, steps=len(x), duration_sec=video.duration,
            model_ms=float(np.median(neural)),
            model_transfer_decoder_ms=float(np.median(pipeline))))
        del x, output
        if (j+1) % 10 == 0:
            print(f"[speed] {variant}: {j+1}/{len(video_ids)} videos", flush=True)
    median_steps = int(round(np.median([r["steps"] for r in records])))
    flops = mba_flops(model, median_steps, config, device)
    common_flops = mba_flops(model, 1092, config, device)
    trainable = model.trainable_parameters()
    total = sum(p.numel() for p in model.parameters())
    inactive = sum(p.numel() for p in model.direct_head.parameters())
    result = dict(variant=variant, checkpoint="seed0_fold1/best.pt",
        trainable_params=trainable, total_params=total,
        inactive_legacy_head_params=inactive,
        inference_path_params=total-inactive,
        gpu=torch.cuda.get_device_name(device),
        precision="bf16" if config.precision != "fp32" and torch.cuda.is_bf16_supported() else "fp32",
        batch_size=1, feature_hz=config.target_hz,
        mean_model_ms=float(np.mean([r["model_ms"] for r in records])),
        median_model_ms=float(np.median([r["model_ms"] for r in records])),
        p95_model_ms=float(np.percentile([r["model_ms"] for r in records], 95)),
        mean_model_transfer_decoder_ms=float(np.mean([r["model_transfer_decoder_ms"] for r in records])),
        feature_steps_per_second=sum(r["steps"] for r in records)/sum(r["model_ms"]/1000 for r in records),
        mean_video_duration_sec=float(np.mean([r["duration_sec"] for r in records])),
        mean_feature_steps=float(np.mean([r["steps"] for r in records])),
        flops_median_length=flops, flops_common_1092=common_flops,
        warmup_per_video=MB_ABL_PROFILE_WARMUP, repeats=MB_ABL_PROFILE_REPEATS,
        timing_scope="model: GPU-resident I3D -> logits; pipeline adds logits D2H + CPU decoder; excludes feature extraction/loading/H2D",
        videos=records)
    mb_write_json(Path(out)/variant/"efficiency.json", result)
    del model
    torch.cuda.empty_cache()
    return result


def mba_decoder_controls(rows, videos, folds, spec, out, device):
    # No extra training: ten FULL checkpoints, official held-out test per run.
    # Removes decoder refinement / min-gap / offset separately.
    # For changed refinement or gap, retune lambda+offset on validation only.
    controls = {key: [] for key in ("no_offset", "no_refinement", "no_min_gap")}
    for row in rows:
        seed, fold = row["seed"], row["fold"]
        config = mba_config("full", seed)
        model = MBAblModel(spec, config, variant="full").to(device)
        path = Path(out)/"full"/"full"/spec.name/f"seed{seed}_fold{fold}"/"best.pt"
        model.load_state_dict(mb_load_own_checkpoint(path)["model"])
        ids = folds[fold]
        val, _, _ = mb_predict(model, ids["validation"], videos, config, device)
        test, _, _ = mb_predict(model, ids["test"], videos, config, device)
        for key in controls:
            cfg = replace(config, refine=False) if key == "no_refinement" else (
                replace(config, min_gap_override_sec=0.) if key == "no_min_gap" else config)
            cal = row["calibration"] if key == "no_offset" else mb_calibrate(val, videos, spec, cfg)
            report = mb_evaluate_logits(test, videos, spec, cfg, cal["lam"],
                                       0. if key == "no_offset" else cal["offset_sec"])
            controls[key].append(dict(seed=seed, fold=fold, metrics=report["metrics"], calibration=cal))
        del model
        torch.cuda.empty_cache()
    for key, rr in controls.items():
        vals = np.array([r["metrics"]["F1@1"] for r in rr])
        print(f"[decoder] {key}: F1@1={vals.mean():.4f}+-{vals.std(ddof=1):.4f}", flush=True)
    mb_write_json(Path(out)/"decoder_controls.json", controls)
    return controls


def mba_run():
    if MB_ABL_SEEDS != (0, 1):
        raise ValueError("Full protocol requires seeds (0,1)")
    if len(set(MB_ABL_VARIANTS)) != len(MB_ABL_VARIANTS) or set(MB_ABL_VARIANTS)-set(MB_ABL_DESCRIPTIONS):
        raise ValueError("Invalid/repeated variant")
    spec = MB_DATASETS["50salads"]
    if MB_ABL_CONFIG.input_dim != 2048 or MB_ABL_CONFIG.context_dim != 64 or MB_ABL_CONFIG.target_hz != 15:
        raise ValueError("This protocol uses input2048, context64, 15Hz")
    out = Path(MB_ABL_OUTPUT)
    if str(out).startswith("/gdrive/") and not Path("/gdrive/MyDrive").is_dir():
        raise RuntimeError("First mount Drive at /gdrive")
    root = mb_final_root("50salads", MB_ABL_ROOT)
    folds, videos = mb_dataset_preflight(root, spec, replace(MB_ABL_CONFIG, seed=0))
    protocol = mb_jsonable(dict(version=MB_ABL_VERSION, root=str(root), split_seed=0,
        ids=folds, seeds=MB_ABL_SEEDS, descriptions=MB_ABL_DESCRIPTIONS,
        configs={v:asdict(mba_config(v,0)) for v in MB_ABL_VARIANTS}))
    out.mkdir(parents=True, exist_ok=True)
    protocol_path = out/"ablation_protocol.json"
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
        raise RuntimeError("Protocol changed; use a new MB_ABL_OUTPUT folder")
    mb_write_json(protocol_path, protocol)
    device = mb_runtime_preflight(MB_ABL_CONFIG)
    for variant in MB_ABL_VARIANTS:
        mba_check(variant, spec, mba_config(variant,0), device)
    print(f"{len(MB_ABL_VARIANTS)} variants x 5 folds x 2 seeds = {10*len(MB_ABL_VARIANTS)} runs", flush=True)
    results, table = {}, []
    for variant in MB_ABL_VARIANTS:
        print(f"\n=== {variant}: {MB_ABL_DESCRIPTIONS[variant]} ===", flush=True)
        trainer = mba_trainer(variant)
        rows = []
        for seed in MB_ABL_SEEDS:
            for fold in range(1,6):
                rows.append(trainer(spec, mba_config(variant,seed), "full",
                                    fold, folds[fold], videos, out/variant, device))
                mb_write_json(out/variant/"partial_results.json", rows)
        summary = mb_final_summary(rows, spec)
        efficiency = mba_profile(variant, spec, mba_config(variant,0), videos, out, device)
        results[variant] = dict(summary=summary, rows=rows, efficiency=efficiency)
        entry = dict(variant=variant,
            f1_05=summary["calibrated"]["F1@0.5"]["mean"],
            f1_1=summary["calibrated"]["F1@1"]["mean"],
            f1_1_run_sd=summary["calibrated"]["F1@1"]["run_sd"],
            f1_2=summary["calibrated"]["F1@2"]["mean"],
            params_trainable=efficiency["trainable_params"],
            params_total=efficiency["total_params"],
            model_ms=efficiency["mean_model_ms"],
            model_decoder_ms=efficiency["mean_model_transfer_decoder_ms"],
            estimated_gflops_1092=efficiency["flops_common_1092"]["total_estimated_gflops"],
            mamba_proxy_gflops_1092=efficiency["flops_common_1092"]["mamba_theoretical_proxy_gflops"])
        table.append(entry)
        print("[RESULT]", json.dumps(entry), flush=True)
        mb_write_json(out/"all_results.json", results)
        with (out/"ablation_table.csv").open("w", newline="", encoding="utf-8") as f:
            writer=csv.DictWriter(f, fieldnames=list(entry))
            writer.writeheader(); writer.writerows(table)
    if "full" in results:
        baseline = {(r["seed"],r["fold"]):r["calibrated"]["metrics"]["F1@1"]
                    for r in results["full"]["rows"]}
        paired = {}
        for variant, item in results.items():
            delta = [r["calibrated"]["metrics"]["F1@1"]-baseline[r["seed"],r["fold"]]
                     for r in item["rows"]]
            paired[variant] = dict(mean=float(np.mean(delta)),
                run_sd=float(np.std(delta, ddof=1)),
                positive_runs=sum(d>0 for d in delta), deltas=delta,
                note="Overlapping CV training sets; descriptive, not independent significance samples")
        mb_write_json(out/"paired_vs_full.json", paired)
        mba_decoder_controls(results["full"]["rows"], videos, folds, spec, out, device)
    print("\nFINAL TABLE — FLOPs are estimates; times measured at real video lengths", flush=True)
    for entry in table:
        print(entry, flush=True)
    print(f"[done] {out}", flush=True)
    return results


if MB_ABL_AUTORUN:
    MB_COMPONENT_RESULTS = mba_run()
