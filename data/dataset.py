"""Public loading CODE, not a dataset distribution.

root/features/<id>.npy; root/groundTruth/<id>.txt;
root/splits/{train,test}.split<fold>.bundle.
The same strict loader is used by the offline experiment runners.
"""
from dataclasses import replace
from pathlib import Path
from experiments import offline_main as main
DATASET_SPECS = main.MB_DATASETS

def get_runner(protocol):
    if protocol == 'main': return main
    if protocol == 'ablation':
        from experiments import offline_ablation
        return offline_ablation
    raise ValueError(protocol)

class VideoSample:
    """Lazy real-data view. Features: float32 [T,2048] sampled at 15 Hz.

    Transition GT is derived from source framewise labels, before subsampling.
    No truncation/interpolation/averaging or synthetic data substitution.
    """
    def __init__(self, video_id, data_root, dataset, config=None):
        self._video = main.MBVideo(video_id,Path(data_root),DATASET_SPECS[dataset],
            config or replace(main.M5_CONFIG,stage_features=False))
    @property
    def video_id(self): return self._video.vid
    @property
    def features(self): return self._video.features()
    @property
    def times(self): return self._video.times
    @property
    def ground_truth_boundaries(self): return self._video.gt.tolist()
    @property
    def duration_sec(self): return self._video.duration
    @property
    def n_frames(self): return len(self._video.times)

def load_dataset_split(dataset,data_root,fold=1,split_kind='train',seed=0,protocol='main'):
    """Val is held out from official train, never test. Paper runs also audit ALL folds.

    Main split seed = training seed; ablation split seed = 0.
    """
    if dataset not in DATASET_SPECS or not 1 <= fold <= DATASET_SPECS[dataset].folds:
        raise ValueError((dataset,fold))
    runner=get_runner(protocol)
    config=replace(runner.MBTrainConfig(),seed=seed if protocol=='main' else 0,stage_features=False)
    ids=runner.mb_fold_ids(Path(data_root),DATASET_SPECS[dataset],fold,config)
    kind={'val':'validation'}.get(split_kind,split_kind)
    if kind not in ids: raise ValueError(split_kind)
    return [VideoSample(vid,data_root,dataset,config) for vid in ids[kind]]

def load_protocol_data(dataset,data_root,protocol='main'):
    """Return (runner, ids_by_seed, videos), after official-count/rate/fold audit.

    Canonical videos use features(); VideoSample's convenience API instead
    exposes a features property. Both entry points consume this function.
    """
    runner=get_runner(protocol); root=Path(data_root).expanduser().resolve()
    # The original notebooks staged features under /content. Keep identical
    # sampling but use a writable local cache outside Colab as well.
    import tempfile
    cache=str(Path(tempfile.gettempdir())/'mamboa_feature_cache')
    if protocol=='main':runner.M5_CONFIG=replace(runner.M5_CONFIG,feature_cache_dir=cache)
    else:runner.MB_ABL_CONFIG=replace(runner.MB_ABL_CONFIG,feature_cache_dir=cache)
    spec=runner.MB_DATASETS[dataset]
    if protocol=='main':
        by_seed,videos,_=runner.m5_data_preflight(root,spec,runner.M5_CONFIG)
    else:
        folds,videos=runner.mb_dataset_preflight(root,spec,replace(runner.MB_ABL_CONFIG,seed=0))
        by_seed={seed:folds for seed in (0,1)}
    return runner,by_seed,videos
