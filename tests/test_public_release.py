import contextlib
import io
import json
import subprocess
import sys
import tempfile
import unittest
import warnings
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from oracle import MambaDouble
from data.dataset import load_dataset_split, load_protocol_data
from experiments import offline_main as a, offline_ablation as b
from evaluation.matching import compute_boundary_prf

torch.set_num_threads(1)
ROOT=Path(__file__).resolve().parents[1]


def fixture(root):
    for name in ('features','groundTruth','splits'): (root/name).mkdir()
    ids=[f'video{i}' for i in range(28)]
    for i,vid in enumerate(ids):
        x=np.arange(45*2048,dtype=np.float32).reshape(45,2048)/10000
        np.save(root/'features'/f'{vid}.npy',x.T if i%2 else x)
        (root/'groundTruth'/f'{vid}.txt').write_text('\n'.join(['a']*15+['b']*15+['a']*15))
    for fold in range(1,5):
        test=ids[(fold-1)*7:fold*7];train=[v for v in ids if v not in test]
        for name,values in [('train',train),('test',test)]:
            (root/'splits'/f'{name}.split{fold}.bundle').write_text('\n'.join(v+'.txt' for v in values))


class ReleaseTests(unittest.TestCase):
    def test_loader_and_complete_official_partition(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);fixture(root)
            with contextlib.redirect_stdout(io.StringIO()):
                _,by_seed,videos=load_protocol_data('gtea',root,'main')
            self.assertEqual(len(videos),28)
            self.assertNotEqual(by_seed[0][1]['validation'],by_seed[1][1]['validation'])
            _,fixed,_=load_protocol_data('gtea',root,'ablation')
            self.assertEqual(fixed[0],fixed[1])
            samples=load_dataset_split('gtea',root,split_kind='test')
            for v in samples:
                self.assertEqual(v.features.shape,(45,2048))
                np.testing.assert_allclose(v.times,np.arange(45)/15)
                np.testing.assert_allclose(v.ground_truth_boundaries,[14.5/15,29.5/15])
            # Duplicate/overlapping official bundles must fail, not generate mocks.
            (root/'splits/test.split1.bundle').write_text('video0.txt\nvideo0.txt')
            with self.assertRaises(ValueError):load_dataset_split('gtea',root,split_kind='test')

    def test_source_rate_subsampling_and_gt_midpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for d in ('features','groundTruth'): (root/d).mkdir()
            x=np.arange(60*2048,dtype=np.float32).reshape(60,2048)
            np.save(root/'features/v.npy',x)
            (root/'groundTruth/v.txt').write_text('\n'.join(['a']*30+['b']*30))
            from data.dataset import VideoSample
            v=VideoSample('v',root,'50salads')
            np.testing.assert_array_equal(v.features.numpy(),x[::2])
            np.testing.assert_allclose(v.times,np.arange(30)/15)
            self.assertEqual(v.ground_truth_boundaries,[29.5/30])
            np.save(root/'features/v.npy',x[:10])
            with self.assertRaises(ValueError):VideoSample('v',root,'50salads')

    def test_missing_data_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):load_dataset_split('gtea',tmp)

    def test_main_and_ablation_recipe_are_distinct(self):
        a.m5_guard()
        self.assertEqual((a.M5_CONFIG.epochs,a.M5_CONFIG.precision,a.M5_CONFIG.weight_decay),(50,'fp32',0.))
        self.assertEqual((b.MB_ABL_CONFIG.epochs,b.MB_ABL_CONFIG.precision,b.MB_ABL_CONFIG.weight_decay),(60,'auto',.01))
        self.assertFalse(b.MB_ABL_AUTORUN)
        self.assertEqual(a.M5_PRIMARY_METRIC,'legacy_greedy')
        self.assertEqual(b.MB_SPLIT_SEED,0)

    def test_assignment_cardinality_and_greedy_disagree(self):
        greedy=compute_boundary_prf([1.,2.],[0.,1.1],1.,'legacy_greedy')
        optimal=compute_boundary_prf([1.,2.],[0.,1.1],1.,'optimal_assignment')
        self.assertEqual((greedy['tp'],optimal['tp']),(1,2))
        for policy in ('legacy_greedy','optimal_assignment','streaming_absolute_cost'):
            self.assertEqual(compute_boundary_prf([],[],1.,policy)['f1'],0.)

    def test_calibration_rules_are_not_interchangeable(self):
        # Validation references far from all predictions enter nearest-GT offset,
        # but are excluded by tolerance-matched calibration.
        from types import SimpleNamespace
        config=replace(a.M5_CONFIG,lambda_grid=(0.,))
        videos={'v':SimpleNamespace(gt=np.array([1.,100.]))}
        report={'metrics':{'F1@1':.5},'videos':[{'signed_errors':[.2]}]}
        with patch.object(a,'m5_decode_logits',return_value={'v':[1.2]}),patch.object(a,'m5_score',return_value=report):
            main=a.m5_calibrate({},videos,a.MB_DATASETS['50salads'],config,'legacy_greedy')
        with patch.object(b,'mb_evaluate_logits',return_value=report):
            old=b.mb_calibrate({},videos,b.MB_DATASETS['50salads'],config)
        self.assertLess(main['offset_sec'],-40)
        self.assertAlmostEqual(old['offset_sec'],.2)

    def test_model_keys_counts_and_forward_share_reference(self):
        from models.full_pipeline import FullBoundaryModel
        with patch.object(a,'mb_get_mamba_class',return_value=MambaDouble):
            model=FullBoundaryModel('50salads')
            ref=a.MamboaV2(a.MB_DATASETS['50salads'],a.M5_CONFIG,'full')
            self.assertEqual(sum(p.numel() for p in model.parameters()),695234)
            self.assertEqual(model.trainable_parameters(),693121)
            model.load_state_dict(ref.state_dict())
            x=torch.randn(7,2048)
            torch.testing.assert_close(model(x)['logits'],ref(x)['logit'],rtol=0,atol=0)
            self.assertTrue(all(not scan.D.requires_grad and not scan.D.any() for scan in model.scan.scans))

    def test_fallback_is_rank_one_and_streaming_buffers_are_bounded(self):
        from models import groupwise_mamba as gm
        from models.streaming_pipeline import StreamingBoundaryModel,StreamingDetector
        with patch.object(gm,'MAMBA_AVAILABLE',False),warnings.catch_warnings():
            warnings.simplefilter('ignore',RuntimeWarning)
            model=StreamingBoundaryModel(input_dim=16,n_groups=4,context_dim=8)
            self.assertEqual(model.scan.scans[0].dt_rank,1)
            detector=StreamingDetector(model,peak_threshold=-100,min_gap_sec=0.)
            memory=detector.get_memory_bytes()
            for _ in range(300):detector.push_feature(torch.randn(16))
            self.assertEqual(detector.get_memory_bytes(),memory)
            self.assertLessEqual(len(detector.score_history),3)
            self.assertEqual(detector.detected_boundaries,[])

    def test_dp_matches_authoritative_decoder(self):
        from decoding.dp_decoder import decode_boundaries_dp
        logits=np.random.RandomState(7).normal(size=100)
        times=np.arange(100)/15
        pred=decode_boundaries_dp(logits,times,100/15,penalty_lambda=.6,min_gap_sec=.8)
        ref=a.mb_decode(logits,times,100/15,a.MB_DATASETS['gtea'],a.M5_CONFIG,.6)
        np.testing.assert_array_equal(pred,ref)

    def test_cli_imports_and_no_implicit_mock_fallback(self):
        for name in ('train.py','eval.py','train_streaming.py'):
            result=subprocess.run([sys.executable,str(ROOT/name),'--help'],capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
        self.assertNotIn('Generating synthetic', (ROOT/'train.py').read_text())
        self.assertNotIn('Generating synthetic', (ROOT/'eval.py').read_text())

    def test_main_checkpoint_to_eval_interface(self):
        # One deliberately tiny CPU test-double run, not a paper experiment.
        import importlib
        evaluator=importlib.import_module('eval')
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'data';root.mkdir();fixture(root)
            with contextlib.redirect_stdout(io.StringIO()):
                _,by_seed,videos=load_protocol_data('gtea',root,'main')
            ids={'train':[by_seed[0][1]['train'][0]],
                 'validation':[by_seed[0][1]['validation'][0]],
                 'test':[by_seed[0][1]['test'][0]]}
            config=replace(a.M5_CONFIG,epochs=1,shuffle_repeats=0)
            out=Path(tmp)/'output'
            with patch.object(a,'mb_get_mamba_class',return_value=MambaDouble),contextlib.redirect_stdout(io.StringIO()):
                original=a.m5_train_fold(a.MB_DATASETS['gtea'],config,1,ids,videos,out,torch.device('cpu'))
                run=out/'full/gtea/seed0_fold1'
                argv=['eval.py','--dataset','gtea','--data_root',str(root),'--run_dir',str(run)]
                with patch.object(sys,'argv',argv),patch.object(evaluator,'load_protocol_data',return_value=(a,by_seed,videos)),patch.object(a,'mb_runtime_preflight',return_value=torch.device('cpu')):
                    evaluator.main()
            recomputed=json.loads((run/'evaluation_recomputed.json').read_text())
            self.assertEqual(recomputed['metrics'],original['calibrated']['metrics'])


if __name__=='__main__':unittest.main()
