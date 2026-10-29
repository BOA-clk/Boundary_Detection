"""Exercise real CLI bodies, including full matrices and real summary functions.

Data/GPU setup and training are controlled stubs. Trainer argument binding uses
the actual runner signatures. These tests perform no model training.
"""
import contextlib
import copy
import inspect
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import train
from experiments import offline_main as main_runner, offline_ablation as ablation_runner


def report_row(seed,fold):
    report={'metrics':{'F1@0.5':.4,'F1@1':.5,'F1@2':.6}}
    return dict(seed=seed,fold=fold,n_params=693121,n_params_total=695234,
        raw=copy.deepcopy(report),tuned_uncalibrated=copy.deepcopy(report),
        calibrated=copy.deepcopy(report),hungarian_same_predictions=copy.deepcopy(report),
        primary_metric_policy='legacy_greedy',best_epoch=1,stopping_epoch=11,
        reached_epoch_limit=False,stopped_by_patience=True,
        shuffle=[{kind:copy.deepcopy(report) for kind in ('timeline','restored','count_matched_random')}])


class TrainingCliTests(unittest.TestCase):
    def exercise(self,protocol,complete):
        runner=main_runner if protocol=='main' else ablation_runner
        # Creating the actual ablation trainer is safe; no training is started.
        signature=inspect.signature(main_runner.m5_train_fold if protocol=='main'
                                    else ablation_runner.mba_trainer('linear_head'))
        ids={'train':['tr'],'validation':['va'],'test':['te']}
        by_seed={s:{f:copy.deepcopy(ids) for f in range(1,6)} for s in (0,1)}
        videos={};calls=[]
        def controlled_trainer(*args,**kwargs):
            bound=signature.bind(*args,**kwargs)
            bound.apply_defaults();v=bound.arguments
            self.assertIsInstance(v['fold'],int)
            self.assertEqual(v['ids'],ids)
            self.assertIs(v['videos'],videos)
            if protocol=='ablation':
                self.assertEqual(v['mode'],'full')
                # Reproduce the real trainer's path operation; reversed mode/fold
                # must fail here before this regression could reach a summary.
                path=Path(v['output_dir'])/v['mode']/v['spec'].name
                self.assertEqual(path.parts[-3:],('linear_head','full','50salads'))
            else:
                self.assertEqual(v['policy'],'legacy_greedy')
            calls.append((v['config'].seed,v['fold']))
            return report_row(v['config'].seed,v['fold'])
        with tempfile.TemporaryDirectory() as tmp,contextlib.ExitStack() as stack:
            argv=['train.py','--protocol',protocol,'--dataset','50salads',
                  '--data_root','/controlled/test/data','--output_dir',tmp]
            if protocol=='ablation':argv+=['--variant','linear_head']
            if not complete:argv+=['--fold','3','--seed','1']
            stack.enter_context(patch.object(sys,'argv',argv))
            stack.enter_context(patch.object(train,'load_protocol_data',return_value=(runner,by_seed,videos)))
            stack.enter_context(patch.object(runner,'mb_runtime_preflight',return_value=torch.device('cpu')))
            if protocol=='main':
                stack.enter_context(patch.object(runner,'m5_preflight'))
                stack.enter_context(patch.object(runner,'m5_longest_shape_check'))
                stack.enter_context(patch.object(runner,'m5_train_fold',side_effect=controlled_trainer))
            else:
                stack.enter_context(patch.object(runner,'mba_check'))
                stack.enter_context(patch.object(runner,'mba_trainer',return_value=controlled_trainer))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            # Neither summary function is mocked: the real function must exist
            # and accept the complete matrix emitted by this CLI.
            train.main()
            base=Path(tmp)/protocol/('full' if protocol=='main' else 'linear_head')/'50salads'
            rows=json.loads((base/'selected_run_results.json').read_text())
            if complete:
                self.assertEqual(calls,[(s,f) for s in (0,1) for f in range(1,6)])
                self.assertEqual(len(rows),10)
                summary=json.loads((base/'summary.json').read_text())
                self.assertEqual(summary['calibrated']['F1@1']['mean'],.5)
            else:
                self.assertEqual(calls,[(1,3)])
                self.assertEqual(len(rows),1)
                self.assertFalse((base/'summary.json').exists())
            for seed,fold in calls:
                run=base/f'seed{seed}_fold{fold}' if protocol=='main' else base.parent/'full'/'50salads'/f'seed{seed}_fold{fold}'
                self.assertTrue((run/'raw.log').exists())

    def test_main_single_run_cli(self):self.exercise('main',False)
    def test_main_full_matrix_cli_and_summary(self):self.exercise('main',True)
    def test_ablation_single_run_cli_argument_binding(self):self.exercise('ablation',False)
    def test_ablation_full_matrix_cli_and_summary(self):self.exercise('ablation',True)


if __name__=='__main__':unittest.main()
