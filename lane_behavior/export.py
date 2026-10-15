"""Export a positively validated mixed CD210 best without legacy reselection.

The selected immutable checkpoint must have a committed optimizer step and a
matching mixed promotion receipt. Existing ONNX tensor/parity and device-build
helpers are reused; legacy selection and release pointers are never called.
This command prepares local files only, without SSH or device activation.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, fields
import json
from pathlib import Path
import pickle
import shutil

import onnx
import torch

from lane_dataset.config import sha256_file, write_json
from lane_dataset.review_hold import assert_checkpoint_not_held, assert_episode_not_held
from lane_training.cd210_model import load_checkpoint
from lane_training.checkpoint_store import checkpoint_path, immutable_copy, immutable_json
from lane_release import export as legacy_export
from .mixed import (OBJECTIVE_VERSION,SELECTION_METRIC,Recipe,EpisodeRef,fingerprint,
    qualification,discover,open_episode,can_promote_mixed)


VERSION = 'validated_mixed_cd210_export_v1'
OBJECTIVE_SOURCES = tuple('lane_behavior/'+name+'.py' for name in (
    'mixed','training','loss','future_loss','loader','records','window','target','search'))+tuple(
    'lane_training/'+name+'.py' for name in ('centerline','checkpoint_store','adapters','cd210','rollout','targets'))


def _read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _same(a,b,message):
    if fingerprint(a)!=fingerprint(b):raise ValueError(message)


def select_validated_checkpoint(run):
    """Validate exact saved selection identity and live data holds; no forward pass.

    Scores are the immutable per-step heldout results, not a fresh legacy-only
    evaluation. Extra newly collected data cannot silently redefine this run's
    cohort, recipe, or winner.
    """
    run = Path(run).resolve()
    best_path = run/'best.json'
    if not best_path.exists():raise ValueError('no validated mixed best is available for export')
    best_bytes = best_path.read_bytes();best=json.loads(best_bytes)
    if (type(best.get('step')) is not int or best['step']<=0
            or best.get('objective_version')!=OBJECTIVE_VERSION
            or best.get('selection_metric')!=SELECTION_METRIC
            or best.get('kind')!='mixed_validated_candidate'):
        raise ValueError('export requires a positively stepped, validated mixed candidate')
    if not can_promote_mixed(best):
        raise ValueError('mixed candidate must improve stock without either heldout cohort regressing')
    checkpoint = checkpoint_path(run,best)
    if sha256_file(checkpoint)!=best['sha256']:raise ValueError('mixed best checkpoint integrity failed')
    run_document=_read(run/'run.json')
    step_path=run/'steps'/f'{best["step"]:06d}.json'
    step=_read(step_path)
    follow_run=run_document.get('schema')=='cd210_mixed_follow_run_v1'
    if follow_run:
        if run_document.get('snapshot_policy')!='immutable_epoch_at_cycle_boundary' or run_document.get('fixed_snapshot') is not False:
            raise ValueError('follow export requires immutable epoch bindings')
        binding=step['state']['epoch_binding']
        _same(best['epoch_binding'],binding,'best epoch differs from its committed step')
        index_path=(run/binding['path']).resolve()
        if index_path.parent!=run/'epochs' or sha256_file(index_path)!=binding['sha256']:
            raise ValueError('mixed follow epoch integrity failed')
        index=_read(index_path)
    else:
        if run_document.get('schema')!='cd210_mixed_bounded_run_v1' or run_document.get('fixed_snapshot') is not True:
            raise ValueError('export requires an explicit immutable mixed run identity')
        index_path=run/'index.json';index=_read(index_path)
    recipe_document=run_document['recipe']
    recipe=Recipe(**{f.name:recipe_document[f.name] for f in fields(Recipe)})
    _same(recipe.document(),recipe_document,'mixed objective recipe/version differs from its implementation')
    if fingerprint(recipe_document)!=run_document.get('recipe_sha256') or best['recipe_sha256']!=run_document['recipe_sha256']:
        raise ValueError('mixed recipe fingerprint changed')
    rows=[EpisodeRef(**row) for row in index['rows']]
    current_qualification=qualification(rows,recipe)
    _same(current_qualification,index['qualification'],'frozen mixed index qualification changed')
    expected_index=current_qualification['index_sha256'] if follow_run else run_document['index_sha256']
    if not current_qualification['ready'] or current_qualification['index_sha256']!=expected_index:
        raise ValueError('mixed index lacks both admitted train and heldout cohorts')
    validation_path=run/'validation'/f'{best["step"]:06d}.json'
    promotion_path=run/'preserved'/'promotions'/f'{best["step"]:06d}-{best["sha256"]}.json'
    validation,promotion=_read(validation_path),_read(promotion_path)
    for key in ('checkpoint','sha256','step'):
        if step.get(key)!=best[key]:raise ValueError('best does not match its immutable committed step')
    _same(promotion,best,'best differs from immutable mixed promotion receipt')
    package=torch.load(checkpoint,map_location='cpu',weights_only=True)
    from lane_training.fetch_cd210 import COMMIT, EXPORTS
    if (package.get('format')!='cd210-centerline-v1' or package.get('upstream_commit')!=COMMIT
            or package.get('source_hashes')!={key:value['sha256'] for key,value in EXPORTS.items()}):
        raise ValueError('mixed checkpoint is not the pinned CD210 weight format')
    state=package.get('training_state')
    if not isinstance(state,dict) or package.get('step')!=best['step'] or state.get('step')!=best['step']:
        raise ValueError('checkpoint step/training state does not match validated candidate')
    _same(state,step['state'],'checkpoint state differs from immutable committed step')
    _same(state['config'],run_document,'checkpoint lost its exact mixed run identity')
    _same(state['source_hashes'],run_document['source_hashes'],'checkpoint training implementation binding changed')
    project=Path(__file__).resolve().parents[1]
    source_hashes={str(name).replace('\\','/'):value for name,value in state['source_hashes'].items()}
    if len(source_hashes)!=len(state['source_hashes']):raise ValueError('duplicate normalized source binding')
    required_sources=OBJECTIVE_SOURCES+(('lane_behavior/mixed_follow.py','lane_behavior/mixed_runner.py') if follow_run else ())
    for relative in required_sources:
        if source_hashes.get(relative)!=sha256_file(project/relative):
            raise ValueError('mixed objective implementation changed; revalidation required before export: '+relative)
    if (state.get('run_id')!=run_document['run_id'] or state.get('objective_version')!=OBJECTIVE_VERSION
            or state.get('selection_metric')!=SELECTION_METRIC):
        raise ValueError('checkpoint objective identity is not the selected mixed run')
    samples=state.get('sample_counts',{})
    training={row.key:row for row in rows if row.split=='train'}
    if (any(key not in training or type(count) is not int or count<0 for key,count in samples.items())
            or any(sum(count for key,count in samples.items() if training[key].kind==kind)!=best['step']
                   for kind in ('legacy','behavior'))):
        raise ValueError('committed step lacks both mixed training objective contributions')
    _same(dict(state['last_validation'],step=best['step']),validation,'checkpoint validation receipt changed')
    _same({key:best.get(key) for key in validation},validation,'best scores differ from committed validation')
    expected_validation=sorted(r.sha256 for r in rows if r.split=='validation')
    actual_validation=sorted(r['manifest_sha256'] for r in validation['rows'])
    if expected_validation!=actual_validation or validation.get('test_split_used') is not False:
        raise ValueError('mixed heldout membership changed or included test data')
    if validation['cohort']!=fingerprint([r.sha256 for r in sorted(rows,key=lambda r:r.key) if r.split=='validation']):
        raise ValueError('mixed heldout cohort fingerprint changed')
    frozen={row.sha256:row for row in rows if row.split=='validation'}
    for row in validation['rows']:
        original=frozen[row['manifest_sha256']]
        if (row.get('kind')!=original.kind or row.get('episode')!=original.path
                or type(row.get('labels')) is not int or not 0<row['labels']<=original.labels):
            raise ValueError('mixed validation row identity or support changed')
    root=Path(run_document['corpus']).resolve()
    assert_checkpoint_not_held(root,state)
    live,_=discover(root,verify_payloads=False)
    available={row.key:row for row in live}
    for row in rows:
        if row.split=='test' or (row.split=='train' and not state['counts'].get(row.sha256,0)):continue
        if row.key not in available or asdict(available[row.key])!=asdict(row):
            raise ValueError('frozen admitted mixed training/validation manifest is missing or changed')
        assert_episode_not_held(Path(row.path).parent/'manifest.json',corpus=root)
        if row.split=='validation':
            with open_episode(row,root):pass
    # A new hold can arrive while archive inventories are being checked.
    assert_checkpoint_not_held(root,state)
    for row in rows:
        if row.split=='validation':assert_episode_not_held(Path(row.path).parent/'manifest.json',corpus=root)
    if best_path.read_bytes()!=best_bytes:raise ValueError('mixed best advanced during export selection; retry')
    selection=dict(best,kind='mixed_fine_tuned',source_checkpoint_sha256=best['sha256'],
        source_checkpoint=str(checkpoint),source_run_id=run_document['run_id'],objective_recipe=recipe_document,
        stock_parameter_sha256=state['stock_parameter_sha256'],improves_original=True,test_split_used=False,
        score_scope='immutable_committed_mixed_heldout_validation_no_legacy_reselection',
        source_receipts={name:dict(path=str(path),sha256=sha256_file(path)) for name,path in (
            ('run',run/'run.json'),('index',index_path),('step',step_path),
            ('validation',validation_path),('promotion',promotion_path))},export_adapter_version=VERSION)
    return selection,rows


def _stamp_identity(output,roles,selection):
    """Preserve the mixed recipe in ONNX, PKL metadata and transfer payload."""
    checkpoint=selection['source_checkpoint_sha256']
    identity=f'local/cd210-mixed/{selection["recipe_sha256"]}/{checkpoint}'
    values=dict(model_checkpoint=identity,fine_tuning_objective=OBJECTIVE_VERSION,
        fine_tuning_recipe_sha256=selection['recipe_sha256'],fine_tuning_run_id=selection['source_run_id'],
        fine_tuning_selection_metric=SELECTION_METRIC,fine_tuning_validation_cohort=selection['cohort'],
        fine_tuning_stock_parameter_sha256=selection['stock_parameter_sha256'],
        fine_tuning_recipe=json.dumps(selection['objective_recipe'],sort_keys=True,separators=(',',':')),
        fine_tuning_heldout_cohorts=json.dumps(selection['cohorts'],sort_keys=True,separators=(',',':')))
    for role,entry in roles.items():
        path=output/entry['onnx'];graph=onnx.load(path)
        props={item.key:item.value for item in graph.metadata_props};props.update(values)
        onnx.helper.set_model_props(graph,props);onnx.checker.check_model(graph);onnx.save(graph,path)
        metadata_path=output/entry['metadata']
        with metadata_path.open('rb') as stream:metadata=pickle.load(stream)
        metadata.update(values)
        stem=f'driving_{role}_cd210_mixed_s{selection["step"]}_{checkpoint[:12]}'
        renamed=output/(stem+'_metadata.pkl')
        with renamed.open('xb') as stream:pickle.dump(metadata,stream)
        metadata_path.unlink()
        entry.update(metadata=renamed.name,artifact=stem+'_tinygrad.pkl',onnx_sha256=sha256_file(path))
    report=_read(output/'export.json')
    for role,entry in roles.items():report['exports'][role]['sha256']=entry['onnx_sha256']
    report.update(objective_version=OBJECTIVE_VERSION,recipe_sha256=selection['recipe_sha256'],
        mixed_run_id=selection['source_run_id'],selection_metric=SELECTION_METRIC,model_checkpoint=identity)
    write_json(output/'export.json',report)
    return identity


def prepare(run, output):
    """Create one immutable local ONNX/device-build package, never activate it."""
    output=Path(output).resolve()
    if output.exists():raise FileExistsError('mixed export output must be an unused directory')
    selection,rows=select_validated_checkpoint(run)
    output.mkdir(parents=True,exist_ok=False)
    write_json(output/'status.json',dict(stage='exporting_validated_mixed_candidate',runtime_ready=False,driving_qualified=False))
    try:
        checkpoint=output/'checkpoint.pt'
        immutable_copy(selection['source_checkpoint'],checkpoint,selection['source_checkpoint_sha256'])
        model,package=load_checkpoint(checkpoint)
        assert_checkpoint_not_held(package['training_state']['config']['corpus'],package['training_state'])
        roles=legacy_export.export_models(model,selection,output)
        identity=_stamp_identity(output,roles,selection)
        legacy_validation=[asdict(row) for row in rows if row.kind=='legacy' and row.split=='validation']
        verification=legacy_export.verification_trace(output,roles,legacy_validation)
        # Confirm all exact selection receipts and live holds after potentially
        # lengthy ONNX generation. New best advancement asks for a fresh request.
        reselected,_=select_validated_checkpoint(run)
        if reselected['source_checkpoint_sha256']!=selection['source_checkpoint_sha256']:
            raise ValueError('a newer mixed best appeared during export; completed files retained')
        selection.update(export_checkpoint='checkpoint.pt',export_checkpoint_sha256=sha256_file(checkpoint),model_checkpoint=identity)
        immutable_json(output/'selection.json',dict(selected=selection,candidates=[selection]))
        shutil.copyfile(Path(legacy_export.__file__).with_name('device_build.py'),output/'device_build.py')
        bundle=legacy_export.bundle_template(roles,selection)
        ref=f'cd210-mixed-s{selection["step"]}-{selection["source_checkpoint_sha256"][:12]}'
        bundle.update(displayName=f'CD210 Mixed s{selection["step"]}',internalName=ref,ref=ref)
        names=['device_build.py','verification.npz','selection.json','export.json','onnx-verification.json']
        names += [entry[key] for entry in roles.values() for key in ('onnx','metadata')]
        manifest=dict(schema_version=1,device='c3x',backend='QCOM',roles=roles,bundle=bundle,
            selected_checkpoint_sha256=selection['source_checkpoint_sha256'],step=selection['step'],
            files={name:sha256_file(output/name) for name in names},onnx_verification=verification,
            objective_version=OBJECTIVE_VERSION,recipe_sha256=selection['recipe_sha256'],
            objective_recipe=selection['objective_recipe'],source_run_id=selection['source_run_id'],
            selection_metric=SELECTION_METRIC,heldout_cohorts=selection['cohorts'],model_checkpoint=identity)
        immutable_json(output/'payload.json',manifest)
        write_json(output/'status.json',dict(stage='mixed_onnx_ready_device_compile_pending',onnx_ready=True,
            runtime_ready=False,driving_qualified=False,step=selection['step'],objective_version=OBJECTIVE_VERSION,
            recipe_sha256=selection['recipe_sha256'],selection_metric=SELECTION_METRIC))
        return output
    except BaseException as error:
        write_json(output/'status.json',dict(stage='mixed_export_failed_files_retained',error=str(error),
            runtime_ready=False,driving_qualified=False))
        raise


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(argv)
    output=prepare(args.run_dir,args.output)
    print(json.dumps(dict(output=str(output),runtime_ready=False,driving_qualified=False)))
    return 0


if __name__=='__main__':raise SystemExit(main())
