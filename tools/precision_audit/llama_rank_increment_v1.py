#!/usr/bin/env python3
"""Frozen DA+GI r16 -> r32 interventions on four Llama layer groups.

Only new diagnostic outputs are written. No solve, collection, root rebuild,
requantization, group deletion, or test-selected rank allocation is performed.
"""
from __future__ import annotations
import argparse
import copy
import json
import math
from pathlib import Path
import signal
import sys
import tarfile
import time
from types import SimpleNamespace

sys.dont_write_bytecode = True
import torch
import diag_a_fp64_dual_v1 as source

dual, single, core, prev = source.dual, source.single, source.core, source.prev
VERSION = 'llama_rank_increment_v1'
GROUPS = ((0, 7), (8, 15), (16, 23), (24, 31))
ARMS = ('DA_R16', 'DA_R32', 'DA_L00_07_R32', 'DA_L08_15_R32',
        'DA_L16_23_R32', 'DA_L24_31_R32')
PROJECTIONS = ('self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj',
               'self_attn.o_proj', 'mlp.gate_proj', 'mlp.up_proj', 'mlp.down_proj')
MODULES = tuple(f'model.layers.{i}.{p}' for i in range(32) for p in PROJECTIONS)
HELPERS = ('diag_a_fp64_dual_v1.py', 'full_a_all_ranks_dual_ppl_v1.py',
           'full_a_all_precision_r8_v1.py', 'full_g_precision_r8_v1.py',
           'full_g_target_probe_v1.py', 'full_g_rank_audit_v1.py', 'frozen_harness_word_ppl.py')
WINDOWS, LENGTH = 138, 2048
UNIT_TOL, PPL_TOL = 1e-3, 1e-5


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def rank_for(arm, name):
    require(arm in ARMS and name in MODULES, 'Unknown fixed arm/module')
    if arm in ARMS[:2]:
        return 16 if arm == ARMS[0] else 32
    lo, hi = GROUPS[ARMS.index(arm)-2]
    return 32 if lo <= int(name.split('.')[2]) <= hi else 16


def compare_rows(rows, reference, kind):
    require(kind in ('token', 'word'), 'Unknown scoring protocol')
    key, den = ('window', 'tokens') if kind == 'token' else ('document', 'words')
    require(rows and len(rows) == len(reference), 'Pair coverage mismatch')
    require(len({r[key] for r in rows}) == len(rows), 'Duplicate scoring units')
    delta = []
    for a, b in zip(rows, reference):
        fields = (key, den) if kind == 'token' else (key, den, 'document_sha256')
        require(all(a[k] == b[k] for k in fields), 'Pair inputs/counts differ')
        require(a[den] >= 0 and all(math.isfinite(r['nll_sum']) and r['nll_sum'] >= 0
                                   for r in (a, b)), 'Invalid paired NLL/count')
        delta.append(a['nll_sum']-b['nll_sum'])
    count = sum(r[den] for r in rows)
    require(count > 0, 'Empty scored corpus')
    return {'delta_nll': math.fsum(delta), 'delta_mean_nll': math.fsum(delta)/count,
            'delta_ppl': math.exp(sum(r['nll_sum'] for r in rows)/count)
                         - math.exp(sum(r['nll_sum'] for r in reference)/count),
            'max_unit_nll_difference': max(map(abs, delta)), 'units': len(rows),
            'improved_units': sum(x < 0 for x in delta), 'worsened_units': sum(x > 0 for x in delta),
            'tied_units': sum(x == 0 for x in delta)}


def control(rows, reference, kind):
    result = compare_rows(rows, reference, kind)
    result.update(unit_nll_tolerance=UNIT_TOL, ppl_tolerance=PPL_TOL)
    result['status'] = 'PASS' if result['max_unit_nll_difference'] <= UNIT_TOL and abs(result['delta_ppl']) <= PPL_TOL else 'FAIL'
    return result


def expected_deployment(ctx, arm, kind):
    result = copy.deepcopy(ctx.historical[(kind, 16)]['deployment'])
    result['correction_bits'] = {n: ctx.historical[(kind, rank_for(arm, n))]['deployment']['correction_bits'][n]
                                 for n in ctx.tasks}
    return result


def validate_deployment(ctx, arm, kind, deployed):
    require(deployed == expected_deployment(ctx, arm, kind),
            'Actual Wq / parameters / buffers / device map / factor prefixes differ from frozen source')


class ForwardAudit:
    """Every successful model forward must execute each correction exactly once."""
    def __init__(self, names, stop):
        self.names, self.stop = tuple(names), stop
        self.current, self.success, self.aborted = None, 0, 0

    def begin(self, _model, _args):
        self.stop.check()
        if self.current is not None:
            self.aborted += 1  # A harness OOM probe can abort before the model post-hook.
        self.current = dict.fromkeys(self.names, 0)

    def hit(self, name):
        require(self.current is not None and name in self.current, 'Untracked correction')
        self.current[name] += 1
        require(self.current[name] == 1, 'Repeated correction: '+name)

    def end(self, _model, _args, _out):
        require(self.current is not None and all(v == 1 for v in self.current.values()), 'Missing correction')
        self.success += 1
        self.current = None

    def snapshot(self):
        require(self.success > 0 and self.current is None, 'No completed forward / unfinished forward')
        return {'status': 'PASS', 'targets': len(self.names), 'successful_forwards': self.success,
                'aborted_probe_forwards': self.aborted, 'exactly_once_each_target_per_successful_forward': True}


def install(ctx, model, arm, kind, stop, handles):
    require(not any(m._forward_hooks or m._forward_pre_hooks for m in model.modules()), 'Unexpected existing forward hooks')
    audit = ForwardAudit(ctx.tasks, stop)
    handles.append(model.register_forward_pre_hook(audit.begin))
    bits = {}
    with torch.no_grad():
        for name, binding in ctx.tasks.items():
            mod = model.get_submodule(name)
            q = ctx.inputs.tensors(binding['quant'])['weight_q']
            require(q.dtype == mod.weight.dtype == torch.bfloat16 and q.shape == mod.weight.shape, 'Invalid Wq')
            mod.weight.copy_(q.to(mod.weight.device))
            require(single.tensor_record(q) == single.tensor_record(mod.weight), 'Wq bits changed during installation')
            values = ctx.inputs.tensors(ctx.factors[name]['file'])
            rank = rank_for(arm, name)
            # Match frozen dual.install: contiguous BF16 prefixes and two GEMMs.
            a = values['A'][:, :rank].to(mod.weight.device).contiguous()
            b = values['B'][:rank].to(mod.weight.device).contiguous()
            require(a.dtype == b.dtype == torch.bfloat16 and a.shape == (mod.in_features, rank)
                    and b.shape == (rank, mod.out_features) and torch.isfinite(a).all()
                    and torch.isfinite(b).all(), 'Invalid BF16 factor prefix')
            bits[name] = {'A': single.tensor_record(a), 'B': single.tensor_record(b)}
            def hook(_m, args, out, a=a, b=b, name=name):
                audit.hit(name)
                return out + (args[0] @ a) @ b
            handles.append(mod.register_forward_hook(hook))
    handles.append(model.register_forward_hook(audit.end))
    deployed = {'parameter_bits': {n: single.tensor_record(t) for n, t in model.named_parameters()},
                'buffer_bits': {n: single.tensor_record(t) for n, t in model.named_buffers()},
                'device_map': model.hf_device_map, 'correction_bits': bits}
    validate_deployment(ctx, arm, kind, deployed)
    return deployed, audit


def verify_execution(cert, targets):
    require(cert.get('status') == 'PASS' and cert.get('targets') == targets
            and cert.get('successful_forwards', 0) > 0
            and cert.get('exactly_once_each_target_per_successful_forward') is True, 'Invalid execution certificate')


def read_token(ctx, arm, pilot=False):
    root = ctx.output / ('pilot' if pilot else 'token')
    path = root / (arm+'.json')
    total = 8 if pilot else WINDOWS
    if not path.exists():
        return {'experiment_identity': ctx.identity, 'arm': arm, 'windows': total,
                'records': [], 'batches': [], 'complete': False}
    s = core.read_json(path)
    require(s.get('experiment_identity') == ctx.identity and s.get('arm') == arm
            and s.get('windows') == total, 'Token checkpoint identity mismatch')
    require(s['payload_sha256'] == core.fingerprint({k:v for k,v in s.items() if k != 'payload_sha256'}), 'Token payload changed')
    dep = s['deployment_file']
    require(Path(dep['path']).resolve() == (root/(arm+'_deployment.json')).resolve(), 'Deployment path changed')
    ctx.inputs.verify(dep)
    validate_deployment(ctx, arm, 'token', core.read_json(dep['path']))
    rr = s['records']
    require(len(rr) <= total and s['complete'] == (len(rr) == total), 'Token completion mismatch')
    require([r['window'] for r in rr] == list(range(len(rr))), 'Token order mismatch')
    require(all(r['tokens'] == LENGTH-1 and math.isfinite(r['nll_sum']) and r['nll_sum'] >= 0 for r in rr), 'Invalid window NLL')
    cursor = 0
    for batch in s['batches']:
        end = min(cursor+8, total)
        require(batch['start'] == cursor and batch['end'] == end and end <= len(rr), 'Batch coverage mismatch')
        require(batch['successful_forward_delta'] == 1, 'Missing fresh forward certificate')
        verify_execution(batch['execution'], len(ctx.tasks))
        expected = root/'tokens'/arm/f'batch_{cursor:04d}.safetensors'
        require(Path(batch['file']['path']).resolve() == expected.resolve(), 'Token tensor path mismatch')
        values = ctx.inputs.tensors(batch['file'])['nll']
        require(values.shape == (end-cursor, LENGTH-1) and torch.isfinite(values).all()
                and (values >= 0).all(), 'Invalid saved token losses')
        for i, row in enumerate(rr[cursor:end]):
            require(float(values[i].double().sum()) == row['token_nll_sum_fp64'], 'Token FP64 diagnostic sum mismatch')
        cursor = end
    require(cursor == len(rr), 'Incomplete committed batch coverage')
    return s


def endpoint_gate(ctx, arm, kind, rows, pilot=False):
    if arm not in ARMS[:2]:
        return
    rank = 16 if arm == ARMS[0] else 32
    reference = ctx.refs[(kind, rank)][:8] if pilot else ctx.refs[(kind, rank)]
    result = control(rows, reference, kind)
    root = ctx.output / ('pilot' if pilot else kind)
    core.atomic_json(root/(arm+'_control.json'), {'experiment_identity': ctx.identity, **result})
    require(result['status'] == 'PASS', 'Frozen '+kind+' endpoint replay failed: '+arm)


def token_arm(ctx, arm, stop, pilot=False):
    root = ctx.output / ('pilot' if pilot else 'token')
    root.mkdir(parents=True, exist_ok=True)
    s = read_token(ctx, arm, pilot)
    total = s['windows']
    if not s['complete']:
        data = ctx.inputs.tensors(ctx.manifest['payload']['data']['wikitext2'])
        require(data['input_ids'].shape == data['attention_mask'].shape == (WINDOWS, LENGTH)
                and bool((data['attention_mask'] == 1).all()), 'Frozen token protocol changed')
        loading = dict(ctx.manifest['payload']['source_config'])
        loading['max_memory'] = ctx.manifest['payload']['config']['eval_max_memory']
        model, handles = None, []
        try:
            stop.check()
            torch.manual_seed(1234)
            with core.heartbeat('TOKEN load/install '+arm):
                model = ctx.legacy.load_model(loading, 'bfloat16', 'balanced')
                dual.resident(model)
                deployed, audit = install(ctx, model, arm, 'token', stop, handles)
                dep_path = root/(arm+'_deployment.json')
                dual.freeze_json(dep_path, deployed)
                s['deployment_file'] = single.file_record(dep_path)
            device = ctx.legacy._input_device(model)
            directory = root/'tokens'/arm
            directory.mkdir(parents=True, exist_ok=True)
            with torch.inference_mode():
                for start in range(len(s['records']), total, 8):
                    stop.check()
                    end = min(start+8, total)
                    before = audit.success
                    with core.heartbeat(f'TOKEN {arm} windows={start+1}-{end}/{total}'):
                        ids, mask = (data[k][start:end].to(device) for k in ('input_ids', 'attention_mask'))
                        logits = model(input_ids=ids, attention_mask=mask, use_cache=False).logits
                        metrics, losses = single.nll_with_tokens(logits, ids, mask, 256)
                        if start == 0:
                            require(metrics == ctx.legacy._chunked_window_nll(logits, ids, mask, 256), 'Legacy scoring reduction changed')
                    del logits, ids, mask
                    require(len(metrics) == end-start and all(n == 2047 and math.isfinite(v) and v >= 0 for v,n in metrics), 'Invalid evaluated NLL')
                    require(audit.success-before == 1, 'Unexpected model forward count')
                    s['records'].extend({'window':start+i, 'tokens':n, 'nll_sum':v,
                                         'token_nll_sum_fp64':float(losses[i].double().sum())} for i,(v,n) in enumerate(metrics))
                    s['batches'].append({'start':start, 'end':end, 'execution':audit.snapshot(),
                        'successful_forward_delta':audit.success-before,
                        'file':single.atomic_tensors(directory/f'batch_{start:04d}.safetensors', {'nll':losses})})
                    s['complete'] = end == total
                    s.pop('payload_sha256', None)
                    s['payload_sha256'] = core.fingerprint(s)
                    core.atomic_json(root/(arm+'.json'), s)
                    stop.done += 1
                    core.log(f'COMMITTED TOKEN {arm} {end}/{total}')
        finally:
            for handle in handles:
                handle.remove()
            handles.clear()
            model = None
            dual.clear_gpu()
    endpoint_gate(ctx, arm, 'token', s['records'], pilot)
    return s['records']


def read_word(ctx, arm):
    folder = ctx.output/'word'/arm
    marker = folder/'complete.json'
    if not marker.exists():
        return None
    s = core.read_json(marker)
    require(s.get('experiment_identity') == ctx.identity and s.get('arm') == arm and s.get('status') == 'PASS', 'Word identity mismatch')
    for key, filename in (('results_file','results.json'), ('documents_file','documents.json'), ('deployment_file','deployment.json')):
        require(Path(s[key]['path']).resolve() == (folder/filename).resolve(), 'Word checkpoint path changed')
        ctx.inputs.verify(s[key])
    validate_deployment(ctx, arm, 'word', core.read_json(folder/'deployment.json'))
    verify_execution(s['execution'], len(ctx.tasks))
    summary, rows = dual.word_documents(core.read_json(folder/'results.json'), ctx.word.task, ctx.word.helper)
    require(summary == s['summary'] and rows == core.read_json(folder/'documents.json'), 'Word summary differs from full raw harness result')
    require(summary['documents'] == 62 and summary['scored_words'] == 241335, 'Incomplete word corpus')
    return rows


def word_arm(ctx, arm, stop):
    rows = read_word(ctx, arm)
    if rows is None:
        from transformers import AutoModelForCausalLM
        from accelerate import dispatch_model
        from qera.utils import create_device_map
        from lm_eval.models.huggingface import HFLM
        from lm_eval.evaluator import simple_evaluate
        folder = ctx.output/'word'/arm
        folder.mkdir(parents=True, exist_ok=True)
        model, lm, handles = None, None, []
        try:
            stop.check()
            torch.manual_seed(1234)
            with core.heartbeat('WORD load/install '+arm):
                model = AutoModelForCausalLM.from_pretrained(ctx.manifest['payload']['source_config']['model_path'],
                    torch_dtype=torch.bfloat16, local_files_only=True, _attn_implementation='eager', max_position_embeddings=4096)
                model.eval()
                model.config.use_cache = False
                model = dispatch_model(model, device_map=create_device_map(model, 'auto-balanced'))
                dual.resident(model)
                deployed, audit = install(ctx, model, arm, 'word', stop, handles)
                dual.freeze_json(folder/'deployment.json', deployed)
                lm = HFLM(model)
                require(lm.max_length == 4096, 'Harness context changed')
                dual.freeze_json(folder/'runtime.json', {'experiment_identity':ctx.identity,
                    'hflm_batch_size':lm.batch_size, 'context':lm.max_length, 'device_map':model.hf_device_map})
            with core.heartbeat('WORD '+arm+' all 62 documents'), torch.no_grad():
                result = simple_evaluate(model=lm, tasks=[ctx.word.task], task_manager=ctx.word.manager,
                    num_fewshot=None, batch_size='auto', limit=None, use_cache=None, bootstrap_iters=0,
                    log_samples=True, apply_chat_template=False, random_seed=0, numpy_random_seed=1234,
                    torch_random_seed=1234, fewshot_random_seed=1234)
            summary, rows = dual.word_documents(result, ctx.word.task, ctx.word.helper)
            require(summary['documents'] == 62 and summary['scored_words'] == 241335, 'Incomplete word evaluation')
            def convert(x):
                item = getattr(x, 'item', None)
                return item() if callable(item) else str(x)
            core.atomic_json(folder/'results.json', json.loads(json.dumps(result, default=convert)))
            core.atomic_json(folder/'documents.json', rows)
            record = {'experiment_identity':ctx.identity, 'arm':arm, 'status':'PASS', 'summary':summary,
                'execution':audit.snapshot(), **{key:single.file_record(folder/name) for key,name in (
                    ('results_file','results.json'), ('documents_file','documents.json'), ('deployment_file','deployment.json'))}}
            core.atomic_json(folder/'complete.json', record)
            stop.done += 1
            core.log('COMMITTED WORD '+arm)
        finally:
            for handle in handles:
                handle.remove()
            handles.clear()
            lm = model = None
            dual.clear_gpu()
    endpoint_gate(ctx, arm, 'word', rows)
    return rows


def export_results(ctx, kind, all_rows):
    folder = ctx.output/kind
    folder.mkdir(parents=True, exist_ok=True)
    den, unit = ('tokens','window') if kind == 'token' else ('words','document')
    summary, detail, comparisons, pairs = [], [], [], []
    for arm, rows in all_rows.items():
        count = sum(r[den] for r in rows)
        nll = sum(r['nll_sum'] for r in rows)
        added = sum((rank_for(arm,n)-16)*sum(b['shape']) for n,b in ctx.tasks.items())
        summary.append({'arm':arm, 'protocol':kind, 'ppl':math.exp(nll/count), 'nll_sum':nll,
                        'mean_nll':nll/count, 'scored_count':count, 'units':len(rows),
                        'upgraded_modules':sum(rank_for(arm,n)==32 for n in ctx.tasks),
                        'added_correction_parameters_vs_r16':added})
        detail.extend({'arm':arm, **r} for r in rows)
        if arm != ARMS[0] and ARMS[0] in all_rows:
            base = all_rows[ARMS[0]]
            comparisons.append({'comparison':arm+'_minus_DA_R16', **compare_rows(rows,base,kind)})
            pairs.extend({'arm':arm, unit:r[unit], 'denominator':r[den],
                          'delta_nll':r['nll_sum']-b['nll_sum']} for r,b in zip(rows,base))
    for name, values in (('ppl_summary',summary), ('per_unit',detail), ('comparisons',comparisons), ('paired_deltas',pairs)):
        prev.write_csv(folder/(name+'.csv'), values)
    if set(all_rows) == set(ARMS):
        effects = {r['comparison']:r['delta_nll'] for r in comparisons}
        joint = effects['DA_R32_minus_DA_R16']
        separate = math.fsum(effects[a+'_minus_DA_R16'] for a in ARMS[2:])
        core.atomic_json(folder/'nonadditivity.json', {'joint_delta_nll':joint,
            'sum_group_delta_nll':separate, 'joint_minus_sum':joint-separate,
            'note':'Finite-intervention nonadditivity, not a unique attribution to pairwise layer interactions.'})
    core.atomic_json(folder/'status.json', {'experiment_identity':ctx.identity,
        'status':'COMPLETE' if set(all_rows)==set(ARMS) else 'INCOMPLETE',
        'completed_arms':list(all_rows), 'expected_arms':list(ARMS), 'posthoc_diagnostic':True})


def summarize(ctx):
    counts = {}
    for kind in ('token','word'):
        values = {}
        for arm in ARMS:
            if kind == 'token':
                state = read_token(ctx, arm)
                rows = state['records'] if state['complete'] else None
            else:
                rows = read_word(ctx, arm)
            if rows is not None:
                endpoint_gate(ctx,arm,kind,rows)
                values[arm] = rows
        require(not any(a in values for a in ARMS[2:]) or all(a in values for a in ARMS[:2]), 'Group results without full endpoint gates')
        export_results(ctx,kind,values)
        counts[kind] = len(values)
    status = {'experiment_identity':ctx.identity, 'complete':all(n==6 for n in counts.values()),
              'completed_arms':counts, 'expected_per_protocol':6, 'posthoc_diagnostic':True}
    core.atomic_json(ctx.output/'status.json',status)
    return status


def pack(ctx):
    status = summarize(ctx)
    require(status['complete'], 'Both protocols must finish before packing a complete summary')
    names = ['experiment.json','source_audit.json','status.json','pilot_gate.json']
    for kind in ('token','word'):
        names += [f'{kind}/{n}' for n in ('ppl_summary.csv','per_unit.csv','comparisons.csv','paired_deltas.csv','nonadditivity.json','status.json')]
        names += [f'{kind}/{a}_control.json' for a in ARMS[:2]]
    destination = ctx.output/(VERSION+'_summary.tar.gz')
    with tarfile.open(destination, 'w:gz') as archive:
        for name in names:
            archive.add(ctx.output/name,arcname=name,recursive=False)
    core.log('SUMMARY PACKAGE: '+str(destination))


def setup(args, stop):
    directory = Path(__file__).resolve().parent
    pins = {}
    for line in (directory/'SHA256SUMS.llama_rank_increment').read_text().splitlines():
        digest, name = line.split('  ',1)
        require(Path(name).name == name and name not in pins and core.sha(directory/name)==digest, 'Bundle checksum mismatch: '+name)
        pins[name] = digest
    require(all(n in pins for n in (*HELPERS,Path(__file__).name)), 'Incomplete bundle manifest')
    source_dir = args.source_da_dir.resolve()
    output = args.output_dir.resolve()
    core.disjoint(output,[source_dir])
    exp_path = source_dir/'experiment.json'
    old_exp = core.read_json(exp_path)
    require(old_exp.get('version') == source.VERSION and old_exp.get('protocol_selection') == 'both', 'Need completed DA dual-protocol source')
    # Read-only source audit; never call source.prepare_one / solve / run.
    src = source.setup(SimpleNamespace(**{**vars(args), 'command':'run', 'protocol':'both'}))
    require(src.experiment == old_exp, 'DA source experiment differs from current frozen provenance/environment')
    require(set(src.tasks) == set(MODULES), 'Expected all 224 Llama projections')
    src.output = source_dir
    factors = {}
    metrics = []
    for name in src.tasks:
        stop.check()
        rec = source.read_checked(src,'diag_gi',name)
        require(rec is not None and rec.get('solve_rank') == 64, 'Missing completed DA+GI rank64 factors')
        values = {r['rank']:r['sse_after_fp64'] for r in rec['metrics']}
        require(all(math.isfinite(v) and v >= 0 for v in values.values())
                and values[32] <= values[16] + max(values[16],1e-30)*1e-8, 'Source local objective rank check failed')
        factors[name] = rec
        metrics.append({'module':name, 'sse_r16':values[16], 'sse_r32':values[32],
                        'factor_metadata':single.file_record(source.record_path(src,'diag_gi',name))})
    historical, refs, files = {}, {}, {}
    with source.evaluation_binding():
        for rank in (16,32):
            stop.check()
            name = dual.label('diag_gi',rank)
            routes = dual.artifact_routes(src,'diag_gi',rank)
            for kind in ('token','word'):
                folder = source_dir/(kind+'_ppl')
                dep_path = folder/('deployment_'+name+'.json')
                dep = core.read_json(dep_path)
                require(dep.get('experiment_identity')==src.identity and dep.get('arm')==name
                        and dep.get('route_sha256')==core.fingerprint(routes), 'Historical route/deployment mismatch')
                expected = {}
                for module, rec in factors.items():
                    v = src.inputs.tensors(rec['file'])
                    expected[module] = {'A':single.tensor_record(v['A'][:,:rank].contiguous()),
                                        'B':single.tensor_record(v['B'][:rank].contiguous())}
                require(dep['deployment']['correction_bits'] == expected, 'Historical factors differ from actual source prefixes')
                if kind == 'token':
                    state = single.read_state(folder,name,src.identity,src.inputs)
                    require(state['complete'] and state['route_sha256']==core.fingerprint(routes), 'Incomplete/mismatched token source')
                    rows = state['records']
                    state_path = folder/(name+'.json')
                else:
                    state = dual.word_existing(src,name)
                    require(state is not None, 'Incomplete word source')
                    rows = core.read_json(state['documents_file']['path'])
                    require(state['summary']['documents']==62 and state['summary']['scored_words']==241335, 'Historical word coverage changed')
                    state_path = folder/name/'complete.json'
                historical[(kind,rank)], refs[(kind,rank)] = dep, rows
                files[f'{kind}_r{rank}'] = {'deployment':single.file_record(dep_path), 'state':single.file_record(state_path)}
    for kind in ('token','word'):
        a,b = (historical[(kind,r)]['deployment'] for r in (16,32))
        require({k:v for k,v in a.items() if k!='correction_bits'} == {k:v for k,v in b.items() if k!='correction_bits'}, 'Historical common model differs')
        compare_rows(refs[(kind,32)],refs[(kind,16)],kind)
    experiment = {'version':VERSION, 'source_experiment':single.file_record(exp_path),
        'source_identity':src.identity, 'tool_hashes':pins, 'source_endpoints':files,
        'source_factors':{n:r['file'] for n,r in factors.items()}, 'groups':[list(g) for g in GROUPS], 'arms':list(ARMS),
        'policy':'DA+GI only; fixed BF16 rank64 prefixes; independent r16-to-r32 group interventions; no new solve/statistics/roots',
        'token_protocol':old_exp['token_protocol'], 'word_protocol':old_exp['word_protocol'],
        'control_tolerances':{'unit_nll':UNIT_TOL,'ppl':PPL_TOL}, 'posthoc_diagnostic':True}
    ctx = copy.copy(src)
    ctx.output, ctx.experiment, ctx.identity = output, experiment, core.fingerprint(experiment)
    ctx.factors, ctx.historical, ctx.refs = factors, historical, refs
    ctx.source_audit = {'experiment_identity':ctx.identity, 'status':'PASS', 'module_count':224,
        'source_metrics':metrics, 'source_endpoints':files,
        'historical_rows':{f'{k}_r{r}':v for (k,r),v in refs.items()},
        'note':'Stored FP64 solve certificates and actual BF16 factor prefixes checked. No fresh SVD, root construction, or forward in source audit. Historical hook counts were not recorded; new forwards are instrumented.'}
    return ctx


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('doctor','pilot','run','pack'))
    for key in ('run-dir','repo-dir','source-fp64-dir','source-da-dir','output-dir',
                'official-qera-root','harness-source','word-reference-dir'):
        parser.add_argument('--'+key,required=True,type=Path)
    parser.add_argument('--protocol',choices=('both','token','word'),default='both')
    parser.add_argument('--max-hours',type=float,default=10)
    parser.add_argument('--max-new-units',type=int)
    args = parser.parse_args(argv)
    if not math.isfinite(args.max_hours) or args.max_hours <= 0 or (args.max_new_units is not None and args.max_new_units < 1):
        parser.error('Positive finite budgets required')
    stop = prev.Budget(args.max_hours,args.max_new_units)
    signal.signal(signal.SIGINT,stop.signal)
    signal.signal(signal.SIGTERM,stop.signal)
    try:
        ctx = setup(args,stop)
    except prev.Paused:
        core.log('PAUSED during read-only source audit; repeat identical command')
        return 75
    if args.command == 'doctor':
        core.log('DOCTOR PASS: DA+GI source factors and both endpoints verified; no model forward executed')
        return 0
    require(not ctx.output.exists() or (ctx.output/'experiment.json').exists() or not any(ctx.output.iterdir()), 'Refusing unknown nonempty output')
    ctx.output.mkdir(parents=True,exist_ok=True)
    gate = {'experiment_identity':ctx.identity,'status':'PASS','arms':list(ARMS),'token_windows_per_arm':8,'separate_from_main':True}
    with core.audit_lock(ctx.output):
        dual.freeze_json(ctx.output/'experiment.json',ctx.experiment)
        dual.freeze_json(ctx.output/'source_audit.json',ctx.source_audit)
        try:
            if args.command == 'pilot':
                for arm in ARMS:
                    token_arm(ctx,arm,stop,pilot=True)
                dual.freeze_json(ctx.output/'pilot_gate.json',gate)
                core.log('PILOT COMPLETE: six arms x eight windows; full endpoint gates still required')
                return 0
            require(core.read_json(ctx.output/'pilot_gate.json')==gate,'Run matching pilot first')
            if args.command == 'pack':
                pack(ctx)
                return 0
            kinds = ('token','word') if args.protocol == 'both' else (args.protocol,)
            for kind in kinds:
                values = {}
                for arm in ARMS:  # Complete BOTH original endpoints before group interventions.
                    values[arm] = token_arm(ctx,arm,stop) if kind=='token' else word_arm(ctx,arm,stop)
                    export_results(ctx,kind,values)
            if summarize(ctx)['complete']:
                pack(ctx)
                core.log('EXPERIMENT COMPLETE; report all four groups, including null or opposite effects')
            else:
                core.log('REQUESTED PROTOCOL COMPLETE; run the remaining protocol before packing')
        except prev.Paused:
            core.log('PAUSED (75): repeat identical command; token resumes per batch, word restarts current arm')
            return 75
        except Exception as exc:
            core.atomic_json(ctx.output/'last_failure.json',{'experiment_identity':ctx.identity,
                'type':type(exc).__name__,'message':str(exc)})
            raise
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
