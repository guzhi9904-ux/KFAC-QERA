"""Bounded old-label replay diagnostics; never constructs candidates or accepts a pilot."""
import argparse
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--windows', type=int, nargs='+', default=list(range(8)))
    args = parser.parse_args()
    assert len(set(args.windows)) == len(args.windows) and set(args.windows) <= set(range(8))
    os.environ['QER_PORTABLE_CONFIG'] = str(Path(args.config).resolve())
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'experiments/qer_functional_gradient_4090_v1'))
    from bridge import CONFIG, EXP03, PLAN, torch, read_tensors, save_json, source_identity, slug, capture, hidden_forward, mo, sha_file
    from runtime import Runtime
    assert torch.cuda.device_count() == 2
    assert not subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip(), 'GPU already in use'
    root = Path(args.output).resolve()
    parent = Path(CONFIG['output_parent']).resolve()
    assert root.is_relative_to(parent) and root != parent
    for protected in (Path(CONFIG['assets']).resolve(), Path(CONFIG['model']).resolve()):
        assert not root.is_relative_to(protected)
    root.mkdir(parents=True, exist_ok=False)
    save_json(root/'identity.json', source_identity())
    report = dict(diagnostic_only=True, full_pilot_executed=False,
                  thresholds_changed=False, tolerances=PLAN['portability'],
                  diagnostic_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), rows=[])
    e = Runtime(root, PLAN['parent_modules'][0], time.time()+480)
    try:
        e.checked_model()
        for name in PLAN['parent_modules']:
            e.name = name
            e.compute_device = int(int(name.split('.')[2]) >= 16)
            torch.cuda.set_device(e.compute_device)
            manifest = e.s_manifest()
            for c in args.windows:
                e.boundary()
                with torch.no_grad(), capture(e.model.get_submodule(name)) as state:
                    hidden = hidden_forward(e.model, e.fit['input_ids'][c:c+1]).detach()
                x = state['x'].detach()
                oldx, xm = read_tensors(EXP03/'cache'/slug(name)/f'x_w{c:02d}.safetensors')
                assert mo.digest_tensor(oldx['x']) == xm['input_hash']
                assert xm['token_hash'] == mo.digest_tensor(e.fit['input_ids'][c])
                labels, lm = e.old_label(c, 0)
                sr = next(r for r in manifest if r['window'] == c and r['replicate'] == 0)
                assert sha_file(sr['path']) == sr['file_sha256']
                old, sm = read_tensors(sr['path'])
                assert mo.digest_tensor(old['S']) == sr['S_hash'] and sr['label_hash'] == lm['label_hash']
                g, audit = e.gradient(name, e.fit['input_ids'][c:c+1], hidden, x, labels, weight_check=True)
                s = (g.T@x.reshape(2048, -1).double()).cpu()
                row = dict(module=name, window=c, replicate=0,
                    x_relative_error=mo.relative(x.cpu(), oldx['x']),
                    S_relative_error=mo.relative(s, old['S']),
                    gradient_gram_relative_error=mo.relative((g.T@g).cpu(), old['gradient_gram']),
                    hidden_hash_equal=mo.digest_tensor(hidden) == xm['teacher_hidden_hash'],
                    gradient_audit=audit, parent_file_sha256=sr['file_sha256'], label_hash=lm['label_hash'])
                row['current_gate_passed'] = (row['x_relative_error'] <= PLAN['portability']['parent_input_relative_tolerance']
                    and row['S_relative_error'] <= PLAN['portability']['parent_S_relative_tolerance'])
                report['rows'].append(row)
                save_json(root/'diagnosis.json', report)
                print('REPLAY', __import__('json').dumps(row), flush=True)
                del g, s, old, oldx, x, hidden, state
        report['complete'] = True
        report['all_current_gates_passed'] = all(r['current_gate_passed'] for r in report['rows'])
        report['max_x_relative_error'] = max(r['x_relative_error'] for r in report['rows'])
        report['max_S_relative_error'] = max(r['S_relative_error'] for r in report['rows'])
        save_json(root/'diagnosis.json', report)
        print('SUMMARY', __import__('json').dumps({k:v for k,v in report.items() if k != 'rows'}), flush=True)
    finally:
        e.unload()


if __name__ == '__main__':
    main()
