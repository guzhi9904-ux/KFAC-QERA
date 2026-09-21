"""A fixed test prefix for a cheap joint-deployment check, not a full test result."""
from contextlib import contextmanager
import math
from bridge import (require, load_record, commit, save_csv, save_json,
                    sha_file, slug, mo)
import evaluation


def subset_report(root, config, identity, count, original_windows):
    frozen = load_record(root/'candidate_freeze.json', identity)
    freeze_hash = sha_file(root/'candidate_freeze.json')
    rows = []
    for role in ('validation', 'test'):
        ids = original_windows(root, role, identity)
        expected = len(ids) if role == 'validation' else count
        require(expected <= len(ids), 'Test subset exceeds frozen dataset')
        jobs = evaluation.evaluation_jobs(config['modules'], frozen['keys'], role)
        hashes = [mo.digest_tensor(ids[i]) for i in range(expected)]
        for scope, key, names in jobs + [('teacher', 'teacher', [])]:
            values = []
            for i in range(expected):
                folder = root/'scores'/role/f'w{i:04d}'
                path = folder/('teacher.json' if scope == 'teacher' else slug(scope)+'___'+key+'.json')
                record = load_record(path, identity)
                require(record['token_hash'] == hashes[i], 'Score token binding changed')
                if scope != 'teacher':
                    require(record['freeze_hash'] == freeze_hash and record['role'] == role
                            and record['window'] == i and record['scope'] == scope
                            and record['candidate'] == key and record['modules'] == names,
                            'Score candidate binding changed')
                values.append(record['scores'])
            tokens = sum(v['tokens'] for v in values)
            nll = sum(v['NLL_sum'] for v in values)/tokens
            rows.append(dict(role=role, scope=scope, candidate=key, windows=expected,
                             tokens=tokens, KL=sum(v['KL']*v['tokens'] for v in values)/tokens,
                             NLL=nll, PPL=math.exp(nll)))
    baselines = {(r['role'], r['scope']): r['KL'] for r in rows if r['candidate'] == 'None'}
    for row in rows:
        base = baselines.get((row['role'], row['scope']))
        row['KL_recovery_percent'] = None if base is None or base <= 1e-12 else 100*(1-row['KL']/base)
    out = root/'summary'/f'test_subset_{count}'
    save_csv(out/'results.csv', rows)
    save_json(out/'results.json', dict(identity=identity, rows=rows,
              test_window_indices=list(range(count)), full_test_complete=False,
              meaning='Validation is single-module; test subset jointly deploys all12 modules. Not a per-module test comparison.'))
    commit(out/'complete.json', identity, passed=True, test_windows=count,
           full_test_complete=False, candidate_freeze_sha256=freeze_hash,
           results_sha256=sha_file(out/'results.json'),
           limitations='Fixed prefix selected to reduce runtime, not a random full-test estimate; no full-test completion claim.')
    print('TEST_SUBSET_COMPLETE', count, 'FULL_TEST_NOT_RUN', flush=True)


@contextmanager
def limited_test(count):
    require(count in (16, 32), 'Only16 or32 test windows are supported')
    old_windows, old_report = evaluation.windows, evaluation.report
    def selected_windows(root, role, identity):
        ids = old_windows(root, role, identity)
        if role != 'test':return ids
        require(count <= len(ids), 'Test subset exceeds frozen dataset')
        path = root/'evaluation_subsets'/f'test_first_{count}.json'
        selection = dict(test_window_indices=list(range(count)), selection='Fixed prefix; chosen for runtime before comparing test scores',
                         source_windows=len(ids), token_hashes=[mo.digest_tensor(ids[i]) for i in range(count)],
                         evaluator_sha256=sha_file(__file__), full_test_complete=False)
        if path.exists():
            record = load_record(path, identity)
            require(all(record[k] == v for k, v in selection.items()), 'Subset selection changed')
        else:commit(path, identity, **selection)
        return ids[:count]
    evaluation.windows = selected_windows
    evaluation.report = lambda root, config, identity: subset_report(root, config, identity, count, old_windows)
    try:yield
    finally:evaluation.windows, evaluation.report = old_windows, old_report
