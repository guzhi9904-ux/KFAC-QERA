"""Fixed-baseline alpha analysis; does not fit curvature or choose platforms."""
import json
import math
from collections import defaultdict
from pathlib import Path

MODULES = ('model.layers.31.self_attn.q_proj', 'model.layers.10.self_attn.v_proj')
DIRECTIONS = ('R_none', 'R_svd64', 'R_A64')
SMALL = (.05, .1, .2)
NEW = (.3, .5, .75, 1.)
ALPHAS = SMALL + NEW


def analyze(parent, extension, baseline):
    """Use token weights globally, and a separate small-alpha q0 per window."""
    q0 = {(r['module'], r['direction']): r for r in baseline}
    assert set(q0) == {(m, d) for m in MODULES for d in DIRECTIONS}
    groups = defaultdict(list)
    seen = set()
    for source, rows in (('parent', parent), ('extension', extension)):
        for r in rows:
            if r['alpha'] == 0:
                continue
            key = (r['module'], r['direction'], r['window'], r['alpha'])
            if key in seen:
                raise ValueError('Duplicate measurement key')
            seen.add(key)
            groups[key[:2]+(key[3],)].append(r | {'source': source})
    expected = {(m, d, w, a) for m in MODULES for d in DIRECTIONS for w in range(8) for a in ALPHAS}
    if seen != expected:
        raise ValueError(f'Incomplete/unexpected grid: missing={len(expected-seen)}, extra={len(seen-expected)}')
    per_window = []
    pooled = []
    for m in MODULES:
        for d in DIRECTIONS:
            base = q0[(m, d)]
            assert base['q_KL'] > 0 and base['local_alphas'] == list(SMALL)
            window_q0 = {}
            for w in range(8):
                local = [next(r for r in groups[(m, d, a)] if r['window'] == w) for a in SMALL]
                if not all(r['valid'] and r['KL_mean'] > 0 for r in local):
                    raise ValueError('Invalid parent window baseline')
                window_q0[w] = sum(r['KL_mean']/r['alpha']**2 for r in local)/3
            for a in ALPHAS:
                rows = groups[(m, d, a)]
                assert sorted(r['window'] for r in rows) == list(range(8))
                total = sum(r['T'] for r in rows)
                actual = sum(r['KL_sum'] for r in rows)/total
                valid = all(r['valid'] for r in rows)
                predicted = a*a*base['q_KL']
                rho = actual/predicted if valid else None
                pooled.append({'module': m, 'direction': d, 'alpha': a, 'T': total,
                    'actual_KL': actual, 'q0': base['q_KL'], 'quadratic_KL': predicted,
                    'rho': rho, 'signed_bias': None if rho is None else rho-1,
                    'valid': valid, 'valid_windows': sum(r['valid'] for r in rows),
                    'invalid_reasons': sorted({x for r in rows for x in r['invalid_reasons']}),
                    'MC_quadratic': a*a*base['q_hat'],
                    'MC_quadratic_low': a*a*base['ci_low'], 'MC_quadratic_high': a*a*base['ci_high']})
                for r in rows:
                    qw = window_q0[r['window']]
                    rw = r['KL_mean']/(a*a*qw) if r['valid'] else None
                    per_window.append({'module': m, 'direction': d, 'window': r['window'], 'alpha': a,
                        'T': r['T'], 'actual_KL': r['KL_mean'], 'q0_window': qw,
                        'quadratic_KL': a*a*qw, 'rho': rw, 'signed_bias': None if rw is None else rw-1,
                        'valid': r['valid'], 'invalid_reasons': r['invalid_reasons'], 'source': r['source']})
    index = {(r['module'], r['direction'], r['alpha']): r for r in pooled}
    windows = {(r['module'], r['direction'], r['alpha'], r['window']): r for r in per_window}
    advantages, window_advantages = [], []
    for m in MODULES:
        for a in ALPHAS:
            svd, aw = index[(m, 'R_svd64', a)], index[(m, 'R_A64', a)]
            comparisons = []
            for w in range(8):
                sw, aa = windows[(m, 'R_svd64', a, w)], windows[(m, 'R_A64', a, w)]
                valid = sw['valid'] and aa['valid']
                delta = sw['actual_KL']-aa['actual_KL']
                pred = sw['quadratic_KL']-aa['quadratic_KL']
                winner = ('A64' if delta > 0 else 'SVD64' if delta < 0 else 'tie') if valid else 'invalid'
                row = {'module': m, 'alpha': a, 'window': w, 'valid': valid,
                    'actual_delta': delta, 'quadratic_delta': pred, 'winner': winner,
                    'rank_reversal': (delta*pred < 0) if valid else None,
                    'A64_KL_reduction': delta/sw['actual_KL'] if valid and sw['actual_KL'] > 0 else None}
                comparisons.append(row); window_advantages.append(row)
            valid = svd['valid'] and aw['valid']
            delta = svd['actual_KL']-aw['actual_KL']
            pred = svd['quadratic_KL']-aw['quadratic_KL']
            advantages.append({'module': m, 'alpha': a, 'valid': valid,
                'actual_delta': delta, 'quadratic_delta': pred,
                'actual_A64_KL_reduction': delta/svd['actual_KL'] if valid and svd['actual_KL'] > 0 else None,
                'quadratic_A64_KL_reduction': pred/svd['quadratic_KL'],
                'advantage_ratio_to_quadratic': delta/pred if valid and pred != 0 else None,
                'rank_reversal': delta*pred < 0 if valid else None,
                'A64_wins': sum(r['winner'] == 'A64' for r in comparisons),
                'SVD64_wins': sum(r['winner'] == 'SVD64' for r in comparisons),
                'ties': sum(r['winner'] == 'tie' for r in comparisons),
                'invalid_windows': sum(not r['valid'] for r in comparisons),
                'window_rank_reversals': sum(r['rank_reversal'] is True for r in comparisons)})
    return {'pooled': pooled, 'windows': per_window, 'advantages': advantages,
            'window_advantages': window_advantages}


def write_report(experiment):
    from storage import read_json, save_json, save_csv, atomic_bytes
    root = experiment.root
    parent = [read_json(p) for p in sorted((root/'parent_snapshot/records/kl').glob('*.json'))]
    extension = [experiment.read_record(p) for p in sorted((root/'records/kl').glob('*.json'))]
    baseline = read_json(root/'parent_snapshot/summary.json')
    result = analyze(parent, extension, baseline)
    save_json(root/'analysis.json', result)
    for key, rows in result.items(): save_csv(root/(key+'.csv'), rows)
    flat = []
    for r in extension:
        flat.append({k: v for k, v in r.items() if k not in ('weight_audit', 'output_audit')}
                    | {'weight_'+k: v for k, v in r['weight_audit'].items()}
                    | {'output_'+k: v for k, v in r['output_audit'].items()})
    save_csv(root/'extension_kl_path.csv', flat)
    save_json(root/'alpha1.json', {'directions': [r for r in result['pooled'] if r['alpha'] == 1.],
                                  'advantages': [r for r in result['advantages'] if r['alpha'] == 1.]})
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figdir = root/'figures'; figdir.mkdir(exist_ok=True)
    for m in MODULES:
        fig, ax = plt.subplots(figsize=(8.6, 5.0), constrained_layout=True)
        for d, color in zip(DIRECTIONS, ('#657487', '#cc7722', '#167f77')):
            rows = [r for r in result['pooled'] if r['module'] == m and r['direction'] == d]
            ax.plot([r['alpha'] for r in rows], [r['rho'] if r['valid'] else math.nan for r in rows],
                    marker='o', color=color, label=d)
        ax.axhline(1, color='black', linestyle='--', linewidth=1, label='Fixed small-alpha prediction')
        ax.axvspan(.05, .2, color='#8196b0', alpha=.10, label='Original baseline range')
        ax.set_xlabel('alpha'); ax.set_ylabel('rho = KL / (alpha² q0)')
        ax.set_title(m.replace('model.layers.', 'L')+' | fixed q0, 8 validation windows')
        ax.grid(alpha=.2); ax.legend(fontsize=8); ax.set_xlim(.025, 1.025)
        fig.savefig(figdir/(m.split('.')[2]+'_rho.png'), dpi=180)
        fig.savefig(figdir/(m.split('.')[2]+'_rho.pdf')); plt.close(fig)
    fmt = lambda x: '无效/未确定' if x is None else f'{x:.7g}'
    pct = lambda x: '无效/未确定' if x is None else f'{100*x:+.3f}%'
    nonzero = [r for r in extension if r['alpha'] > 0]
    invalid = [r for r in nonzero if not r['valid']]
    text = ['# 实验一：完整残差幅度扩展', '',
        f'已完成 {len(nonzero)}/192 个新增非零点；无效点 {len(invalid)} 个。原实验保持不变。', '',
        '固定原 alpha=0.05、0.1、0.2 平台的 q_KL 为 q0；没有使用扩展点重拟合或重新选平台。rho−1 为有符号预测偏差；未另设扩展实验的通过阈值。', '',
        '| 模块 | 方向 | alpha=1 实际 KL | 二次预测 | rho−1 | 有效 |',
        '|---|---|---:|---:|---:|---|']
    for r in result['pooled']:
        if r['alpha'] == 1:
            text.append(f"| {r['module']} | {r['direction']} | {fmt(r['actual_KL'])} | {fmt(r['quadratic_KL'])} | {pct(r['signed_bias'])} | {r['valid']} |")
    text += ['', '| 模块 | alpha | 实际 KL 差值 SVD64−A64 | 二次预测差值 | A64 实际 KL 降幅 | A64 胜/有效窗口 | 总体排序反转 |',
             '|---|---:|---:|---:|---:|---:|---|']
    for r in result['advantages']:
        text.append(f"| {r['module']} | {r['alpha']} | {fmt(r['actual_delta'])} | {fmt(r['quadratic_delta'])} | {pct(r['actual_A64_KL_reduction'])} | {r['A64_wins']}/{8-r['invalid_windows']} | {r['rank_reversal']} |")
    text += ['', '![L31 rho](figures/31_rho.png)', '', '![L10 rho](figures/10_rho.png)', '',
        '逐窗口 q0 单独由该窗口原来的三个小幅度点计算；见 windows.csv。逐窗口比较和反转记录见 window_advantages.csv。总体按有效预测 token 数加权，所有窗口路径有效才解释总体 rho；无效点不作为非线性证据。', '',
        '原 MC 二次预测及近似区间仅作为辅助字段保存在 pooled.csv，主分析使用固定小幅度 KL 基准。R_none 是无补偿控制，A64/SVD64 才是同 rank=64 比较。', '',
        'alpha=1 是单个模块沿冻结 FP32 残差的完整扰动；理想算术下对应 Wq+C。该实验不等于全模型同时量化或 BF16 两次 GEMM 部署；结论仅适用于当前模块、方向和八个窗口。', '',
        '资源与路径审计：resource_usage.csv、extension_kl_path.csv、pilot/complete.json、parent_integrity.json。原模型、输入、量化来源和方向身份已核验；仅做前向，没有采集 A/G、分解 SVD 或运行 MC 反传。']
    atomic_bytes(root/'REPORT.md', ('\n'.join(text)+'\n').encode())
    experiment.status('COLLECTION_COMPLETE', nonzero_points=len(nonzero), control_points=len(extension)-len(nonzero),
                      invalid_nonzero_points=len(invalid), alpha1=[r for r in result['pooled'] if r['alpha'] == 1.])
