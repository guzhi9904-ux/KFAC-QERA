"""Independent record validation and CPU-only complete scientific tables."""
from pathlib import Path
import math
import numpy as np
from common import PLAN,read,save_json,save_csv,sha_file,atomic_bytes,require
from records import load_record,checked_files
from analysis import summarize


def number(v):return '不可归一化' if v is None else f'{v:.6g}'


def report(root,verify=True):
    root=Path(root);manifest=read(root/'manifest.json');identity=manifest['identity']
    if not (root/'summary/diagnostics.json').exists():
        counts={r:len(list((root/'cache'/(r+'_S')).glob('*.safetensors'))) for r in ('fit','eval')}
        atomic_bytes(root/'READOUT.md',f'# 实验尚未完成\n\n已提交缓存：{counts}。不能把缺失结果当成零收益或方法失败。\n'.encode())
        return False
    freeze=load_record(root/'candidate_freeze.json',identity);keys=freeze['keys'];index=read(root/'data/sample_index.json')
    require(len(keys)==15 and len(index['fit'])==224 and len(index['eval'])==256,'Protocol cardinality differs')
    require([len(index['budgets'][b]) for b in PLAN['budgets']]==[32,128,128],'Fit budgets differ')
    require(set(index['budgets']['S0'])<=set(index['budgets']['S1']) and set(index['budgets']['S0'])<=set(index['budgets']['S2']),'Fit nesting violated')
    if verify:
        checked_files(root,freeze['files']);checked_files(root,read(root/'data/data_freeze.json')['files'])
        # One sequential checksum pass over the cache. No repeated 60GiB hashing per candidate.
        for role in ('fit','eval'):
            for row in index[role]:
                p=root/'cache'/(role+'_S')/(row['id']+'.safetensors');receipt=read(p.with_suffix('.json'))
                require(receipt['identity']==identity and receipt['sample']==row,'Cache sample binding differs')
                require(sha_file(p)==receipt['file_sha256'],'Cache file changed: '+str(p))
                lp=root/'data/labels'/role/(row['id']+'.safetensors')
                require(sha_file(lp)==receipt['label_file_sha256'],'Cached gradient label changed')
                if role=='fit':
                    gp=root/'cache/fit_g'/(row['id']+'.safetensors');gm=read(gp.with_suffix('.json'))
                    require(sha_file(gp)==gm['file_sha256'] and gm['sample']==row,'Fit g cache differs')
        for role,count in [('fit',32),('eval',16)]:
            for c in range(count):
                p=root/'cache'/(role+'_x')/f'w{c:02d}.safetensors'
                require(sha_file(p)==read(p.with_suffix('.json'))['file_sha256'],'Input cache changed')
    q=[];qrows=[];labels=set();columns=set(keys)
    for row in index['eval']:
        record=load_record(root/'scores/atomic_q'/(row['id']+'.json'),identity)
        require(record['sample']==row and set(record['scores'])==columns,'Projection set/sample differs')
        require(record['candidate_freeze_sha256']==sha_file(root/'candidate_freeze.json'),'Scored candidate freeze differs')
        receipt=read((root/'cache/eval_S'/(row['id']+'.safetensors')).with_suffix('.json'))
        require(record['S_hash']==receipt['tensor_hashes']['S'] and record['label_hash']==receipt['label_hash'],'Projection S or label differs')
        for key,v in record['scores'].items():
            require(v['T']==PLAN['T'] and v['squared']==v['d']*v['d'] and v['q']==v['squared']/(2*PLAN['T']),'Incorrect score normalization')
            qrows.append(dict(sample=row['id'],window=row['window'],replicate=row['replicate'],token_hash=row['token_hash'],label_hash=record['label_hash'],candidate=key,**v))
        q.append([record['scores'][key]['q'] for key in keys]);labels.add(row['id'])
    require(len(qrows)==3840 and len(labels)==256,'Expected 3840 logical scores')
    kl=[];klrows=[];checks=[];guard=0.
    for c in range(PLAN['eval_windows']):
        values=[];selfcheck=load_record(root/'scores/self_KL'/f'w{c:02d}.json',identity)
        require(abs(selfcheck['value'])<=PLAN['tolerances']['self_KL'],'Self-KL failed')
        for key in keys:
            row=load_record(root/'scores/atomic_kl'/f'w{c:02d}_{key}.json',identity)
            require(row['window']==c and row['candidate']==key,'KL record binding differs')
            receipt=read((root/'corrections'/(key+'.safetensors')).with_suffix('.json'))
            require(row['W_hash']==receipt['W_hash'],'KL used wrong deployment')
            require(math.isfinite(row['value']) and row['value']>=0,'Nonfinite KL')
            if c==0:
                require(row['repeat']['absolute_error']<=row['repeat']['allowed'],'KL repeat failed')
                require(row['output_path']['absolute_error']<=row['output_path']['allowed'],'KL output perturbation path failed')
                guard=max(guard,row['repeat']['absolute_error'],row['repeat']['allowed'])
                checks.append(dict(candidate=key,repeat=row['repeat'],output_path=row['output_path']))
            values.append(row['value']);klrows.append(row)
        kl.append(values)
    require(len(klrows)==240,'Expected 240 main KL rows')
    q=np.asarray(q,dtype=np.float64).reshape(16,16,15);stats=summarize(q,kl,keys,guard)
    save_csv(root/'scores/q_per_sample.csv',qrows);save_csv(root/'scores/kl_per_window.csv',klrows)
    save_csv(root/'scores/q_per_window.csv',[dict(window=c,candidate=key,q=float(q[c,:,i].mean())) for c in range(16) for i,key in enumerate(keys)])
    save_json(root/'summary/statistics.json',dict(identity=identity,**stats))
    save_csv(root/'summary/candidate_recovery.csv',stats['candidates'])
    save_csv(root/'summary/paired_comparisons.csv',[r for r in stats['comparisons'] if r['kind']!='sample_budget'])
    save_csv(root/'summary/sample_budget_effects.csv',[r for r in stats['comparisons'] if r['kind']=='sample_budget'])
    diag=read(root/'summary/diagnostics.json');require(diag['identity']==identity and all(r['passed'] for r in diag['bounds']),'Incomplete bound diagnostics')
    pilot=load_record(root/'pilot.json',identity);require(pilot['passed'],'Pilot missing acceptance')
    a0=read(root/'statistics/S0.json');a1=read(root/'statistics/S1.json')
    require(a0['A_asset']==a1['A_asset'] and a0['A_hash']==a1['A_hash'],'S0/S1 input-statistics sharing failed')
    for key in keys:
        if key=='None':continue
        m=read((root/'corrections'/(key+'.safetensors')).with_suffix('.json'))
        require(m['audit']['deployment_relative_drift']<=PLAN['tolerances']['deployment'],'Deployment drift failed')
    verification=dict(identity=identity,passed=True,fit_gradients=224,eval_gradients=256,q_rows=3840,KL_main_rows=240,
        cache_checksums_verified=verify,all_candidates_evaluated=True,A8_shared_exact=True,bound_checks=len(diag['bounds']),
        self_KL_windows=16,KL_path_checks=checks,pilot_passed=True)
    if verify:save_json(root/'verification.json',verification)
    else:require(read(root/'verification.json')['passed'],'No prior full verification')
    resources=read(root/'resource_usage.json');hours=resources['active_seconds']/3600
    lookup={(r['candidate'],r['metric']):r for r in stats['candidates']}
    lines=['# Kronecker sketch / sample-budget pilot','',f'完整验收：通过。L31.q_proj，固定 MXINT3 与 rank64。累计活动时间 {hours:.3f} 小时。',
        '', '所有 15 个部署均评价，无赢家筛选。拟合预算：S0=8×4，S1=8×16，S2=32×4；独立评价=16×16。',
        '', '## 独立评价损伤修复比例','', '| 候选 | q 恢复率 | q 文章95%区间 | KL恢复率 | KL文章95%区间 |','|---|---:|---|---:|---|']
    for key in keys:
        a=lookup[key,'q'];b=lookup[key,'KL']
        lines.append(f"| {key} | {number(a['recovery'])} | [{number(a['ci_low'])}, {number(a['ci_high'])}] | {number(b['recovery'])} | [{number(b['ci_low'])}, {number(b['ci_high'])}] |")
    lines += ['', '恢复率为 1−候选均值/None均值；不是准确率，不裁剪负值。',
        '', '## 预登记方法比较：正数代表比同预算 Marginal 更好','',
        '| 比较 | 指标 | d | 文章95%区间 | 2pp判定 | 条件标签95%区间 |','|---|---|---:|---|---|---|']
    for row in stats['comparisons']:
        if row['kind']!='primary':continue
        ci=row['article'];mc=row.get('conditional_labels');mct=f"[{number(mc['ci_low'])}, {number(mc['ci_high'])}]" if mc else '不适用'
        lines.append(f"| {row['label']} | {row['metric']} | {number(row['d'])} | [{number(ci['ci_low'])}, {number(ci['ci_high'])}] | {ci['status']} | {mct} |")
    lines += ['', '## 等梯度预算：增加文章还是增加标签','',
        '| S2 对 S1 | 指标 | d | 文章95%区间 | 判定 |','|---|---|---:|---|---|']
    for row in stats['comparisons']:
        if row['kind']!='sample_budget' or not row['baseline'].startswith('S1__'):continue
        ci=row['article'];lines.append(f"| {row['label']} | {row['metric']} | {number(row['d'])} | [{number(ci['ci_low'])}, {number(ci['ci_high'])}] | {ci['status']} |")
    lines += ['', 'S1−S0、S2−S0 与 A32−A8 的全部数值见 [sample_budget_effects.csv](summary/sample_budget_effects.csv)。S2/S1 是等梯度预算配置比较，不是纯单因素实验。',
        '', '## 同一评价曲率上的近似质量','',
        '| 候选 | 因子 | cosine | 相对误差 | s* | qK(E) | qK(Rdeploy) |','|---|---|---:|---:|---:|---:|---:|']
    for row in diag['quality']:
        if row['reference']!='eval':continue
        lines.append(f"| {row['candidate']} | {row['representation']} | {row['cosine']:.6g} | {row['relative_error']:.6g} | {row['s_star']:.6g} | {row['qK_E']:.6g} | {row['qK_Rdeploy']:.6g} |")
    lines += ['', '拟合与评价 reference 的全部 raw/solve 收缩见 [curvature_quality.csv](summary/curvature_quality.csv)。不同拟合预算的 Hhat 不同，不能将其原始误差直接横比作总体改善。',
        '', '## 理论界：理想 FP64 残差','',
        '| 候选 | 参照 | 经验超额损伤 | 代理项 | 误差项 | U/None |','|---|---|---:|---:|---:|---:|']
    for row in diag['bounds']:
        if row['reference']!='eval':continue
        lines.append(f"| {row['candidate']} | {row['baseline']} | {row['empirical_excess']:.6g} | {row['scaled_proxy_improvement']:.6g} | {row['error_term']:.6g} | {number(row['U_over_None'])} |")
    bound_ratios=[r['U_over_None'] for r in diag['bounds'] if r['reference']=='eval' and r['U_over_None'] is not None]
    bound_note=('未补偿 q 损伤低于归一化门限，仅报告绝对界值。' if not bound_ratios else
        '这些界均大于整个未补偿损伤，在本例不具备实用的紧约束力。' if min(bound_ratios)>1 else '界的松紧必须结合 U/None 与实际差值判断。')
    lines += ['', '全部经验界校验通过。'+bound_note,
        '', '## 解释边界与资源','',stats['caveats'],
        '', '判定门限是未补偿损伤的 2 个百分点；未确定不等于相等。区间不包含重新拟合随机性，多重逐对区间不控制整体错误率。',
        '', '每个原始因子的最后预定轮次才是主候选；不从评价选择迭代数。J 不含 ||H||²，可以为负；不据此声称误差下降百分比或全局最优。',
        '', '这里只检验一个模块、同语料的新文章，不能外推全模型 PPL、任务准确率或通用算法优势。',
        '', f"输出体积约 {resources.get('output_GiB',0):.3f} GiB；GPU峰值 {resources.get('GPU_peaks',{})}。完整阶段计时见 resource_usage.json。未启动后续实验。"]
    atomic_bytes(root/'RESULTS.md',('\n'.join(lines)+'\n').encode())
    short=['# 本轮实验简读','',f'完整完成：224 份拟合梯度、256 份评价梯度、15 个候选、240 次主 KL；累计 {hours*60:.1f} 分钟。','',
        '下表逐一报告预登记比较；正数是多修复了多少未补偿损伤，乘100为百分点。不能只挑最好一项宣布方法普遍有效。','',
        '| 预算/方法对 Marginal | q差值 | q判定 | KL差值 | KL判定 |','|---|---:|---|---:|---|']
    for budget in PLAN['budgets']:
        for method in PLAN['methods'][1:]:
            pair=[r for r in stats['comparisons'] if r['kind']=='primary' and r['candidate']==budget+'__'+method]
            a=next(r for r in pair if r['metric']=='q');b=next(r for r in pair if r['metric']=='KL')
            short.append(f"| {budget}/{method} | {number(a['d'])} | {a['article']['status']} | {number(b['d'])} | {b['article']['status']} |")
    short += ['', '请结合 RESULTS.md 中的样本预算、curvature cosine 和经验界对照。曲率拟合更像，不保证固定 rank 补丁更好；q 更好也不自动保证真实 KL 更好。',
        '', 'S1 固定8篇文章增加标签；S2 用相同128份梯度覆盖32篇文章。两者比较只能说明这次数据配置的差异。']
    atomic_bytes(root/'READOUT.md',('\n'.join(short)+'\n').encode())
    return True
