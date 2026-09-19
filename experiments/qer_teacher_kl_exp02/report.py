"""Persist all predeclared paired summaries and figures, regardless of scientific state."""
import hashlib
import io
import json
import numpy as np
from stats_ops import summarize,bootstrap_indices,DIRECTIONS,METHODS,STEPS


def write_report(experiment,records):
    from storage import read_json,save_json,save_csv,atomic_bytes
    root=experiment.root
    experiment.status('ANALYZING_FIXED_K64')
    baseline=read_json(root/'parent_snapshot/summary.json')
    alpha1=read_json(root/'extension_snapshot/alpha1.json')['directions']
    draws=bootstrap_indices()
    buffer=io.BytesIO();np.savez_compressed(buffer,indices=draws.astype(np.uint8))
    atomic_bytes(root/'bootstrap_indices.npz',buffer.getvalue())
    analysis=summarize(records,baseline,alpha1,draws)
    save_json(root/'analysis.json',analysis)
    for name,rows in analysis.items():save_csv(root/(name+'.csv'),rows)
    flat=[]
    for r in records:
        for d,v in r['directions'].items():
            flat.append({k:value for k,value in r.items() if k not in ('directions','timings')}
                        |dict(direction=d,**v,**r['timings']))
    assert len(flat)==3072
    save_csv(root/'scores.csv',flat)
    save_json(root/'bootstrap.json',{'count':2000,'seed':2026091802,'generator':'numpy.default_rng/PCG64',
        'sample_unit':'within each fixed window, 64 replicate indices, jointly for every score/direction/module',
        'window_resampling':False,'quantile_method':'linear','ratio_denominator_floor':1e-12,
        'indices_sha256':hashlib.sha256(buffer.getvalue()).hexdigest()})
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    modules=list(dict.fromkeys(r['module'] for r in baseline))
    colors=('#456789','#d17a26','#168174')
    fig,axes=plt.subplots(1,3,figsize=(14,5.2),constrained_layout=True)
    errors=[r for r in analysis['summary'] if r['kind']=='score_error']
    labels=[m.replace('model.layers.','L')+' / '+d for m in modules for d in DIRECTIONS]
    for si,step in enumerate(STEPS):
        ax=axes[si]
        rows=[next(r for r in errors if r['module']==m and r['direction']==d and r['quantity']==step) for m in modules for d in DIRECTIONS]
        means=np.array([r['relative_to_q0'] for r in rows]);low=np.array([r['relative_ci_low'] for r in rows]);high=np.array([r['relative_ci_high'] for r in rows])
        ax.axvspan(-.1,.1,color='#75a594',alpha=.17,label='Predeclared small-error band')
        ax.axvline(0,color='black',linewidth=.8)
        ax.errorbar(means,np.arange(6),xerr=np.array([means-low,high-means]),fmt='o',capsize=3,color=colors[si])
        ax.set_yticks(np.arange(6),labels if si==0 else ['']*6);ax.invert_yaxis()
        ax.set_title({'position':'pos - full','dependence':'sep - pos','total':'sep - full'}[step])
        ax.set_xlabel('Signed error / frozen q0');ax.grid(axis='x',alpha=.2)
    folder=root/'figures';folder.mkdir(exist_ok=True)
    fig.savefig(folder/'paired_structure_errors.png',dpi=180);fig.savefig(folder/'paired_structure_errors.pdf');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4.2),constrained_layout=True)
    for ax,module in zip(axes,modules):
        rows=[next(r for r in analysis['normalized'] if r['module']==module and r['kind']=='gamma' and r['quantity']==m) for m in METHODS]
        for i,r in enumerate(rows):
            if r['mean'] is not None:
                ax.vlines(i,r['ci_low'],r['ci_high'],color=colors[i]);ax.plot(i,r['mean'],'o',color=colors[i])
                ax.hlines([r['ci_low'],r['ci_high']],i-.045,i+.045,color=colors[i])
        ax.axhline(0,color='black',linewidth=.8);ax.set_xticks(range(3),METHODS);ax.set_ylabel('gamma = (q_SVD64 - q_A64) / q_none')
        ax.set_title(module.replace('model.layers.','L'));ax.grid(axis='y',alpha=.2)
    fig.savefig(folder/'normalized_benefit.png',dpi=180);fig.savefig(folder/'normalized_benefit.pdf');plt.close(fig)
    def f(x):return 'unresolved' if x is None else f'{x:.6g}'
    def interval(r):return f"{f(r['mean'])} [{f(r['ci_low'])}, {f(r['ci_high'])}]"
    text=['# QER 实验二：位置交叉项与池化边缘乘积', '',
        '采集完成：1024 个原子单元、3072 行方向记录；固定 K=64。full 复现核验与 sep 两种 FP64 收缩等价性均通过。采集完成不等同于结构近似足够。', '',
        '所有区间仅描述条件于八个固定窗口的标签采样不确定性，不是泛化或多重比较同时覆盖区间。差值使用同窗口、同标签先相减再估计 SE；归一化指标使用 2000 次窗口内配对 bootstrap。', '',
        '| 模块 | 方向 | full：均值 [95%区间] | pos | sep | 原 q0 | alpha=1 KL |',
        '|---|---|---:|---:|---:|---:|---:|']
    for module in modules:
        for d in DIRECTIONS:
            rows=[next(r for r in analysis['summary'] if r['module']==module and r['direction']==d and r['kind']=='score' and r['quantity']==m) for m in METHODS]
            text.append('| '+module+' | '+d+' | '+' | '.join(interval(r) for r in rows)+f" | {f(rows[0]['q0'])} | {f(rows[0]['alpha1_KL'])} |")
    text+=['','![两步及总误差](figures/paired_structure_errors.png)','',
           '| 模块 | 方向 | 步骤 | 有符号误差/q0 [95%区间] | 判定 |', '|---|---|---|---:|---|']
    for r in errors:
        text.append(f"| {r['module']} | {r['direction']} | {r['quantity']} | {f(r['relative_to_q0'])} [{f(r['relative_ci_low'])}, {f(r['relative_ci_high'])}] | {r['status']} |")
    text+=['','| 模块 | 类型 | 评分/步骤 | 同 rank 收益或配对收益误差 [95%区间] | 原局部 KL 收益 | alpha=1 实际收益 | 判定 |',
           '|---|---|---|---:|---:|---:|---|']
    for r in analysis['advantages']:
        text.append(f"| {r['module']} | {r['kind']} | {r['quantity']} | {interval(r)} | {f(r['delta0'])} | {f(r['delta1'])} | {r['status']} |")
    text+=['','![归一化收益](figures/normalized_benefit.png)','',
           '| 模块 | 归一化量 | 评分/步骤 | 均值 [bootstrap区间] | 判定 |', '|---|---|---|---:|---|']
    for r in analysis['normalized']:
        if r['kind'] in ('gamma','gamma_difference'):
            text.append(f"| {r['module']} | {r['kind']} | {r['quantity']} | {interval(r)} | {r['status']} |")
    text+=['','nu、nu 步骤差异及所有 gamma 区间见 normalized.csv；逐窗口评分、误差、收益与排序见 windows.csv。原始评分误差带为 ±0.1 q0，收益误差带为 ±0.2 Δ0，gamma 差异误差带为 ±0.05。区间穿越边界时保留 UNRESOLVED_AT_K64。','',
           '误差抵消检查：']
    for r in analysis['cancellation']:text.append(f"- {r['module']} / {r['direction']}：{r['status']}")
    text+=['', '诊断 A 每模块计数 16384，仅作数值对称化，未中心化、加阻尼、截断或通道对角化。M_R 与 sep 收缩均为 FP64，具体实现、实测耗时和矩阵大小保存在 implementation.json、direction_metrics/*.json、resource_usage.csv。','',
           '数值与身份审计：parent_verification.json、numerical_audit.json、pilot.json、parent_integrity.json。保存的原标签跨模块复用；未重新量化、求解补偿或测量 KL。','',
           '结论限于当前两个模块的三个冻结方向。普通边缘乘积失配不能证明单个 Kronecker 家族不足，也不能证明需要 sum-of-Kronecker；原始评分差异大而归一化收益接近时，只能称与共同缩放相容。','',
           '理论背景：[Martens & Grosse 2015](https://proceedings.mlr.press/v37/martens15.html)、[共享权重下的 K-FAC](https://arxiv.org/html/2311.00636v2)。本轮是沿实际残差方向的结构误差诊断，不宣称新的 Fisher 恒等式或重构算法。']
    atomic_bytes(root/'RESULTS.md',('\n'.join(text)+'\n').encode())
    experiment.status('COLLECTION_COMPLETE',records=1024,direction_rows=3072,numerical_replay_passed=True,
                      summary_states={state:sum(r.get('status')==state for r in analysis['summary']) for state in ('SMALL_WITHIN_BUDGET','MATERIAL_DIFFERENCE','UNRESOLVED_AT_K64')})
