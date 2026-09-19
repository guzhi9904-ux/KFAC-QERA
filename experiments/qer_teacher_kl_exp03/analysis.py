"""Fixed-window paired inference, with separate correction and metric decisions."""
from pathlib import Path
import json
import math
import time
import numpy as np

CANDIDATES=('None','SVD64','A64','Marginal-AG64','Full-fit-AG64')


def estimate(v):
    v=np.asarray(v,dtype=np.float64)
    assert v.ndim==2 and v.shape[1]>=2 and np.isfinite(v).all()
    mean=float(v.mean());se=float(np.sqrt(v.var(axis=1,ddof=1).sum()/v.shape[1]/v.shape[0]**2))
    return {'mean':mean,'SE':se,'ci_low':mean-1.96*se,'ci_high':mean+1.96*se}


def decision(lo,hi,tau):
    if lo>tau:return 'IMPROVED'
    if hi<-tau:return 'DEGRADED'
    if lo>=-tau and hi<=tau:return 'SMALL_WITHIN_BUDGET'
    return 'UNRESOLVED_AT_K64'


def ratio(point,draws,den,bootden,floor):
    if den<=floor or np.any(bootden<=floor):return {'mean':None,'ci_low':None,'ci_high':None,'status':'RATIO_UNRESOLVED'}
    sample=draws/bootden;lo,hi=np.quantile(sample,[.025,.975],method='linear')
    return {'mean':float(point/den),'ci_low':float(lo),'ci_high':float(hi),'status':'DESCRIPTIVE'}


def summarize(records,klrows,metricrows,plan):
    modules=plan['modules'];grid={}
    for r in records:
        key=(r['module'],r['window'],r['replicate'])
        if key in grid:raise ValueError('Duplicate evaluation record')
        grid[key]=r
    assert set(grid)=={(m,c,k) for m in modules for c in range(8) for k in range(64)}
    indices=np.random.Generator(np.random.PCG64(plan['bootstrap_seed'])).integers(0,64,size=(2000,8,64),dtype=np.int64)
    quality=[];paired=[];errors=[];windows=[]
    for m in modules:
        v=np.array([[[grid[(m,c,k)]['scores'][d]['b'] for d in CANDIDATES] for k in range(64)] for c in range(8)])
        means=v.mean(axis=(0,1));boot=np.zeros((2000,5))
        for c in range(8):boot+=v[c][indices[:,c]].mean(axis=1)/8
        valid=bool(means[0]>plan['ratio_denominator_floor'] and np.all(boot[:,0]>plan['ratio_denominator_floor']))
        gamma=1-means/means[0] if valid else None
        bgamma=1-boot/boot[:,0,None] if valid else None
        kl=np.array([[next(r['KL'] for r in klrows if (r['module'],r['window'],r['candidate'])==(m,c,d)) for d in CANDIDATES] for c in range(8)])
        km=kl.mean(axis=0);klvalid=km[0]>plan['ratio_denominator_floor']
        for i,d in enumerate(CANDIDATES):
            s=estimate(v[:,:,i]);recovery=ratio(means[0]-means[i],boot[:,0]-boot[:,i],means[0],boot[:,0],plan['ratio_denominator_floor'])
            quality.append({'module':m,'candidate':d,'q_H':s['mean'],'q_H_SE':s['SE'],'q_H_ci_low':s['ci_low'],'q_H_ci_high':s['ci_high'],
                'KL':float(km[i]),'gamma_H':recovery['mean'],'gamma_H_ci_low':recovery['ci_low'],'gamma_H_ci_high':recovery['ci_high'],
                'gamma_KL':float(1-km[i]/km[0]) if klvalid else None,'ratio_status':recovery['status']})
            for c in range(8):
                w=estimate(v[c:c+1,:,i]);windows.append({'module':m,'window':c,'candidate':d,'q_H':w['mean'],'SE':w['SE'],
                    'KL':float(kl[c,i]),'gamma_KL':float(1-kl[c,i]/kl[c,0]) if kl[c,0]>plan['ratio_denominator_floor'] else None})
        raw=estimate(v[:,:,3]-v[:,:,4])
        d=ratio(means[3]-means[4],boot[:,3]-boot[:,4],means[0],boot[:,0],plan['ratio_denominator_floor'])
        hstatus=decision(d['ci_low'],d['ci_high'],plan['tau_H']) if valid else 'RATIO_UNRESOLVED'
        paired.append({'module':m,'quantity':'correction_d_H','mean':d['mean'],'ci_low':d['ci_low'],'ci_high':d['ci_high'],
                       'tau':plan['tau_H'],'status':hstatus,'raw_difference':raw['mean'],'raw_SE':raw['SE'],'raw_ci_low':raw['ci_low'],'raw_ci_high':raw['ci_high']})
        noise=max([r['repeat_difference'] for r in klrows if r['module']==m]+[plan['KL_repeat_absolute_floor']])
        delta=float(km[3]-km[4]);value=delta/km[0] if klvalid else None
        bound=2*noise/km[0] if klvalid else None
        state=decision(value-bound,value+bound,plan['tau_KL']) if klvalid else 'RATIO_UNRESOLVED'
        if state=='UNRESOLVED_AT_K64':state='NUMERICALLY_UNRESOLVED'
        paired.append({'module':m,'quantity':'correction_d_KL','mean':value,'ci_low':None,'ci_high':None,'tau':plan['tau_KL'],
                       'status':state,'raw_difference':delta,'numerical_repeat_guard':bound,'uncertainty':'deterministic fixed windows; no MC confidence interval'})
        for variant in ('raw','solve'):
            maes={};bmaes={}
            for metric in ('marginal','full_fit'):
                predictions=np.array([next(r['gamma_K'] for r in metricrows if (r['module'],r['metric'],r['variant'],r['candidate'])==(m,metric,variant,d)) for d in CANDIDATES[1:]],dtype=float)
                if not valid or not np.isfinite(predictions).all():raise ValueError('Unresolved metric recovery denominator')
                signed=predictions-gamma[1:];bsigned=predictions[None,:]-bgamma[:,1:]
                maes[metric]=float(np.abs(signed).mean());bmaes[metric]=np.abs(bsigned).mean(axis=1)
                for j,dname in enumerate(CANDIDATES[1:]):
                    lo,hi=np.quantile(bsigned[:,j],[.025,.975]);alo,ahi=np.quantile(np.abs(bsigned[:,j]),[.025,.975])
                    errors.append({'module':m,'variant':variant,'metric':metric,'candidate':dname,'gamma_K':float(predictions[j]),
                        'gamma_H':float(gamma[j+1]),'signed_error':float(signed[j]),'absolute_error':float(abs(signed[j])),
                        'signed_ci_low':float(lo),'signed_ci_high':float(hi),'absolute_ci_low':float(alo),'absolute_ci_high':float(ahi),
                        'candidate_set_equal_weight_MAE':maes[metric]})
            improvement=maes['marginal']-maes['full_fit'];draw=bmaes['marginal']-bmaes['full_fit']
            lo,hi=np.quantile(draw,[.025,.975])
            paired.append({'module':m,'quantity':'metric_MAE_improvement_'+variant,'mean':float(improvement),'ci_low':float(lo),'ci_high':float(hi),
                'tau':plan['tau_metric'],'status':decision(lo,hi,plan['tau_metric']),'marginal_MAE':maes['marginal'],'full_fit_MAE':maes['full_fit']})
    return {'correction_quality':quality,'paired_comparisons':paired,'metric_prediction_error':errors,'window_quality':windows},indices


def report(e):
    from storage import save_json,save_csv,sha_file
    records=[json.loads(p.read_text()) for p in sorted((e.root/'eval/records').glob('*.json'))]
    klrows=[json.loads(p.read_text()) for p in sorted((e.root/'eval/kl').glob('*.json'))]
    metrics=[r for p in sorted((e.root/'eval').glob('*_metric.json')) for r in json.loads(p.read_text())['scores']]
    assert len(records)==1024 and len(klrows)==80 and len(metrics)==40
    a,indices=summarize(records,klrows,metrics,e.plan)
    for name,rows in a.items():save_csv(e.root/'summary'/(name+'.csv'),rows)
    save_json(e.root/'analysis.json',a);np.savez_compressed(e.root/'bootstrap_indices.npz',indices=indices.astype(np.uint8))
    save_json(e.root/'bootstrap.json',{'count':2000,'seed':e.plan['bootstrap_seed'],'rng':'PCG64','scope':'resample 64 labels within each fixed window; common across candidates and metrics; windows fixed'})
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    folder=e.root/'figures';folder.mkdir(exist_ok=True)
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    for ax,m in zip(axes,e.plan['modules']):
        rows=[r for r in a['correction_quality'] if r['module']==m and r['candidate']!='None'];x=np.arange(4)
        for i,r in enumerate(rows):
            ax.vlines(i-.07,r['gamma_H_ci_low'],r['gamma_H_ci_high'],color='#376780',lw=2)
        ax.scatter(x-.07,[r['gamma_H'] for r in rows],color='#376780',label='Full curvature recovery')
        ax.scatter(x+.07,[r['gamma_KL'] for r in rows],marker='D',color='#be702c',label='Actual KL recovery')
        ax.axhline(0,color='black',lw=.6);ax.set_xticks(x,['SVD64','A64','Marginal-AG64','Full-fit-AG64'],rotation=12)
        ax.set_title(m.replace('model.layers.','L'));ax.set_ylabel('Recovery relative to None');ax.grid(axis='y',alpha=.2);ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(folder/'correction_recovery.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    for ax,m in zip(axes,e.plan['modules']):
        for j,metric in enumerate(('marginal','full_fit')):
            rows=[r for r in a['metric_prediction_error'] if r['module']==m and r['variant']=='solve' and r['metric']==metric]
            x=np.arange(4)+(j-.5)*.12
            ax.vlines(x,[r['signed_ci_low'] for r in rows],[r['signed_ci_high'] for r in rows],color=('#376780','#be702c')[j],lw=1.5)
            ax.scatter(x,[r['signed_error'] for r in rows],color=('#376780','#be702c')[j],label=metric)
        ax.axhline(0,color='black',lw=.6);ax.set_xticks(np.arange(4),['SVD64','A64','Marginal-AG64','Full-fit-AG64'],rotation=12)
        ax.set_title(m.replace('model.layers.','L'));ax.set_ylabel('Predicted minus full-curvature recovery');ax.grid(axis='y',alpha=.2);ax.legend()
    fig.tight_layout();fig.savefig(folder/'metric_prediction_errors.png',dpi=170);plt.close(fig)
    fmt=lambda x:'未确定' if x is None else f'{x:.7g}'
    lines=['# QER 实验三结果','', '采集与计算完成：2 个模块 × 8 个历史固定评价窗口 × 64 份标签，1024 次配对反传、5120 行候选评分；5 个候选，80 个逐窗口 KL。', '',
           '拟合集为 8 个新的 WT2 train 文章窗口，每窗口 4 份固定 teacher 标签。两个 AG 组共享全部拟合样本。主初始化仅为 canonical marginal；不使用评价集选参。', '',
           '本轮 gamma 是相对 None 的恢复率，与实验二的 A64–SVD64 归一化优势不同。q_H 区间仅反映固定窗口与固定补偿下的标签采样误差；KL 没有 MC 区间。', '',
           '| 模块 | 核心比较 | 点估计 | 95%配对区间 | 阈值 | 判定 |','|---|---|---:|---|---:|---|']
    for r in a['paired_comparisons']:
        interval='不适用（固定 KL）' if r['ci_low'] is None else f"[{fmt(r['ci_low'])}, {fmt(r['ci_high'])}]"
        lines.append(f"| {r['module']} | {r['quantity']} | {fmt(r['mean'])} | {interval} | {r['tau']} | {r['status']} |")
    lines+=['','所有改善量的正值均表示 Full-fit 更好；solve metric 的 MAE 为预测准确度主比较，raw 为解释性比较。','',
            '| 模块 | 补偿 | q_H | q_H 95%区间 | alpha=1 KL | gamma_H | gamma_KL |','|---|---|---:|---|---:|---:|---:|']
    for r in a['correction_quality']:
        lines.append(f"| {r['module']} | {r['candidate']} | {fmt(r['q_H'])} | [{fmt(r['q_H_ci_low'])}, {fmt(r['q_H_ci_high'])}] | {fmt(r['KL'])} | {fmt(r['gamma_H'])} | {fmt(r['gamma_KL'])} |")
    lines+=['','![补偿恢复率](figures/correction_recovery.png)','','![冻结 metric 的收益预测误差](figures/metric_prediction_errors.png)','',
        '**求解与数值验收**','',
        '- H 使用 1/T；marginal canonical G 使用 L/T。所有统计和拟合收缩为完整通道 FP64，raw 不加阻尼；solve 两侧均加 eta=0.001 的 trace-relative damping。',
        '- 完整 dense SVD，rank=64。评价使用实际 FP32 部署权重的精确残差，保存在 FP64 中。旧基线的 KL 在窗口 0 重放后复用历史同部署结果；两个新候选在全部窗口各重复两次 KL。',
        '- ALS 逐块监测 J 的单调性，J 可以为负，不是误差范数。初始化和最大迭代数事先固定，不按评价效果挑选 checkpoint。']
    for m in e.plan['modules']:
        from run import slug
        p=e.root/'factors'/slug(m)/'history.json';hist=json.loads(p.read_text())['iterations']
        from storage import read_tensors
        _,meta=read_tensors(e.root/'factors'/slug(m)/'full_fit_raw.safetensors')
        lines.append(f"- {m}：{len(hist)} 个 ALS 周期；停止原因 {meta['stop_reason']}；J 从 {fmt(hist[0]['J_before'])} 到 {fmt(hist[-1]['J_after_A'])}。不声称全局最优。")
    resources=e.resources
    allocated=max(max(x['allocated_peak_GiB'].values(),default=0) for x in resources)
    rss=max(x['rss_lifetime_peak_GiB'] for x in resources)
    cgroup=max(int(x.get('cgroup_memory',{}).get('memory.peak',0)) for x in resources)/2**30
    lines+=['',f'实测 GPU allocated 峰值 {allocated:.2f} GiB；进程 RSS 峰值 {rss:.2f} GiB；作业 cgroup memory.peak 最大值 {cgroup:.2f} GiB。各阶段耗时见 resource_records.json；调度退出码与释放另存 scheduler_receipt.json。', '',
        '全部原始评分、逐窗口 KL、metric 交叉评分、恢复率区间、MAE 及配对判定见 eval/ 和 summary/。数据、因子、补偿和源码在 evaluation_freeze.json 和 identity.json 中固定。', '',
        '结论只覆盖这些拟合样本、两个模块、rank-64 和四个补偿候选。拟合 Frobenius 目标改善不保证补偿或预测 MAE 改善；任何退化不否定整个 single-Kronecker 家族。不进行全模型 PPL 或新方法搜索。', '',
        '方法来源：[Koroko 等，2022](https://arxiv.org/abs/2201.10285)、[Van Loan 与 Pitsianis](https://users.cs.duke.edu/~nikos/reprints/C-001-KronApprox.pdf)。']
    (e.root/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    e.status('COMPUTATION_COMPLETE',records=1024,candidate_rows=5120,KL_rows=80,
             states=[{'module':r['module'],'quantity':r['quantity'],'status':r['status']} for r in a['paired_comparisons']])
