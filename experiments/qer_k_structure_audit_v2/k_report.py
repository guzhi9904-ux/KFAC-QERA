import math
import statistics
from k_common import (LAYERS,WINDOWS,METHODS,save_csv,save_json,load_record,require,file_table,commit)


def report(root,ident,resources,baselines):
    allrows={k:[] for k in ('grouped','spectra','post','direction')};paths=[]
    for w in WINDOWS:
        for layer in LAYERS:
            p=root/'probes'/f'w{w:04d}_L{layer}.json';r=load_record(p,ident)
            require(r['window']==w and r['layer']==layer and r['passed'],'Bad probe commit')
            for key in allrows:allrows[key].extend(r[key])
            paths.append(p)
    require(len(allrows['grouped'])==160,'Wrong grouped count')
    spectra=[r for r in allrows['spectra'] if not r['pilot_edge']]
    require(len(spectra)==8192 and len(allrows['post'])==1024 and len(allrows['direction'])==5120,'Wrong probe scope')
    for r in allrows['grouped']:
        base=next(b for b in allrows['grouped'] if b['layer']==r['layer'] and b['window']==r['window'] and b['method']=='None')
        for metric in ('full','source','query'):
            r[metric+'_ratio_to_E']=None if base[metric]<=1e-24 else r[metric]/base[metric]
    save_csv(root/'diagnostics/grouped_scores.csv',allrows['grouped'])
    save_csv(root/'diagnostics/separability.csv',spectra)
    save_csv(root/'diagnostics/pilot_edge_spectra.csv',[r for r in allrows['spectra'] if r['pilot_edge']])
    save_csv(root/'diagnostics/post_activation_error.csv',allrows['post'])
    save_csv(root/'diagnostics/residual_direction_error.csv',allrows['direction'])
    summaries=[];ranks=[];differences=[];geometry=[]
    for layer in LAYERS:
        for metric in ('full','source','query'):
            totals={method:sum(r[metric] for r in allrows['grouped'] if r['layer']==layer and r['method']==method) for method in METHODS}
            for method in METHODS:
                values=[r[metric] for r in allrows['grouped'] if r['layer']==layer and r['method']==method]
                total=totals[method]
                summaries.append(dict(layer=layer,metric=metric,method=method,score=total,
                                      ratio_to_E=None if totals['None']<=1e-24 else total/totals['None'],
                                      largest_window_share=None if total<=1e-24 else max(values)/total))
            ranks.append(dict(layer=layer,metric=metric,order=' < '.join(sorted(METHODS[1:],key=totals.get))))
            for i,m in enumerate(METHODS[1:]):
                for other in METHODS[1:][i+1:]:
                    for w in WINDOWS:
                        vals={r['method']:r[metric] for r in allrows['grouped'] if r['layer']==layer and r['window']==w}
                        differences.append(dict(layer=layer,metric=metric,window=w,left=m,right=other,left_minus_right=vals[m]-vals[other]))
        rows=[r for r in spectra if r['layer']==layer];energy=sum(r['energy'] for r in rows)
        import numpy as np
        for k in (1,2,4,8):
            values=[r[f'rho{k}'] for r in rows if not r['zero_energy']]
            quant=np.quantile(values,[.1,.5,.9]).tolist() if values else [None]*3
            geometry.append(dict(layer=layer,k=k,objects=len(rows),zero_energy=sum(r['zero_energy'] for r in rows),
                                 total_energy=energy,p10=quant[0],median=quant[1],p90=quant[2],
                                 energy_weighted_rho=None if energy<=1e-24 else sum(r['energy']*(r[f'rho{k}'] or 0) for r in rows)/energy))
    save_csv(root/'diagnostics/grouped_summary.csv',summaries);save_csv(root/'diagnostics/grouped_rankings.csv',ranks)
    save_csv(root/'diagnostics/paired_score_differences.csv',differences)
    save_csv(root/'diagnostics/separability_summary.csv',geometry)
    errors=[]
    for key in ('post','direction'):
        for layer in LAYERS:
            for k in (1,4):
                for method in (METHODS if key=='direction' else ['all']):
                    rows=[r for r in allrows[key] if r['layer']==layer and r['k']==k and (key=='post' or r['method']==method)]
                    num=sum(r['error2'] for r in rows);den=sum(r['norm2'] for r in rows)
                    errors.append(dict(kind=key,layer=layer,k=k,method=method,objects=len(rows),error2=num,norm2=den,
                                       absolute_error=math.sqrt(num),epsilon=None if den<=1e-24 else math.sqrt(num/den)))
    save_csv(root/'diagnostics/aggregated_errors.csv',errors)
    validation=[]
    for layer in LAYERS:
        name=f'model.layers.{layer}.self_attn.k_proj'
        kl={m:statistics.mean(r['KL'] for r in baselines if r['module']==name and r['method']==m) for m in METHODS}
        for method in METHODS:validation.append(dict(layer=layer,method=method,KL=kl[method],recovery_percent=100*(1-kl[method]/kl['None'])))
    save_csv(root/'diagnostics/old_validation_summary.csv',validation)
    timing=resources.data['timings'];counts={s:sum(t['stage']==s and t['completed'] for t in timing) for s in ('shared_capture','full_teacher_backward','local_MLP_VJP')}
    lines=['# K structure audit A/B', '',
           '1. 恒等式：两个 pilot 窗口四层 source/cache/full autograd/RoPE/GQA 验收通过；其余窗口继续核验缓存与分组收缩。详见 acceptance 和 probes。',
           '', '2. 固定部署残差的训练侧分组分数（越小越好；不是实际 KL）：', '',
           '|层|方法|q_full|q_src|q_qry|旧 validation KL|旧恢复率|','|---|---|---:|---:|---:|---:|---:|']
    for layer in LAYERS:
        for m in METHODS:
            vals={r['metric']:r['score'] for r in summaries if r['layer']==layer and r['method']==m}
            v=next(r for r in validation if r['layer']==layer and r['method']==m)
            lines.append(f"|{layer}|{m}|{vals['full']:.7g}|{vals['source']:.7g}|{vals['query']:.7g}|{v['KL']:.7g}|{v['recovery_percent']:.3f}%|")
    lines+=['','逐窗口贡献及最大窗口占比见 grouped_summary.csv；配对差值见 paired_score_differences.csv。full−source/query 为带符号交叉项，不是丢失的非负能量。',
            '', '3. RoPE 分离性及乘 X 后误差：', '', '|层|能量加权 rho1|能量加权 rho4|epsilon_M1|epsilon_M4|','|---|---:|---:|---:|---:|']
    for layer in LAYERS:
        rho={r['k']:r['energy_weighted_rho'] for r in geometry if r['layer']==layer}
        eps={r['k']:r['epsilon'] for r in errors if r['layer']==layer and r['kind']=='post'}
        lines.append(f"|{layer}|{rho[1]}|{rho[4]}|{eps[1]}|{eps[4]}|")
    lines+=['','固定残差方向误差见 aggregated_errors.csv 的 direction 行；它是逐对象平方再聚合，不能解释为完整 q_full 误差证书。',
            '', '4. 边界：只有8个固定训练窗口×1套采样标签；窗口不等于独立文章，无法区分文本与标签噪声。旧 validation 已多轮使用。本轮无新实际KL、无新test。',
            '', '5. 候选定义：本审计没有冻结新 Query-aware 方法。还需明确真实RoPE低秩项到A/G的统计映射、缩放规范、跨项/跨head舍弃、全位置计数及N256成本；谱集中不能替代这些定义。不得自动进入C/D。',
            '',f'6. 已完成计时事件次数：{counts}。正常无恢复路径为8次共享前向、2次完整反向、32次局部VJP；恢复后重做的事件也计入。',
            '真实各阶段计时及GPU峰值见 resources/；嵌套计时不能相加当作墙钟时间。增量磁盘仅含小诊断和收缩向量。']
    (root/'READOUT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    resources.flush()
    save_json(root/'resources/timings.json',dict(active_seconds=resources.data['active_seconds'],timings=timing,counts=counts,
              note='Nested timing rows overlap; use active_seconds for total'))
    used=sum(p.stat().st_size for p in root.rglob('*') if p.is_file())/2**30
    save_json(root/'resources/peak_memory.json',dict(GPU_peak_allocated_GiB=resources.data.get('GPU_peak_allocated_GiB'),output_GiB=used))
    paths+=list((root/'diagnostics').glob('*.csv'))+[root/'READOUT.md',root/'resources/timings.json',root/'resources/peak_memory.json']
    commit(root/'complete.json',ident,passed=True,stages=['A','B'],new_K_method_validated=False,N256_started=False,files=file_table(root,paths))
    print('K_STRUCTURE_AUDIT_AB_COMPLETE',flush=True)
