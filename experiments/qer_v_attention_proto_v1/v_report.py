import hashlib
import time
import numpy as np
from v_common import MODULES,slug,load_record,sha_file,require,mo,read,save_json,save_csv,commit,file_table


def paired(a,m,s,b,indices):
    denominator=float(np.mean(b));out={}
    for name,reference in (('vs_Marginal',m),('vs_Sequence',s)):
        if denominator<=1e-12:out[name]=dict(point=None,low=None,high=None,denominator_degenerate=True);continue
        point=float(np.mean(reference-a)/denominator);den=b[indices].mean(axis=1)
        boot=np.divide((reference-a)[indices].mean(axis=1),den,out=np.full(len(indices),np.nan),where=den>1e-12)
        if not np.isfinite(boot).all():low=high=None
        else:low,high=map(float,np.quantile(boot,[.025,.975]))
        out[name]=dict(point=point,low=low,high=high,denominator_degenerate=False,
                       within_practical_band=low is not None and low>=-.02 and high<=.02,
                       practical_threshold=.02,bootstrap_valid=int(np.isfinite(boot).sum()))
    return out


def report(root,config,ident,resources,source):
    load_record(root/'scores/baseline_replay.json',ident)
    audit=load_record(root/'parent_assets_audit.json',ident);freeze=sha_file(root/'candidate_freeze.json')
    rows=[];per_window=[];comparisons=[];bindings=[]
    seed=int.from_bytes(hashlib.sha256(b'qer-vattn-proto-v1/bootstrap').digest()[:8],'little')
    indices=np.random.default_rng(seed).integers(0,16,size=(2000,16))
    save_json(root/'summary/bootstrap.json',dict(seed=seed,indices=indices.tolist(),interpretation='Exploratory paired window bootstrap; window dependence and fit resampling not covered; no across-layer guarantee'))
    for n in MODULES:
        series={}
        for method in ['Marginal','Attention-aware','Sequence-one-step','None']:
            values=[]
            for w in range(16):
                if method=='Attention-aware':
                    p=root/'scores/validation'/f'w{w:04d}'/(slug(n)+'.json');r=load_record(p,ident)
                    require(r['freeze_hash']==freeze and r['module']==n and r['window']==w,'New record binding changed')
                else:
                    key='None' if method=='None' else 'N256__'+method
                    p=source.root/'scores/validation'/f'w{w:04d}'/(slug(n)+'___'+key+'.json')
                    require(sha_file(p)==audit['baseline_record_hashes'][str(p)],'Imported baseline changed')
                    r=load_record(p,source.identity)
                require(r['token_hash']==mo.digest_tensor(source.validation[w]) and r['scores']['tokens']==2047,'Evaluation token binding changed')
                value=r['scores']['KL'];values.append(value);bindings.append(dict(path=str(p),sha256=sha_file(p)))
                per_window.append(dict(module=n,method=method,window=w,KL=value,tokens=2047))
            series[method]=np.asarray(values,dtype=np.float64)
        baseline=float(series['None'].mean())
        for method,values in series.items():
            value=float(values.mean());rows.append(dict(module=n,method=method,KL=value,KL_None=baseline,
                recovery_percent=None if baseline<=1e-12 else 100*(1-value/baseline),windows=16,tokens=32752))
        effects=paired(series['Attention-aware'],series['Marginal'],series['Sequence-one-step'],series['None'],indices)
        for comparison,values in effects.items():comparisons.append(dict(module=n,comparison=comparison,**values,
            interpretation='Exploratory only; 16 reused validation windows; no article-independence assumption'))
        for w in range(16):
            b=series['None'][w]
            for record in per_window:
                if record['module']==n and record['window']==w:
                    record['recovery_percent']=None if b<=1e-12 else 100*(1-record['KL']/b)
                    record['d_AM']=None if b<=1e-12 else float((series['Marginal'][w]-series['Attention-aware'][w])/b)
                    record['d_AS']=None if b<=1e-12 else float((series['Sequence-one-step'][w]-series['Attention-aware'][w])/b)
    save_csv(root/'summary/results.csv',rows);save_csv(root/'summary/per_window.csv',per_window)
    save_csv(root/'summary/paired_comparisons.csv',comparisons)
    save_json(root/'summary/results.json',dict(identity=ident,rows=rows,paired_comparisons=comparisons,source_records=bindings))
    total=resources.base+time.monotonic()-resources.started
    lines=['# V attention-aware prototype','', '四层均保留；256冻结fit窗口、一套原标签、全矩阵rank64、同16 validation窗口。实际teacher-KL，不是q_full。','',
           '| 模块 | 方法 | KL | 恢复率 |','|---|---|---:|---:|']
    for r in rows:
        recovery='不可判定' if r['recovery_percent'] is None else f"{r['recovery_percent']:.4f}%"
        lines.append(f"| {r['module']} | {r['method']} | {r['KL']:.10g} | {recovery} |")
    lines += ['', '## 配对效应（正值支持Attention-aware；单位为未补偿KL的比例）','', '| 模块 | 对比 | 点估计 | 探索性95%区间 | ±0.02带内 |','|---|---|---:|---|---|']
    for r in comparisons:lines.append(f"| {r['module']} | {r['comparison']} | {r['point']} | [{r['low']}, {r['high']}] | {r.get('within_practical_band')} |")
    lines += ['',f'累计活跃时间约{total/3600:.3f}小时；分阶段真实计时见resource_usage.json（并行工作量不能相加当墙钟时间）。',
              '', 'G为8个KV-head块对角矩阵，A为全部query heads共享的输入度量。它不是完整GQA Fisher的精确分解。',
              'FP64重排和FP32 autograd检查见pilot/*.json；阻尼/条件数/根/SVD/部署检查见factors/*/solve_audit.json。',
              '2000次配对窗口bootstrap仅为探索性，未控制窗口相关性、不包含fit重采样、不保证四层整体显著性。',
              '只有d_AS整个区间位于[-0.02,+0.02]才描述为当前实用门限内接近Sequence；区间宽则未确定。',
              '本prototype同时改变变量与head相关处理；效果不能唯一归因为某一机制。负结果不等于梯度恒等式错误。']
    (root/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    (root/'READOUT.md').write_text('\n'.join(lines[:4]+lines[-7:])+'\n',encoding='utf-8')
    commit(root/'verification.json',ident,passed=True,new_KL=64,reused_baseline_KL=192,baseline_replays=12,new_repeats=4,
           check1=read(root/'pilot/complete.json'),head_mapping=read(root/'head_mapping.json'),no_test=True,active_seconds=total)
    paths=[root/'RESULTS.md',root/'summary/results.json',root/'summary/results.csv',root/'summary/paired_comparisons.csv',root/'verification.json']
    commit(root/'complete.json',ident,passed=True,layers=4,files=file_table(root,paths))
    print('V_ATTENTION_PROTOTYPE_COMPLETE',flush=True)
