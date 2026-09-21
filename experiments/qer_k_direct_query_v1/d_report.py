import hashlib
import time
import numpy as np
from d_common import (MODULES,METHOD,BASELINES,slug,load_record,sha_file,require,mo,read,save_json,
                      save_csv,commit,file_table)


def report(root,config,ident,resources,source):
    load_record(root/'scores/baseline_replay.json',ident)
    audit=load_record(root/'parent_assets_audit.json',ident);freeze=sha_file(root/'candidate_freeze.json')
    rows=[];per=[];pairs=[];bindings=[]
    seed=int.from_bytes(hashlib.sha256(b'k-direct-query-v1/bootstrap').digest()[:8],'little')
    indices=np.random.default_rng(seed).integers(0,16,size=(2000,16))
    save_json(root/'summary/bootstrap.json',dict(seed=seed,indices=indices.tolist(),exploratory=True))
    for name in MODULES:
        series={}
        for method in BASELINES+[METHOD]:
            values=[]
            for w in range(16):
                if method==METHOD:
                    p=root/'scores/validation'/f'w{w:04d}'/(slug(name)+'.json');r=load_record(p,ident)
                    require(r['freeze_hash']==freeze and r['module']==name and r['window']==w and r['candidate']==METHOD,'New score binding differs')
                else:
                    key='None' if method=='None' else 'N256__'+method
                    p=source.root/'scores/validation'/f'w{w:04d}'/(slug(name)+'___'+key+'.json')
                    require(sha_file(p)==audit['baseline_record_hashes'][str(p)],'Old baseline record changed')
                    r=load_record(p,source.identity)
                require(r['token_hash']==mo.digest_tensor(source.validation[w]) and r['scores']['tokens']==2047,'Window/token mismatch')
                val=r['scores']['KL'];require(np.isfinite(val) and val>=0,'Invalid KL')
                values.append(val);bindings.append(dict(path=str(p),sha256=sha_file(p)))
                per.append(dict(module=name,method=method,window=w,KL=val,tokens=2047))
            series[method]=np.asarray(values,dtype=np.float64)
        baseline=float(series['None'].mean());require(baseline>1e-12,'Degenerate baseline')
        marginal=float(series['Marginal'].mean())
        for method,values in series.items():
            val=float(values.mean())
            rows.append(dict(module=name,method=method,KL=val,KL_None=baseline,recovery_percent=100*(1-val/baseline),
                gain_vs_Marginal_pp=100*(marginal-val)/baseline,
                remaining_KL_reduction_vs_Marginal_percent=None if marginal<=1e-12 else 100*(marginal-val)/marginal,
                windows=16,tokens=32752))
        for method in BASELINES[:-1]:
            delta=series[method]-series[METHOD];den=series['None'][indices].mean(axis=1)
            boot=100*delta[indices].mean(axis=1)/den;lo,hi=np.quantile(boot,[.025,.975])
            pairs.append(dict(module=name,reference=method,absolute_KL_difference=float(delta.mean()),
                recovery_gain_pp=float(100*delta.mean()/baseline),wins=int((delta>0).sum()),
                exploratory_low_pp=float(lo),exploratory_high_pp=float(hi)))
    save_csv(root/'summary/results.csv',rows);save_csv(root/'summary/per_window.csv',per)
    save_csv(root/'summary/paired_comparisons.csv',pairs)
    save_json(root/'summary/results.json',dict(identity=ident,rows=rows,paired_comparisons=pairs,source_records=bindings))
    newer=[r for r in rows if r['method']==METHOD];wins=sum(r['gain_vs_Marginal_pp']>0 for r in newer)
    lines=['# Direct Query-Marginal K', '',
           f'唯一主问题：相同N256/rank64下，新构造是否优于标准Marginal-K？本轮四层中{wins}层的实际teacher-KL点估计低于Marginal；逐层差异如下，不能只用层平均代替。',
           '', '| 模块 | A-only | Marginal | Token-joint | Sequence-one-step | Direct Query-Marginal |',
           '|---|---:|---:|---:|---:|---:|']
    for name in MODULES:
        vals={r['method']:r for r in rows if r['module']==name}
        lines.append('| '+name+' | '+' | '.join(f"{vals[m]['recovery_percent']:.4f}%" for m in BASELINES[:-1]+[METHOD])+' |')
    lines+=['', '|模块|新方法绝对KL|相对Marginal恢复率变化pp|相对Marginal剩余KL降低|','|---|---:|---:|---:|']
    for r in newer:
        lines.append(f"|{r['module']}|{r['KL']:.10g}|{r['gain_vs_Marginal_pp']:+.4f}|{r['remaining_KL_reduction_vs_Marginal_percent']}%|")
    lines+=['','实际teacher保留RoPE/GQA，仅代理几何忽略source-dependent相对旋转。G是KV块对角；各query/head外积分开累加；分离q和有效输入的联合依赖。',
            '这是整个候选构造的性能比较，不是单独source/query grouping的因果消融，不是精确Fisher重组。',
            '原320个验证分数复用，64个新KL点新算；同16验证窗口，各2047个预测token，先汇总KL再求恢复率。',
            '2000次配对窗口bootstrap仅为探索性，不涵盖训练随机性、文本相关性或多轮验证选择；本轮没有test。',
            '负结果结束当前候选，不自动转ALS；层间差异不用于事后构造层选择器。',
            f'累计活跃时间约{(resources.base+time.monotonic()-resources.started)/3600:.3f}小时；详细资源见resource_usage.json。']
    text='\n'.join(lines)+'\n'
    for name in ('READOUT.md','RESULTS.md'):(root/name).write_text(text,encoding='utf-8')
    commit(root/'verification.json',ident,passed=True,new_KL=64,reused_KL=320,pilot=read(root/'pilot/complete.json'),
           no_test=True,approximate_RoPE_geometry=True)
    paths=[root/'READOUT.md',root/'RESULTS.md',root/'verification.json']+list((root/'summary').glob('*'))
    commit(root/'complete.json',ident,passed=True,layers=4,fit_windows=256,rank=64,files=file_table(root,paths))
    print('DIRECT_QUERY_EXPERIMENT_COMPLETE',flush=True)
