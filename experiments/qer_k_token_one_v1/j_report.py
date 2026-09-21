import hashlib
import time
import numpy as np
from j_common import (MODULES,METHOD,BASELINES,TABLE_METHODS,slug,load_record,sha_file,require,mo,read,
                      save_json,save_csv,commit,file_table)


def paired(left,right,none,indices):
    delta=right-left;boot=100*delta[indices].mean(axis=1)/none[indices].mean(axis=1)
    low,high=np.quantile(boot,[.025,.975]);reference=float(right.mean())
    return dict(KL_reduction_right_minus_left=float(delta.mean()),recovery_gain_left_minus_right_pp=float(100*delta.mean()/none.mean()),
                remaining_KL_reduction_percent=None if reference<=1e-12 else float(100*delta.mean()/reference),
                wins=int((delta>0).sum()),exploratory_low_pp=float(low),exploratory_high_pp=float(high))


def report(root,config,ident,resources,source):
    require(load_record(root/'scores/baseline_replay.json',ident)['passed'],'Missing replay acceptance')
    audit=load_record(root/'parent_assets_audit.json',ident);freeze=sha_file(root/'candidate_freeze.json')
    rows=[];per=[];pairs=[];pair_windows=[];bindings=[]
    seed=int.from_bytes(hashlib.sha256(b'k-token-one-v1/bootstrap').digest()[:8],'little')
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
                    p,pid=source.baseline_path(name,method,w)
                    require(sha_file(p)==audit['baseline_record_hashes'][str(p)],'Baseline changed');r=load_record(p,pid)
                require(r['token_hash']==mo.digest_tensor(source.validation[w]) and r['scores']['tokens']==2047,'Window/token mismatch')
                val=r['scores']['KL'];require(np.isfinite(val) and val>=0,'Invalid KL')
                values.append(val);bindings.append(dict(path=str(p),sha256=sha_file(p)))
                per.append(dict(module=name,method=method,window=w,KL=val,tokens=2047))
            series[method]=np.asarray(values,dtype=np.float64)
        none=series['None'];den=float(none.mean());require(den>1e-12 and (none>0).all(),'Degenerate None denominator')
        for method,values in series.items():
            kl=float(values.mean());rows.append(dict(module=name,method=method,KL=kl,KL_None=den,recovery_percent=100*(1-kl/den),
                gain_vs_Marginal_pp=100*(series['Marginal'].mean()-kl)/den,windows=16,tokens=32752))
        for left,right in [(METHOD,'Sensitivity-A'),(METHOD,'Marginal'),('Token-joint-3',METHOD),(METHOD,'Sequence')]:
            pairs.append(dict(module=name,left=left,right=right,**paired(series[left],series[right],none,indices)))
            for w in range(16):pair_windows.append(dict(module=name,left=left,right=right,window=w,KL_left=series[left][w],KL_right=series[right][w],KL_reduction_right_minus_left=series[right][w]-series[left][w]))
    save_csv(root/'summary/results.csv',rows);save_csv(root/'summary/per_window.csv',per)
    save_csv(root/'summary/paired_comparisons.csv',pairs);save_csv(root/'summary/paired_window_differences.csv',pair_windows)
    save_json(root/'summary/results.json',dict(identity=ident,rows=rows,paired_comparisons=pairs,source_records=bindings))
    stages=[]
    for stage in sorted({r['stage'] for r in resources.data['timings']}):
        values=[r['seconds'] for r in resources.data['timings'] if r['stage']==stage and r['completed']]
        stages.append(dict(stage=stage,calls=len(values),summed_call_seconds=sum(values),note='Parallel calls overlap; sum is not wall time'))
    save_csv(root/'summary/stage_costs.csv',stages)
    wins=sum(p['recovery_gain_left_minus_right_pp']>0 for p in pairs if p['left']==METHOD and p['right']=='Sensitivity-A')
    lines=['# K Token-joint-one','',f'第一轮G替换是否比Sensitivity-A更好？四层中{wins}层的实际teacher-KL点估计降低；后两轮的增量逐层列出，不预设方法排序。','',
           '|模块|Marginal|Sensitivity-A|Token-joint-one|Token-joint-3|Sequence|','|---|---:|---:|---:|---:|---:|']
    for name in MODULES:
        data={r['method']:r for r in rows if r['module']==name}
        lines.append('|'+name+'|'+'|'.join(f"{data[m]['recovery_percent']:.4f}%" for m in TABLE_METHODS)+'|')
    lines+=['','|模块|仅量化KL|Joint1绝对KL|Joint1−SensA恢复率pp|Joint1−Marginal恢复率pp|Token3−Joint1恢复率pp|','|---|---:|---:|---:|---:|---:|']
    for name in MODULES:
        vals={r['method']:r for r in rows if r['module']==name}
        r=vals[METHOD];j=r['recovery_percent']
        lines.append(f"|{name}|{r['KL_None']:.10g}|{r['KL']:.10g}|{j-vals['Sensitivity-A']['recovery_percent']:+.4f}|{j-vals['Marginal']['recovery_percent']:+.4f}|{vals['Token-joint-3']['recovery_percent']-j:+.4f}|")
    lines+=['','恢复率越高越好，实际KL越低越好。先按相同预测token数合并KL，再算恢复率。完整绝对KL、剩余KL变化、逐窗口差值与配对区间见summary。',
            'SensA→Joint1保持A方向一致，只改变第一轮G；Joint1→Token3同时改变A/G。后两轮没有引入新样本，重加权同一缓存。',
            'A0为原始未阻尼A_M，G0为I；只做一次同步更新，A1从父U恢复，G1使用旧A_M。完整rank64，继承原阻尼和部署规则。',
            '16验证窗口和2000次配对bootstrap仅作探索性；不覆盖训练采样随机性、相关文本和多轮开发选择。差异不显著不等于等效。',
            '复用320条旧KL，只新增64个主要KL点；没有test、第二轮新候选、其他初始化或事后层选择器。',
            f'累计活跃墙钟时间约{(resources.base+time.monotonic()-resources.started)/3600:.3f}小时。并行调用耗时不可直接相加。']
    (root/'READOUT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    commit(root/'verification.json',ident,passed=True,new_KL=64,reused_KL=320,pilot=read(root/'pilot/one_step_equivalence.json'),rounds=1,no_test=True)
    paths=[root/'READOUT.md',root/'verification.json']+list((root/'summary').glob('*'))
    commit(root/'complete.json',ident,passed=True,layers=4,fit_windows=256,rank=64,files=file_table(root,paths))
    print('TOKEN_ONE_EXPERIMENT_COMPLETE',flush=True)
