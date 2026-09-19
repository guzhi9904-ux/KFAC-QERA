"""Aggregate completed/partial modules honestly; never replace missing results by zero."""
from pathlib import Path
import json


def aggregate(root):
    from bridge import read,PLAN,save_json,save_csv,read_tensors,sha_file,slug
    root=Path(root);completed=[];constructed=[];newrows=[];klrows=[];fitrows=[];directions=[];curves=[];paired=[];resources=[];checks={}
    for module in PLAN['modules']:
        folder=root/'modules'/slug(module)
        if not folder.exists():continue
        if (folder/'resource_records.json').exists():resources.extend(dict(module=module,**r) if 'module' not in r else r for r in read(folder/'resource_records.json'))
        if (folder/'candidate_freeze.json').exists():
            constructed.append(module);directions.extend(read(folder/'fit/direction_statistics.json'))
            checks[module]=read(folder/'construction/numerical_checks.json')
            records=read(folder/'fit/records.json')['records']
            fitrows.extend(dict(module=module,window=r['window'],replicate=r['replicate'],candidate=n,
                label_hash=r['label_hash'],S_hash=r['S_hash'],**score) for r in records for n,score in r['scores'].items())
            fit=read(folder/'fit/summary.json')
            curves.extend(dict(domain='fit_empirical',**r) for r in fit['curves'])
            paired.extend(dict(domain='fit_empirical',**r) for r in fit['paired'])
        for p in sorted((folder/'new_labels/records').glob('*.json')):
            r=read(p)
            newrows.extend(dict(module=module,window=r['window'],replicate=r['replicate'],candidate=n,
                label_hash=r['label_hash'],input_hash=r['input_hash'],T=2047,**score) for n,score in r['scores'].items())
        klrows.extend(read(p) for p in sorted((folder/'kl/records').glob('*.json')))
        if (folder/'complete.json').exists():
            completed.append(module);fresh=read(folder/'new_labels/summary.json');kl=read(folder/'kl/summary.json')
            curves.extend(dict(domain='new_labels',**r) for r in fresh['curves'])
            paired.extend(dict(domain='new_labels',**r) for r in fresh['paired'])
            paired.extend(dict(domain='actual_KL',**r) for r in kl['paired'])
    for name,rows in [('fit/direction_statistics.csv',directions),('fit/scores.csv',fitrows),('new_labels/scores.csv',newrows),
                      ('kl/by_window.csv',klrows),('summary/rank_curves.csv',curves),('summary/paired_comparisons.csv',paired)]:save_csv(root/name,rows)
    labels=[]
    for p in sorted((root/'new_labels/samples').glob('*.safetensors')):
        t,m=read_tensors(p);labels.append(dict(path=str(p.relative_to(root)),file_sha256=sha_file(p),**m))
    save_json(root/'new_labels/sample_manifest.json',dict(samples=labels))
    attempts=read(root/'execution_attempts.json') if (root/'execution_attempts.json').exists() else []
    verified=read(root/'verification.json') if (root/'verification.json').exists() else {}
    formal=[r for r in attempts if r['stage']=='formal']
    all_passed=len(completed)==28 and verified.get('passed') and formal and formal[-1]['status']=='COMPLETE'
    state='COMPLETE_VERIFIED' if all_passed else 'PILOT_ACCEPTED' if not formal and (root/'pilot_acceptance.json').exists() else 'INCOMPLETE'
    status=dict(status=state,completed_modules=completed,constructed_modules=constructed,
        missing_modules=[m for m in PLAN['modules'] if m not in completed],observed_fit_score_rows=len(fitrows),
        observed_new_module_records=len(newrows)//17,observed_new_score_rows=len(newrows),observed_KL_rows=len(klrows),
        expected_modules=28,expected_new_module_records=3584,expected_new_score_rows=60928,expected_KL_rows=1120)
    save_json(root/'status.json',status);save_json(root/'resource_usage.json',dict(attempts=attempts,module_records=resources))
    save_json(root/'numerical_checks.json',dict(all_modules_complete=bool(all_passed),modules=checks,
        scalar_verification=verified,small_checks=read(root/'small_checks.json') if (root/'small_checks.json').exists() else None))
    fmt=lambda x:'—' if x is None else f'{x:.7g}'
    lines=['# Fixed-geometry functional-gradient SVD / Step 1','',f'状态：**{state}**。完成 {len(completed)}/28 个模块，构造 {len(constructed)}/28 个模块。',
        '',f'已记录新标签模块样本 {len(newrows)//17}/3584，候选评分 {len(newrows)}/60928，rank-64 KL 主评分 {len(klrows)}/1120。未完成模块保持缺失，不补零、不当作方法失败。','',
        '固定代码层号0、10、20、31，每层七个Linear；逐次只修改一个模块。Marginal solve几何、原8×4构造样本、四个rank、beta和全部部署权重在评价前冻结。',
        '', '## rank-64 主对照','',
        'd=(Residual损伤−Gradient损伤)/None损伤，正值支持Gradient；raw和calibrated分别配对。新标签95%区间仅覆盖固定文本和候选下的标签采样误差。KL是固定文本上的确定性测量，只有重复数值误差保护，不使用标签bootstrap区间。',
        '', '| 模块 | 比较 | d_new | 95%配对区间 | 新标签判定 | d_KL | 相对Residual的KL改善% | KL判定 |',
        '|---|---|---:|---|---|---:|---:|---|']
    for m in completed:
        for comparison in ('raw','calibrated'):
            a=next(r for r in paired if r['domain']=='new_labels' and r['module']==m and r['rank']==64 and r['comparison']==comparison)
            b=next(r for r in paired if r['domain']=='actual_KL' and r['module']==m and r['comparison']==comparison)
            interval=f"[{fmt(a.get('ci_low'))}, {fmt(a.get('ci_high'))}]"
            lines.append(f"| {m} | {comparison} | {fmt(a['d'])} | {interval} | {a['status']} | {fmt(b['d'])} | {fmt(b['relative_to_residual_percent'])} | {b['status']} |")
    lines+=['','## 解释边界','',
        '- 原拟合经验下降只是数值验收；不作为样本外成功证据，也不给参与构造的32份标签套泛化置信区间。',
        '- 若raw较差、共同幅度校准后更好，支持该补偿射线存在幅度失配；不等于证明整个行列子空间更优。若校准后相近，原优势可能主要来自幅度。',
        '- 新标签与KL均改善，才支持当前固定几何下该构造更有效；新标签改善但KL不改善，需报告二次目标未转化为实际损伤收益。',
        '- 不声称新文本泛化、全局rank-r最优、学到了更好的左右几何或已证明方法新颖性。不得仅挑改善模块或rank。',
        '- 恢复率不裁剪。归一化分母≤1e-12时不解释比值；原始损伤保留。2pp阈值判断与是否排除零分别保存。',
        '', '## 低成本方向诊断','',
        '[方向统计](fit/direction_statistics.csv)包含a、b、beta、范数、代理曲率比、代理收益、理想与部署q_fit，以及舍入漂移。'
        '[前缀曲线](summary/rank_curves.csv)含四组、四个预定rank的经验／新标签恢复率和区间。'
        '[配对比较](summary/paired_comparisons.csv)保留每模块每rank raw/calibrated比较。',
        '', '## 未完成清单','']
    lines.extend('- '+m for m in status['missing_modules'])
    if not status['missing_modules']:lines.append('无。')
    lines+=['','资源和检查见 [resource_usage.json](resource_usage.json)、[numerical_checks.json](numerical_checks.json)、[verification.json](verification.json)。资源租约由外层gpu命令管理，本程序不会自动申请、强制释放或进行管理员操作。','']
    if all_passed:
        plot_curves(root,curves,PLAN['layers'])
        for layer in PLAN['layers']:
            for domain in ('fit_empirical','new_labels'):
                lines+=['',f'![L{layer} {domain}](figures/L{layer}_{domain}.png)','']
    (root/'RESULTS.md').write_text('\n'.join(lines),encoding='utf8')
    return status


def plot_curves(root,curves,layers):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from functional_analysis import METHODS,RANKS
    folder=root/'figures';folder.mkdir(exist_ok=True)
    for layer in layers:
        for domain in ('fit_empirical','new_labels'):
            modules=sorted({r['module'] for r in curves if r['domain']==domain and r['module'].startswith(f'model.layers.{layer}.')})
            fig,axes=plt.subplots(3,3,figsize=(14,11))
            for ax,module in zip(axes.flat,modules):
                for method in METHODS:
                    rows=[next(r for r in curves if r['domain']==domain and r['module']==module and r['method']==method and r['rank']==rank) for rank in RANKS]
                    if any(r['gamma'] is None for r in rows):continue
                    line=ax.plot(RANKS,[100*r['gamma'] for r in rows],marker='o',label=method)[0]
                    if domain=='new_labels':
                        ax.vlines(RANKS,[100*r['gamma_ci_low'] for r in rows],[100*r['gamma_ci_high'] for r in rows],color=line.get_color(),alpha=.7)
                ax.set_title(module.split(f'layers.{layer}.')[1]);ax.axhline(0,color='black',lw=.7)
                ax.set_yscale('symlog',linthresh=100);ax.set_xticks(RANKS);ax.set_xlabel('rank (64 primary)')
                ax.set_ylabel('Recovery %; symlog outside +/-100%');ax.grid(alpha=.2)
            for ax in list(axes.flat)[len(modules):]:ax.set_visible(False)
            handles,labels=axes.flat[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=4)
            fig.suptitle(f'Layer {layer}: {domain}; fixed marginal geometry')
            fig.tight_layout(rect=(0,.04,1,.95));fig.savefig(folder/f'L{layer}_{domain}.png',dpi=160);fig.savefig(folder/f'L{layer}_{domain}.pdf');plt.close(fig)
