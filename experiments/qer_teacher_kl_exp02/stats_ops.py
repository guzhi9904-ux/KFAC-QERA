"""Paired finite-window analysis. No scale fitting and no adaptive sample count."""
import math
import numpy as np

DIRECTIONS = ('R_none', 'R_svd64', 'R_A64')
METHODS = ('full', 'pos', 'sep')
STEPS = ('position', 'dependence', 'total')
BOOTSTRAP_SEED = 2026091802
RATIO_FLOOR = 1e-12


def estimate(values):
    v = np.asarray(values, dtype=np.float64)
    if v.ndim != 2 or v.shape[1] < 2 or not np.isfinite(v).all():
        raise ValueError('Expected finite window-by-replicate values with K>=2')
    n, k = v.shape
    mean = float(v.mean(axis=1).mean())
    se = float(np.sqrt(v.var(axis=1, ddof=1).sum()/k/n**2))
    return dict(mean=mean, SE=se, ci_low=mean-1.96*se, ci_high=mean+1.96*se)


def classify(low, high, band):
    if not (math.isfinite(low) and math.isfinite(high) and low <= high and band > 0):
        raise ValueError('Invalid interval or tolerance')
    if low >= -band and high <= band: return 'SMALL_WITHIN_BUDGET'
    if low > band or high < -band: return 'MATERIAL_DIFFERENCE'
    return 'UNRESOLVED_AT_K64'


def ranking(low, high):
    return 'A64_PREFERRED' if low > 0 else 'SVD64_PREFERRED' if high < 0 else 'RANKING_UNRESOLVED'


def differences(v):
    return np.stack((v[..., 1]-v[..., 0], v[..., 2]-v[..., 1], v[..., 2]-v[..., 0]), axis=-1)


def record_array(records, module, k=64):
    rows = [r for r in records if r['module'] == module]
    keys = [(r['window'], r['replicate']) for r in rows]
    if len(set(keys)) != len(keys): raise ValueError('Duplicate MC unit')
    if set(keys) != {(c,j) for c in range(8) for j in range(k)}: raise ValueError('Incomplete MC grid')
    values = np.empty((8,k,3,3), dtype=np.float64)
    for r in rows:
        if r['L'] != 2048 or r['T'] != 2047 or set(r['directions']) != set(DIRECTIONS):
            raise ValueError('Wrong normalization or incomplete unit')
        for di,d in enumerate(DIRECTIONS):
            for mi,m in enumerate(METHODS): values[r['window'],r['replicate'],di,mi] = r['directions'][d]['b_'+m]
    if not np.isfinite(values).all(): raise ValueError('Nonfinite score')
    return values


def bootstrap_indices(n=8, k=64, count=2000):
    # Reused across modules, all scores, all directions and all differences.
    return np.random.default_rng(BOOTSTRAP_SEED).integers(0,k,size=(count,n,k),dtype=np.int64)


def bootstrap_means(values, indices):
    v = np.asarray(values, dtype=np.float64)
    if indices.shape[1:] != v.shape[:2]: raise ValueError('Bootstrap shape mismatch')
    result = np.zeros((len(indices),)+v.shape[2:], dtype=np.float64)
    for c in range(v.shape[0]): result += v[c][indices[:,c]].mean(axis=1)/v.shape[0]
    return result


def ratio_summary(values, draws, floor=RATIO_FLOOR):
    mean = values.mean(axis=(0,1))
    samples = bootstrap_means(values, draws)
    rows = []
    nu, gamma, boot_nu, boot_gamma = {}, {}, {}, {}
    for mi,m in enumerate(METHODS):
        denominator = mean[0,mi]
        resolved = denominator > floor and np.all(samples[:,0,mi] > floor)
        for di,d in enumerate(DIRECTIONS):
            key = (d,m)
            nu[key] = float(mean[di,mi]/denominator) if resolved else None
            boot_nu[key] = samples[:,di,mi]/samples[:,0,mi] if resolved else None
        gamma[m] = float((mean[1,mi]-mean[2,mi])/denominator) if resolved else None
        boot_gamma[m] = (samples[:,1,mi]-samples[:,2,mi])/samples[:,0,mi] if resolved else None
    def append(kind, direction, quantity, point, samples, band=None):
        if point is None or samples is None or not np.isfinite(samples).all():
            rows.append(dict(kind=kind,direction=direction,quantity=quantity,mean=None,ci_low=None,ci_high=None,
                             status='RATIO_UNRESOLVED',band=band)); return
        lo,hi = [float(x) for x in np.quantile(samples,[.025,.975],method='linear')]
        rows.append(dict(kind=kind,direction=direction,quantity=quantity,mean=point,ci_low=lo,ci_high=hi,
                         status=classify(lo,hi,band) if band else 'DESCRIPTIVE',band=band))
    for d in DIRECTIONS:
        for m in METHODS: append('nu',d,m,nu[(d,m)],boot_nu[(d,m)])
        for step,(first,last) in zip(STEPS,(('full','pos'),('pos','sep'),('full','sep'))):
            ready = nu[(d,first)] is not None and nu[(d,last)] is not None
            append('nu_difference',d,step,nu[(d,last)]-nu[(d,first)] if ready else None,
                   boot_nu[(d,last)]-boot_nu[(d,first)] if ready else None)
    for m in METHODS: append('gamma','SVD64_minus_A64',m,gamma[m],boot_gamma[m])
    for step,(first,last) in zip(STEPS,(('full','pos'),('pos','sep'),('full','sep'))):
        ready = gamma[first] is not None and gamma[last] is not None
        append('gamma_difference','SVD64_minus_A64',step,gamma[last]-gamma[first] if ready else None,
               boot_gamma[last]-boot_gamma[first] if ready else None,.05)
    return rows


def summarize(records, baselines, alpha1, draws):
    modules = list(dict.fromkeys(r['module'] for r in baselines))
    summary, advantages, normalized, windows, cancellation = [], [], [], [], []
    for module in modules:
        values = record_array(records,module)
        errors = differences(values)
        q0 = {r['direction']:r['q_KL'] for r in baselines if r['module']==module}
        kl1 = {r['direction']:r['actual_KL'] for r in alpha1 if r['module']==module}
        delta0,delta1 = q0['R_svd64']-q0['R_A64'],kl1['R_svd64']-kl1['R_A64']
        if delta0 <= 0: raise ValueError('Nonpositive frozen KL benefit')
        for di,d in enumerate(DIRECTIONS):
            component_stats = []
            for kind,names,data in (('score',METHODS,values),('score_error',STEPS,errors)):
                for si,name in enumerate(names):
                    stat=estimate(data[:,:,di,si])
                    row=dict(module=module,direction=d,kind=kind,quantity=name,**stat,q0=q0[d],alpha1_KL=kl1[d],
                             relative_to_q0=stat['mean']/q0[d],relative_ci_low=stat['ci_low']/q0[d],relative_ci_high=stat['ci_high']/q0[d])
                    if kind=='score_error':
                        row.update(band=.1*q0[d],status=classify(stat['ci_low'],stat['ci_high'],.1*q0[d]))
                        component_stats.append(row)
                    summary.append(row)
                    for c in range(8):
                        ws=estimate(data[c:c+1,:,di,si])
                        windows.append(dict(module=module,window=c,direction=d,kind=kind,quantity=name,**ws))
            p,dep,total=component_stats
            opposite=(p['ci_low']>0 and dep['ci_high']<0) or (p['ci_high']<0 and dep['ci_low']>0)
            flag='ERROR_CANCELLATION' if opposite and total['status']=='SMALL_WITHIN_BUDGET' else (
                 'OPPOSING_COMPONENTS_TOTAL_UNRESOLVED' if opposite and total['status']=='UNRESOLVED_AT_K64' else 'NOT_ESTABLISHED')
            cancellation.append(dict(module=module,direction=d,opposite_sign_intervals=opposite,status=flag))
        gains=values[:,:,1,:]-values[:,:,2,:]
        gain_errors=differences(gains)
        for kind,names,data in (('benefit',METHODS,gains),('benefit_error',STEPS,gain_errors)):
            for si,name in enumerate(names):
                stat=estimate(data[:,:,si])
                row=dict(module=module,kind=kind,quantity=name,**stat,delta0=delta0,delta1=delta1,
                         relative_to_delta0=stat['mean']/delta0)
                if kind=='benefit':row['status']=ranking(stat['ci_low'],stat['ci_high'])
                else:row.update(band=.2*delta0,status=classify(stat['ci_low'],stat['ci_high'],.2*delta0))
                advantages.append(row)
                for c in range(8):
                    ws=estimate(data[c:c+1,:,si])
                    windows.append(dict(module=module,window=c,direction='SVD64_minus_A64',kind=kind,quantity=name,**ws,
                                        status=ranking(ws['ci_low'],ws['ci_high']) if kind=='benefit' else 'DESCRIPTIVE'))
        normalized += [dict(module=module,**r) for r in ratio_summary(values,draws)]
    return dict(summary=summary,advantages=advantages,normalized=normalized,windows=windows,cancellation=cancellation)
