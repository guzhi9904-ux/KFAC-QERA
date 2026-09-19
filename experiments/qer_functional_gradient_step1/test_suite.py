"""CPU formula and independent-statistics tests; never connects to a server."""
import json
import math
import torch
from functional_math import small_checks,decompose,calibrated_ray
from functional_analysis import CANDIDATES,test_statistics,summarize,summarize_kl,METHODS
from verify import verify_summary


def full_shape_algebra_toy():
    rng=torch.Generator().manual_seed(910613)
    a=torch.randn(20,20,dtype=torch.float64,generator=rng);a=a@a.T+torch.eye(20)
    g=torch.randn(24,24,dtype=torch.float64,generator=rng);g=g@g.T+torch.eye(24)
    e=torch.randn(24,20,dtype=torch.float64,generator=rng)
    factors,audits=decompose(e,g@e@a,a,g,16)
    b=factors['Residual']['P']@factors['Residual']['Q'];f=factors['Gradient']['P']@factors['Gradient']['Q']
    assert float((b-f).norm()/b.norm())<1e-10
    for method in ('Residual','Gradient'):
        assert audits[method]['16']['tail_check_error']<1e-8
    zero=calibrated_ray(torch.ones(3,dtype=torch.float64),torch.zeros(3,dtype=torch.float64),1.,7)
    assert zero['a']==zero['b']==zero['beta']==0


def independent_statistics():
    import numpy as np
    rng=np.random.Generator(np.random.PCG64(42));records=[]
    for c in range(8):
        for k in range(16):
            projection=rng.normal(size=17)+(c+1)*.1
            records.append(dict(window=c,replicate=k,scores={name:dict(d=float(d),b=float(d*d/(2*2047))) for name,d in zip(CANDIDATES,projection)}))
    verify_summary(records,summarize('toy',records,16,True),16)
    old=[r for r in records if r['replicate']<4]
    verify_summary(old,summarize('toy',old,4,False),4)
    rows=[dict(window=c,candidate=name,KL=(c+1)*(1-.1*i),repeat_difference=0.) for c in range(8) for i,name in enumerate(('None',)+tuple(f'{m}-r64' for m in METHODS))]
    result=summarize_kl('toy',rows)
    assert all(abs(r['d']-.2)<1e-12 and r['status']=='IMPROVED' for r in result['paired'])
    assert all('ci_low' not in r for r in result['paired'])


if __name__=='__main__':
    torch.set_num_threads(2)
    result=small_checks();full_shape_algebra_toy();test_statistics();independent_statistics()
    print(json.dumps(dict(passed=True,mathematical_checks=result,independent_bootstrap=True,
                         actual_GPU_pilot_executed=False),indent=2))
