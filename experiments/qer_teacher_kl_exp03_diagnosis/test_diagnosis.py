import numpy as np
from diag_math import small_checks
from diag_analysis import CANDIDATES,summarize,decision


def test_statistics():
    rows=[]
    for c in range(8):
        for k in range(16):
            for i,d in enumerate(CANDIDATES):
                rows.append(dict(module='test',window=c,replicate=k,candidate=d,b=(c+1)*(k+1)*(1-.1*i)))
    indices=np.random.Generator(np.random.PCG64(2026091805)).integers(0,16,size=(2000,8,16))
    a,_=summarize(rows,['test'],16,indices)
    assert abs(a['paired'][0]['d']-.1)<1e-12
    assert abs(a['paired'][0]['ci_low']-.1)<1e-12
    assert abs(a['paired'][0]['ci_high']-.1)<1e-12
    assert decision(-.1,.1)=='UNRESOLVED_AT_K16'
    assert decision(-.01,.01)=='SMALL_WITHIN_BUDGET'
    assert decision(-.1,-.03)=='DEGRADED'
    assert decision(.03,.1)=='IMPROVED'
    z=[dict(r,b=0.) for r in rows]
    assert summarize(z,['test'],16,indices)[0]['paired'][0]['status']=='RATIO_UNRESOLVED'
    try:summarize(rows+rows[:1],['test'],16,indices)
    except AssertionError:pass
    else:raise AssertionError('Duplicate acceptance')
    old=[r for r in rows if r['replicate']<4]
    a,_=summarize(old,['test'],4)
    assert 'ci_low' not in a['paired'][0] and 'SE' not in a['paired'][0]


if __name__=='__main__':
    print(small_checks());test_statistics();print('All diagnosis numerical/statistical tests passed')
