"""Teacher-only streaming estimators; no raw x/g/P/delta archive."""
import contextlib
from functools import wraps
import gc
import time
import torch
from torch.utils.checkpoint import checkpoint
from fm_common import *
from data import checked_ids, label

def functional_capture(model, ids, labels, layers, chunk=64):
    states = {}; probabilities = {}; handles = []; originals = []; phase = {'recompute': False}
    names = [name(i,k) for i in layers for k in ('q','k','v','o')]
    @contextlib.contextmanager
    def replay():
        phase['recompute'] = True
        try:
            yield
        finally:
            phase['recompute'] = False
    def target(key):
        def hook(module, args, output):
            if not phase['recompute']:
                require(key not in states, 'Repeated original target forward')
                states[key] = {'x': args[0].detach().cpu(), 'h': output}
                if key.endswith('o_proj'):
                    states[key]['live_input'] = args[0]
        return hook
    def force(module, args, kw):
        require(not kw.get('use_cache',False), 'Unexpected KV cache')
        return args, dict(kw, output_attentions=True)
    def save_p(i):
        def hook(module, args, out):
            if not phase['recompute']:
                require(out[1] is not None and out[1].dtype == torch.float32, 'Native eager probabilities absent')
                probabilities[i] = out[1][0].detach().cpu()
        return hook
    try:
        handles.append(model.model.embed_tokens.register_forward_hook(lambda m,a,o: o.detach().requires_grad_(True)))
        for key in names:
            handles.append(model.get_submodule(key).register_forward_hook(target(key)))
        for i in layers:
            attn = model.model.layers[i].self_attn
            handles += [attn.register_forward_pre_hook(force,with_kwargs=True), attn.register_forward_hook(save_p(i))]
        for layer in model.model.layers:
            original = layer.forward
            @wraps(original)
            def wrapped(*args, _forward=original, **kwargs):
                return checkpoint(_forward,*args,use_reentrant=False,preserve_rng_state=True,
                                  context_fn=lambda:(contextlib.nullcontext(),replay()),**kwargs)
            originals.append((layer,original)); layer.forward = wrapped
        h = hidden_forward(model,ids)
        seed = mo.sampled_seed(h.detach(),model.lm_head.weight,labels,chunk)
        inputs = [states[n]['h'] for n in names] + [states[name(i,'o')]['live_input'] for i in layers]
        gs = torch.autograd.grad(h,inputs,grad_outputs=seed)
        result = {n: dict(x=states[n]['x'],g=g.detach().reshape(L,-1).cpu()) for n,g in zip(names,gs[:len(names)])}
        for i,g in zip(layers,gs[len(names):]):
            result[name(i,'v')].update(prob=probabilities[i],delta=g.detach().reshape(L,-1).cpu())
        result_hidden = h.detach().cpu()
    finally:
        for handle in handles:
            handle.remove()
        for layer, original in originals:
            layer.forward = original
    require(set(result)==set(names), 'Incomplete gradient targets')
    for row in result.values():
        require(row['x'].dtype == row['g'].dtype == torch.float32 and torch.isfinite(row['g']).all(), 'Invalid native gradient')
        require(float(row['g'][-1].abs().max()) == 0., 'Last no-loss position has nonzero gradient')
    return result_hidden, result

def ordinary_forward(model, ids, layers):
    terms = {}; handles=[]
    def gram(i,family):
        def hook(module,args):
            x=args[0].detach().reshape(L,-1).double()
            terms[f'{i}.{family}']=(x.T@x).cpu()
        return hook
    try:
        for i in layers:
            for family,kind in GROUP_KEYS.items():
                handles.append(model.get_submodule(name(i,kind)).register_forward_pre_hook(gram(i,family)))
        with torch.no_grad():
            hidden=hidden_forward(model,ids).detach()
    finally:
        for handle in handles:
            handle.remove()
    return hidden,terms

def functional_terms(rows,layers,ordinary,explicit=False):
    terms={}; checks={}
    for i in layers:
        device=f'cuda:{i>=16:d}'
        for kind in ('q','k','v','o'):
            g=rows[name(i,kind)]['g'].to(device,dtype=torch.float64)
            terms[f'{i}.G_{kind}']=(g.T@g).cpu()
        row=rows[name(i,'k')]
        x=row['x'].reshape(L,-1).to(device,dtype=torch.float64)
        g=row['g'].to(device,dtype=torch.float64); a=ordinary[i].to(device)
        weights=g.square().sum(-1); quadratic=((x@a)*x).sum(-1)
        require(float(quadratic.min()) >= -1e-10*float(quadratic.abs().max()), 'Significantly negative x A_M x; no clipping')
        terms[f'{i}.K_A']=(x.T@(x*weights[:,None])).cpu()
        terms[f'{i}.K_G']=(g.T@(g*quadratic[:,None])).cpu()
        v=rows[name(i,'v')]
        eff=effective(v['prob'],v['x'],v['delta'],[h//4 for h in range(32)],8,device,explicit=explicit)
        checks[str(i)]={'g_v':compare(eff['g_v'],v['g'],1e-5)}
        if explicit:
            checks[str(i)]['A_equivalence']=compare(eff['A_sum'],eff['explicit_A_sum'],1e-10)
        terms[f'{i}.V_A']=eff['A_sum'].cpu(); terms[f'{i}.V_G']=eff['G_blocks_sum'].cpu()
        del x,g,a,eff
    return terms,checks

def group_ranges(size):
    require(size in (4,8), 'Only contiguous 4/8-layer engineering groups are supported')
    return [list(range(i,min(i+size,32))) for i in range(0,32,size)]

def collect_a(ctx):
    require(ctx.done('verification/pilot_complete.json'), 'Pilot gate not passed')
    if ctx.done('statistics/a_complete.json', windows=N, groups=4):
        return
    ids=checked_ids(ctx,'calibration'); ctx.teacher.load(); files=[]
    for layers in group_ranges(ctx.config['pass_a_group_layers']):
        cp=f'statistics/a_group_{layers[0]:02d}.json'
        if ctx.done(cp,windows=N,layers=layers):
            files.append(ctx.root/cp);continue
        shapes={f'{i}.{family}':(14336,14336) if family=='down' else (4096,4096)
                for i in layers for family in GROUP_KEYS}
        acc=Accumulator(ctx,f'a_{layers[0]:02d}',shapes,dict(data=sha(ctx.root/'data_manifest.json'),layers=layers,stage='ordinary_A'))
        start=time.monotonic(); initial=acc.count
        for window in range(acc.count,N):
            ctx.check()
            hidden,terms=ordinary_forward(ctx.teacher.model,ids[window:window+1],layers)
            if layers[0]==0:
                label(ctx,window,hidden)
            acc.add(terms,window);del terms,hidden
            log('PASS_A_WINDOW',layers=layers,window=acc.count,total=N)
            if acc.count%ctx.config['checkpoint_every']==0 or acc.count==N:
                with ctx.timed('checkpoint_a',layers=layers,windows=acc.count):acc.save()
                seconds=time.monotonic()-start
                log('PASS_A_PROGRESS',layers=layers,window=acc.count,total=N,
                    group_eta_seconds=(N-acc.count)*seconds/(acc.count-initial),resources=ctx.resource_snapshot())
        paths=[]
        for i in layers:
            for family in GROUP_KEYS:
                p=ctx.stat(i,'A_'+family)
                tensors(p,{'A':sm.sym(acc.values[f'{i}.{family}']/(N*L))});paths.append(p)
        ctx.commit(cp,paths,windows=N,layers=layers,normalization=N*L,all_module_positions=True)
        ctx.cleanup_temporary(acc.folder); del acc;gc.collect();files.append(ctx.root/cp)
    labels=torch.stack([label(ctx,w) for w in range(N)])
    tensors(ctx.root/'data/predictive_labels.safetensors',{'labels':labels,'prediction_mask':torch.ones_like(labels)})
    ctx.commit('data/labels_complete.json',[ctx.root/'data/predictive_labels.safetensors',
               *sorted((ctx.root/'data/labels').glob('*'))],windows=N,positions=T)
    files.append(ctx.root/'data/labels_complete.json')
    ctx.commit('statistics/a_complete.json',files,windows=N,groups=4)
    ctx.teacher.unload()

def collect_functional(ctx):
    require(ctx.done('statistics/a_complete.json',windows=N,groups=4), 'Complete ordinary A is required before K G1')
    require(ctx.done('data/labels_complete.json',windows=N,positions=T), 'Frozen predictive labels absent')
    if ctx.done('statistics/functional_complete.json',windows=N,groups=8):
        return
    ids=checked_ids(ctx,'calibration');ctx.teacher.load();files=[]
    for layers in group_ranges(ctx.config['pass_b_group_layers']):
        cp=f'statistics/functional_group_{layers[0]:02d}.json'
        if ctx.done(cp,windows=N,layers=layers):
            files.append(ctx.root/cp);continue
        ordinary={i:load_file(str(ctx.stat(i,'A_qkv')))['A'] for i in layers}
        shapes={f'{i}.G_{kind}':(1024,1024) if kind in ('k','v') else (4096,4096)
                for i in layers for kind in ('q','k','v','o')}
        for i in layers:
            shapes.update({f'{i}.K_A':(4096,4096),f'{i}.K_G':(1024,1024),
                           f'{i}.V_A':(4096,4096),f'{i}.V_G':(8,128,128)})
        binding=dict(data=sha(ctx.root/'data_manifest.json'),labels=sha(ctx.root/'data/labels_complete.json'),
                     ordinary={str(i):sha(ctx.stat(i,'A_qkv')) for i in layers},layers=layers)
        acc=Accumulator(ctx,f'functional_{layers[0]:02d}',shapes,binding)
        start=time.monotonic();initial=acc.count
        for window in range(acc.count,N):
            ctx.check()
            _,rows=functional_capture(ctx.teacher.model,ids[window:window+1],label(ctx,window),layers,ctx.config['vocab_chunk'])
            terms,checks=functional_terms(rows,layers,ordinary,explicit=False)
            acc.add(terms,window);del rows,terms
            log('PASS_B_WINDOW',layers=layers,window=acc.count,total=N)
            write(ctx.root/'verification/functional'/f'L{layers[0]:02d}_w{window:04d}.json',checks)
            if acc.count%ctx.config['checkpoint_every']==0 or acc.count==N:
                with ctx.timed('checkpoint_functional',layers=layers,windows=acc.count):acc.save()
                elapsed=time.monotonic()-start
                log('PASS_B_PROGRESS',layers=layers,window=acc.count,total=N,
                    group_eta_seconds=(N-acc.count)*elapsed/(acc.count-initial),resources=ctx.resource_snapshot())
        paths=[]
        for i in layers:
            for kind in ('q','k','v','o'):
                p=ctx.stat(i,'G_'+kind);tensors(p,{'G':sm.sym(acc.values[f'{i}.G_{kind}']/(N*T))});paths.append(p)
            p=ctx.stat(i,'K_one');tensors(p,dict(A=sm.sym(acc.values[f'{i}.K_A']/(N*T*1024)),
                          G=sm.sym(acc.values[f'{i}.K_G']/(N*T*ordinary[i].square().sum()))));paths.append(p)
            av,gv=normalize(acc.values[f'{i}.V_A'],acc.values[f'{i}.V_G'],N,L,32)
            p=ctx.stat(i,'V_attention');tensors(p,dict(A=sm.sym(av),G=sm.sym(gv)));paths.append(p)
        ctx.commit(cp,paths,windows=N,layers=layers,K_initialization='complete raw N256 A_M,I',K_rounds=1,
                   G_denominator=N*T,V_A_denominator=N*L*32,labels_sha256=binding['labels'])
        ctx.cleanup_temporary(acc.folder);del acc,ordinary;gc.collect();files.append(ctx.root/cp)
    ctx.commit('statistics/functional_complete.json',files,windows=N,groups=8)
    ctx.teacher.unload()
