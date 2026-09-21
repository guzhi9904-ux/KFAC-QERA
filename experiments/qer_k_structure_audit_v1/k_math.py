import math
import torch
from k_common import require,METHODS,QUERIES


def half(x):
    a,b=x.chunk(2,dim=-1);return torch.cat((-b,a),dim=-1)


def rotate(x,c,s):return x*c+half(x)*s


def transpose_rotate(y,c,s):
    # Exact transpose of diag(c)+diag(s)*J, including unequal half coefficients.
    return y*c-half(y*s)


def compare(actual,reference,tolerance,scale=None):
    a=actual.detach().double();b=reference.detach().to(a.device).double();d=a-b
    norm=float(b.norm());absolute=float(d.norm());scale=max(1e-12,norm if scale is None else float(scale))
    degenerate=norm<=1e-12*scale
    bound=tolerance*scale if degenerate else tolerance*norm
    row=dict(reference_norm=norm,absolute_error=absolute,max_element_error=float(d.abs().max()),
             relative_error=absolute/max(norm,1e-300),cancellation_scale=scale,degenerate=degenerate,
             absolute_bound=bound,tolerance=tolerance,passed=absolute<=bound)
    require(row['passed'],'Numerical check failed: '+str(row));return row


def spectrum(f):
    gram=f@f.T;values,vectors=torch.linalg.eigh((gram+gram.T)*.5)
    energy=float((f*f).sum());require(float(values[0])>=-1e-10*max(energy,1e-24),'Negative small Gram spectrum')
    values=values.clamp_min(0);zero=energy<=1e-24
    return dict(energy=energy,zero_energy=zero,**{f'rho{k}':None if zero else float(values[-k:].sum())/energy for k in (1,2,4,8)}),vectors


def probe_layer(state,delta,cached_g,residuals,device,window,layer,window_index,resources,pilot=False,full=None):
    length=2048;d=128;hq=32;hkv=8
    x=state['x'].reshape(length,-1).to(device).double()
    c=state['cos'].to(device).double();s=state['sin'].to(device).double()
    delta=delta.reshape(length,hq,d).to(device)
    v=state['v'].reshape(length,hkv,d).to(device)
    z=state['z'].reshape(length,hq,d).to(device)
    residuals=torch.stack([residuals[m] for m in METHODS]).to(device)
    with resources.timed('five_residual_Rx',window=window,layer=layer):
        rx=torch.matmul(x,residuals.transpose(1,2)).reshape(5,length,hkv,d)
    g=torch.zeros((hkv,length,d),device=device,dtype=torch.float64)
    head_score=torch.zeros((hkv,d,x.shape[1]),device=device,dtype=torch.float64)
    query=torch.zeros((5,length),device=device,dtype=torch.float64)
    cancellation=torch.zeros(5,device=device,dtype=torch.float64)
    checks=[];spectra=[];post=[];direction=[];row_audits=[]
    for a in range(hq):
        resources.boundary();b=a//4
        with resources.timed('native_softmax_and_K_reconstruction',window=window,layer=layer,head=a):
            p=state['prob'][a].to(device);da=delta[:,a,:];va=v[:,b,:]
            dp=da@va.T
            e=torch._softmax_backward_data(dp.contiguous(),p.contiguous(),-1,torch.float32)
            centered=p*(dp-(da*z[:,a,:]).sum(-1,keepdim=True))
            check=compare(centered,e,1e-5,scale=float((p*dp).double().norm()))
            require(torch.count_nonzero(p.triu(1))==0 and torch.count_nonzero(e.triu(1))==0,'Masked edge contributes')
            row_sum=e.double().sum(-1);row_scale=(p.double()*dp.double().abs()).sum(-1)
            # Native FP32 softmax backward rounding, frozen 8eps forward-scale bound.
            row_bound=8*torch.finfo(torch.float32).eps*row_scale+1e-20
            require(bool((row_sum.abs()<=row_bound).all()),'Softmax signed row-sum failed')
            require(torch.count_nonzero(e[-1])==0,'Loss-free last query has nonzero gradient')
            row_audits.append(dict(head=a,centered_formula=check,max_signed_row_sum=float(row_sum.abs().max()),
                                  max_row_bound=float(row_bound.max()),zero_gradient_rows=int((e==0).all(-1).sum()),
                                  all_mask_rows=int((p==0).all(-1).sum()),t0_abs=float(e[0].abs().max()),last_abs=float(e[-1].abs().max())))
            e=e.double();qt=state['qt'][a].to(device).double()
            gh=transpose_rotate((e.T@qt)/math.sqrt(d),c,s)
            g[b].add_(gh)
            if pilot:head_score[b].add_(gh.T@x)
        with resources.timed('five_grouped_contractions',window=window,layer=layer,head=a):
            for j in range(5):
                rotated=rotate(rx[j,:,b,:],c,s)
                edge=e*((qt@rotated.T)/math.sqrt(d))
                query[j].add_(edge.sum(-1));cancellation[j].add_(edge.abs().sum())
            del edge,rotated
        with resources.timed('small_spectra',window=window,layer=layer,head=a):
            for t in QUERIES+([0,2047] if pilot else []):
                # F has source columns; transpose is applied using captured native cos/sin.
                f=(transpose_rotate(qt[t].expand(t+1,-1),c[:t+1],s[:t+1])*(e[t,:t+1,None]/math.sqrt(d))).T
                spec,u=spectrum(f)
                q=state['q'][a,t].to(device).double()
                control=q[:,None]*(e[t,:t+1][None,:]/math.sqrt(d))
                control_spec,_=spectrum(control)
                if not control_spec['zero_energy']:require(abs(control_spec['rho1']-1)<=1e-10,'No-RoPE rank1 control failed')
                spectra.append(dict(layer=layer,window=window,head=a,query=t,pilot_edge=t in (0,2047),
                                    **spec,no_rope_energy=control_spec['energy'],no_rope_rho1=control_spec['rho1']))
                if t in (383,1663) and a%4==window_index%4:
                    with resources.timed('post_activation_and_residual_diagnostics',window=window,layer=layer,head=a,query=t):
                        m=f@x[:t+1];mnorm=float((m*m).sum())
                        for k in (1,4):
                            uk=u[:,-k:];approx=uk@(uk.T@m);err=m-approx
                            error2=float((err*err).sum())
                            post.append(dict(layer=layer,window=window,head=a,query=t,k=k,norm2=mnorm,error2=error2,
                                             absolute_error=math.sqrt(error2),epsilon=None if mnorm<=1e-24 else math.sqrt(error2/mnorm)))
                            for j,method in enumerate(METHODS):
                                rb=residuals[j,b*d:(b+1)*d,:]
                                val=float((m*rb).sum());er=float((err*rb).sum())
                                direction.append(dict(layer=layer,window=window,head=a,query=t,k=k,method=method,
                                                      signed_reference=val,signed_error=er,norm2=val*val,error2=er*er,
                                                      absolute_error=abs(er),epsilon=None if val*val<=1e-24 else abs(er/val)))
                        del m,approx,err
        del p,dp,centered,e,qt,gh
    g=g.permute(1,0,2).reshape(length,-1)
    with resources.timed('score_identity_and_grouped_summary',window=window,layer=layer):
        checks.append(dict(name='reconstructed_K_vs_cache',**compare(g,cached_g,1e-5)))
        source=(rx.reshape(5,length,-1)*g[None,:,:]).sum(-1)
        cached_source=(rx.reshape(5,length,-1)*cached_g.to(device).double()[None,:,:]).sum(-1)
        checks.append(dict(name='source_contractions_vs_cache',**compare(source,cached_source,1e-5)))
        if pilot:
            score=g.T@x;cached_score=cached_g.to(device).double().T@x
            checks.append(dict(name='FP64_head_sum_score',**compare(head_score.reshape_as(score),score,1e-10)))
            checks.append(dict(name='source_cache_vs_full_weight_autograd',**compare(cached_score,full,1e-5)))
            checks.append(dict(name='edge_score_vs_full_weight_autograd',**compare(score,full,1e-5)))
        grouped=[]
        for j,method in enumerate(METHODS):
            checks.append(dict(name='FP64_query_source_total_'+method,
                               **compare(query[j].sum(),source[j].sum(),1e-10,scale=float(cancellation[j]))))
            fullscore=float(source[j].sum())**2/(2*8*2047)
            src=float(source[j].square().sum())/(2*8*2047)
            qry=float(query[j].square().sum())/(2*8*2047)
            grouped.append(dict(layer=layer,window=window,method=method,full=fullscore,source=src,query=qry,
                                full_minus_source=fullscore-src,full_minus_query=fullscore-qry,
                                signed_full=float(source[j].sum()),normalization='per-window contribution / (2*8*2047)'))
    return dict(checks=checks,softmax=row_audits,grouped=grouped,spectra=spectra,post=post,direction=direction),dict(source=source.cpu(),query=query.cpu(),cached_source=cached_source.cpu())
