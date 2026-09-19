"""Sequential geometry construction, search freeze, then independent evaluation."""
import gc
import math
from pathlib import Path
import sys
import torch
from common import (PLAN, HERE, read, save_csv, save_tensors, read_tensors,
                    sha_file, mo, slug, seed, require)
from geometry import spectrum, construct, candidate_key, choose, scalar_check, relative
from records import commit, load_record, checked_files, file_table
from teacher import Teacher

ROLES = ['None', 'A-only', 'Marginal-AG', 'Selected']


class Experiment:
    def __init__(self, root, config, identity, assets, resources, teacher=None, device='cuda:0'):
        self.root = root; self.config = config; self.identity = identity; self.assets = assets
        self.resources = resources; self.device = device
        self.teacher = teacher or Teacher(config, root, identity, resources.timed)
        self.data = read(root/'data/article_manifest.json')
        require(self.data['identity'] == identity, 'Wrong data identity')
        checked_files(root/'data', {f'{r}_windows.safetensors':v for r,v in self.data['files'].items()})
        self.grid = {candidate_key(a,b):[a,b] for a in PLAN['exponents'] for b in PLAN['exponents']}

    def store_tensors(self, *args, **kwargs):
        with self.resources.io(): return save_tensors(*args, **kwargs)

    def record(self, *args, **kwargs):
        with self.resources.io(): return commit(*args, **kwargs)

    def folder(self, name): return self.root/'modules'/slug(name)

    def quantized(self, name):
        path = Path(self.config['assets'])/'exp01/quantized'/(slug(name)+'.safetensors')
        t, m = read_tensors(path)
        require(all(mo.digest_tensor(t[k]) == m[k+'_hash'] for k in ('W0','Wq')), 'W0/Wq changed')
        require(t['W0'].dtype == t['Wq'].dtype == torch.float32, 'Teacher and quantizer must be FP32')
        return t

    def old_label(self, c, k, ids):
        path = Path(self.config['assets'])/'exp03/data/fit_samples'/f'w{c:02d}_k{k:03d}.safetensors'
        t, m = read_tensors(path)
        require(m['identity'] == PLAN['parent_identity'] and m['label_hash'] == mo.digest_tensor(t['labels']), 'Old label changed')
        require(m['input_hash'] == mo.digest_tensor(ids[0]), 'Old label input differs')
        return t['labels']

    def factors(self, name):
        source = self.assets['factors'][name]
        if not source['rebuilt']:
            require(sha_file(source['path']) == source['sha256'], 'Parent factors changed')
            return read_tensors(source['path'])[0], source
        folder = self.folder(name); result = folder/'rebuilt_marginal.safetensors'
        if result.exists():
            t, m = read_tensors(result); require(m['identity'] == self.identity, 'Rebuilt factor identity')
            require(all(mo.digest_tensor(t[k]) == m['hashes'][k] for k in t), 'Rebuilt factor changed')
            return t, dict(source, output_sha256=sha_file(result), reconstruction=m)
        fit, _ = read_tensors(Path(self.config['assets'])/'exp03/data/fit_windows.safetensors')
        progress = folder/'factor_rebuild_progress.safetensors'
        if progress.exists():
            t, m = read_tensors(progress); require(m['identity'] == self.identity, 'Rebuild progress identity')
            require(all(mo.digest_tensor(t[k]) == m['hashes'][k] for k in t), 'Rebuild progress changed')
            a = t['A_sum'].to(self.device); gsum = t['G_sum'].to(self.device); start = m['next_sample']
        else: a = gsum = None; start = 0
        for c in range(start//4, 8):
            self.resources.boundary(); ids = fit['input_ids'][c:c+1]
            reference, x, h = self.teacher.reference(name, ids); xd = x.reshape(PLAN['L'], -1).double()
            for k in range(4):
                index = 4*c+k
                if index < start: continue
                with self.resources.timed('historical_factor_rebuild', module=name, window=c, replicate=k):
                    if k == 0:
                        term = (xd.T@xd).to(self.device)
                        a = term if a is None else a+term
                    g, _ = self.teacher.gradient(name, ids, reference, x, self.old_label(c,k,ids), audit=(index==0))
                    term = (g.T@g).to(self.device); gsum = term if gsum is None else gsum+term
                    tensors = dict(A_sum=a, G_sum=gsum)
                    self.store_tensors(progress, tensors, dict(identity=self.identity, next_sample=index+1,
                        hashes={key:mo.digest_tensor(value) for key,value in tensors.items()}))
                    del g, term
            del reference, x, h, xd
        self.teacher.unload()
        sys.path.insert(0, str(HERE.parent/'qer_teacher_kl_exp03'))
        import ag_math
        sys.path.insert(0, str(HERE))
        a, gsum = ag_math.gauge(ag_math.sym(a/(8*PLAN['L'])), ag_math.sym(gsum/(32*PLAN['T'])))
        with self.resources.timed('historical_factor_damping', module=name):
            aa, _, am = ag_math.damp(a, .001); gg, _, gm = ag_math.damp(gsum, .001)
            tensors = dict(A_solve=aa, G_solve=gg)
            metadata = dict(identity=self.identity, original_windows=8, original_labels=4,
                normalization='A_sum/(8L), G_sum/(32T), then original unit-Frobenius-A gauge',
                damping_A=am, damping_G=gm, new_search_or_test_data_used=False,
                hashes={key:mo.digest_tensor(value) for key,value in tensors.items()})
            self.store_tensors(result, tensors, metadata)
        return {k:v.cpu() for k,v in tensors.items()}, dict(source, output_sha256=sha_file(result), reconstruction=metadata)

    def candidates(self, name):
        folder = self.folder(name); path = folder/'candidate_manifest.json'
        if path.exists():
            record = load_record(path, self.identity); checked_files(folder, record['files'])
            require(set(record['candidates']) == set(self.grid)|{'None'}, 'Candidate set changed')
            return record
        factors, provenance = self.factors(name)
        self.teacher.unload(); gc.collect()
        q = self.quantized(name); error = (q['W0'].double()-q['Wq'].double()).to(self.device)
        eigpath = folder/'eigenbasis.safetensors'
        if eigpath.exists():
            t, m = read_tensors(eigpath); require(m['identity'] == self.identity, 'Eigenbasis identity')
            require(all(mo.digest_tensor(t[k]) == m['hashes'][k] for k in t), 'Eigenbasis changed')
            av, au, gv, gu = [t[k].to(self.device) for k in ('av','au','gv','gu')]
        else:
            with self.resources.timed('factor_eigendecompositions', module=name):
                av, au, ai = spectrum(factors['A_solve'].to(self.device))
                gv, gu, gi = spectrum(factors['G_solve'].to(self.device))
                t = dict(av=av, au=au, gv=gv, gu=gu)
                m = dict(identity=self.identity, A=ai, G=gi, provenance=provenance,
                    hashes={k:mo.digest_tensor(v) for k,v in t.items()})
                self.store_tensors(eigpath, t, m)
        self.record(folder/'geometry_manifest.json', self.identity, module=name, spectrum=m,
            eigenbasis_sha256=sha_file(eigpath), exponents=PLAN['exponents'], rank=PLAN['rank'],
            extra_damping=False, factor_source='teacher sampled-label Marginal solve')
        coordinates = gu.T@error@au
        candidate_info = {}
        for key, (a,b) in self.grid.items():
            self.resources.boundary(); cp = folder/'candidates'/(key+'.safetensors')
            if not cp.exists():
                with self.resources.timed('dense_SVD_candidate', module=name, candidate=key):
                    p, r, audit = construct(error, av, au, gv, gu, a, b, PLAN['rank'], coordinates)
                    correction = p@r
                    # Exact parent deployment: cast compensation to FP32 before the FP32 add.
                    weight = q['Wq'].to(self.device)+correction.float()
                    if (a,b) == (1.,1.):
                        old = Path(self.config['assets'])/'exp03/corrections'/slug(name)/'marginal.safetensors'
                        if old.exists():
                            ot, _ = read_tensors(old); oldc = ot['C64'].to(self.device)
                            obj = float((gv.sqrt()[:,None]*(gu.T@(error-oldc)@au)*av.sqrt()[None,:]).square().sum()/2)
                            audit['parent_endpoint_objective'] = scalar_check(obj, audit['objective'], float(error.square().sum()), 1e-8)
                            audit['parent_C_relative_error'] = relative(correction, oldc)
                            if audit['compensation_unique']:
                                require(audit['parent_C_relative_error'] <= 1e-7, 'Unique parent Marginal endpoint differs')
                            del ot, oldc
                        else: audit['parent_endpoint_note'] = 'No old compensation file; same solve factors and spectral objective verified'
                    tensors = dict(P64=p, Q64=r, W_deploy=weight)
                    cm = dict(identity=self.identity, module=name, candidate=key, parameters=[a,b], audit=audit,
                        deployment='FP32(Wq + FP32(P64@Q64)); same as parent; rank claim is ideal only',
                        hashes={k:mo.digest_tensor(v) for k,v in tensors.items()})
                    self.store_tensors(cp, tensors, cm)
                    del tensors, p, r, correction, weight
            t, cm = read_tensors(cp)
            require(cm['identity'] == self.identity and cm['candidate'] == key, 'Candidate identity differs')
            require(all(mo.digest_tensor(t[k]) == cm['hashes'][k] for k in t), 'Candidate tensor changed')
            candidate_info[key] = dict(path=cp.relative_to(folder).as_posix(), sha256=sha_file(cp),
                weight_hash=cm['hashes']['W_deploy'], parameters=[a,b], audit=cm['audit'])
            del t
        nonepath = folder/'candidates/None.safetensors'
        if not nonepath.exists():
            self.store_tensors(nonepath, dict(W_deploy=q['Wq']), dict(identity=self.identity, candidate='None', weight_hash=mo.digest_tensor(q['Wq'])))
        nt, nm = read_tensors(nonepath)
        require(nm['identity'] == self.identity and torch.equal(nt['W_deploy'], q['Wq']), 'None deployment differs')
        candidate_info['None'] = dict(path='candidates/None.safetensors', sha256=sha_file(nonepath),
            weight_hash=mo.digest_tensor(q['Wq']), parameters=None)
        files = file_table(folder, [folder/v['path'] for v in candidate_info.values()]+[eigpath,folder/'geometry_manifest.json'])
        record = self.record(path, self.identity, module=name, candidates=candidate_info, files=files,
            new_S_saved=False, stage='ALL_CANDIDATES_FROZEN_BEFORE_SEARCH')
        del av, au, gv, gu, coordinates, factors, error, q, nt
        gc.collect()
        if torch.cuda.is_initialized(): torch.cuda.empty_cache()
        return record

    def parent_replay(self, name):
        path = self.folder(name)/'parent_replay.json'
        if path.exists(): return load_record(path, self.identity)
        old = Path(self.config['assets'])/'exp03'
        fit, _ = read_tensors(old/'data/fit_windows.safetensors'); ids = fit['input_ids'][:1]
        self.resources.boundary()
        with self.resources.timed('historical_numerical_replay', module=name):
            ref, x, h = self.teacher.reference(name, ids)
            g, audit = self.teacher.gradient(name, ids, ref, x, self.old_label(0,0,ids), audit=True)
            s = g.T@x.reshape(PLAN['L'], -1).double()
            xp = old/'cache'/slug(name)/'x_w00.safetensors'
            sp = old/'cache'/slug(name)/'w00_k000.safetensors'
            if xp.exists():
                xt, xm = read_tensors(xp)
                require(mo.digest_tensor(xt['x']) == xm['input_hash'], 'Parent x hash mismatch')
                audit['parent_x_relative'] = relative(x.cpu().double(), xt['x'].double())
                require(audit['parent_x_relative'] <= PLAN['tolerances']['parent_x'], 'Parent x replay tolerance exceeded')
            if sp.exists():
                st, sm = read_tensors(sp)
                require(mo.digest_tensor(st['S']) == sm['S_hash'], 'Parent S hash mismatch')
                audit['parent_S_relative'] = relative(s.cpu(), st['S'])
                require(audit['parent_S_relative'] <= PLAN['tolerances']['parent_S'], 'Parent S replay tolerance exceeded')
            else: audit['parent_S_note'] = 'Cache absent; reproduced original-label shared-weight autograd audit instead'
        return self.record(path, self.identity, module=name, checks=audit, historical_gradients=1,
            new_sample_budget_used=0, optional_old_S_present=sp.exists())

    def labels(self, name, role, c, ids, reference, count):
        if role == 'test':
            freeze = self.verify_selection(name)
            require(freeze['selection']['selected'] is not None, 'Test requested without a selected geometry')
        row = next(w for w in self.data['windows'] if w['role'] == role and w['window'] == c)
        paths = [self.root/'data/labels'/role/f'w{c:02d}_k{k:03d}.safetensors' for k in range(count)]
        missing = [k for k,p in enumerate(paths) if not p.exists()]
        if missing:
            with self.resources.timed('label_sampling', role=role, window=c):
                samples = self.teacher.labels(reference, [seed(role,row['article_id'],k) for k in missing])
                for k, labels in zip(missing, samples):
                    self.store_tensors(paths[k], dict(labels=labels), dict(identity=self.identity, role=role, window=c,
                        article_id=row['article_id'], replicate=k, seed=seed(role,row['article_id'],k),
                        input_hash=mo.digest_tensor(ids[0]), mask_hash=mo.digest_tensor(torch.ones_like(ids[0])),
                        label_hash=mo.digest_tensor(labels), positions=PLAN['T'], distribution='full-vocabulary teacher FP64 inverse CDF; fixed text inputs'))
        result = []
        for k,p in enumerate(paths):
            t, m = read_tensors(p)
            require(m['identity'] == self.identity and m['role'] == role and m['article_id'] == row['article_id'], 'Label identity differs')
            require(m['replicate'] == k and m['seed'] == seed(role,row['article_id'],k), 'Label stream differs')
            require(m['input_hash'] == mo.digest_tensor(ids[0]) and m['label_hash'] == mo.digest_tensor(t['labels']), 'Label/tokens changed')
            require(t['labels'].dtype == torch.int64 and t['labels'].shape == (PLAN['T'],), 'Label shape/dtype differs')
            result.append((t['labels'], dict(m, file_sha256=sha_file(p))))
        return result

    def phase_paths(self, name, role):
        return [self.folder(name)/'atomic'/role/f'w{c:02d}_k{k:03d}.json'
            for c in range(PLAN[role+'_windows']) for k in range(PLAN[role+'_labels'])]

    def score(self, name, role, cm, mapping):
        folder = self.folder(name); unique = list(dict.fromkeys(mapping.values()))
        windows, _ = read_tensors(self.root/'data'/f'{role}_windows.safetensors')
        rows = []; q = self.quantized(name); error = q['W0'].double()-q['Wq'].double()
        for c in range(PLAN[role+'_windows']):
            paths = [folder/'atomic'/role/f'w{c:02d}_k{k:03d}.json' for k in range(PLAN[role+'_labels'])]
            if all(p.exists() for p in paths):
                rows.extend(load_record(p,self.identity) for p in paths); continue
            self.resources.boundary(); ids = windows['input_ids'][c:c+1]
            with self.resources.timed('reference_and_self_KL', module=name, role=role, window=c):
                reference, x, h = self.teacher.reference(name, ids)
                self_kl = self.teacher.kl(reference, reference)
                require(abs(self_kl) <= PLAN['tolerances']['self_KL'], 'Teacher self-KL nonzero')
                self.record(folder/'atomic'/role/f'self_w{c:02d}.json', self.identity, value=self_kl,
                    module=name, role=role, window=c, input_hash=mo.digest_tensor(ids[0]))
            labels = self.labels(name, role, c, ids, reference, PLAN[role+'_labels'])
            xd = x.reshape(PLAN['L'], -1).double(); projected = {}; ideal_projected = {}
            with self.resources.timed('host_staged_Rx', module=name, role=role, window=c):
                for key in unique:
                    cp = folder/cm['candidates'][key]['path']
                    require(sha_file(cp) == cm['candidates'][key]['sha256'], 'Candidate changed before projection')
                    t, _ = read_tensors(cp)
                    residual = (q['W0'].double()-t['W_deploy'].double()).to(x.device)
                    projected[key] = (xd@residual.T).cpu()
                    if key != 'None':
                        ideal = error.to(x.device)-t['P64'].to(x.device)@t['Q64'].to(x.device)
                        ideal_projected[key] = (xd@ideal.T).cpu()
                        del ideal
                    del t, residual
            for k, path in enumerate(paths):
                if path.exists(): rows.append(load_record(path,self.identity)); continue
                self.resources.boundary(); sample, lm = labels[k]
                with self.resources.timed('gradient_and_all_projections', module=name, role=role, window=c, replicate=k):
                    audit_sample = role == 'search' and c == 0 and k < 2
                    g, checks = self.teacher.gradient(name, ids, reference, x, sample, audit=audit_sample)
                    s = g.T@xd if audit_sample else None
                    physical = {}
                    for key in unique:
                        rx = projected[key].to(g.device); d = float((g*rx).sum())
                        require(math.isfinite(d), 'Nonfinite projection')
                        item = dict(d=d, squared=d*d, score=d*d/(2*PLAN['T']), T=PLAN['T'],
                            weight_hash=cm['candidates'][key]['weight_hash'], parameters=cm['candidates'][key]['parameters'])
                        if key != 'None':
                            di = float((g*ideal_projected[key].to(g.device)).sum())
                            item.update(ideal_d=di, ideal_score=di*di/(2*PLAN['T']))
                        if audit_sample:
                            t, _ = read_tensors(folder/cm['candidates'][key]['path'])
                            r = (q['W0'].double()-t['W_deploy'].double()).to(g.device)
                            direct = float((s*r).sum()); scale = float(s.norm()*r.norm())
                            item['projection_check'] = scalar_check(d, direct, scale, PLAN['tolerances']['projection'])
                            del t, r
                        physical[key] = item; del rx
                    row = self.record(path, self.identity, module=name, role=role, window=c, replicate=k,
                        article_id=lm['article_id'], input_hash=lm['input_hash'], label_hash=lm['label_hash'],
                        label_file_sha256=lm['file_sha256'], candidate_manifest_sha256=sha_file(folder/'candidate_manifest.json'),
                        role_mapping=mapping, physical=physical, checks=checks, shared_gradient_count=1)
                    rows.append(row); del g, s
                print(f'{name} {role}: {c*PLAN[role+"_labels"]+k+1}/{PLAN[role+"_windows"]*PLAN[role+"_labels"]} gradients committed', flush=True)
            del reference, x, h, xd, projected, ideal_projected, labels
            self.resources.estimate(self.config["modules"])
        # Validate all input bindings even when every atomic sample already existed.
        for row in rows:
            require(row['role_mapping'] == mapping and row['candidate_manifest_sha256'] == sha_file(folder/'candidate_manifest.json'), 'Projection candidate binding changed')
            lp = self.root/'data/labels'/role/f'w{row["window"]:02d}_k{row["replicate"]:03d}.safetensors'
            require(sha_file(lp) == row['label_file_sha256'], 'Scored label file changed')
        flattened = [dict(module=name, role=role, article_id=r['article_id'], window=r['window'], replicate=r['replicate'],
            label_hash=r['label_hash'], input_hash=r['input_hash'], candidate=label, geometry=key, **r['physical'][key])
            for r in rows for label,key in mapping.items()]
        save_csv(folder/(role+'_scores.csv'), flattened)
        scores = {key:math.fsum(r['physical'][key]['score'] for r in rows)/len(rows) for key in unique}
        drifts = {}
        for key in unique:
            if key == 'None': continue
            ideal = math.fsum(r['physical'][key]['ideal_score'] for r in rows)/len(rows)
            absolute = abs(scores[key]-ideal)
            scale = max(ideal, scores[key], PLAN['denominator_floor'])
            drifts[key] = dict(ideal=ideal, deployed=scores[key], absolute=absolute,
                relative_with_floor=absolute/scale, floor=PLAN['denominator_floor'],
                passed=absolute <= PLAN['tolerances']['deployment']*scale)
        check = self.record(folder/(role+'_numerical_checks.json'), self.identity, module=name, role=role,
            deployment_drift=drifts, passed=all(v['passed'] for v in drifts.values()),
            sample_count=len(rows), logical_scores=len(flattened), audit_samples=2 if role=='search' else 0)
        require(check['passed'], 'Deployment functional drift >1e-4; saved scores retained for diagnosis, no hidden repair')
        return rows, scores

    def freeze(self, name, cm, scores):
        folder = self.folder(name)
        selection = choose({k:scores[k] for k in self.grid}, self.grid, scores['None'], PLAN['denominator_floor'])
        selected = selection['selected']
        selection['baseline_selected'] = ('Marginal-AG' if selected == candidate_key(1,1) else
                                         'A-only' if selected == candidate_key(1,0) else None)
        paths = self.phase_paths(name, 'search')
        files = file_table(self.root, paths+[folder/'candidate_manifest.json', self.root/'data/article_manifest.json'])
        files.update(file_table(self.root, (self.root/'data/labels/search').glob('*.safetensors')))
        record = self.record(folder/'selection_freeze.json', self.identity, module=name, selection=selection,
            search_scores=scores, files=files, test_accessed_before_selection=False,
            source_manifest_sha256=sha_file(self.root/'manifest.json'))
        save_csv(folder/'search_summary.csv', [dict(candidate=key, parameters=cm['candidates'][key]['parameters'],
            score=value, ratio=value/scores['None'] if scores['None']>PLAN['denominator_floor'] else None,
            selected=key==selected) for key,value in scores.items()])
        return record

    def verify_selection(self, name):
        folder = self.folder(name); record = load_record(folder/'selection_freeze.json', self.identity)
        checked_files(self.root, record['files'])
        require(record['source_manifest_sha256'] == sha_file(self.root/'manifest.json'), 'Source/configuration changed after selection')
        expected = choose({k:record['search_scores'][k] for k in self.grid}, self.grid,
                          record['search_scores']['None'], PLAN['denominator_floor'])
        require(record['selection']['selected'] == expected['selected'], 'Selected geometry is not frozen search argmin')
        return record

    def actual_kl(self, name, cm, mapping):
        self.verify_selection(name); folder = self.folder(name); unique = list(dict.fromkeys(mapping.values()))
        windows, _ = read_tensors(self.root/'data/test_windows.safetensors'); q = self.quantized(name); rows = []
        for c in range(PLAN['test_windows']):
            paths = [folder/'atomic/kl'/f'w{c:02d}_{key}.json' for key in unique]
            if all(p.exists() for p in paths):
                rows.extend(load_record(p, self.identity) for p in paths); continue
            self.resources.boundary(); ids = windows['input_ids'][c:c+1]
            reference, x, h = self.teacher.reference(name, ids)
            self_kl = self.teacher.kl(reference, reference)
            require(abs(self_kl) <= PLAN['tolerances']['self_KL'], 'KL teacher self check failed')
            for key,path in zip(unique,paths):
                if path.exists(): rows.append(load_record(path,self.identity)); continue
                cp = folder/cm['candidates'][key]['path']
                require(sha_file(cp) == cm['candidates'][key]['sha256'], 'KL deployment changed')
                t,_ = read_tensors(cp); weight = t['W_deploy']; residual = q['W0'].double()-weight.double()
                with self.resources.timed('actual_KL', module=name, window=c, candidate=key):
                    value, audit = self.teacher.intervention_kl(name, ids, reference, x, h, weight, residual)
                    require(math.isfinite(value) and value >= 0, 'Invalid stable KL')
                    repeat = None
                    if c == 0:
                        second, _ = self.teacher.intervention_kl(name, ids, reference, x, h, weight, residual)
                        limit = max(PLAN['tolerances']['KL_repeat_absolute'], PLAN['tolerances']['KL_repeat_relative']*abs(value))
                        repeat = dict(value=second, difference=abs(second-value), allowed=limit)
                        require(repeat['difference'] <= limit, 'KL repeat failed')
                    row = self.record(path, self.identity, module=name, window=c, geometry=key, value=value,
                        input_hash=mo.digest_tensor(ids[0]), weight_hash=cm['candidates'][key]['weight_hash'],
                        candidate_file_sha256=cm['candidates'][key]['sha256'], self_KL=self_kl, repeat=repeat, intervention=audit)
                    rows.append(row)
                del t, weight, residual
            del reference,x,h
        for row in rows:
            require(row['candidate_file_sha256'] == cm['candidates'][row['geometry']]['sha256'], 'KL binding changed')
        save_csv(folder/'kl_by_window.csv', [dict(candidate=role, **row)
            for row in rows for role,key in mapping.items() if key==row['geometry']])
        return rows

    def run_module(self, name):
        self.parent_replay(name)
        cm = self.candidates(name)
        search_mapping = {key:key for key in ['None']+list(self.grid)}
        _, scores = self.score(name, 'search', cm, search_mapping)
        frozen = self.freeze(name, cm, scores)
        selected = frozen['selection']['selected']
        if selected is None:
            self.record(self.folder(name)/'completion.json', self.identity, status='DENOMINATOR_UNRESOLVED',
                reason='Search None <=1e-12; no forced choice and no Selected test claim')
            self.teacher.unload(); return
        mapping = dict(zip(ROLES, ['None',candidate_key(1,0),candidate_key(1,1),selected]))
        self.score(name, 'test', cm, mapping)
        self.actual_kl(name, cm, mapping)
        self.verify_selection(name)
        self.record(self.folder(name)/'completion.json', self.identity, status='COMPLETE',
            role_mapping=mapping, logical_new_gradients=256, test_unique_candidates=len(set(mapping.values())),
            selection_freeze_sha256=sha_file(self.folder(name)/'selection_freeze.json'))
        self.teacher.unload()
