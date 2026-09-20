"""Offline CPU regression tests for the 2026-09-20 sketch/sample protocol.

Run from qera_mxint4_full_ag:
    D:/anaconda/python.exe -B experiments/qer_kronecker_sketch_sample_v1/test_math.py

Only tiny test oracles materialize H (column-major vec). Production contractions
are checked against that independent matrix, never against another contraction.
All experiment caches below are in memory; no teacher, GPU, network or files are
created. PLAN patches and process settings are restored, including after failure.
"""
import builtins
import copy
from contextlib import ExitStack, contextmanager, nullcontext
import hashlib
import io
import json
import math
import os
from pathlib import Path
import socket
import sys
import unittest
from unittest.mock import Mock, patch

sys.dont_write_bytecode = True
_IMPORT_PATH = sys.path[:]
sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    import numpy as np
    import torch
    import common
    import sketch_math as sm
    import analysis
    import data
    import experiment
finally:
    sys.path[:] = _IMPORT_PATH

DTYPE = torch.float64
PLAN = common.PLAN
_GUARDS = None


def setUpModule():
    global _GUARDS
    _GUARDS = ExitStack()
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    _GUARDS.callback(torch.set_num_threads, threads)
    # Fail immediately if any tested path unexpectedly attempts external IO.
    for owner, name in ((socket, 'create_connection'), (socket, 'getaddrinfo'),
                        (socket.socket, 'connect'), (socket.socket, 'connect_ex'),
                        (torch.cuda, '_lazy_init')):
        _GUARDS.enter_context(patch.object(owner, name, side_effect=AssertionError(
            'CPU/offline test attempted network or CUDA: ' + name)))
    for owner in (builtins, io):
        original = owner.open

        def read_only(file, mode='r', *args, _open=original, **kwargs):
            if any(flag in mode for flag in 'wax+'):
                raise AssertionError('Test attempted a filesystem write: ' + str(file))
            return _open(file, mode, *args, **kwargs)

        _GUARDS.enter_context(patch.object(owner, 'open', read_only))
    original_os_open = os.open

    def read_only_fd(path, flags, *args, **kwargs):
        if flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND):
            raise AssertionError('Test attempted a filesystem write: ' + str(path))
        return original_os_open(path, flags, *args, **kwargs)

    _GUARDS.enter_context(patch.object(os, 'open', read_only_fd))


def tearDownModule():
    _GUARDS.close()


def eye(n):
    return torch.eye(n, dtype=DTYPE, device='cpu')


def vec(matrix):
    """Explicit column-major ordering, including noncontiguous inputs."""
    return torch.stack([matrix[i, j] for j in range(matrix.shape[1])
                        for i in range(matrix.shape[0])])


def dense_h(sequences, T):
    vectors = [vec(s) for s in sequences]
    return sum(torch.outer(v, v) for v in vectors) / (len(vectors) * T)


def dense_pos(pairs, T):
    # Cache tensors may be FP32; the protocol promotes BEFORE every product.
    terms = [vec(torch.outer(z[t].double(), x[t].double())) for x, z in pairs
             for t in range(len(x))]
    return sum(torch.outer(v, v) for v in terms) / (len(pairs) * T)


def hq(h, residual):
    v = vec(residual)
    return float(v @ h @ v / 2)


def hcontract(h, factor, side, m, n):
    """Partial inner product of explicit H; intentionally no S or x/g here."""
    if side == 'A':
        return torch.tensor([[sum(h[i + m*j, k + m*l] * factor[i, k]
                                  for i in range(m) for k in range(m))
                              for l in range(n)] for j in range(n)], dtype=DTYPE)
    return torch.tensor([[sum(h[i + m*j, k + m*l] * factor[j, l]
                              for j in range(n) for l in range(n))
                          for k in range(m)] for i in range(m)], dtype=DTYPE)


def canonical_gauge(a, g):
    norm = math.sqrt(float((a * a).sum()))
    return a / norm, g * norm


def objective(h, a, g):
    k = torch.kron(a.contiguous(), g.contiguous())
    return float(((h - k)**2).sum() - (h**2).sum())


def oracle_seed(role, token_hash, replicate):
    raw = json.dumps(['qer-ksample-v1/' + role, token_hash, replicate],
                     separators=(',', ':')).encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], 'little') % (2**63 - 1)


class MemoryStore:
    """TensorStore substitute, with snapshots so history cannot alias."""
    def __init__(self):
        self.records = {}
        self.writes = []

    def put(self, path, tensors, **metadata):
        self.records[Path(path)] = copy.deepcopy((tensors, metadata))
        self.writes.append(Path(path))

    def get(self, path):
        return self.records[Path(path)]


class CPUCase(unittest.TestCase):
    def setUp(self):
        saved = copy.deepcopy(PLAN)

        def restore_plan():
            PLAN.clear()
            PLAN.update(saved)

        self.addCleanup(restore_plan)
        self.rng = torch.Generator(device='cpu').manual_seed(20260920)

    def rand(self, *shape):
        return torch.randn(*shape, generator=self.rng, dtype=DTYPE, device='cpu')

    def close(self, actual, expected, rtol=2e-10, atol=2e-12):
        torch.testing.assert_close(torch.as_tensor(actual, dtype=DTYPE),
                                   torch.as_tensor(expected, dtype=DTYPE),
                                   rtol=rtol, atol=atol)


class MathTests(CPUCase):
    def setUp(self):
        super().setUp()
        self.m, self.n, self.L, self.T = 3, 5, 4, 3
        self.D, self.M = 3, 2
        self.x = [self.rand(self.L, self.n) + 0.35*c for c in range(self.D)]
        self.pairs = [(x, self.rand(self.L, self.m) + 0.2*(c+k))
                      for c, x in enumerate(self.x) for k in range(self.M)]
        self.sequences = [sum(torch.outer(z[t], x[t]) for t in range(self.L))
                          for x, z in self.pairs]
        self.h = dense_h(self.sequences, self.T)
        self.hp = dense_pos(self.pairs, self.T)
        self.a = sum(torch.outer(row, row) for x in self.x for row in x) / (self.D*self.L)
        self.g = sum(torch.outer(row, row) for _, z in self.pairs for row in z) / (len(self.pairs)*self.T)
        self.stream = lambda: iter(self.sequences)
        self.tokens = lambda: iter(self.pairs)

    def test_column_vec_and_canonical_full_pos_normalization(self):
        r = self.rand(self.m, self.n)
        self.close(vec(torch.tensor([[1., 2., 3.], [4., 5., 6.]])),
                   [1., 4., 2., 5., 3., 6.])
        b = [float((s*r).sum()) for s in self.sequences]
        self.close(hq(self.h, r), sum(v*v for v in b)/(2*len(b)*self.T))
        pos_q = sum(float(z[t] @ r @ x[t])**2 for x, z in self.pairs
                    for t in range(self.L))/(2*len(self.pairs)*self.T)
        self.close(hq(self.hp, r), pos_q)
        k = torch.kron(self.a, self.g)
        self.close(sm.qmetric(r, self.a, self.g), hq(k, r))
        self.assertGreater(abs(hq(k, r) - float(r.reshape(-1) @ k @ r.reshape(-1)/2)), 1e-3)
        self.assertGreater(abs(hq(self.h, r) - hq(self.hp, r)), 1e-3)

    def test_token_joint_three_rounds_both_sides_use_old_pair(self):
        a, g = self.a.clone(), eye(self.m)
        for iteration in range(3):
            with self.subTest(round=iteration+1):
                old_a, old_g = a.clone(), g.clone()
                expected_a = hcontract(self.hp, old_g, 'A', self.m, self.n)/old_g.square().sum()
                expected_g = hcontract(self.hp, old_a, 'G', self.m, self.n)/old_a.square().sum()
                new_a, new_g = sm.token_step(self.tokens, a, g, self.T)
                self.close(new_a, expected_a)
                self.close(new_g, expected_g)
                self.close(a, old_a, rtol=0, atol=0)
                self.close(g, old_g, rtol=0, atol=0)
                sequential_g = hcontract(self.hp, expected_a, 'G', self.m, self.n)/expected_a.square().sum()
                self.assertGreater(float((new_g-sequential_g).norm()), 1e-5)
                a, g = sm.gauge(new_a, new_g)
                ea, eg = canonical_gauge(expected_a, expected_g)
                self.close(a, ea)
                self.close(g, eg)

    def test_sequence_one_step_independent_identity_denominators(self):
        ma, mg, count = sm.sequence_moments(self.stream, self.T, 'cpu')
        self.assertEqual(count, self.D*self.M)
        self.close(ma, hcontract(self.h, eye(self.m), 'A', self.m, self.n))
        self.close(mg, hcontract(self.h, eye(self.n), 'G', self.m, self.n))
        a, g = sm.sequence_one_step(ma, mg)
        self.close(a, ma/self.m)
        self.close(g, mg/self.n)
        sequential_g = hcontract(self.h, a, 'G', self.m, self.n)/a.square().sum()
        self.assertGreater(float((g-sequential_g).norm()), 1e-3)

    def test_full_fit_alternates_and_each_half_step_is_monotonic(self):
        a, g = self.a.clone(), self.g.clone()
        last_error = math.inf
        for iteration in range(6):
            with self.subTest(round=iteration+1):
                old_a, old_g = canonical_gauge(a, g)
                next_g = hcontract(self.h, old_a, 'G', self.m, self.n)/old_a.square().sum()
                next_a = hcontract(self.h, next_g, 'A', self.m, self.n)/next_g.square().sum()
                expected_j = [objective(self.h, old_a, old_g),
                              objective(self.h, old_a, next_g),
                              objective(self.h, next_a, next_g)]
                a, g, audit = sm.full_step(self.stream, a, g, self.T)
                ea, eg = canonical_gauge(next_a, next_g)
                self.close(a, ea)
                self.close(g, eg)
                self.close([audit[k] for k in ('J_before', 'J_after_G', 'J_after_A')], expected_j)
                self.assertLessEqual(expected_j[1], expected_j[0]+1e-9)
                self.assertLessEqual(expected_j[2], expected_j[1]+1e-9)
                error = float((self.h-torch.kron(a, g)).square().sum())
                self.assertLessEqual(error, last_error+1e-9)
                last_error = error
                product_change = float((torch.kron(a, g)-torch.kron(old_a, old_g)).norm()/torch.kron(old_a, old_g).norm())
                self.close(audit['product_relative_change'], product_change, atol=1e-8)
                scale = max(abs(v) for v in expected_j)
                self.close(audit['relative_J_improvement'], (expected_j[0]-expected_j[2])/scale)
                self.assertFalse(audit['synchronous'])
                self.assertEqual(audit['samples'], len(self.sequences))
        self.assertLess(expected_j[2], 0, 'Negative J is valid; it excludes ||H||^2')

    def test_full_step_separates_contraction_and_fixed_seconds(self):
        # start, G start/end, A start/end, finish (including both spectra).
        with patch.object(sm.time, 'perf_counter', side_effect=[100., 102., 105., 110., 117., 130.]) as clock, \
                patch.object(sm.parent, 'spectrum', wraps=sm.parent.spectrum) as spectrum, \
                patch.object(torch.cuda, 'synchronize', side_effect=AssertionError('CPU test synchronized CUDA')):
            _, _, audit = sm.full_step(self.stream, self.a, self.g, self.T)
        self.assertEqual(clock.call_count, 6)
        self.assertEqual(spectrum.call_count, 2)
        self.close(audit['contraction_seconds'], 3.+7.)
        self.close(audit['fixed_seconds'], 20.)
        self.close(audit['contraction_seconds']+audit['fixed_seconds'], 30.)

    def test_gauge_preserves_product_proxy_and_product_distance(self):
        a, g = sm.gauge(self.a*7.3, self.g/7.3)
        self.close(a.norm(), 1.)
        self.close(torch.kron(a, g), torch.kron(self.a, self.g))
        r = self.rand(self.m, self.n)
        self.close(sm.qmetric(r, a, g), hq(torch.kron(self.a, self.g), r))
        change, _ = sm.parent.product_change(1.2*a, g, a, g)
        self.close(change, 0.2)
        same, _ = sm.parent.product_change(a, g, 2*a, g/2)
        self.close(same, 0., atol=3e-8)
        with self.assertRaises(RuntimeError):
            sm.gauge(torch.zeros_like(a), g)

    def test_spd_relative_damping_roots_and_no_raw_clipping(self):
        for raw in (self.a, torch.diag(torch.tensor([-1e-12, 0., 2.], dtype=DTYPE))):
            with self.subTest(shape=raw.shape):
                original = raw.clone()
                solved, root, audit = sm.parent.damp(raw, 1e-3)
                lam = 1e-3*float(raw.trace())/len(raw)
                self.close(solved, raw+lam*eye(len(raw)))
                self.close(root, root.T)
                self.close(root@root, solved)
                self.close(raw, original, rtol=0, atol=0)
                self.assertGreater(float(torch.linalg.eigvalsh(solved).min()), 0)
                self.assertLessEqual(audit['condition'], 1e8)
                self.assertEqual(audit['eigenvalue_clipping'], 'none')
                self.close(audit['lambda'], lam)

    def test_invalid_psd_symmetry_and_condition_fail_without_retuning(self):
        for raw in (torch.diag(torch.tensor([-0.01, 2.], dtype=DTYPE)),
                    torch.zeros((2, 2), dtype=DTYPE)):
            with self.subTest(raw=raw.tolist()):
                with self.assertRaises(RuntimeError):
                    sm.parent.spectrum(raw)
                with self.assertRaises(RuntimeError):
                    sm.parent.damp(raw, 1e-3)
        with self.assertRaises(RuntimeError):
            sm.parent.damp(torch.diag(torch.tensor([1e-12, 1.], dtype=DTYPE)), 0.)
        with self.assertRaises(RuntimeError):
            sm.sym(torch.tensor([[1., 0.1], [0., 1.]], dtype=DTYPE))

    def test_weighted_svd_dense_oracle_rank_tail_and_deployment(self):
        wq = self.rand(self.m, self.n).float()
        w0 = (wq.double()+0.25*self.rand(self.m, self.n)).float()
        error = w0.double()-wq.double()
        rank = 2
        t, correction, audit = sm.solve(error, self.a, self.g, rank, wq, w0)
        a, g = canonical_gauge(self.a, self.g)
        aa = a+PLAN['eta_A']*a.trace()/self.n*eye(self.n)
        gg = g+PLAN['eta_G']*g.trace()/self.m*eye(self.m)
        av, au = torch.linalg.eigh(aa)
        gv, gu = torch.linalg.eigh(gg)
        ar, gr = (au*av.sqrt())@au.T, (gu*gv.sqrt())@gu.T
        ai, gi = (au/av.sqrt())@au.T, (gu/gv.sqrt())@gu.T
        u, s, vh = torch.linalg.svd(gr@error@ar, full_matrices=False)
        expected_c = gi@((u[:, :rank]*s[:rank])@vh[:rank])@ai
        self.close(t['C64'], expected_c)
        self.close(t['P64']@t['Q64'], expected_c)
        self.assertEqual(t['P64'].shape, (self.m, rank))
        self.assertEqual(t['Q64'].shape, (rank, self.n))
        self.assertEqual(int(torch.linalg.matrix_rank(t['C64'])), rank)
        self.close(t['A_solve'], aa)
        self.close(t['G_solve'], gg)
        self.close(audit['tail_energy_half'], s[rank:].square().sum()/2)
        self.close(audit['proxy_ideal'], hq(torch.kron(aa, gg), error-expected_c))
        deployed = wq+t['C64'].float()  # Frozen parent dense path, cast AFTER P64@Q64.
        self.assertEqual(correction['W_deploy'].dtype, torch.float32)
        self.close(correction['W_deploy'], deployed, rtol=0, atol=0)
        self.close(correction['R64'], error-t['C64'], rtol=0, atol=0)
        self.close(correction['R_deploy'], w0.double()-deployed.double(), rtol=0, atol=0)
        self.assertGreater(float((correction['R_deploy']-correction['R64']).norm()), 0.)
        self.close(audit['proxy_deployed'], hq(torch.kron(aa, gg), correction['R_deploy']))
        self.assertLessEqual(audit['deployment_relative_drift'], PLAN['tolerances']['deployment'])
        self.assertEqual(audit['deployment'], PLAN['deployment'])
        # Feasible random rank-r alternatives cannot beat the whitened SVD optimum.
        for _ in range(8):
            c0 = self.rand(self.m, rank)@self.rand(rank, self.n)
            self.assertLessEqual(audit['proxy_ideal'], hq(torch.kron(aa, gg), error-c0)+1e-10)

    def test_svd_solution_is_gauge_and_positive_scale_invariant(self):
        error = self.rand(self.m, self.n)
        first, _ = sm.parent.weighted_svd(error, self.a, self.g, 2, 1e-3, 1e-3)
        for a, g in ((self.a*9., self.g/9.), (self.a*4., self.g*3.)):
            other, _ = sm.parent.weighted_svd(error, a, g, 2, 1e-3, 1e-3)
            self.close(other['C64'], first['C64'])

    def test_gram_blocking_norm_inner_cosine_sstar_delta(self):
        expected_gram = torch.tensor([[float(vec(x)@vec(y)) for y in self.sequences]
                                      for x in self.sequences], dtype=DTYPE)
        paths = list(range(len(self.sequences)))
        for block in (1, 2, 4, 20):
            calls = []
            with self.subTest(block=block):
                gram, norm2 = sm.gram_blocked(paths, self.sequences.__getitem__, self.T,
                                             'cpu', block, lambda: calls.append(True))
                self.close(gram, expected_gram)
                self.close(norm2, self.h.square().sum())
                self.assertTrue(calls)
        k = torch.kron(self.a, self.g)
        inner = sm.raw_metric_inner(self.stream, self.a, self.g, self.T)
        self.close(inner, (self.h*k).sum())
        quality = sm.curvature_quality(norm2, inner, self.a, self.g)
        scale = float((self.h*k).sum()/k.square().sum())
        self.close(quality['H_norm'], self.h.norm())
        self.close(quality['K_norm'], k.norm())
        self.close(quality['cosine'], (self.h*k).sum()/(self.h.norm()*k.norm()))
        self.close(quality['s_star'], scale)
        self.close(quality['delta_F'], (self.h-scale*k).norm())
        self.close(quality['relative_error'], (self.h-scale*k).norm()/self.h.norm())
        for factor in (0.5, 2.):
            self.assertLessEqual(quality['delta_F'], float((self.h-factor*scale*k).norm()))

    def test_curvature_roundoff_guard_and_degeneracy(self):
        hnorm2 = float(torch.kron(self.a, self.g).square().sum())
        good = sm.curvature_quality(hnorm2, hnorm2*(1+1e-12), self.a, self.g)
        self.assertTrue(good['roundoff_guard_used'])
        self.assertLess(good['relative_error_square_before_guard'], 0)
        self.assertEqual(good['delta_F'], 0)
        with self.assertRaises(RuntimeError):
            sm.curvature_quality(hnorm2, hnorm2*1.01, self.a, self.g)
        for h2, inner in ((0., 1.), (1., 0.), (1., -1.)):
            with self.subTest(h2=h2, inner=inner), self.assertRaises(RuntimeError):
                sm.curvature_quality(h2, inner, self.a, self.g)

    def test_damping_contraction_expansion_includes_all_four_terms(self):
        ma, mg, _ = sm.sequence_moments(self.stream, self.T, 'cpu')
        raw = sm.raw_metric_inner(self.stream, self.a, self.g, self.T)
        for la, lg in ((0., 0.), (0.07, 0.), (0., 0.13), (0.07, 0.13)):
            with self.subTest(lambda_A=la, lambda_G=lg):
                actual = sm.solve_metric_inner(raw, self.a, self.g, ma, mg, la, lg)
                k = torch.kron(self.a+la*eye(self.n), self.g+lg*eye(self.m))
                self.close(actual, (self.h*k).sum())
        self.close(ma.trace(), mg.trace())

    def test_empirical_excess_bound_for_marginal_and_a_only_references(self):
        error = self.rand(self.m, self.n)
        ma, mg, _ = sm.sequence_moments(self.stream, self.T, 'cpu')
        sa, sg = sm.sequence_one_step(ma, mg)
        candidate, _ = sm.parent.weighted_svd(error, sa, sg, 2, 1e-3, 1e-3)
        a, g = candidate['A_solve'], candidate['G_solve']
        k = torch.kron(a, g)
        quality = sm.curvature_quality(float(self.h.square().sum()), float((self.h*k).sum()), a, g)
        rk = error-candidate['C64']
        for refg in (self.g, eye(self.m)):
            reference, _ = sm.parent.weighted_svd(error, self.a, refg, 2, 1e-3, 1e-3)
            r0 = error-reference['C64']
            qk, q0, qe = hq(self.h, rk), hq(self.h, r0), hq(self.h, error)
            result = sm.excess_bound(quality, rk, r0, a, g, qk, q0, qe)
            proxy = quality['s_star']*(hq(k, rk)-hq(k, r0))
            delta = float((self.h-quality['s_star']*k).norm())
            penalty = 0.5*delta*float(rk.square().sum()+r0.square().sum())
            self.close(result['empirical_excess'], qk-q0)
            self.close(result['scaled_proxy_improvement'], proxy)
            self.close(result['error_term'], penalty)
            self.close(result['U'], proxy+penalty)
            self.close(result['U_over_None'], (proxy+penalty)/qe)
            self.assertLessEqual(qk-q0, result['U']+1e-10)
            self.assertLessEqual(proxy, 1e-10)
            degenerate = sm.excess_bound(quality, rk, r0, a, g, qk, q0, 1e-12)
            self.assertIsNone(degenerate['U_over_None'])
            with self.assertRaisesRegex(RuntimeError, 'bound violated'):
                sm.excess_bound(quality, rk, r0, a, g, q0+abs(result['U'])+100., q0, qe)

    def test_sequence_cross_position_terms_are_preserved(self):
        x = torch.tensor([[1., 2., -1.], [1., 2., -1.]], dtype=DTYPE)
        z = torch.tensor([[2., -3.], [2., -3.]], dtype=DTYPE)
        s = sum(torch.outer(z[t], x[t]) for t in range(2))
        full, pos = dense_h([s], 1), dense_pos([(x, z)], 1)
        self.close(full, 2*pos)
        a, g = eye(3), eye(2)
        inner = sm.raw_metric_inner(lambda: iter([s]), a, g, 1)
        self.close(inner, (full*torch.kron(a, g)).sum())
        self.assertGreater(abs(inner-float((pos*torch.kron(a, g)).sum())), 1.)
        # Opposite positions cancel in full H, but have strictly positive H_pos.
        z[1] = -z[0]
        s = sum(torch.outer(z[t], x[t]) for t in range(2))
        self.close(dense_h([s], 1), torch.zeros((6, 6), dtype=DTYPE))
        self.assertGreater(float(dense_pos([(x, z)], 1).norm()), 0)
        ma, mg, _ = sm.sequence_moments(lambda: iter([s]), 1, 'cpu')
        self.close(ma, torch.zeros((3, 3), dtype=DTYPE))
        self.close(mg, torch.zeros((2, 2), dtype=DTYPE))

    def test_empty_stream_and_scalar_mismatch_are_rejected(self):
        empty = lambda: iter(())
        for call in (lambda: sm.token_step(empty, self.a, self.g, self.T),
                     lambda: sm.sequence_moments(empty, self.T, 'cpu'),
                     lambda: sm.raw_metric_inner(empty, self.a, self.g, self.T),
                     lambda: sm.full_step(empty, self.a, self.g, self.T)):
            with self.assertRaises(RuntimeError):
                call()
        with self.assertRaises(RuntimeError):
            sm.scalar_check(1., 1.01, 1.)
        self.assertLessEqual(sm.scalar_check(1., 1.+1e-12, 1.)['absolute_error'], 1e-10)


class ExperimentCPUIntegrationTests(CPUCase):
    def setUp(self):
        super().setUp()
        PLAN.update(rank=2, L=4, T=3, shape=[3, 5])
        self.e = experiment.Experiment.__new__(experiment.Experiment)
        self.e.root = Path(__file__).resolve().parent/'__memory_only_test_root__'
        self.e.device = 'cpu'
        self.e.identity = 'cpu-test'
        self.e.store = MemoryStore()
        self.e.resources = type('Resources', (), {
            'boundary': lambda _: None, 'timed': lambda *args, **kwargs: nullcontext()})()
        self.e.index = data.sample_views([{'token_hash': f'fit-{c}'} for c in range(32)],
                                         [{'token_hash': f'eval-{c}'} for c in range(16)])
        self.e.entries = {r['id']: r for r in self.e.index['fit']}
        for c in range(32):
            x = (self.rand(4, 5)+c/9).float()
            self.e.store.put(self.e.path('fit_x', f'w{c:02d}'), {'x': x})
        for key, row in self.e.entries.items():
            z = self.rand(4, 3).float()
            x = self.e.store.get(self.e.path('fit_x', f'w{row["window"]:02d}'))[0]['x']
            s = sum(torch.outer(z[t].double(), x[t].double()) for t in range(4))
            self.e.store.put(self.e.path('fit_g', key), {'g': z})
            self.e.store.put(self.e.path('fit_S', key), {'S': s})
        exists = patch.object(Path, 'exists', lambda p: p in self.e.store.records)
        exists.start()
        self.addCleanup(exists.stop)
        saves = patch.object(experiment, 'save_json')
        self.save_json = saves.start()
        self.addCleanup(saves.stop)

    def test_marginal_canonical_normalization_and_shared_a8_asset(self):
        for budget in ('S0', 'S1', 'S2'):
            keys = self.e.index['budgets'][budget]
            result = self.e.marginal_stats(keys, budget)
            windows = sorted({self.e.entries[k]['window'] for k in keys})
            expected_a = sum(torch.outer(row.double(), row.double())
                             for c in windows for row in self.e.store.get(
                                 self.e.path('fit_x', f'w{c:02d}'))[0]['x'])/(len(windows)*4)
            gm = sum(torch.outer(row.double(), row.double()) for k in keys
                     for row in self.e.store.get(self.e.path('fit_g', k))[0]['g'])/(len(keys)*4)
            self.close(result['A_m'], expected_a)
            self.close(result['G_canonical'], (4/3)*gm)
        s0, m0 = self.e.store.get(self.e.root/'statistics/S0.safetensors')
        s1, m1 = self.e.store.get(self.e.root/'statistics/S1.safetensors')
        self.close(s0['A_m'], s1['A_m'], rtol=0, atol=0)
        self.assertEqual(m0['A_asset'], m1['A_asset'])
        self.assertEqual(m0['A_hash'], m1['A_hash'])
        self.assertEqual(self.e.store.writes.count(self.e.root/'statistics/A8.safetensors'), 1)
        self.assertEqual(self.e.store.writes.count(self.e.root/'statistics/A32.safetensors'), 1)

    def test_raw_factor_dispatch_three_rounds_and_one_step(self):
        keys = self.e.index['budgets']['S0']
        pairs = list(self.e.token_stream(keys)())
        hp = dense_pos(pairs, PLAN['T'])
        h = dense_h(list(self.e.stream(keys)()), PLAN['T'])
        marginal = self.e.marginal_stats(keys, 'S0')
        a, g = marginal['A_m'], eye(3)
        for _ in range(3):
            aa = hcontract(hp, g, 'A', 3, 5)/g.square().sum()
            gg = hcontract(hp, a, 'G', 3, 5)/a.square().sum()
            a, g = canonical_gauge(aa, gg)
        result = self.e.raw_factors('S0', 'Token-joint')
        self.close(result['A_raw'], a)
        self.close(result['G_raw'], g)
        _, audit = self.e.store.get(self.e.root/'factors/S0/Token-joint/raw.safetensors')
        self.assertEqual(len(audit['history']), 3)
        self.assertTrue(all(row['synchronous'] for row in audit['history']))
        sa = hcontract(h, eye(3), 'A', 3, 5)/3
        sg = hcontract(h, eye(5), 'G', 3, 5)/5
        sa, sg = canonical_gauge(sa, sg)
        result = self.e.raw_factors('S0', 'Sequence-one-step')
        self.close(result['A_raw'], sa)
        self.close(result['G_raw'], sg)

    def test_full_fit_fixed_stop_rule_and_final_checkpoint(self):
        result = self.e.raw_factors('S0', 'Full-fit')
        _, audit = self.e.store.get(self.e.root/'factors/S0/Full-fit/raw.safetensors')
        history = audit['history']
        self.assertGreaterEqual(len(history), 2)
        self.assertLessEqual(len(history), 20)
        stable = 0
        for index, row in enumerate(history):
            stable = stable+1 if (row['relative_J_improvement'] <= 1e-6 and
                                  row['product_relative_change'] <= 1e-4) else 0
            if index < len(history)-1:
                self.assertLess(stable, 2)
        if audit['stop'] == 'TWO_CONSECUTIVE_CONVERGED_CYCLES':
            self.assertEqual(stable, 2)
        else:
            self.assertEqual(audit['stop'], 'MAX_ITERATIONS_REACHED')
            self.assertEqual(len(history), 20)
        tensors, _ = self.e.store.get(self.e.root/f'factors/S0/Full-fit/iteration_{len(history):02d}.safetensors')
        self.close(torch.kron(result['A_raw'], result['G_raw']), torch.kron(tensors['A'], tensors['G']))

    def test_projection_keeps_full_sequence_and_ideal_deployed_distinct(self):
        r64, rd = self.rand(3, 5), self.rand(3, 5)
        self.e.store.put(self.e.root/'corrections/test.safetensors', {'R64': r64, 'R_deploy': rd})
        keys = self.e.index['budgets']['S0']
        sequences = list(self.e.stream(keys)())
        values = [self.e.project(s, 'test') for s in sequences]
        h = dense_h(sequences, PLAN['T'])
        self.close(sum(v['q'] for v in values)/len(values), hq(h, rd))
        self.close(sum(v['ideal_q'] for v in values)/len(values), hq(h, r64))
        for s, v in zip(sequences, values):
            self.close(v['d'], vec(s)@vec(rd))
            self.close(v['q'], v['d']**2/(2*PLAN['T']))
            self.close(v['squared'], v['d']**2)

    def test_pilot_repeats_only_two_saved_samples_for_gram_throughput(self):
        e = self.e
        keys = [e.index['fit'][i]['id'] for i in range(2)]
        self.assertTrue(all(e.entries[k]['source'] == 'parent_saved' for k in keys))
        sequences = [e.s(k).clone() for k in keys]
        expected_hnorm2 = float(dense_h(sequences, PLAN['T']).square().sum())
        expected_packed = torch.stack([s.reshape(-1) for s in sequences])
        wq = self.rand(3, 5).float()
        w0 = (wq.double()+self.rand(3, 5)*.25).float()
        e.quantized = {'W0': w0, 'Wq': wq}
        e.error = w0.double()-wq.double()
        e.name = 'tiny-cpu-module'
        e.windows = {'fit': torch.arange(32*4).reshape(32, 4)}
        e.teacher = Mock()
        e.teacher.reference.return_value = (None, None, None)
        e.teacher.intervention_kl.return_value = (.125, {'mocked_teacher': True})
        e.teacher.output_perturbation_kl.return_value = .125
        e.teacher.labels.side_effect = AssertionError('Pilot requested new teacher labels')
        e.teacher.gradient.side_effect = AssertionError('Throughput benchmark requested a new gradient')
        e.resources.data = {'timings': [], 'active_seconds': 0., 'GPU_peaks': {}}
        e.resources.flush = Mock()
        active_stage = [None]
        stages, reads, repeats = [], [], []

        @contextmanager
        def timed(stage, **metadata):
            stages.append(stage)
            previous = active_stage[0]
            active_stage[0] = stage
            try:
                yield
            finally:
                active_stage[0] = previous

        original_s, original_repeat = e.s, torch.Tensor.repeat

        def read_sample(key, role='fit'):
            self.assertEqual(role, 'fit')
            self.assertIn(key, keys)
            reads.append((active_stage[0], key))
            return original_s(key, role)

        def repeat(tensor, *args, **kwargs):
            result = original_repeat(tensor, *args, **kwargs)
            repeats.append((active_stage[0], tensor.clone(), result.clone()))
            return result

        e.resources.timed = timed
        with ExitStack() as stack:
            collect = stack.enter_context(patch.object(e, 'collect_fit'))
            stack.enter_context(patch.object(e, 'offline'))
            labels = stack.enter_context(patch.object(e, 'labels', side_effect=AssertionError('New labels')))
            estimate = stack.enter_context(patch.object(e, 'estimate'))
            # MemoryStore models persisted tensors; clearing a real LRU does not delete them.
            clear = stack.enter_context(patch.object(e.store, 'clear', create=True))
            stack.enter_context(patch.object(e, 's', side_effect=read_sample))
            stack.enter_context(patch.object(torch.Tensor, 'repeat', repeat))
            gram = stack.enter_context(patch.object(sm, 'gram_blocked', wraps=sm.gram_blocked))
            commit = stack.enter_context(patch.object(experiment, 'commit', side_effect=lambda p, i, **fields: fields))
            stack.enter_context(patch.object(torch.cuda, 'is_available', return_value=False))
            for block in (1, 7, 8):
                with self.subTest(block=block):
                    stages.clear(); reads.clear(); repeats.clear()
                    e.config = {'gram_block': block}
                    result = e.pilot()
                    collect.assert_called_with(keys)
                    self.assertEqual(result['fit_samples_reused'], keys)
                    self.assertTrue(result['fit_only'])
                    self.assertFalse(result['extra_evaluation_data_seen'])
                    self.close(result['H_norm2'], expected_hnorm2)
                    self.assertEqual([key for stage, key in reads if stage == 'pilot_Gram_two_sample_load'], keys)
                    self.assertFalse(any(stage == 'pilot_Gram_block_compute' for stage, _ in reads))
                    self.assertEqual(len(repeats), 1)
                    stage, packed, repeated = repeats[0]
                    self.assertEqual(stage, 'pilot_Gram_block_compute')
                    self.close(packed, expected_packed, rtol=0, atol=0)
                    expected = torch.stack([expected_packed[i % 2] for i in range(block)])
                    self.close(repeated[:block], expected, rtol=0, atol=0)
                    self.assertEqual(stages.count('pilot_token_contraction'), 1)
                    self.assertEqual(stages.count('pilot_token_fixed'), 1)
                    self.assertEqual(stages.count('pilot_Gram_two_sample_load'), 1)
                    self.assertEqual(stages.count('pilot_Gram_block_compute'), 1)
                    self.assertEqual(gram.call_args.args[0], keys)
                    self.assertEqual(gram.call_args.kwargs['block'], 2)
                    self.assertEqual(commit.call_args.args, (e.root/'pilot.json', e.identity))
            self.assertEqual(collect.call_count, 3)
            self.assertEqual(gram.call_count, 3)
            self.assertEqual(estimate.call_count, 3)
            self.assertEqual(clear.call_count, 6)
            labels.assert_not_called()
            e.teacher.labels.assert_not_called()
            e.teacher.gradient.assert_not_called()


class EstimateTests(CPUCase):
    """Frozen operation counts with synthetic timings; no resource/IO constructor."""
    def setUp(self):
        super().setUp()
        self.e = experiment.Experiment.__new__(experiment.Experiment)
        self.e.root = Path(__file__).resolve().parent/'__memory_only_estimate__'
        self.e.identity = 'cpu-estimate-test'
        self.e.config = {'gram_block': 8, 'budget_hours': 100.}
        self.times = {
            'fit_gradient_cache': 2., 'pilot_token_contraction': 3.,
            'pilot_token_fixed': 5., 'pilot_sequence_moments': 7.,
            'pilot_SVD': 11., 'pilot_KL': 13.,
            'pilot_Gram_two_sample_load': 17., 'pilot_Gram_block_compute': 19.,
            'pilot_metric_contractions': 23.,
        }
        self.full = {'contraction_seconds': 29., 'fixed_seconds': 31.}
        self.e.resources = Mock()
        self.e.resources.data = {'timings': [], 'active_seconds': 123.}
        reader = patch.object(experiment, 'read', return_value={'full_fit_round': self.full})
        self.read = reader.start()
        self.addCleanup(reader.stop)
        writer = patch.object(experiment, 'commit', side_effect=lambda p, i, **fields: fields)
        self.commit = writer.start()
        self.addCleanup(writer.stop)
        self.set_timings()

    def set_timings(self):
        # Two completed observations average to the chosen time. Incomplete and
        # obsolete aggregate timings must not contaminate any estimate.
        self.e.resources.data['timings'] = [
            {'stage': stage, 'seconds': seconds*factor, 'completed': completed}
            for stage, seconds in self.times.items()
            for factor, completed in ((.5, True), (1.5, True), (1e9, False))
        ] + [{'stage': stage, 'seconds': 1e12, 'completed': True}
             for stage in ('pilot_token_round', 'pilot_full_round', 'pilot_Gram')]

    def test_estimate_constant_counts_and_separate_fixed_costs(self):
        result = self.e.estimate()
        expected = {
            'fit_collection': 224*2.,
            'token_joint_9_rounds': 432*3.+9*5.,
            'full_fit_max_60_rounds': 2880*29.+60*31.,
            'sequence_moments_fit_and_eval': 272*7.,
            'all_14_SVD': 14*11.,
            'eval_gradients': 256*2.,
            'KL_240_plus_checks': 135*13.,
            # 28 and 32 blocks: 406 + 528 lower-triangle pairs, four
            # two-sample loads per full block, one compute per pair.
            'Gram_fit_eval': 3736*17.+934*19.,
            'curvature_metric_contractions': 1056*23.,
        }
        self.assertEqual(result['stage_estimated_seconds'], expected)
        self.close(result['measured_upper_estimate_seconds'], sum(expected.values())*1.35+600.)
        self.assertEqual(result['active_seconds_at_freeze'], 123.)
        self.assertEqual(result['budget_hours'], 100.)
        self.assertEqual(result['safety_multiplier'], 1.35)
        self.assertEqual(result['extra_overhead_seconds'], 600)
        self.assertTrue(result['fits_budget'])
        self.read.assert_called_once_with(self.e.root/'pilot.json')
        self.commit.assert_called_once_with(self.e.root/'budget_freeze.json', self.e.identity, **result)

    def test_fixed_eigenspectrum_costs_scale_by_rounds_not_samples(self):
        baseline = self.e.estimate()['stage_estimated_seconds']
        self.full['fixed_seconds'] += 1.
        self.times['pilot_token_fixed'] += 1.
        self.set_timings()
        fixed = self.e.estimate()['stage_estimated_seconds']
        self.close(fixed['full_fit_max_60_rounds']-baseline['full_fit_max_60_rounds'], 60.)
        self.close(fixed['token_joint_9_rounds']-baseline['token_joint_9_rounds'], 9.)
        self.full['contraction_seconds'] += 1.
        self.times['pilot_token_contraction'] += 1.
        self.set_timings()
        contraction = self.e.estimate()['stage_estimated_seconds']
        self.close(contraction['full_fit_max_60_rounds']-fixed['full_fit_max_60_rounds'], 2880.)
        self.close(contraction['token_joint_9_rounds']-fixed['token_joint_9_rounds'], 432.)
        for stage in baseline.keys()-{'full_fit_max_60_rounds', 'token_joint_9_rounds'}:
            self.close(contraction[stage], baseline[stage])

    def test_gram_load_and_compute_counts_include_partial_blocks(self):
        for block in (1, 7, 8, 257):
            with self.subTest(block=block):
                self.e.config['gram_block'] = block
                # Enumerate the actual lower-triangle block traversal independently.
                pairs = [(i, j) for samples in (224, 256)
                         for i in range(0, samples, block) for j in range(0, i+1, block)]
                expected = sum(block/2*self.times['pilot_Gram_two_sample_load']+
                               self.times['pilot_Gram_block_compute'] for _ in pairs)
                result = self.e.estimate()
                self.close(result['stage_estimated_seconds']['Gram_fit_eval'], expected)

    def test_budget_gate_includes_elapsed_time_overhead_and_strict_boundary(self):
        # Zero measured work leaves exactly the fixed 600-second reservation;
        # exact integer seconds avoid a floating-point boundary ambiguity.
        self.times = {stage: 0. for stage in self.times}
        self.full.update(contraction_seconds=0., fixed_seconds=0.)
        self.set_timings()
        self.e.config['budget_hours'] = 1.
        for elapsed, expected in ((2999., True), (3000., False), (3001., False)):
            with self.subTest(elapsed=elapsed):
                self.e.resources.data['active_seconds'] = elapsed
                result = self.e.estimate()
                self.assertEqual(result['measured_upper_estimate_seconds'], 600.)
                self.assertEqual(result['active_seconds_at_freeze'], elapsed)
                self.assertIs(result['fits_budget'], expected)
        self.e.config['budget_hours'] = 2.
        self.assertTrue(self.e.estimate()['fits_budget'])

    def test_missing_or_only_incomplete_timing_cannot_freeze_budget(self):
        for stage in self.times:
            for mode in ('missing', 'incomplete'):
                with self.subTest(stage=stage, mode=mode):
                    self.set_timings()
                    rows = self.e.resources.data['timings']
                    if mode == 'missing':
                        rows[:] = [row for row in rows if row['stage'] != stage]
                    else:
                        for row in rows:
                            if row['stage'] == stage:
                                row['completed'] = False
                    self.commit.reset_mock()
                    with self.assertRaisesRegex(RuntimeError, 'Missing pilot timing '+stage):
                        self.e.estimate()
                    self.commit.assert_not_called()


class BootstrapTests(CPUCase):
    def setUp(self):
        super().setUp()
        self.keys = [b+'__'+m for b in ('S0', 'S1', 'S2') for m in
                     ('Marginal', 'Token-joint', 'Sequence-one-step', 'Full-fit')]+['A8', 'A32', 'None']

    def oracle_draws(self, q, kl):
        n, m, _ = q.shape
        b = 2000
        article = np.random.Generator(np.random.PCG64(oracle_seed('bootstrap', '', 0))).integers(n, size=(b, n))
        label = np.random.Generator(np.random.PCG64(oracle_seed('bootstrap', '', 1))).integers(m, size=(b, n, m))
        # Loop over replicates to avoid copying the production broadcasting formula.
        q_article = np.array([q.mean(axis=1)[indices].mean(axis=0) for indices in article])
        kl_article = np.array([kl[indices].mean(axis=0) for indices in article])
        q_mc = np.array([np.stack([q[c, draw[c]].mean(axis=0) for c in range(n)]).mean(axis=0)
                         for draw in label])
        return {'q': (q_article, q_mc), 'KL': (kl_article,)}

    def test_all_preregistered_contrasts_and_direction(self):
        pairs = analysis.contrasts()
        self.assertEqual(len(pairs), 34)
        self.assertEqual(sum(p['kind'] == 'primary' for p in pairs), 9)
        self.assertEqual(sum(p['kind'] == 'A_only' for p in pairs), 12)
        self.assertEqual(len({p['label'] for p in pairs}), 34)
        for row in pairs:
            self.assertIn(row['baseline'], self.keys)
            self.assertIn(row['candidate'], self.keys)
            if row['kind'] == 'A_only':
                self.assertEqual(row['baseline'], 'A32' if row['candidate'].startswith('S2') else 'A8')
            if row['kind'] == 'sample_budget' and row['candidate'] != 'A32':
                self.assertIn((row['baseline'][:2], row['candidate'][:2]), [('S0', 'S1'), ('S0', 'S2'), ('S1', 'S2')])

    def test_bootstrap_common_draws_ratio_of_means_and_conditional_labels(self):
        rng = np.random.default_rng(1701)
        q = rng.uniform(.2, 2., size=(5, 4, 15))
        q[..., -1] *= np.arange(1, 6)[:, None]**2
        kl = rng.uniform(.3, 3., size=(5, 15))
        kl[:, -1] *= np.arange(1, 6)**2
        result = analysis.summarize(q, kl, self.keys)
        self.assertEqual(result, analysis.summarize(q, kl, self.keys))
        self.assertEqual(result['bootstrap_count'], 2000)
        self.assertEqual(result['bootstrap_seed'], oracle_seed('bootstrap', '', 0))
        self.assertEqual(result['label_bootstrap_seed'], oracle_seed('bootstrap', '', 1))
        draws = self.oracle_draws(q, kl)
        means = {'q': q.mean(axis=(0, 1)), 'KL': kl.mean(axis=0)}
        self.assertEqual(len(result['comparisons']), 68)
        self.assertEqual(len(result['candidates']), 30)
        for row in result['comparisons']:
            old, new = self.keys.index(row['baseline']), self.keys.index(row['candidate'])
            mu = means[row['metric']]
            self.close(row['Delta'], mu[old]-mu[new])
            self.close(row['d'], (mu[old]-mu[new])/mu[-1])
            scopes = ['article', 'conditional_labels'] if row['metric'] == 'q' else ['article']
            for scope, sample in zip(scopes, draws[row['metric']]):
                delta = sample[:, old]-sample[:, new]
                expected = np.quantile(delta/sample[:, -1], [.025, .975])
                self.close([row[scope]['ci_low'], row[scope]['ci_high']], expected)
                self.close(row[scope]['absolute_ci'], np.quantile(delta, [.025, .975]))
        for row in result['candidates']:
            i = self.keys.index(row['candidate'])
            sample = draws[row['metric']][0]
            self.close([row['ci_low'], row['ci_high']], np.quantile(1-sample[:, i]/sample[:, -1], [.025, .975]))
        first = result['comparisons'][0]
        old, new = self.keys.index(first['baseline']), self.keys.index(first['candidate'])
        wrong = np.mean((q[..., old]-q[..., new])/q[..., -1])
        self.assertGreater(abs(first['d']-wrong), 1e-4)

    def test_exact_2pp_boundaries_and_unresolved_are_not_equal(self):
        examples = [(.021, .06, 'IMPROVED_OVER_2PP'), (-.06, -.021, 'DEGRADED_OVER_2PP'),
                    (-.02, .02, 'DIFFERENCE_WITHIN_2PP'), (.02, .05, 'UNRESOLVED'),
                    (-.05, -.02, 'UNRESOLVED'), (-.03, .03, 'UNRESOLVED'),
                    (.01, .04, 'UNRESOLVED')]
        for lo, hi, expected in examples:
            with self.subTest(lo=lo, hi=hi):
                self.assertEqual(analysis.state(lo, hi), expected)
        self.assertEqual(analysis.state(.021, .04, guard=.002), 'UNRESOLVED')

    def test_2pp_is_none_damage_and_negative_recovery_is_not_clipped(self):
        q = np.full((4, 3, 15), 30.)
        q[..., -1] = 100.
        q[..., self.keys.index('S0__Marginal')] = 50.
        q[..., self.keys.index('S0__Token-joint')] = 48.5  # 1.5 pp, not 3%.
        q[..., self.keys.index('S0__Sequence-one-step')] = 47.
        q[..., self.keys.index('S0__Full-fit')] = 120.
        result = analysis.summarize(q, q.mean(axis=1), self.keys)
        rows = {r['candidate']: r for r in result['comparisons'] if r['metric'] == 'q' and r['kind'] == 'primary'}
        self.close(rows['S0__Token-joint']['d'], .015)
        self.assertEqual(rows['S0__Token-joint']['article']['status'], 'DIFFERENCE_WITHIN_2PP')
        self.assertEqual(rows['S0__Sequence-one-step']['article']['status'], 'IMPROVED_OVER_2PP')
        self.assertEqual(rows['S0__Full-fit']['article']['status'], 'DEGRADED_OVER_2PP')
        candidate = next(r for r in result['candidates'] if r['metric'] == 'q' and r['candidate'] == 'S0__Full-fit')
        self.close(candidate['recovery'], -.2)

    def test_degenerate_denominators_preserve_absolute_differences(self):
        for denominator in (0., 1e-12):
            q = np.full((4, 3, 15), 3.)
            q[..., 0] = 5.
            q[..., -1] = denominator
            result = analysis.summarize(q, q.mean(axis=1), self.keys)
            for row in result['comparisons']:
                self.assertIsNone(row['d'])
                for scope in ('article', 'conditional_labels') if row['metric'] == 'q' else ('article',):
                    self.assertEqual(row[scope]['status'], 'DENOMINATOR_UNRESOLVED')
                    self.assertIsNone(row[scope]['ci_low'])
                    self.assertIsNone(row[scope]['ci_high'])
                    self.close(row[scope]['absolute_ci'], [row['Delta'], row['Delta']])
            self.close(result['comparisons'][0]['Delta'], 2.)
            for row in result['candidates']:
                self.assertIsNone(row['recovery'])
                self.assertIsNone(row['ci_low'])

    def test_draw_level_degeneracy_does_not_clamp_or_change_pairing(self):
        q = np.full((2, 2, 15), 2.)
        q[..., 0] = np.array([[4., 5.], [7., 9.]])
        q[..., -1] = np.array([[0., 0.], [0., 4.]])
        kl = q.mean(axis=1)
        result = analysis.summarize(q, kl, self.keys)
        draws = self.oracle_draws(q, kl)
        for row in result['comparisons']:
            old, new = self.keys.index(row['baseline']), self.keys.index(row['candidate'])
            self.assertIsNotNone(row['d'])  # Overall None mean is nonzero.
            scopes = ['article', 'conditional_labels'] if row['metric'] == 'q' else ['article']
            for scope, sample in zip(scopes, draws[row['metric']]):
                self.assertTrue(np.any(sample[:, -1] <= 1e-12))
                self.assertEqual(row[scope]['status'], 'DENOMINATOR_UNRESOLVED')
                self.assertIsNone(row[scope]['ci_low'])
                self.close(row[scope]['absolute_ci'], np.quantile(sample[:, old]-sample[:, new], [.025, .975]))

    def test_invalid_loss_arrays_fail(self):
        good = np.ones((3, 2, 15))
        for value in (np.nan, np.inf, -1.):
            q = good.copy()
            q[0, 0, 0] = value
            with self.subTest(value=value), self.assertRaises(RuntimeError):
                analysis.summarize(q, good.mean(axis=1), self.keys)
        with self.assertRaises(RuntimeError):
            analysis.summarize(good, np.ones((3, 14)), self.keys)


class DataTests(CPUCase):
    def test_nested_224_fit_256_eval_views_and_same_a_indices(self):
        fit = [{'token_hash': f'fit-{c}'} for c in range(32)]
        evaluation = [{'token_hash': f'eval-{c}'} for c in range(16)]
        result = data.sample_views(fit, evaluation)
        self.assertEqual(result, data.sample_views(fit, evaluation))
        self.assertEqual(len(result['fit']), 224)
        self.assertEqual(len(result['eval']), 256)
        entries = {r['id']: r for r in result['fit']}
        self.assertEqual(len(entries), 224)
        sets = {b: set(ids) for b, ids in result['budgets'].items()}
        self.assertEqual([len(sets[b]) for b in ('S0', 'S1', 'S2')], [32, 128, 128])
        self.assertEqual(sets['S1'] & sets['S2'], sets['S0'])
        self.assertEqual(sets['S1'] | sets['S2'], set(entries))
        for budget, (windows, labels) in {'S0': (8, 4), 'S1': (8, 16), 'S2': (32, 4)}.items():
            expected = {(c, k) for c in range(windows) for k in range(labels)}
            actual = {(entries[key]['window'], entries[key]['replicate']) for key in sets[budget]}
            self.assertEqual(actual, expected)
        a0 = {entries[k]['window'] for k in sets['S0']}
        a1 = {entries[k]['window'] for k in sets['S1']}
        self.assertEqual(a0, a1)
        self.assertEqual(result['A_views'], {'S0': 'A8', 'S1': 'A8', 'S2': 'A32'})
        self.assertEqual({(r['window'], r['replicate']) for r in result['eval']},
                         {(c, k) for c in range(16) for k in range(16)})
        self.assertFalse(set(entries) & {r['id'] for r in result['eval']})
        for r in result['fit'] + result['eval']:
            if r['id'] in sets['S0']:
                self.assertEqual(r['source'], 'parent_saved')
                self.assertIsNone(r['seed'])
            else:
                self.assertEqual(r['source'], 'new_teacher_sample')
                self.assertEqual(r['seed'], oracle_seed(r['role'], r['token_hash'], r['replicate']))
        self.assertNotEqual(common.seed('fit', 'same-hash', 0), common.seed('eval', 'same-hash', 0))

    def test_article_parser_retains_sections_and_stable_ranges(self):
        rows = [{'text': t} for t in ['preface', '= Alpha =', 'body', '== Section ==',
                                      'more', '= Beta =', 'last']]
        self.assertEqual(list(data.articles(rows)), [
            (0, 1, 'preamble', 'preface'),
            (1, 5, '= Alpha =', '= Alpha =\nbody\n== Section ==\nmore'),
            (5, 7, '= Beta =', '= Beta =\nlast')])
        self.assertEqual(list(data.articles([])), [])

    def candidate(self, name, start, count=90, title=None, text_hash=None):
        return ({'article_id': name, 'article_title': title or name,
                 'article_text_sha256': text_hash or hashlib.sha256(name.encode()).hexdigest()},
                list(range(start, start+count)))

    def selection_key(self, item):
        article_id = item[0]['article_id']
        return hashlib.sha256(('qer-ksample-v1/data'+article_id).encode()).hexdigest(), article_id

    def test_selection_reproducible_sha_order_first_windows_24_plus_16(self):
        candidates = [self.candidate(f'article-{i:03d}', 1000*i) for i in range(48)]
        selected, rejected = data.select(candidates, set(), set(), set(), count=40, length=80)
        reordered, other_rejected = data.select(candidates[::-1], set(), set(), set(), count=40, length=80)
        expected = sorted(candidates, key=self.selection_key)[:40]
        self.assertEqual([r['article_id'] for r, _ in selected], [r['article_id'] for r, _ in expected])
        self.assertEqual([r for r, _ in selected], [r for r, _ in reordered])
        self.assertEqual(rejected, other_rejected)
        self.assertEqual(rejected, [])
        self.assertEqual(len(selected[:24]), 24)
        self.assertEqual(len(selected[24:]), 16)
        for (row, tokens), (_, source) in zip(selected, expected):
            self.assertEqual(tokens.tolist(), source[:80])
            self.assertEqual(row['token_start'], 0)
            self.assertEqual(row['token_stop'], 80)
            self.assertEqual(row['article_token_count'], 90)
            self.assertEqual(row['token_hash'], common.mo.digest_tensor(tokens))

    def test_historical_title_document_hash_and_full_article_64_spans(self):
        historical = list(range(100000, 100080))
        forbidden = data.spans(historical)
        candidates = [self.candidate('used-title', 0, title='history-title'),
                      self.candidate('used-text', 1000, text_hash='history-text'),
                      self.candidate('short', 2000, count=79),
                      self.candidate('historical-prefix', 100000),
                      self.candidate('historical-tail', 3000),
                      self.candidate('fresh', 4000)]
        # Historical overlap only AFTER the exported prefix must also be excluded.
        candidates[4][1].extend(historical[:64])
        titles, texts = {'history-title'}, {'history-text'}
        before = (forbidden.copy(), titles.copy(), texts.copy())
        selected, rejected = data.select(candidates, forbidden, titles, texts, count=40, length=80)
        self.assertEqual([r['article_id'] for r, _ in selected], ['fresh'])
        reasons = {r['article_id']: r['reason'] for r in rejected}
        self.assertEqual(reasons, {'used-title': 'used_or_duplicate_document',
                                  'used-text': 'used_or_duplicate_document', 'short': 'short',
                                  'historical-prefix': 'historical_64_token_span',
                                  'historical-tail': 'historical_64_token_span'})
        self.assertEqual((forbidden, titles, texts), before)
        self.assertLess(len(selected), 40)  # Shortfall is not filled with duplicates.

    def test_new_collection_title_text_token_and_tail_overlap_dedup(self):
        groups = [
            [self.candidate('title-a', 1000, title='same'), self.candidate('title-b', 2000, title='same')],
            [self.candidate('text-a', 3000, text_hash='same-text'), self.candidate('text-b', 4000, text_hash='same-text')],
            [self.candidate('tokens-a', 5000), self.candidate('tokens-b', 5000)],
            [self.candidate('tail-a', 6000), self.candidate('tail-b', 7000)],
        ]
        for item in groups[-1]:
            item[1].extend(range(8000, 8064))
        for index, group in enumerate(groups):
            with self.subTest(group=index):
                accepted, rejected = data.select(group, set(), set(), set(), count=40, length=80)
                self.assertEqual(len(accepted), 1)
                self.assertEqual(len(rejected), 1)
                self.assertEqual(accepted[0][0]['article_id'], sorted(group, key=self.selection_key)[0][0]['article_id'])
                self.assertEqual(rejected[0]['reason'], 'used_or_duplicate_document' if index < 2
                                 else 'new_collection_64_token_span')

    def test_span_width_is_exactly_64_and_slide_includes_last_start(self):
        self.assertEqual(data.spans(list(range(63))), set())
        self.assertEqual(len(data.spans(list(range(64)))), 1)
        spans = data.spans(list(range(65)))
        expected = {b''.join(int(i).to_bytes(4, 'little') for i in range(start, start+64))
                    for start in (0, 1)}
        self.assertEqual(spans, expected)


if __name__ == '__main__':
    unittest.main(verbosity=2)
