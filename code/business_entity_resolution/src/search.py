"""Chunked top-K similarity search (GPU dense, GPU sparse SpMM, CPU sparse).

All functions return (idx int32 [nq, K], score float32 [nq, K]); rows with
fewer than K hits are padded with idx=-1, score=0.
"""

from __future__ import annotations

import warnings

import numpy as np
import scipy.sparse as sp
import torch

warnings.filterwarnings("ignore", message=".*[Ss]parse.*")
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _merge(best_s, best_i, s, i, k):
    if best_s is None:
        return s, i
    s = torch.cat([best_s, s], 1)
    i = torch.cat([best_i, i], 1)
    top_s, pos = torch.topk(s, min(k, s.shape[1]), dim=1)
    return top_s, torch.gather(i, 1, pos)


def _pad(idx, score, k):
    nq, kk = idx.shape
    if kk == k:
        return idx, score
    pi = np.full((nq, k), -1, np.int32)
    ps = np.zeros((nq, k), np.float32)
    pi[:, :kk], ps[:, :kk] = idx, score
    return pi, ps


@torch.no_grad()
def dense_topk(Q: np.ndarray, C: np.ndarray, k: int, q_chunk=512,
               c_slice=1_000_000, min_score=None) -> tuple[np.ndarray, np.ndarray]:
    """Cosine top-k for L2-normalised float16 embeddings. C is uploaded to GPU once."""
    if len(C) == 0 or len(Q) == 0:
        return np.full((len(Q), k), -1, np.int32), np.zeros((len(Q), k), np.float32)
    Cg = torch.from_numpy(np.ascontiguousarray(C)).to(DEV, torch.float16)
    out_i, out_s = [], []
    for q0 in range(0, len(Q), q_chunk):
        q = torch.from_numpy(np.ascontiguousarray(Q[q0:q0 + q_chunk])).to(DEV, torch.float16)
        bs = bi = None
        for c0 in range(0, len(Cg), c_slice):
            s = q @ Cg[c0:c0 + c_slice].T
            ts, ti = torch.topk(s, min(k, s.shape[1]), dim=1)
            bs, bi = _merge(bs, bi, ts.float(), ti + c0, k)
        out_i.append(bi.int().cpu().numpy())
        out_s.append(bs.cpu().numpy())
    idx, score = np.vstack(out_i), np.vstack(out_s)
    del Cg
    torch.cuda.empty_cache()
    return _pad(idx, score, k)


def _to_torch_csr(m: sp.csr_matrix):
    m = m.tocsr()
    return torch.sparse_csr_tensor(
        torch.from_numpy(m.indptr.astype(np.int64)), torch.from_numpy(m.indices.astype(np.int64)),
        torch.from_numpy(m.data.astype(np.float32)), size=m.shape, device=DEV)


@torch.no_grad()
def sparse_topk_gpu(Q: sp.csr_matrix, C: sp.csr_matrix, k: int, q_chunk=256,
                    c_slice=1_000_000) -> tuple[np.ndarray, np.ndarray]:
    """Exact sparse cosine top-k: (C_slice CSR on GPU) @ (dense Q chunk)^T via cuSPARSE."""
    if C.shape[0] == 0 or Q.shape[0] == 0:
        return np.full((Q.shape[0], k), -1, np.int32), np.zeros((Q.shape[0], k), np.float32)
    slices = [(c0, _to_torch_csr(C[c0:c0 + c_slice])) for c0 in range(0, C.shape[0], c_slice)]
    out_i, out_s = [], []
    for q0 in range(0, Q.shape[0], q_chunk):
        qd = torch.from_numpy(Q[q0:q0 + q_chunk].T.toarray().astype(np.float32)).to(DEV)
        bs = bi = None
        for c0, cs in slices:
            # SpMM gives [c_slice, q] row-major; top-k over a strided view is very slow,
            # so transpose into a contiguous half-precision buffer first.
            s = torch.sparse.mm(cs, qd).T.half().contiguous()  # [q, c_slice]
            ts, ti = torch.topk(s, min(k, s.shape[1]), dim=1)
            del s
            bs, bi = _merge(bs, bi, ts.float(), ti + c0, k)
        out_i.append(bi.int().cpu().numpy())
        out_s.append(bs.cpu().numpy())
    idx, score = np.vstack(out_i), np.vstack(out_s)
    idx[score <= 0] = -1
    score[score <= 0] = 0
    del slices
    torch.cuda.empty_cache()
    return _pad(idx, score, k)


def sparse_topk_cpu(Q: sp.csr_matrix, CT: sp.csr_matrix, k: int,
                    q_chunk=4000) -> tuple[np.ndarray, np.ndarray]:
    """Top-k of Q @ CT for very sparse products (rare-token inverted index).

    CT is the transposed corpus matrix (vocab x N) in CSR form.
    """
    nq = Q.shape[0]
    idx = np.full((nq, k), -1, np.int32)
    score = np.zeros((nq, k), np.float32)
    for q0 in range(0, nq, q_chunk):
        r = (Q[q0:q0 + q_chunk] @ CT).tocsr()
        r.sum_duplicates()
        if r.nnz == 0:
            continue
        # segmented top-k on GPU: sort by (row asc, score desc) in one key
        indptr = torch.from_numpy(r.indptr.astype(np.int64)).to(DEV)
        data = torch.from_numpy(r.data.astype(np.float32)).to(DEV)
        rows = torch.repeat_interleave(torch.arange(r.shape[0], device=DEV), indptr.diff())
        dmax = float(data.max()) + 1.0
        key = rows.double() * (2 * dmax) + (dmax - data.double())
        order = torch.sort(key).indices
        rows_s = rows[order]
        rank = torch.arange(len(order), device=DEV) - indptr[rows_s]
        keep = rank < k
        o = order[keep].cpu().numpy()
        rr, kk = rows_s[keep].cpu().numpy(), rank[keep].cpu().numpy()
        idx[q0 + rr, kk] = r.indices[o]
        score[q0 + rr, kk] = r.data[o]
    return idx, score
