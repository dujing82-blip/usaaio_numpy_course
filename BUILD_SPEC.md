# USAAIO Matrix-Operations Practice Repository — Build Specification

**Purpose.** Build a GitHub repository of Google Colab notebooks where a 14-year-old beginner practices every NumPy / PyTorch array operation that the USA AI Olympiad (USAAIO / USA-NA-AIO) has tested so far. Each exercise is small, auto-graded, and traceable to a past exam problem.

**Audience.** One student, no prior NumPy background, learning from a companion slide deck (40 slides, same six sections as below). The repository is the "lab"; the deck is the "lecture".

**Source material analysed.** 2024 IAIO (theory paper); 2025 USA-NA-AIO Round 1 (3 problems, 41 parts) and Round 2 (3 problems, 46 parts); 2026 USAAIO Round 1 sample set (13 problems) and the 2026 Round 1 contest (9 problems). Official solutions exist for 2024–2025; the 2026 sets have community solutions only.

**Language of all content:** English. **Libraries:** NumPy, PyTorch (CPU), matplotlib, pandas, scikit-learn only where the exam allowed it. **Runtime:** every notebook must run top-to-bottom on a free Colab CPU runtime in under 2 minutes with no downloads except the datasets explicitly listed in §7.

---

## 1. Knowledge map (what the exams actually test)

Each line: **topic → operations → where it appeared**. Use this as the backbone; every exercise in §5 is tagged with one topic id.

### A. Arrays and shapes
| id | Topic | Operations | Past problems |
|---|---|---|---|
| A1 | ndim / shape / size / dtype | `np.array`, `.shape`, `.ndim`, `.size`, `.dtype`, `.astype`, `torch.tensor`, `.float()` | 2025 R1 P1.9 (int32 output), P2.12 (print ndim and shape), R2 P1.3, P3.4 |
| A2 | Creating arrays | `zeros/ones/eye/arange/linspace`, `np.random.seed/randn/rand/permutation`, `np.random.seed()` reset, `torch.manual_seed/randn/rand/linspace/meshgrid` | 2026 Sample P4.1, 2025 R1 P2.4, P2.12, R2 P1.4–1.7, P3.2 |
| A3 | Indexing and slicing | `X[i]`, `X[:, j]`, `X[a:b, c:d]`, negative indices, `...`, boolean masks `arr[arr > 1] = 100`, fancy indexing with a permutation, `.iloc` | 2026 Sample P4.5, P9.14; 2025 R1 P3.8; R2 P1.10 (`[:, 0]`, `[:, 1]`) |
| A4 | Reshape family (order-preserving) | `reshape`, `reshape(-1, 1)`, `flatten`, `squeeze`, `expand_dims` / `[:, None]` / `np.newaxis`, `.view`, `.unsqueeze` | 2026 Sample P4.2–4.3; 2025 R1 P1.9, P2.9, P2.13; R2 P1.6, P1.12 |
| A5 | Transpose family (reordering) | `.T`, `swapaxes`, `np.transpose(x, axes)`, `permute`, `transpose(-2, -1)` | 2026 Sample P4.4; 2025 R2 P2.5, P2.7, P2.13, P3.4 |
| A6 | Stack vs concatenate | `np.stack`, `np.concatenate(axis=)`, `torch.stack`, `torch.cat`, column-of-ones trick, `[W] * k` copies | 2025 R1 P3.9; R2 P1.6–1.7, P1.12, P2.10, P2.13, P3.4, P3.9 |

### B. Elementwise operations and broadcasting
| id | Topic | Operations | Past problems |
|---|---|---|---|
| B1 | Elementwise arithmetic and ufuncs | `+ - * / **`, `np.exp/log/sqrt/sin/tanh`, `np.maximum(0, x)`, `torch.max(torch.zeros_like(x), x)`, sigmoid / ReLU / softmax in one line | 2025 R1 P2.10, P2.12, P3.16; 2026 Sample P10 |
| B2 | Broadcasting rules | right-aligned shape matching, length-1 stretching, `(N,1)` vs `(1,N)`, `x[..., None] * W.T` | 2025 R1 P2.3, P2.9; R2 P2.7 (size-1 head axis), P2.10 (`S.reshape(-1,1) * V`) |
| B3 | Outer-product / all-pairs trick | `x[:, None] * x[None, :]`, `W[:, None, :] * W[None, :, :]` summed on the last axis | 2026 Sample P5.5; 2026 R1 P5.7 |
| B4 | Reductions along an axis | `sum/mean/max/min/argmax/argmin/prod` with `axis=` and `keepdims=True`; `dim=` in torch | 2025 R1 P1.9, P2.3, P3.17; 2026 Sample P9.8, P11; R2 P1.9 (`torch.sum` before `autograd.grad`) |
| B5 | Norms and unit vectors without np.linalg | `np.sqrt(np.sum(v**2, axis=1, keepdims=True))`, `torch.norm(dim=-1, keepdim=True)` | 2026 R1 P5.4; 2025 R2 P3.15; 2026 Sample P7.3 |
| B6 | Boolean arrays | `(a == b).mean()` accuracy, `np.sum(mask)`, `np.where(p >= 0.5, 1, 0)`, `&`/`|` with pandas, `(x >= 0).float()`, `torch.prod(dim=1)` as AND | 2025 R1 P3.17; 2026 Sample P9.7, P9.11, P11; 2026 R1 P7.1, P7.4 |

### C. Dot product, matmul, transpose
| id | Topic | Operations | Past problems |
|---|---|---|---|
| C1 | Dot product and cosine | `np.dot`, `x @ y`, `np.sum(x*y)`, cosine similarity | 2026 Sample P1.2, P7.3; 2026 R1 P2.2, P5.5 |
| C2 | Matrix × vector, matrix × matrix, shape rule | `W @ x`, `A @ B`, `np.matmul`, `(M,N)@(N,P)→(M,P)`, `*` vs `@` | 2025 R1 P2.1, P3.17; 2026 R1 P3.1 |
| C3 | Transpose and symmetric matrices | `A.T == A`, `X.T @ X`, `X @ X.T`, `X.T @ r` as a weighted sum over samples | 2025 R1 P1.3, P1.4, P3.12–3.13; 2026 Sample P5.3–5.4 |
| C4 | Matmul without `@`, `.T`, `np.linalg` | `np.sum(X * theta, axis=1)`, `np.sum(r[:, None] * X, axis=0)` | **2026 R1 P4.2 (15 pts)**, P5.7 |
| C5 | Batched matmul | `(B,H,L1,d) @ (B,H,d,L2)`, `transpose(-2,-1)`, broadcasting over batch axes | 2025 R2 P2.5, P2.7, P2.13 |
| C6 | Special matrices | `np.eye`, `np.diag` (both directions), `np.trace`, `np.fill_diagonal`, permutation matrices, `np.outer`, rank-1 terms `Σ uᵢvᵢᵀ` | 2026 Sample P2; 2026 R1 P3.1–3.2; 2025 R1 P3.17 (`np.diag`) |

### D. Linear-algebra toolbox
| id | Topic | Operations | Past problems |
|---|---|---|---|
| D1 | Norms, projections, residuals, hyperplanes | `np.linalg.norm`, `(x @ e) * e`, `x - proj`, midpoint hyperplane | 2026 Sample P1, P7; 2026 R1 P2 |
| D2 | Linear systems and inverses | `np.linalg.solve`, `np.linalg.inv`, `det`, `matrix_rank`, Newton step `inv(H) @ grad` | 2024 IAIO Q6.3; 2025 R1 P3.15, P3.17 |
| D3 | Eigenvalues, spectral decomposition, matrix powers | 2×2 by hand, `np.linalg.eigh/eig`, `Q @ diag(λ**n) @ Q.T`, `lam ** idx[:, None]` | 2025 R1 P1.5–1.9; 2026 R1 P5.12 |
| D4 | SVD | `np.linalg.svd(W, full_matrices=False)` → `(U, s, Vt)`, `WWᵀ = UΣ²Uᵀ`, rank, Frobenius norm `Σσ²` / `Σσ⁴` | 2026 R1 P5.11–5.13; 2026 Sample P5.4; 2025 R2 P2.8–2.10 |
| D5 | Low-rank approximation | `U[:, :r] * s[:r]`, relative Frobenius error `1 − Σ_{i<r} σᵢ⁴ / Σ σᵢ⁴`, error-vs-r plot, factor pair `W_UM @ W_DKV` | 2026 R1 P5.14–5.15; 2025 R2 P2.9–2.10 |

### E. Where it shows up in AI
| id | Topic | Operations | Past problems |
|---|---|---|---|
| E1 | Linear / affine layer | `x @ W.T + b` for any leading dims, `nn.Linear` weight `(out, in)`, parameter counting, tied weights `WᵀW` | 2025 R1 P2.1–2.9; 2026 R1 P7.2–7.3; 2024 IAIO Q4 |
| E2 | Gradients in matrix form; GD loop | `Xᵀ(y − Xθ)`, one loop over iterations, loss history plot | 2026 R1 P4.2; 2025 R1 P3.15–3.17; 2026 Sample P6.2 |
| E3 | Logistic regression | `sigmoid(X @ beta)`, `X.T @ (z − y)`, `X.T @ diag(z(1−z)) @ X`, GD and Newton, `np.where` threshold, accuracy | 2025 R1 P3.10–3.18 |
| E4 | Attention (MHA / GQA / MLA) | head split `reshape + permute`, `Q @ Kᵀ / √d`, `softmax(dim=-1)`, `@ V`, merge heads, `nn.Linear(bias=False)`, `nn.Parameter` | 2025 R2 P2.2–2.14 |
| E5 | Similarity / distance tables, nearest neighbours | `W @ W.T`, `fill_diagonal(-inf)`, `argmax(axis=1)`, `‖a‖²+‖b‖²−2abᵀ`, `argsort`, k-means assignment and centroid update | 2026 R1 P5.8–5.9, P9; 2024 IAIO Q3; 2026 Sample P13 |
| E6 | Images and batches as tensors | `np.array(PIL)`, `torch.from_numpy`, `permute(2,0,1)`, scaling to `[-1,1]`, `torch.stack`, `(B,C,H,W)`, conv parameter counts | 2025 R2 P3.4–3.5; 2026 Sample P12; 2026 R1 P6, P8 |
| E7 | PyTorch specifics | `requires_grad`, `autograd.grad(torch.sum(f), x, create_graph=True)`, `.detach()`, `.item()`, `torch.no_grad()`, `.to(device)`, `nn.Module`/`forward`, `Dataset`/`DataLoader`, padding + attention masks | 2025 R2 P1.4–1.12, P3.8–3.16; 2025 R1 P2.5–2.13 |

### F. Exam rules that must be enforced by the grader
1. No imports beyond the starter cell (not even temporarily).
2. No loops when the task says so (`for`, `while`, comprehensions all count). One explicit loop over iterations/epochs is allowed only when the task says so.
3. No `np.linalg`, no `@`, no `.T` when the task says so.
4. Exact output shapes and dtypes (e.g. `(len(indices),)` with `int32`).
5. Seeded randomness exactly as specified; reset with `np.random.seed()` when told.
6. Print exactly what is asked (shape, ndim, dtype, 4 decimal places).

---

## 2. Repository layout

```
usaaio-matrix-practice/
├── README.md                     # what this is, how to use it, the knowledge map (copy §1), badges
├── LICENSE                       # MIT
├── requirements.txt              # numpy, torch (cpu), matplotlib, pandas, scikit-learn, pytest
├── usaaio_check/                 # tiny grading library, installed in Colab with one pip line
│   ├── __init__.py               # exports grade, check_rules, show_shape
│   ├── checker.py                # §3 code
│   └── datasets.py               # synthetic data generators (seeded) used by exercises and mocks
├── notebooks/
│   ├── 00_setup_and_how_to_use.ipynb
│   ├── 01_arrays_and_shapes.ipynb
│   ├── 02_elementwise_and_broadcasting.ipynb
│   ├── 03_dot_matmul_transpose.ipynb
│   ├── 04_linear_algebra_toolbox.ipynb
│   ├── 05_where_it_shows_up_in_ai.ipynb
│   ├── 06_mock_exam_A.ipynb        # 2026-style, NumPy only
│   ├── 06_mock_exam_B.ipynb        # 2025-style, NumPy + PyTorch classes
│   └── 07_pytorch_bridge.ipynb     # optional: autograd, Dataset/DataLoader, nn.Module
├── solutions/                    # same notebooks with solutions filled in (do NOT link from the student README)
│   └── ... (same names)
├── tests/                        # pytest versions of every exercise, run in CI against solutions/
│   └── test_01.py … test_07.py
├── scripts/
│   ├── strip_solutions.py        # produces notebooks/ from solutions/ by replacing solution cells with stubs
│   └── run_all.py                # executes every notebook headless (nbclient) for CI
└── .github/workflows/ci.yml      # pytest + run_all on push
```

Every notebook starts with:

```python
# @title Setup (run first)
!pip -q install git+https://github.com/<owner>/usaaio-matrix-practice.git
import numpy as np, matplotlib.pyplot as plt
from usaaio_check import grade, check_rules, show_shape
np.set_printoptions(precision=4, suppress=True)
```

Each notebook has an "Open in Colab" badge in README and at the top of the notebook.

---

## 3. Grading library (`usaaio_check/checker.py`) — use this code

Tested; works in Colab/Jupyter because cell source is available to `inspect`.

```python
import ast, inspect, numpy as np

def check_rules(fn, no_loops=False, no_linalg=False, no_matmul=False, no_transpose=False, no_imports=True):
    """Static check of a function's source against exam-style bans."""
    try:
        src = inspect.getsource(fn)
    except OSError:
        print("  (could not read source; rule check skipped)"); return []
    tree = ast.parse(src)
    problems = []
    for node in ast.walk(tree):
        if no_loops and isinstance(node, (ast.For, ast.While, ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
            problems.append("loop or comprehension")
        if no_imports and isinstance(node, (ast.Import, ast.ImportFrom)):
            problems.append("import inside the function")
        if no_linalg and isinstance(node, ast.Attribute) and node.attr == "linalg":
            problems.append("np.linalg")
        if no_matmul and isinstance(node, ast.BinOp) and isinstance(node.op, ast.MatMult):
            problems.append("the @ operator")
        if no_matmul and isinstance(node, ast.Attribute) and node.attr in ("dot", "matmul", "einsum", "inner", "tensordot", "vdot"):
            problems.append(f"np.{node.attr}")
        if no_transpose and isinstance(node, ast.Attribute) and node.attr in ("T", "transpose", "swapaxes", "mT", "moveaxis"):
            problems.append(f".{node.attr}")
    return sorted(set(problems))

def grade(fn, cases, rules=None, atol=1e-6):
    """cases: list of (args_tuple, expected). Checks rules, then shape, dtype (for integer expectations) and values."""
    bad = check_rules(fn, **(rules or {}))
    if bad:
        print(f"RULE VIOLATION in {fn.__name__}: {', '.join(bad)} -> 0 points on the real exam"); return False
    ok = True
    for i, (args, expected) in enumerate(cases):
        try:
            got = fn(*args)
            if isinstance(expected, np.ndarray):
                assert isinstance(got, np.ndarray), f"returned {type(got).__name__}, expected a NumPy array"
                assert got.shape == expected.shape, f"shape {got.shape}, expected {expected.shape}"
                if expected.dtype.kind in "iu":
                    assert got.dtype == expected.dtype, f"dtype {got.dtype}, expected {expected.dtype}"
                assert np.allclose(got, expected, atol=atol), "values differ"
            else:
                assert np.allclose(got, expected, atol=atol), f"got {got}, expected {expected}"
            print(f"  case {i+1}: PASS")
        except Exception as e:
            ok = False; print(f"  case {i+1}: FAIL - {e}")
    print("ALL PASS" if ok else "not yet"); return ok

def show_shape(**arrays):
    """show_shape(X=X, y=y) prints name, shape, ndim, dtype on one line each."""
    for k, v in arrays.items():
        print(f"{k:>12s}: shape {tuple(v.shape)}  ndim {v.ndim}  dtype {v.dtype}")
```

Notes for the builder:
- Expected values in `grade` cases must be computed by a **hidden reference implementation** inside the grading cell (fine for a self-study repo), or loaded from `usaaio_check/datasets.py` so the student cannot read the answer in the stub cell.
- For exercises whose answer is a plot, grade the array that feeds the plot, then show the plot.
- For PyTorch exercises, convert with `.detach().numpy()` before `grade`, or write a `grade_torch` wrapper that accepts tensors.
- A `rules` preset dictionary should exist: `NO_LOOPS = dict(no_loops=True)`, `PURE = dict(no_loops=True, no_linalg=True, no_matmul=True, no_transpose=True)`.

---

## 4. Exercise cell format (identical everywhere)

Every exercise is three cells:

1. **Markdown**: `### Exercise 2.7 — Unit rows without np.linalg` · tags `[B5] [no-loop] [no-linalg]` · source `2026 Round 1, Problem 5.4` · one-paragraph task · **"Predict first:"** a one-line shape question the student answers in a text field before coding (e.g. *What is the shape of `norms` if you use `keepdims=True`? And without it?*).
2. **Code stub**:
   ```python
   def unit_rows(V):
       """V: (N, d) float array. Return W (N, d) whose rows have length 1. No loops, no np.linalg."""
       ### YOUR CODE HERE ###
       return ...
   ```
3. **Grading cell** (student runs, never edits): builds seeded test inputs, computes expected with a reference, calls `grade(unit_rows, cases, rules=dict(no_loops=True, no_linalg=True))`.

Then an optional **"Why it matters"** markdown line (one sentence) and a **hint** in a collapsed `<details>` block.

Difficulty labels: ★ (one call), ★★ (shape reasoning), ★★★ (exam-level, several steps).

---

## 5. Complete exercise list

Reference solutions are given so the builder can write graders and solution notebooks without re-deriving them. All test inputs use `rng = np.random.default_rng(seed)` with the seed stated, except where the exam used `np.random.seed(...)`.

### Notebook 01 — Arrays and shapes

| # | Title | Task (signature → return) | Rules | Reference |
|---|---|---|---|---|
| 1.1 ★ | Describe an array | `describe(a)` → tuple `(a.shape, a.ndim, a.size, str(a.dtype))` | — | as stated |
| 1.2 ★ | Build by recipe | `build()` → `(np.zeros((2,3)), np.ones((3,1)), np.eye(3), np.arange(0,1,0.1), np.linspace(0,1,101))` | — | as stated; grade each |
| 1.3 ★ | Seeded normals (2026 Sample P4.1) | `seeded(seed, shape)` → `np.random.seed(seed); return np.random.randn(*shape)` | — | exact match with same seed |
| 1.4 ★★ | Three looks of the same numbers | `three_looks(v)` with `v` shape `(3,)` → `(v, v.reshape(3,1), v.reshape(1,3))`; grader checks shapes `(3,)`, `(3,1)`, `(1,3)` | — | as stated |
| 1.5 ★ | Row, column, block | `pick(X)` for `X=np.arange(12).reshape(3,4)` → `(X[1], X[:,2], X[0:2,1:3], X[:,-1])` | — | shapes `(4,)`, `(3,)`, `(2,2)`, `(3,)` |
| 1.6 ★★ | Mask assignment | `cap(arr, t, value)` → copy of `arr` with entries `> t` set to `value`; must not modify the input | no-loop | `out = arr.copy(); out[out > t] = value` |
| 1.7 ★★ | Random split (2025 R1 P3.8 / 2026 Sample P9.14) | `split_indices(N, test_size, seed)` → `(train_idx, test_idx)`; `np.random.seed(seed)`, `idx = np.random.permutation(N)`, first `int(N*test_size)` are test; then `np.random.seed()` | — | as stated; grader uses the same seed |
| 1.8 ★ | Reshape family | `regroup(a)` for `a` shape `(6,)` → `(a.reshape(2,3), a.reshape(-1,1), a.reshape(2,3).flatten())` | — | shapes `(2,3)`, `(6,1)`, `(6,)`, order preserved |
| 1.9 ★★ | squeeze / expand_dims / swapaxes (2026 Sample P4.2–4.4) | `shape_walk(arr)` for `(5,8,3,1,2)` → tuple of shapes after `squeeze`, then `expand_dims(axis=2)`, then `swapaxes(0,1)`, then `flatten` | — | `(5,8,3,2)`, `(5,8,1,3,2)`, `(8,5,1,3,2)`, `(240,)` |
| 1.10 ★★ | reshape vs transpose | `which_moved(M)` for `M=np.arange(6).reshape(2,3)` → `(M.reshape(3,2), M.T)`; grader also asks the student (markdown) to mark the cells that differ | — | `[[0,1],[2,3],[4,5]]` vs `[[0,3],[1,4],[2,5]]` |
| 1.11 ★★ | HWC → CHW | `to_chw(img)` for `(H,W,3)` → `np.transpose(img, (2,0,1))`; grader checks that `out[c, i, j] == img[i, j, c]` | no-loop | as stated; also reject `reshape` by checking values |
| 1.12 ★★ | stack vs concatenate | `join(a, b)` for two `(3,)` → `(np.concatenate([a,b]), np.stack([a,b]), np.stack([a,b], axis=1))` | — | shapes `(6,)`, `(2,3)`, `(3,2)` |
| 1.13 ★★ | Bias column (2025 R1 P3.9) | `add_ones(X)` → `np.concatenate([X, np.ones((X.shape[0],1))], axis=1)` | no-loop | shape `(N, d+1)`, last column all ones |
| 1.14 ★★ | Repeat copies (2025 R2 P2.10) | `tile_rows(W, k)` → `np.concatenate([W]*k, axis=0)` | no-loop | shape `(k·r, D)`; rows `i` and `i+r` equal |
| 1.15 ★★★ | PINN grids (2025 R2 P1.6–1.7, P1.12), NumPy version | `ic_points()` → `(101,2)` with `t=0`, `x=linspace(0,1,101)`; `bc_points()` → `(202,2)` with `x∈{0,1}`; `grid_points()` → `(10201,2)` all pairs via `np.meshgrid` + `reshape(-1)` + `stack(axis=1)` | no-loop | as in official solution |

### Notebook 02 — Elementwise operations and broadcasting

| # | Title | Task | Rules | Reference |
|---|---|---|---|---|
| 2.1 ★ | Activations in one line (2025 R1 P2.10, P3.16) | `sigmoid(z)`, `relu(z)`, `tanh_(z)` for any shape | no-loop | `1/(1+np.exp(-z))`, `np.maximum(0,z)`, `np.tanh(z)` |
| 2.2 ★★ | Stable softmax along the last axis | `softmax(z)` | no-loop | `e=np.exp(z-z.max(-1,keepdims=True)); e/e.sum(-1,keepdims=True)`; rows sum to 1 |
| 2.3 ★★ | Broadcast or error? | `broadcast_result(shape_a, shape_b)` → resulting shape tuple or `None` (student implements the right-alignment rule by hand, no NumPy allowed inside; grader also checks against `np.broadcast_shapes`) | — | cases `(2,3)+(3,)`, `(2,3)+(2,)`, `(2,3)+(2,1)`, `(4,1)*(1,3)`, `(B,L,H,d)*(H,1)` |
| 2.4 ★★ | All pairs (2026 Sample P5.5) | `kernel_matrix(x)` → `(1 + P + 2P²)²` with `P = x[:,None]*x[None,:]` | no-loop, no-linalg | symmetric `(N,N)` |
| 2.5 ★★ | Column-wise and row-wise sums | `sums(A)` → `(A.sum(0), A.sum(1), A.sum(1, keepdims=True))` | no-loop | shapes `(3,)`, `(2,)`, `(2,1)` |
| 2.6 ★★ | Min–max scaling per column from scratch (2026 Sample P9.8) | `minmax(X)` → `(X - X.min(0)) / (X.max(0) - X.min(0))` | no-loop | every column in `[0,1]` |
| 2.7 ★★ | Standardize per column (what `StandardScaler` does) | `standardize(X)` → `(X - X.mean(0)) / X.std(0)` | no-loop | column means 0, stds 1 |
| 2.8 ★★ | Unit rows without np.linalg (2026 R1 P5.4) | `unit_rows(V)` | no-loop, no-linalg | `V/np.sqrt(np.sum(V**2,1,keepdims=True))` |
| 2.9 ★★ | Accuracy and threshold (2025 R1 P3.17) | `predict(p)` → `np.where(p>=0.5,1,0)`; `accuracy(y_pred,y)` → `np.mean(y_pred==y)` | no-loop | as stated |
| 2.10 ★★ | Counting with masks (2026 Sample P9.7, P9.11) | `count_unknown(df)` → per-column count of `"unknown"` via `df.isin([...]).sum()`; `odd_married_counts(df)` → `value_counts` of `y` under `(marital=="married") & (age%2==1)` | — | pandas one-liners; use a tiny synthetic DataFrame |
| 2.11 ★★★ | One-hot from scratch (2026 Sample P9.9) | `one_hot(labels, categories)` → `(N, K)` 0/1 array via `(labels[:,None] == np.array(categories)[None,:]).astype(int)` | no-loop | row sums 1 |
| 2.12 ★★★ | Affine over the last axis by broadcasting (2025 R1 P2.3) | `affine_bc(x, W, b)` for `x` of shape `(..., in)` → `np.sum(x[..., None] * W.T, axis=-2) + b` | no-loop, no-matmul | equals `x @ W.T + b`; test with `(in,)`, `(10,in)`, `(10,20,in)`, `(10,20,30,in)` |

### Notebook 03 — Dot product, matmul, transpose

| # | Title | Task | Rules | Reference |
|---|---|---|---|---|
| 3.1 ★ | Dot three ways | `dot3(x,y)` → `(np.dot(x,y), x@y, np.sum(x*y))`; grader checks all equal | — | as stated |
| 3.2 ★★ | Cosine similarity of two vectors | `cosine(x,y)` | no-linalg | `x@y/np.sqrt((x@x)*(y@y))` |
| 3.3 ★★ | Matrix × vector by hand then code (2025 R1 P2.1) | `affine(W,x,b)` → `W@x+b`; first case is the exam's `W (3,5)`, `x`, `b` → `[1,3,20]` | — | as stated |
| 3.4 ★★ | Shape rule quiz | `matmul_shape(sa, sb)` → result shape or `None` for 2-D inputs | — | `(M,N)@(N,P)→(M,P)` |
| 3.5 ★★ | `*` is not `@` | `both(B,C)` → `(B@C, B*C)`; exam numbers `B=[[1,2],[3,4]]`, `C=[[0,1],[1,0]]` | — | `[[2,1],[4,3]]`, `[[0,2],[3,0]]` |
| 3.6 ★★ | Gram matrices | `grams(X)` → `(X.T@X, X@X.T)`; grader checks symmetry and shapes `(d,d)`, `(N,N)` | — | as stated |
| 3.7 ★★ | Weighted sum of rows | `weighted_rows(X, r)` → `X.T @ r` **and** the no-`@`/no-`.T` version `np.sum(r[:,None]*X, axis=0)`; return both | second must pass `PURE` | equal results |
| 3.8 ★★★ | **MSE gradient, pure (2026 R1 P4.2)** | `my_fun_derivative(X,y,theta)` → `∇θ (1/N)Σ(y−θᵀx)²` shape `(d,)` | `PURE` (no loops, no `@`, no `.T`, no `np.linalg`) | `-2/N*np.sum((y-np.sum(X*theta,1))[:,None]*X,0)` |
| 3.9 ★★★ | Similarity matrix, pure (2026 R1 P5.6–5.8) | `sim_matrix(W)` → `np.sum(W[:,None,:]*W[None,:,:], -1)`; grader checks `== W@W.T` and `diag == 1` when rows are unit | `PURE` | as stated |
| 3.10 ★★ | Batched matmul shapes | `attn_scores(Q,K)` for `(B,H,L1,d)`, `(B,H,L2,d)` → `Q @ np.swapaxes(K,-2,-1) / np.sqrt(d)` shape `(B,H,L1,L2)` | no-loop | as stated; grader checks one `(b,h)` slice against a 2-D product |
| 3.11 ★★ | Permutation matrix (2026 Sample P2) | `perm_matrix(order)` → `np.eye(len(order))[order]`; grader checks `P @ x == x[order]` for `order=[2,3,1,0]` | no-loop | as stated |
| 3.12 ★★ | Outer product and rank-1 sum (2026 R1 P3.2) | `outer_sum(us, vs)` → `Σ uᵢ vᵢᵀ` from lists of vectors, computed as `U.T @ V` where `U`, `V` stack the vectors as rows; second case: rebuild the exam's `A (3,4)` with `Ax = [x2−x3, x3, x0+2x1]` from `u = e_i`, `v = row i` | — | `A = [[0,0,1,-1],[0,0,0,1],[1,2,0,0]]` |
| 3.13 ★★ | diag both ways, trace, fill_diagonal | `diag_tools(M, v)` → `(np.diag(v), np.diag(M), np.trace(M))`; `kill_diag(S)` → copy with `-inf` on the diagonal | — | as stated |

### Notebook 04 — Linear-algebra toolbox

| # | Title | Task | Rules | Reference |
|---|---|---|---|---|
| 4.1 ★★ | Unit vector, projection, residual (2026 Sample P1) | `project(x, v)` → `(e, c, proj, r)`; exam case `v=[1,-2,3]`, `x=[-1,-4,6]` | — | `e=v/√14`, `c=25/√14`, `proj=(25/14)v`, `r=(1/14)[-39,-6,9]`, `r@e≈0` |
| 4.2 ★★ | Midpoint hyperplane (2026 Sample P7.3) | `hyperplane(X)` for `X (2,d)` → `(theta_hat, b)` | — | `w=x0-x1; theta=w/‖w‖; b=-theta@(x0+x1)/2`; cases `[(-3,0),(3,0)]→([-1,0],0)` and `[(5,7),(-3,2)]→([8,5]/√89, -30.5/√89)` |
| 4.3 ★★ | Solve vs inverse (2024 IAIO Q6.3) | `solve_system(A,b)` → `np.linalg.solve`; grader checks `A@x≈b` and compares with `inv(A)@b`; use the Bellman system `A=[[0.91,-0.72,0],[-0.09,1,-0.09],[0,-0.09,0.91]]`, `b=[1,2,0]` | — | `x≈[2.904, 2.282, 0.226]` (**note:** the published 2024 key prints 2.094 for the first entry; 2.904 is correct — mention this in the notebook as a lesson in checking `A@x == b`) |
| 4.4 ★★ | 2×2 eigenvalues by hand (2025 R1 P1.5) | `eig2(a,b,c,d)` → the two roots of `λ² − (a+d)λ + (ad−bc) = 0`, descending; grader compares with `np.linalg.eigvals` | no-linalg | quadratic formula |
| 4.5 ★★★ | Spectral decomposition and matrix powers (2025 R1 P1.6–1.9) | `My_Fib` class with attributes `Q (2,2)`, `lambdas (2,)`, method `compute_fib(f0,f1,indices)` → `int32` array `(len(indices),)` using `Q`, `lambdas**n` broadcasting, no loop; `plot_fib` with marker `x`, linestyle `--`, red | no-loop inside `compute_fib` | official solution: `(np.power(lambdas, np.array(indices)[:,None]) / np.sqrt(1+lambdas**2)) @ (Q.T @ [f1,f0])`, `.astype(np.int32)`; test `f0=0,f1=1,range(10)` → `[0,1,1,2,3,5,8,13,21,34]` |
| 4.6 ★★ | eigh round trip | `spectral(A)` for symmetric `A` → `(lam, Q)` with `Q@diag(lam)@Q.T ≈ A` and `Q.T@Q ≈ I` | — | `np.linalg.eigh` |
| 4.7 ★★ | SVD shapes and the Vt trap (2026 R1 P5.11) | `svd_parts(W)` → `(U, s, V)` where `V = Vt.T`; grader checks `U*s@V.T ≈ W`, `s` descending, shapes `(N,d)`, `(d,)`, `(d,d)` | — | `np.linalg.svd(W, full_matrices=False)` |
| 4.8 ★★★ | Spectral decomposition of S from the SVD of W (2026 R1 P5.12–5.13) | `spectral_from_svd(W)` → `(Q, lam)` with `Q=U`, `lam=s**2`; `fro_sq(W)` → `np.sum(s**4)`; grader checks `Q@diag(lam)@Q.T ≈ W@W.T` and `fro_sq == np.sum((W@W.T)**2)` | — | as stated |
| 4.9 ★★★ | Low-rank approximation and the error curve (2026 R1 P5.14–5.15) | `lowrank_error(W, r)` → relative squared Frobenius error of `S_r = W_r W_rᵀ`, `W_r = U[:,:r]*s[:r]`; `error_curve(W)` → array over `r=1..d`; grader checks against `1 - np.sum(s[:r]**4)/np.sum(s**4)` and that the curve is non-increasing and ends at 0; then plot | no-loop in `lowrank_error` | as stated |
| 4.10 ★★★ | GQA → MLA factorization (2025 R2 P2.10) | `GQA_2_MLA(W)` for `(r, D)` with `r | D` → `(W_DKV (r,D), W_UM (D,r))` such that `W_UM @ W_DKV ≈ concatenate([W]*(D//r), axis=0)`; print shapes and MSE; test with `np.random.randn(4,24)` | no-loop | official: `U,S,V=svd(tilde)`, `W_UM=U[:,:r]`, `W_DKV=S.reshape(-1,1)[:r]*V[:r]` |
| 4.11 ★★ | Rank and determinant facts (2026 Sample P5.3–5.4) | `kernel_rank(x)` → `np.linalg.matrix_rank(kernel_matrix(x))` for the degree-4 polynomial kernel; grader expects 5 for `N ≥ 6` distinct points, and `det ≈ 0` | — | as stated |
| 4.12 ★★★ | Pairwise distances two ways (2024 IAIO Q3, 2026 R1 P9) | `pairwise_sq_dist(A,B)` via `‖a‖²+‖b‖²−2abᵀ` and via broadcasting `(A[:,None,:]-B[None,:,:])**2 .sum(-1)`; `kmeans_assign(X, C)` → `argmin(axis=1)`; `kmeans_update(X, labels, k)` → new centroids with `np.array([X[labels==j].mean(0) for j in range(k)])` (loop over k allowed here) ; test on the IAIO table with centers `(1.90,0.97)`, `(3.17,4.96)` → labels `[0,0,0,0,0,1,1,1,1,1]`, new centers `(1.886,1.528)`, `(4.178,3.778)` | no-loop in distances | as stated |

### Notebook 05 — Where it shows up in AI (NumPy first, then the PyTorch twin)

| # | Title | Task | Rules | Reference |
|---|---|---|---|---|
| 5.1 ★★★ | `My_Linear_NumPy` (2025 R1 P2.3–2.4) | class with `in_features`, `out_features`, `weight (out,in)`, `bias (out,)`, `random_seed=42`, `forward(x)` for any leading dims | no-loop | `x @ self.weight.T + self.bias`; print output shapes for the four exam input shapes |
| 5.2 ★★ | Parameter counting (2025 R1 P2.6, 2024 IAIO Q4, 2026 Sample P12.3) | `n_params_linear(in_,out,bias)`, `n_params_mlp(sizes)`, `n_params_conv(in_c,out_c,k,bias)`, `n_params_ffn(d,f)` | — | `out*in+out`, sum over layers, `out*in*k*k(+out)`, `2*d*f` |
| 5.3 ★★ | Tied-weight symmetric model (2025 R1 P2.7–2.9) | `symmetric_forward(W, x)` → `W.T @ W @ x` for `x (in,)` and batched `x (B,in)` via `x @ W.T @ W` | no-loop | rank of `WᵀW` equals rank of `W` (grader checks with `matrix_rank`) |
| 5.4 ★★★ | Gradient descent for linear regression (E2) | `fit_gd(X,y,lr,n_iter)` → `(theta, loss_history)` using `grad = -2/N * X.T @ (y - X@theta)`; only the iteration loop | one loop | loss must decrease; compare `theta` with `np.linalg.lstsq` on well-conditioned data |
| 5.5 ★★★ | `My_Log_Reg` (2025 R1 P3.16–3.18) | `my_sigmoid`, class with `solver∈{GD,Newton}`, `lr`, `num_iter`, `coef_ (d,)`, `fit` (one loop; `grad=X.T@(z-y)`, `H=X.T@np.diag(z*(1-z))@X`, Newton uses `np.linalg.inv(H)@grad`), `predict` with `np.where`, `score` = accuracy; raise `ValueError('Invalid solver')` otherwise | one loop | test on a seeded 2-feature Gaussian blob dataset from `datasets.py`; Newton should reach ≥ 0.9 accuracy in 20 iterations |
| 5.6 ★★★ | Attention in NumPy (2025 R2 P2.5) | `mha_numpy(x, y, WQ, WK, WV, WO, H)` with weights given as `(H·d, D)` matrices: project, `reshape(B,L,H,d)`, `transpose(0,2,1,3)`, scores, softmax, `@V`, merge, out-projection; return `(B,L1,D1)`; **a shape comment after every line is required** (grader checks only values; the markdown asks for comments) | no-loop | official solution logic |
| 5.7 ★★★ | GQA by broadcasting (2025 R2 P2.7) | `gqa_numpy(..., H, G)` with `K,V` having `G` heads; `Q.reshape(B,L1,H//G,G,d)`, `K.reshape(B,L2,1,G,d)` so the size-1 axis broadcasts | no-loop, no copies of K/V | equals `mha_numpy` when `G == H` |
| 5.8 ★★★ | Nearest-token dictionary (2026 R1 P5.9) | `nearest_dict(tokens, W)` → `{token: most similar other token}` using `S=W@W.T`, `fill_diagonal(-inf)`, `argmax(axis=1)` | no-loop except the final dict comprehension | use a synthetic 50-token, 20-dim embedding matrix from `datasets.py`; grader checks the dictionary |
| 5.9 ★★★ | Embedding compression end to end (2026 R1 P5, condensed) | given `vectors (N,100)`: unit rows → `S` → `U,s,Vt` → `Q,lam` → `‖S‖_F²` → error curve → plot | mixed | reuse 2.8, 3.9, 4.7–4.9 |
| 5.10 ★★ | Image batch (2025 R2 P3.4) | `to_batch(list_of_HWC_uint8)` → float array `(B,3,224,224)` in `[-1,1]` via `np.transpose(img,(2,0,1))/255*2-1` and `np.stack`; grader checks range, shape and one pixel | loop over the list allowed | as stated |
| 5.11 ★★ | Triangle classifier with threshold units (2026 Sample P11 / 2026 R1 P7.4) | `inside_triangle(P)` for `P (B,2)` → `(B,)` 0/1 using `W (3,2)`, `b (3,)`, step `(z>=0)`, AND via `np.prod(axis=1)`; vertices `(0,0),(1,0),(0,2)`; second case vertices `(0,0),(3,1),(1,5)` (student derives the three half-plane inequalities) | no-loop | first case `W=[[1,0],[0,1],[-2,-1]]`, `b=[0,0,2]` |
| 5.12 ★★ | Softmax temperature (2025 R2 P3.20) | `softmax_T(z, tau)`; grader checks `tau→0` gives one-hot on the max, `tau→∞` gives uniform | no-loop | `softmax(z/tau)` |
| 5.13 ★★★ | CLIP-style contrastive loss in NumPy (2025 R2 P3.15) | `clip_loss(I, T, log_tau)`: unit-normalize rows, `sim = I @ T.T`, symmetric cross-entropy on the diagonal with `log_softmax` along axis 0 and axis 1 | no-loop | official logic; grader uses a tiny `B=4`, `d=8` case |

### Notebook 07 — PyTorch bridge (optional but recommended before Round 2)

| # | Title | Task | Reference |
|---|---|---|---|
| 7.1 | NumPy ↔ torch vocabulary | redo 1.8, 1.9, 1.12, 2.8, 3.9 with `torch` (`unsqueeze`, `squeeze`, `permute`, `cat`, `stack`, `dim=`, `keepdim=`) | same numbers |
| 7.2 | `My_ReLU`, `My_Threshold`, `My_Linear` as `nn.Module` (2025 R1 P2.10, 2026 R1 P7.1–7.3) | `torch.max(torch.zeros_like(x), x)`; `(x>=0).float()`; `nn.Parameter(torch.randn(out,in)*std)` with `std = 1/sqrt(in)` for unit output variance (2026 R1 P7.2), zero bias, `bias=False` option | shapes and variance test |
| 7.3 | `My_MLP_Model` with `nn.Sequential` + training loop on `y = sin(2πx) + noise` (2025 R1 P2.11–2.13) | MSE, Adam, whole batch, print every 10 epochs with 4 decimals, loss plot | loss decreases |
| 7.4 | `autograd.grad` drills (2025 R2 P1.9) | scalar `f(p,q)=p²+q³+pq²` at `(1,2)`: `f_p=6, f_q=16, f_pp=2, f_qq=12, f_pq=4, f_qqq=6`; elementwise `g=x²` via `autograd.grad(torch.sum(g), x, create_graph=True)` → `2x`, then `2` | exact values |
| 7.5 | `MyMHA` as `nn.Module` (2025 R2 P2.5) and equality test against 5.6 with copied weights (`nn.Parameter`) | relative error < 1e-5 |
| 7.6 | `Dataset`, `DataLoader`, `collate_fn` with padding and attention masks (2025 R2 P1.4–1.5, P3.8–3.9) | `token_id_batch (B,K)`, `attention_mask_batch (B,K)` int64 | shapes and mask correctness |

### Notebooks 06 — Mock exams (timed, 60 minutes each, rules enforced by `grade`)

**Mock A (2026 style, NumPy only, no datasets):**
1. 2026 Sample P4 verbatim (shape walk) — 10 pts.
2. 2026 Sample P5.5 kernel matrix — 10 pts.
3. 2026 Sample P7.3 hyperplane — 10 pts.
4. 2026 R1 P4.2 `my_fun_derivative` with `PURE` rules — 15 pts.
5. 2026 R1 P5.3–5.15 on a **synthetic** `vectors` list (`datasets.random_embeddings(N=300, d=100, seed=2026)`; no gensim): `vectors_arr`, `W` (no linalg), range of `sim`, `S` in one line (no `@`), diagonal, `sim_dict`, invertibility reasoning (text), `U,σ,V`, `Q,λ`, `‖S‖_F²`, `W_r`, error plot — 45 pts.
6. 2026 Sample P2 / 2026 R1 P3: write the matrix and its outer-product decomposition (text + code check) — 10 pts.

**Mock B (2025 style, NumPy + PyTorch classes):**
1. 2025 R1 P1.9–1.10 `My_Fib` — 20 pts.
2. 2025 R1 P2.3–2.4 `My_Linear_NumPy` + shapes — 15 pts.
3. 2025 R1 P2.10–2.11 `My_ReLU`, `My_MLP_Model` — 15 pts.
4. 2025 R1 P3.16–3.18 `My_Log_Reg` on `datasets.titanic_like(seed=2025)` (synthetic, same column structure: `Survived, Sex, Age, SibSp, Parch, Fare, Embarked`) including `get_dummies(drop_first=True, dtype=np.int8)`, manual train/test split, `StandardScaler`, column of ones — 30 pts.
5. 2025 R2 P2.5 + P2.7 `MyMHA`, `MyGQA` with equality test at `G == H` — 20 pts.

A `MOCK_RULES.md` in the notebook folder restates the exam warning box (no extra imports, no temporary imports, CPU only).

---

## 6. `usaaio_check/datasets.py` (synthetic, seeded, no downloads)

```python
import numpy as np, pandas as pd
def random_embeddings(N=300, d=100, seed=2026):
    """Clustered unit-ish vectors + token names so sim_dict has sensible answers."""
    rng = np.random.default_rng(seed)
    centers = rng.standard_normal((10, d))
    labels = rng.integers(0, 10, N)
    V = centers[labels] + 0.3 * rng.standard_normal((N, d))
    tokens = [f"tok{i:03d}" for i in range(N)]
    return tokens, V
def blobs(N=400, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, N)
    X = rng.standard_normal((N, 2)) + np.where(y[:, None] == 1, 2.0, -2.0)
    return X, y.astype(int)
def titanic_like(N=600, seed=2025):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "Survived": rng.integers(0, 2, N), "Pclass": rng.integers(1, 4, N),
        "Sex": rng.choice(["male", "female"], N), "Age": rng.normal(30, 12, N).round(1),
        "SibSp": rng.integers(0, 4, N), "Parch": rng.integers(0, 3, N),
        "Fare": rng.gamma(2, 15, N).round(2), "Embarked": rng.choice(["S", "C", "Q"], N)})
    df.loc[rng.choice(N, 40, replace=False), "Age"] = np.nan   # missing values to drop
    return df
def iaio_kmeans_table():
    X = np.array([[1.90,0.97],[1.76,0.84],[2.32,1.63],[2.31,2.09],[1.14,2.11],
                  [5.02,3.02],[5.74,3.84],[2.25,3.47],[4.71,3.60],[3.17,4.96]])
    C0 = np.array([[1.90,0.97],[3.17,4.96]])
    return X, C0
```

Real datasets the exams used (optional "extra" notebooks only, with a network warning): Titanic CSV (2025 R1 P3), `usaaio-official/2026_USAAIO_samples` bank and breast-cancer CSVs on Hugging Face (2026 Sample P9, P13; 2026 R1 P9), GloVe `glove-wiki-gigaword-100` via gensim (2026 R1 P5). Do not make the core notebooks depend on them.

---

## 7. Writing style rules for the notebooks

- Explain in plain English for a beginner; never assume prior NumPy knowledge in notebook 01–02. One new idea per exercise.
- Every exercise shows the **expected shape** in the docstring and asks the student to predict shapes before running.
- Tiny made-up numbers in examples (2×3 matrices, vectors of length 3) so results can be checked by hand; seeded random data only in graders and mocks.
- Hints go in collapsed `<details><summary>Hint</summary>…</details>` blocks; the first hint names the operation, the second gives the shape trick, never the full line.
- Code comments in English; no `import` statements anywhere except the setup cell and inside grader cells.
- Grader cells print **PASS/FAIL per case** and **RULE VIOLATION** messages verbatim from `grade`; never print the expected arrays.
- Each notebook ends with a "Checklist" markdown cell (the operations learned) and a link to the next notebook.
- README contains: purpose, how to open in Colab, the six-notebook plan (Week 1: 01–02, Week 2: 03–04, Week 3: 05 (+07), Week 4: 06 mocks), the knowledge map of §1, and a one-page cheat sheet (below).

### Cheat sheet to include in README

| Task | NumPy one-liner |
|---|---|
| unit rows, no linalg | `W = V / np.sqrt(np.sum(V**2, axis=1, keepdims=True))` |
| all pairs of a vector | `P = x[:, None] * x[None, :]` |
| similarity of rows | `S = W @ W.T`  or  `np.sum(W[:, None, :] * W[None, :, :], axis=-1)` |
| `X @ theta` without `@` | `np.sum(X * theta, axis=1)` |
| `X.T @ r` without `@` or `.T` | `np.sum(r[:, None] * X, axis=0)` |
| bias column | `np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)` |
| random split | `idx = np.random.permutation(N); test, train = idx[:k], idx[k:]` |
| softmax | `e = np.exp(z - z.max(-1, keepdims=True)); e / e.sum(-1, keepdims=True)` |
| accuracy / threshold | `np.mean(pred == y)` · `np.where(p >= 0.5, 1, 0)` |
| nearest other row | `np.fill_diagonal(S, -np.inf); np.argmax(S, axis=1)` |
| SVD and rank-r | `U, s, Vt = np.linalg.svd(W, full_matrices=False); W_r = U[:, :r] * s[:r]` |
| spectral and Aⁿ | `lam, Q = np.linalg.eigh(A); A_n = Q @ np.diag(lam**n) @ Q.T` |
| heads in / out | `x.reshape(B, L, H, d).permute(0, 2, 1, 3)` · reverse the two steps to merge |
| HWC → CHW | `np.transpose(img, (2, 0, 1))` / `t.permute(2, 0, 1)` |

---

## 8. Acceptance criteria

1. `pytest tests/` passes against `solutions/`.
2. `scripts/run_all.py` executes every notebook in `notebooks/` without error (stubs return `...`, so grader cells must print FAIL, not raise).
3. Every exercise in §5 exists, with the stated tags, rules and at least two grader cases (one hand-checkable small case, one seeded random case).
4. `check_rules` rejects the deliberately wrong solutions listed in `tests/test_rules.py`: a loop version, an `@` version and a `.T` version of exercise 3.8.
5. No notebook imports anything outside `numpy, torch, matplotlib, pandas, sklearn, usaaio_check`.
6. Total time to run all student notebooks on Colab CPU < 10 minutes.
