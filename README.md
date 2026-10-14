# USAAIO NumPy & PyTorch Practice

[![Notebook checks](https://github.com/dujing82-blip/usaaio_numpy_course/actions/workflows/ci.yml/badge.svg)](https://github.com/dujing82-blip/usaaio_numpy_course/actions/workflows/ci.yml)
![CPU only](https://img.shields.io/badge/runtime-CPU-blue)
![License](https://img.shields.io/badge/license-MIT-green)

**Learn to read shapes, write array operations, and debug your own code.**

A beginner-friendly lab for a student preparing for the USA AI Olympiad. Includes **71 practice exercises**, **11 mock-exam problems**, shape predictions, folded hints, and small automatic checks. All instructional content is in English. The scope follows the supplied 2024 IAIO / 2025 USA-NA-AIO / 2026 USAAIO Round 1 knowledge map; this is not a claim to cover later rounds or future exams.

## Start here

1. Open notebook 00 in Colab and choose a CPU runtime.
2. Use **File → Save a copy in Drive** so your work is saved.
3. Run Setup. Read each task and write your predicted shapes.
4. Replace `...` only inside the answer stubs. Run the grader; blank answers print FAIL without crashing the notebook.
5. Keep practicing until each component says ALL PASS. Use the hints when needed.

The first package installation needs internet. Core datasets are synthetic and generated locally; no dataset downloads are required. Free Colab already provides the scientific libraries used here. Runtime targets exclude first-time package installation and service startup delays.

| Notebook | What you practice | Open |
|---|---|---|
| 00_setup_and_how_to_use | Setup and notebook workflow | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dujing82-blip/usaaio_numpy_course/blob/main/notebooks/00_setup_and_how_to_use.ipynb) |
| 01_arrays_and_shapes | Arrays, indexing, reshaping, stacking | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dujing82-blip/usaaio_numpy_course/blob/main/notebooks/01_arrays_and_shapes.ipynb) |
| 02_elementwise_and_broadcasting | Elementwise operations and broadcasting | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dujing82-blip/usaaio_numpy_course/blob/main/notebooks/02_elementwise_and_broadcasting.ipynb) |
| 03_dot_matmul_transpose | Dot products, matrix multiplication, transpose | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dujing82-blip/usaaio_numpy_course/blob/main/notebooks/03_dot_matmul_transpose.ipynb) |
| 04_linear_algebra_toolbox | Projections, systems, eigenvalues, SVD, low rank | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dujing82-blip/usaaio_numpy_course/blob/main/notebooks/04_linear_algebra_toolbox.ipynb) |
| 05_where_it_shows_up_in_ai | Linear models, attention, embeddings, images | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dujing82-blip/usaaio_numpy_course/blob/main/notebooks/05_where_it_shows_up_in_ai.ipynb) |
| 07_pytorch_bridge | Optional: tensors, autograd, modules, data loaders | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dujing82-blip/usaaio_numpy_course/blob/main/notebooks/07_pytorch_bridge.ipynb) |
| 06_mock_exam_A | 60-minute NumPy mock exam | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dujing82-blip/usaaio_numpy_course/blob/main/notebooks/06_mock_exam_A.ipynb) |
| 06_mock_exam_B | 60-minute NumPy + PyTorch mock exam | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dujing82-blip/usaaio_numpy_course/blob/main/notebooks/06_mock_exam_B.ipynb) |

## Four-week plan

| Week | Lessons |
|---|---|
| 1 | 01–02: arrays/shapes and broadcasting |
| 2 | 03–04: multiplication and linear algebra |
| 3 | 05: AI applications; optional 07: PyTorch bridge |
| 4 | Both 06 mock exams, 60 minutes each |

Try three to five exercises per session. `★` = one main operation, `★★` = shape reasoning, `★★★` = multiple steps. Do not rush to train a model before you can explain the shapes.

## Source trail and adaptations

Official past-paper entry point: [USAAIO Past Problems](https://www.usaaio.org/past-problems). Each task carries a source tag and a knowledge-map topic. Detailed problem/part mappings are inherited from the supplied build specification, not independently re-certified against every original paper. Mocks are adaptations with fresh synthetic data, not verbatim official exams. This course is independent of the contest organizers.

The course includes several mathematical and grading repairs:

- `f(p,q)=p²+q³+pq²` has `f_qq=14` at `(1,2),` correcting 12 in the specification.
- Fibonacci values are rounded before int32 conversion; direct truncation can lose one due to floating-point error. Indices are deliberately small enough for int32.
- SVD/eigenvector checks accept sign changes and alternative valid factorizations; they grade reconstruction, shapes, and orthogonality.
- Relative **squared** Frobenius error for `S=W W.T` uses fourth powers of singular values; the formula for W itself uses squares.
- Min–max scaling, standardization, and unit normalization assume nonconstant columns/nonzero rows, as stated in the exercises.
- Synthetic Titanic labels are random. The task checks preprocessing and fitting, with no high-accuracy claim. StandardScaler fits only training rows.
- `log_tau` in the CLIP drill means log(temperature), so logits divide by `exp(log_tau)`.
- PINN grids use rows `(t,x)` and `meshgrid(indexing="ij")`; boundary rows list x=0 then x=1.

The graders are open self-study tools, not a secure exam sandbox. They check common banned syntax inside submitted functions/methods, not arbitrary aliases or external helper code. Shape predictions, explanations, the exact visual style of plots, and the exam-wide import rule require self-review. Student notebooks contain no instructor answer code in the answer cells; grader internals are inspectable for transparency.

## Knowledge map (from the supplied exam analysis)

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

### F. Exam-style rules
1. No imports beyond the starter cell (not even temporarily).
2. No loops when the task says so (`for`, `while`, comprehensions all count). One explicit loop over iterations/epochs is allowed only when the task says so.
3. No `np.linalg`, no `@`, no `.T` when the task says so.
4. Exact output shapes and dtypes (e.g. `(len(indices),)` with `int32`).
5. Seeded randomness exactly as specified; reset with `np.random.seed()` when told.
6. Print exactly what is asked (shape, ndim, dtype, 4 decimal places).

---


## Quick reference

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
| heads in / out (PyTorch) | `x.reshape(B, L, H, d).permute(0, 2, 1, 3)` · reverse the two steps to merge |
| HWC → CHW | `np.transpose(img, (2, 0, 1))` / `t.permute(2, 0, 1)` |

---


## Local development

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
pip install -e .
python -m pytest -q
python -m scripts.run_all --folder both
```

`exercise_manifest.json` lists all exercises, targets, case counts, and rules. `python -m scripts.build_notebooks` regenerates notebook content from the catalog. `python -m scripts.strip_solutions` regenerates student stubs from instructor notebooks. The execution runner skips only the GitHub package-install line, uses the checked-out package, runs all other cells, requires every instructor answer to pass, and checks CPU runtime limits. CI performs the same checks on pushes and pull requests.

MIT applies to the original course code. Exam names and source materials belong to their respective owners.

On hosts that prohibit Jupyter sockets, `python -m scripts.run_all --folder both --in-process` executes through IPython instead. See [VALIDATION.md](VALIDATION.md) for the local verification report.
