"""Generate consistent notebooks from the auditable exercise catalog."""
import ast
import copy
import json
import re
from pathlib import Path
import nbformat as nbf
from usaaio_check.catalog import EXERCISES
ROOT = Path(__file__).resolve().parents[1]
REPO = 'dujing82-blip/usaaio_numpy_course'
NAMES = {
 '0':'00_setup_and_how_to_use', '1':'01_arrays_and_shapes',
 '2':'02_elementwise_and_broadcasting','3':'03_dot_matmul_transpose',
 '4':'04_linear_algebra_toolbox','5':'05_where_it_shows_up_in_ai',
 'A':'06_mock_exam_A','B':'06_mock_exam_B','7':'07_pytorch_bridge'}
TOPICS = {
 '1':['A1','A2','A2','A4','A3','A3','A3','A4','A4 A5','A4 A5','A5 E6','A6','A6','A6','A2 A6'],
 '2':['B1','B1 B4','B2','B3','B4','B4','B4','B5','B6','B6','B2 B6','B2 E1'],
 '3':['C1','C1 B5','C2','C2','C2','C3','C3 C4','C4 E2','C4 B3','C5','C6','C6','C6'],
 '4':['D1','D1','D2','D3','D3','D3','D4','D4','D5','D5 E4','D2','E5'],
 '5':['E1','E1','E1 C3','E2','E3','E4','E4','E5','D4 D5 E5','E6','B6 E1','B1 E4','E4 B5'],
 '7':['A4 A5 A6 B5 C4 E7','B1 E1 E7','E1 E2 E7','E7','C5 E4 E7','E7']}
SHAPES = {
 '1.1':'input a: any shape; output: (shape tuple, integer ndim, integer size, dtype string)',
 '1.2':'output array shapes: (2,3), (3,1), (3,3), (10,), (101,)',
 '1.3':'output: exactly shape, with reproducible float values',
 '1.4':'input: (3,); outputs: (3,), (3,1), (1,3)',
 '1.5':'input: (3,4); outputs: (4,), (3,), (2,2), (3,)',
 '1.6':'input: any array; output: same shape; input unchanged',
 '1.7':'output: train indices (N-k,), test indices (k,); k=int(N*test_size)',
 '1.8':'input: (6,); outputs: (2,3), (6,1), (6,)',
 '1.9':'input: (5,8,3,1,2); output shape tuples: (5,8,3,2), (5,8,1,3,2), (8,5,1,3,2), (240,)',
 '1.10':'input: (2,3); output: two (3,2) arrays with different element order',
 '1.11':'input: (H,W,3); output: (3,H,W)',
 '1.12':'inputs: two (3,) arrays; outputs: (6,), (2,3), (3,2)',
 '1.13':'input: (N,d); output: (N,d+1)',
 '1.14':'input W: (r,D); output: (k*r,D)',
 '1.15':'outputs: initial points (101,2), boundary points (202,2), grid points (10201,2)',
 '2.1':'input z: any shape; output: same shape', '2.2':'input z: (...,K); output: (...,K); last-axis probabilities sum to 1',
 '2.3':'input: two shape tuples; output: shape tuple or None', '2.4':'input x: (N,); output: (N,N)',
 '2.5':'input A: (N,d); outputs: (d,), (N,), (N,1)', '2.6':'input X: (N,d); output: (N,d)',
 '2.7':'input X: (N,d); output: (N,d)', '2.8':'input V: (N,d); output: (N,d)',
 '2.9':'predict input/output: (N,); accuracy output: scalar', '2.10':'input: DataFrame; outputs: Series indexed by columns or target categories',
 '2.11':'labels: (N,), K categories; output: (N,K)', '2.12':'x: (...,in), W: (out,in), b: (out,); output: (...,out)',
 '3.1':'x,y: (d,); output: tuple of three scalars', '3.2':'x,y: (d,); output: scalar',
 '3.3':'W: (out,in), x: (in,), b: (out,); output: (out,)', '3.4':'input: two 2-D shape tuples; output: (M,P) or None',
 '3.5':'B,C: (d,d); output: two (d,d) arrays', '3.6':'X: (N,d); outputs: (d,d), (N,N)',
 '3.7':'X: (N,d), r: (N,); each function output: (d,)', '3.8':'X: (N,d), y: (N,), theta: (d,); output: (d,)',
 '3.9':'W: (N,d); output: (N,N)', '3.10':'Q: (B,H,L1,d), K: (B,H,L2,d); output: (B,H,L1,L2)',
 '3.11':'order: (N,); output: (N,N)', '3.12':'k pairs of u: (M,) and v: (N,); output: (M,N)',
 '3.13':'M: (d,d), v: (d,); diag outputs: (d,d), (d,), scalar; kill_diag output: same shape as S',
 '4.1':'x,v: (d,); outputs: e (d,), coefficient scalar, projection (d,), residual (d,)',
 '4.2':'X: (2,d); outputs: normal (d,), intercept scalar', '4.3':'A: (d,d), b: (d,); output: (d,)',
 '4.4':'four scalars; output: eigenvalue array (2,), descending', '4.5':'indices: (K,); output: (K,), dtype int32; Q: (2,2), lambdas: (2,)',
 '4.6':'A: (d,d); outputs: lam (d,), Q (d,d)', '4.7':'W: (N,d), N>=d; outputs: U (N,d), s (d,), V (d,d)',
 '4.8':'W: (N,d); outputs: Q (N,min(N,d)), lam (min(N,d),); fro_sq scalar',
 '4.9':'W: (N,d), N>=d; lowrank_error scalar; error_curve: (d,)',
 '4.10':'W: (r,D); outputs: W_DKV (r,D), W_UM (D,r)', '4.11':'x: (N,); output: scalar integer rank',
 '4.12':'A: (M,d), B: (N,d); distances: two (M,N) arrays; labels: (M,); centroids: (k,d)',
 '5.1':'x: (...,in); forward output: (...,out); weights: (out,in), bias: (out,)',
 '5.2':'output of each parameter counter: one integer', '5.3':'W: (out,in), x: (in,) or (B,in); output: same shape as x',
 '5.4':'X: (N,d), y: (N,); outputs: theta (d,), history (n_iter,)', '5.5':'X: (N,d), y: (N,); coef_: (d,), predictions: (N,), score: scalar',
 '5.6':'x: (B,L1,D), y: (B,L2,D); output: (B,L1,D1)', '5.7':'x: (B,L1,D), y: (B,L2,D); output: (B,L1,D1)',
 '5.8':'N tokens and W: (N,d); output: dictionary of N nearest-token names',
 '5.9':'vectors: (N,d), N>=d; outputs: W (N,d), S (N,N), U (N,d), s (d,), Vt (d,d), Q (N,d), lam (d,), norm scalar, curve (d,)',
 '5.10':'input: list of B uint8 (224,224,3) images; output: floating (B,3,224,224)',
 '5.11':'P: (B,2); output: integer labels (B,)', '5.12':'z: (...,K), tau>0; output: (...,K)',
 '5.13':'I,T: (B,d), log_tau scalar; output: scalar loss',
 '7.1':'same shapes as NumPy exercises 1.8, 1.9, 1.12, 2.8, 3.9, and HWC -> CHW',
 '7.2':'activations: same shape; linear x: (...,in), output: (...,out)',
 '7.3':'x,y: (N,1); outputs: trained module and loss history (epochs,)',
 '7.4':'scalar drill output: six scalars; vector drill x: (d,), outputs: two (d,) tensors',
 '7.5':'x: (B,L1,D), y: (B,L2,D); module output: (B,L1,D1)',
 '7.6':'output: token IDs (B,K), attention masks (B,K), both int64'}
ALIAS = {'A.1':'1.9','A.2':'2.4','A.3':'4.2','A.4':'3.8','A.6':'3.12','B.1':'4.5','B.2':'5.1','B.3':'7.3','B.4':'5.5','B.5':'7.5'}
SHAPES['A.5']='vectors: (300,100); W: (300,100), S: (300,300), U/Q: (300,100), s/lam: (100,), V: (100,100), rank-r factor: (300,r), error curve: (100,)'
SHAPES['B.4']='prepare_titanic returns Xtrain (Ntrain,d+1), Xtest (Ntest,d+1), ytrain (Ntrain,), ytest (Ntest,); final column is ones'
SHAPES['B.5']='MHA/GQA inputs: (B,L1,D), (B,L2,D); outputs: (B,L1,D1)'
POINTS = {'A':[10,10,10,15,45,10],'B':[20,15,15,30,20]}
INTRO = {
 '1':'''An array is a box of numbers with an address system. A (2, 3) array has two rows and three columns. `ndim` counts axes; `size` counts values. Python starts counting indices at zero. A slice `a:b` includes a but stops before b.

| Code | Meaning | Result shape for X (2,3) |
|---|---|---|
| `X[0]` | first row | (3,) |
| `X[:, 0]` | first column | (2,) |
| `X[:, :1]` | first column, keeping its axis | (2,1) |
| `X.reshape(3,2)` | regroup the existing order | (3,2) |
| `X.T` | swap row and column roles | (3,2) |

Try tiny numbers by hand before using random arrays. `reshape` and transpose can have the same output shape but different values.''',
 '2':'''Elementwise means doing the same calculation at each address. Broadcasting lets a smaller array act on many entries without making explicit copies. Compare shapes from the **right**: each pair must be equal, or one must be 1. Missing leading dimensions act like 1.

| Shapes | Result |
|---|---|
| (2,3) and (3,) | (2,3) |
| (2,3) and (2,1) | (2,3) |
| (2,3) and (2,) | error |
| (4,1) and (1,3) | (4,3) |

A reduction combines many numbers: `sum(axis=1)` adds across columns, leaving one result per row. `keepdims=True` keeps the reduced axis with length one, making later broadcasting easier.''',
 '3':'''A dot product multiplies matching entries and adds the products. Matrix multiplication repeats that calculation for every row/column pair. `(M,N) @ (N,P)` has shape `(M,P)`: the shared N is summed away. `*` never performs that sum.

Samples are usually rows of X. `X @ theta` predicts once per sample. `X.T @ r` weights all sample rows and adds them to produce one number per feature. Both can be rewritten using broadcasting and sum.''',
 '4':'''A matrix can stretch, rotate, project, or mix numbers. Linear algebra gives tools to understand or reverse these changes. `solve` finds x in Ax=b. `eigh` finds orthogonal directions for a symmetric matrix. SVD factors a rectangular matrix into directions and nonnegative strengths.

NumPy returns **Vt**, not V. `(U*s) @ Vt` reconstructs W. For S=W W.T, eigenvalues are s², and its squared Frobenius norm is the sum of s⁴. Keep the largest strengths for a compact approximation.''',
 '5':'''AI models use the same array operations in larger combinations. A linear layer changes the last feature axis. Gradient descent repeatedly updates parameters. Attention compares queries with keys, turns scores into probabilities, and mixes values. Image batches add a batch axis to channel/height/width.

Write a shape beside every intermediate array. The exercises are CPU-sized and use synthetic data; they teach operations, not production model performance.''',
 '7':'''PyTorch tensors are arrays that can also remember how they were computed. `requires_grad=True` starts a computation graph; `backward` or `autograd.grad` follows it to calculate derivatives. `detach()` disconnects a tensor from the graph; `item()` extracts one scalar. `torch.no_grad()` is useful for evaluation or copying parameters.

| NumPy | PyTorch |
|---|---|
| axis, keepdims | dim, keepdim |
| expand_dims | unsqueeze |
| transpose with axes | permute |
| concatenate | cat |
| ndarray | Tensor |

All tasks here run on CPU. You can explicitly use `tensor.to('cpu')`. `torch.from_numpy` can share memory; `torch.tensor` creates a copy.'''}

spec = (ROOT/'BUILD_SPEC.md').read_text()
metadata = {}
for line in spec.splitlines():
    m = re.match(r'\| (\d+\.\d+) (★+) \| (.*?) \| (.*)',line)
    if m:
        id,stars,title,rest = m.groups()
        parts = rest.split(' | ')
        task = parts[0]
        source_matches = re.findall(r'(?:202[456][^;|)]*)',title+' '+task)
        source = '; '.join(dict.fromkeys(source_matches)) or 'Operation practice adapted from the supplied knowledge map'
        metadata[id] = dict(title=title,stars=stars,task=task,source=source)
# Notebook 07's table does not carry stars.
for line in spec.splitlines():
    m = re.match(r'\| (7\.\d+) \| (.*?) \| (.*?) \|',line)
    if m:
        id,title,task = m.groups()
        metadata[id] = dict(title=title,stars='★★★',task=task,source='2025 Round 1/2 and 2026 sample/contest, as tagged in the supplied specification')

class Stubber(ast.NodeTransformer):
    def visit_FunctionDef(self,node):
        doc = node.body[0] if isinstance(node.body[0],ast.Expr) and isinstance(node.body[0].value,ast.Constant) and isinstance(node.body[0].value.value,str) else None
        node.body = ([doc] if doc else [])+[ast.Expr(ast.Constant('YOUR CODE HERE')),ast.Return(ast.Constant(None if node.name=='__init__' else Ellipsis))]
        return node

def decorate_code(code,shape,task):
    tree = ast.parse(code)
    for node in ast.walk(tree):
        if isinstance(node,ast.FunctionDef):
            doc = f'{node.name}: {shape}. See the exercise task for the required behavior.'
            node.body.insert(0,ast.Expr(ast.Constant(doc)))
    # Preserve comments (especially the required attention shape comments) by inserting docs line-wise.
    lines = code.splitlines()
    for node in sorted([n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef)],key=lambda n:n.lineno,reverse=True):
        indent = ' '*(node.col_offset+4)
        lines.insert(node.lineno,indent+repr(ast.get_docstring(node)))
    return '\n'.join(lines)+'\n'

def badge(name,folder='notebooks'):
    url=f'https://colab.research.google.com/github/{REPO}/blob/main/{folder}/{name}.ipynb'
    return f'[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)]({url})'

def setup(section):
    s = '''# @title Setup (run first)
!pip -q install git+https://github.com/dujing82-blip/usaaio_numpy_course.git
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from usaaio_check import grade, check_rules, show_shape, NO_LOOPS, PURE
from usaaio_check.exercise_tests import evaluate, regression_args, mlp_args
from usaaio_check.datasets import random_embeddings, blobs, titanic_like, iaio_kmeans_table
np.set_printoptions(precision=4, suppress=True)
exercise_results = {}
'''
    if section in ('7','B'):
        s += '''import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
torch.set_num_threads(1)
torch.manual_seed(2026)
'''
    if section=='B':
        s += 'from sklearn.preprocessing import StandardScaler\n'
    return s

def make(section):
    name=NAMES[section]
    nb=nbf.v4.new_notebook()
    nb.metadata.update(kernelspec=dict(display_name='Python 3',language='python',name='python3'),language_info=dict(name='python',version='3.11'),colab=dict(name=name+'.ipynb'))
    cells=[]
    title=name.split('_',1)[1].replace('_',' ').title()
    cells.append(nbf.v4.new_markdown_cell(f'# {title}\n\n'+badge(name)+'\n\n**English • CPU • synthetic data • predict shapes before coding**\n\nRun Setup, then work down the page. Edit only YOUR CODE HERE. A blank answer prints FAIL; that is normal. Complete every component until the grader says ALL PASS.'))
    cells.append(nbf.v4.new_code_cell(setup(section),metadata={'tags':['setup']}))
    if section=='0':
        cells.append(nbf.v4.new_markdown_cell('''## Your first five minutes
1. Click Open in Colab. Use **File → Save a copy in Drive** to keep your edits.
2. Choose a CPU runtime. Run Setup once; it installs the tiny grading package from this repository.
3. Read a task, write your shape prediction, replace `...`, and run its grader.
4. Open the hint only if you need it. Mistakes are part of practice.
5. After finishing, use the checklist and open the next notebook.

`np.array([[1,2,3],[4,5,6]])` is two rows of three numbers. Its shape is `(2,3)`, ndim is 2, and size is 6. NumPy uses zero-based indices. A tuple such as `(3,)` is a one-dimensional shape, not a column vector.

The checks are teaching tools, not a secure exam environment. They inspect submitted functions for common prohibited syntax. They cannot prevent deliberate use of aliases or helper functions to bypass rules. Written predictions and explanations are self-reviewed.

The first package installation needs internet. The core exercises download no datasets. If you want an offline local session, install requirements beforehand and run `pip install -e .`.'''))
        cells.append(nbf.v4.new_code_cell('X = np.array([[1,2,3],[4,5,6]])\nshow_shape(X=X, row=X[0], column=X[:,0], column_with_axis=X[:,:1])\nprint(X)'))
        cells.append(nbf.v4.new_markdown_cell('''## Four-week plan
| Week | Work |
|---|---|
| 1 | Arrays/shapes, then elementwise/broadcasting |
| 2 | Dot/matmul/transpose, then linear algebra |
| 3 | AI applications; optional PyTorch bridge |
| 4 | Two 60-minute mock exams |

The `★` tasks need one main call, `★★` tasks need shape reasoning, and `★★★` tasks combine several operations. Work in short sessions: 3–5 tasks at a time is enough.'''))
        cells.append(nbf.v4.new_markdown_cell('## Checklist\n- [ ] I can run a cell and save my notebook.\n- [ ] I know the difference between a row and a column.\n- [ ] I understand that FAIL is feedback.\n\nNext: [Arrays and shapes](01_arrays_and_shapes.ipynb).'))
    else:
        if section in POINTS:
            cells.append(nbf.v4.new_markdown_cell(f'## Timed practice — 60 minutes, 100 points\n\nThese are **adapted practice exams**, not official papers. Read [MOCK_RULES.md](MOCK_RULES.md). Start a timer, avoid reference code, and follow each task’s restrictions. Automated scores cover code; separately review your written answers and figures.\n\n'+('NumPy only; embeddings are synthetic.' if section=='A' else 'NumPy, pandas, sklearn, and CPU PyTorch. No real dataset downloads.')))
        else:
            cells.append(nbf.v4.new_markdown_cell('## Before you start\n\n'+INTRO[section]))
        if section=='7':
            cells.append(nbf.v4.new_markdown_cell('## Vocabulary pit stop\n\n`view` reshapes a contiguous tensor; `reshape` can also copy when needed. `from_numpy` can share storage. A 2-D grid is built by pairing two coordinate axes.'))
            cells.append(nbf.v4.new_code_cell("a = np.array([1.,2.,3.],dtype=np.float32)\nt = torch.from_numpy(a).float().to('cpu')\nshow_shape(t=t, column=t.view(-1,1))\nu = torch.linspace(0,1,3)\ng0,g1 = torch.meshgrid(u,u,indexing='ij')\nshow_shape(grid0=g0, grid1=g1)\nprint(t.detach().numpy())"))
        ids=[id for id in EXERCISES if id.startswith(section+'.')]
        for index,id in enumerate(ids):
            entry=EXERCISES[id]
            base=ALIAS.get(id,id)
            meta=metadata.get(base,dict(title='Embedding analysis exam',stars='★★★',task='Implement all listed stages of the embedding analysis pipeline.',source='2026 Round 1 Problem 5, adapted with synthetic embeddings'))
            shape=SHAPES.get(id,SHAPES.get(base,''))
            topics=TOPICS.get(base.split('.')[0],['E5'])
            topic=topics[int(base.split('.')[1])-1] if base in metadata else 'E5 D4 D5'
            tags=' '.join(f'[{x}]' for x in topic.split())
            restrictions=[]
            for target,rules in entry['rules'].items():
                for k,v in rules.items():
                    if v is True: restrictions.append(f'{target}: '+k.replace('_','-'))
                    elif k=='max_loops': restrictions.append(f'{target}: at most {v} explicit iteration loop')
            rule_text='; '.join(restrictions) or 'Only the starter imports; loops allowed unless a subtask says otherwise.'
            pts=f' — {POINTS[section][index]} points' if section in POINTS else ''
            prediction=entry['prediction'] or f'Before running, write the input and output shapes for each function. Contract: {shape}.'
            task=meta['task']
            if id=='A.5': task='Implement vectors_array, unit_rows, similarity_range (theoretical bounds), sim_matrix, diagonal_of_similarity, nearest_dict, svd_parts, spectral_from_svd, fro_sq, lowrank_factor, and error_curve. Use 300 synthetic 100-dimensional embeddings.'
            if id=='B.4': task += '\n\nAlso implement prepare_titanic: drop missing rows, one-hot encode features, seeded manual split, StandardScaler fit on training rows, then append ones. Fit My_Log_Reg and evaluate held-out accuracy.'
            if id=='B.5': task='Implement MyMHA and MyGQA CPU modules with trainable projection matrices; match NumPy outputs, and verify equality at G == H.'
            text=f'### Exercise {id} — {meta["title"]} {meta["stars"]}{pts}\n\n{tags}\n\n**Source:** {meta["source"]}. Adapted exercise; source tags supplied with the course specification.\n\n**Task:** {task}\n\n**Shape contract:** {shape}\n\n**Rules:** {rule_text}\n\n**Predict first:** {prediction}'
            if entry['note']: text+='\n\n'+entry['note']
            cells.append(nbf.v4.new_markdown_cell(text,metadata={'exercise_id':id}))
            cells.append(nbf.v4.new_markdown_cell('**My prediction:** _Write your answer here before coding._',metadata={'tags':['prediction'],'exercise_id':id}))
            code=decorate_code(entry['code'],shape,task)
            cells.append(nbf.v4.new_code_cell(code,metadata={'tags':['solution'],'exercise_id':id}))
            cells.append(nbf.v4.new_code_cell(f'# @title Grade {id} (run, do not edit)\nexercise_results[{id!r}] = evaluate(globals(), {id!r})',metadata={'tags':['grader'],'exercise_id':id}))
            hint=entry['hint'] or 'Write the shapes first, then combine the operations introduced in this section.'
            cells.append(nbf.v4.new_markdown_cell(f'**Why it matters:** This operation is a building block used by the AI tasks tagged above.\n\n<details><summary>Hint 1: operation</summary>\n\n{hint}\n\n</details>\n\n<details><summary>Hint 2: shape trick</summary>\n\nCheck which axes survive. A missing axis or an extra length-one axis changes broadcasting. Verify with a tiny hand-checkable example.\n\n</details>'))
            if entry['plot']:
                cells.append(nbf.v4.new_code_cell(f'# @title Plot after passing\nif exercise_results.get({id!r}, False):\n'+''.join('    '+l+'\n' for l in entry['plot'].splitlines()),metadata={'tags':['plot'],'exercise_id':id}))
            if id=='4.10':
                cells.append(nbf.v4.new_code_cell("if exercise_results.get('4.10',False):\n    W = np.random.default_rng(2026).normal(size=(4,24))\n    down,up = GQA_2_MLA(W)\n    show_shape(W_DKV=down,W_UM=up)\n    print(f'MSE: {np.mean((up@down-np.tile(W,(6,1)))**2):.4f}')"))
            if id=='5.1':
                cells.append(nbf.v4.new_code_cell("if exercise_results.get('5.1',False):\n    model = My_Linear_NumPy(3,2)\n    show_shape(vector=model.forward(np.zeros(3)),batch=model.forward(np.zeros((10,3))),sequence=model.forward(np.zeros((10,20,3))),extra=model.forward(np.zeros((10,20,30,3))))"))
        if section in POINTS:
            cells.append(nbf.v4.new_code_cell(f'points = {dict(zip(ids,POINTS[section]))!r}\nscore = sum(points[k] for k in points if exercise_results.get(k,False))\nprint(f"Code score: {{score}}/100 (all components of a problem must pass)")'))
        learned=sorted(set(x for topic in TOPICS.get(section,['E5 E7']) for x in topic.split()))
        next_section={'1':'2','2':'3','3':'4','4':'5','5':'7','7':'A','A':'B'}.get(section,'0')
        cells.append(nbf.v4.new_markdown_cell('## Checklist\n'+ '\n'.join('- [ ] I can explain and use '+t+' (see the README knowledge map).' for t in learned)+'\n- [ ] Every exercise says ALL PASS.\n- [ ] I reviewed my shape predictions and can explain my mistakes.\n\nNext: ['+NAMES[next_section].replace('_',' ')+']('+NAMES[next_section]+'.ipynb).'))
    nb.cells=cells
    nbf.write(nb,ROOT/'solutions'/f'{name}.ipynb')
    student=copy.deepcopy(nb)
    for cell in student.cells:
        if 'solution' in cell.metadata.get('tags',[]):
            tree=Stubber().visit(ast.parse(cell.source))
            cell.source=ast.unparse(ast.fix_missing_locations(tree)).replace("'YOUR CODE HERE'",'### YOUR CODE HERE ###')+'\n'
    nbf.write(student,ROOT/'notebooks'/f'{name}.ipynb')

if __name__=='__main__':
    for section in NAMES:
        make(section)
    (ROOT/'exercise_manifest.json').write_text(json.dumps({id:dict(title=metadata.get(ALIAS.get(id,id),{}).get('title','Embedding analysis'),targets=[t[0] for t in e['tests']],cases=sum(len(t[1]) for t in e['tests']),rules=e['rules']) for id,e in EXERCISES.items()},indent=2)+'\n')
    print(f'Generated {len(NAMES)} student + {len(NAMES)} solution notebooks, {len(EXERCISES)} exercises.')
