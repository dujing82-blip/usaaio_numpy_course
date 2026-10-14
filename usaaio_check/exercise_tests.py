"""Seeded, visible self-study test cases; no dataset downloads."""
import contextlib
import io
import linecache
import numpy as np
import pandas as pd
from .checker import grade, check_rules, _equal
from .catalog import EXERCISES
from .datasets import random_embeddings, blobs, titanic_like, iaio_kmeans_table


def normalized(V):
    return V/np.linalg.norm(V,axis=1,keepdims=True)
def spd(rng,d):
    M = rng.normal(size=(d,d))
    return M.T@M+np.eye(d)
def tiny_bank(seed=0,N=4):
    if N==4:
        return pd.DataFrame({'marital':['married','single','married','unknown'], 'age':[31,20,42,27], 'job':['unknown','teacher','doctor','unknown'], 'y':['yes','no','no','yes']})
    rng = np.random.default_rng(seed)
    return pd.DataFrame({'marital':rng.choice(['married','single','unknown'],N), 'age':rng.integers(18,70,N), 'job':rng.choice(['unknown','teacher'],N),'y':rng.choice(['yes','no'],N)})
def regression_args(rng):
    X = rng.normal(size=(80,3))
    y = X@np.array([1.,-2.,0.5])
    return X,y,0.1,150
def attention_args(seed,B,L1,L2,D,H,d):
    rng = np.random.default_rng(seed)
    return (rng.normal(size=(B,L1,D)),rng.normal(size=(B,L2,D)),
            rng.normal(size=(H*d,D))*0.2,rng.normal(size=(H*d,D))*0.2,
            rng.normal(size=(H*d,D))*0.2,rng.normal(size=(D,H*d))*0.2,H)
def gqa_args(seed,H,G):
    x,y,WQ,WK,WV,WO,H = attention_args(seed,2,3,4,6,H,2)
    return x,y,WQ,WK[:G*2],WV[:G*2],WO,H,G
def embedding_args(N,d,seed):
    tokens,V = random_embeddings(N,d,seed)
    return tokens,normalized(V)
def mlp_args(seed,epochs):
    import torch
    g = torch.Generator().manual_seed(seed)
    x = torch.linspace(-1,1,80).unsqueeze(1)
    y = torch.sin(2*np.pi*x)+0.03*torch.randn(80,1,generator=g)
    return x,y,epochs

def spectral_probe(ns,args):
    A, = args
    lam,Q = ns['spectral'](A)
    assert lam.shape==(len(A),) and Q.shape==A.shape
    _equal(Q.T@Q,np.eye(len(A)))
    return Q@np.diag(lam)@Q.T

def svd_probe(ns,args):
    W, = args
    U,s,V = ns['svd_parts'](W)
    n,d = W.shape
    assert (U.shape,s.shape,V.shape)==((n,d),(d,),(d,d))
    assert np.all(s[:-1]>=s[1:]) and np.all(s>=0)
    _equal(U.T@U,np.eye(d)); _equal(V.T@V,np.eye(d))
    return (U*s)@V.T

def gram_spectral_probe(ns,args):
    W, = args
    Q,lam = ns['spectral_from_svd'](W)
    assert Q.shape==(W.shape[0],min(W.shape)) and lam.shape==(min(W.shape),)
    return (Q*lam)@Q.T

def factor_probe(ns,args):
    W, = args
    down,up = ns['GQA_2_MLA'](W)
    r,D = W.shape
    assert down.shape==(r,D) and up.shape==(D,r)
    result = up@down
    target = np.tile(W,(D//r,1))
    _equal(result,target)
    return result

def linear_probe(ns,args):
    model = ns['My_Linear_NumPy'](3,2)
    assert model.in_features==3 and model.out_features==2
    assert model.weight.shape==(2,3) and model.bias.shape==(2,)
    got = model.forward(args[0])
    _equal(got,args[0]@model.weight.T+model.bias)
    return got

def logistic_probe(ns,args):
    seed,solver = args
    X,y = blobs(240,seed)
    # Label noise prevents complete separation and a singular Newton Hessian.
    y[::13] = 1-y[::13]
    X = np.column_stack([X,np.ones(len(X))])
    model = ns['My_Log_Reg'](solver=solver,lr=0.002,num_iter=20)
    model.fit(X,y)
    assert model.coef_.shape==(3,)
    score = model.score(X,y)
    assert score>=0.9, 'accuracy should reach 0.9'
    _equal(score,np.mean(model.predict(X)==y))
    try:
        ns['My_Log_Reg'](solver='invalid')
    except ValueError as e:
        assert str(e)=='Invalid solver'
    else:
        raise AssertionError('invalid solver must raise ValueError')
    return model.predict(X),score

def compression_probe(ns,args):
    V, = args
    W,S,U,s,Vt,Q,lam,norm_sq,curve = ns['compress_embeddings'](V)
    _equal(np.sum(W**2,1),np.ones(len(W)))
    _equal(S,W@W.T)
    _equal((U*s)@Vt,W)
    _equal(Q,U); _equal(lam,s**2)
    _equal((Q*lam)@Q.T,S)
    _equal(norm_sq,np.sum(S**2))
    assert curve.shape==(min(W.shape),) and np.all(np.diff(curve)<=1e-8)
    _equal(curve[-1],0.)
    return W,S,norm_sq,curve

def torch_linear_probe(ns,args):
    import torch
    i,o,bias = args
    torch.manual_seed(42)
    model = ns['My_Linear'](i,o,bias=bias)
    assert isinstance(model,torch.nn.Module)
    assert isinstance(model.weight,torch.nn.Parameter)
    assert model.weight.shape==(o,i)
    if bias:
        assert isinstance(model.bias,torch.nn.Parameter)
        _equal(model.bias,torch.zeros(o))
    else:
        assert model.bias is None
    x = torch.randn(1000,i)
    out = model(x)
    expected = x@model.weight.T
    if bias:
        expected = expected+model.bias
    _equal(out,expected,atol=1e-5)
    if i==64:
        assert 0.85<float(out.detach().var())<1.15, 'unit output variance initialization'
    return tuple(out.shape),len(list(model.parameters())),True

def mlp_probe(ns,args):
    import torch
    x,y,epochs = args
    torch.manual_seed(42)
    with contextlib.redirect_stdout(io.StringIO()) as printed:
        model,history = ns['train_mlp'](x,y,epochs)
    assert isinstance(model,torch.nn.Module)
    assert isinstance(model.layers,torch.nn.Sequential)
    assert history.shape==(epochs,) and np.isfinite(history).all()
    assert history[-1]<history[0]*0.9, 'training loss should decrease'
    import re
    lines = printed.getvalue().strip().splitlines()
    assert len(lines)==epochs//10 and all(re.search(r'\d+\.\d{4}$',line) for line in lines), 'print every 10 epochs to four decimals'
    with torch.no_grad():
        predictions = model(x)
    assert predictions.shape==y.shape
    return True,tuple(predictions.shape)

def torch_attention_probe(ns,args):
    import torch
    seed,B,L1,L2,D,H,d = args
    vals = attention_args(seed,B,L1,L2,D,H,d)
    x,y,WQ,WK,WV,WO,H = vals
    model = ns['MyMHA'](D,D,H,d).double()
    with torch.no_grad():
        for name,W in zip(['WQ','WK','WV','WO'],[WQ,WK,WV,WO]):
            getattr(model,name).copy_(torch.from_numpy(W))
    got = model(torch.from_numpy(x),torch.from_numpy(y))
    reference = reference_namespace('5.6')['mha_numpy'](*vals)
    assert isinstance(got,torch.Tensor) and got.shape==reference.shape
    relative = np.linalg.norm(got.detach().numpy()-reference)/max(np.linalg.norm(reference),1e-12)
    assert relative<1e-5, 'NumPy/PyTorch attention mismatch'
    return got.detach().numpy()

def loader_probe(ns,args):
    import torch
    sequences, = args
    ds = ns['TokenDataset'](sequences)
    assert len(ds)==len(sequences)
    loader = torch.utils.data.DataLoader(ds,batch_size=len(ds),shuffle=False,collate_fn=ns['collate_fn'])
    tokens,mask = next(iter(loader))
    assert tokens.dtype==mask.dtype==torch.int64
    expected = np.zeros((len(sequences),max(map(len,sequences))),dtype=np.int64)
    expected_mask = expected.copy()
    for i,s in enumerate(sequences):
        expected[i,:len(s)] = s
        expected_mask[i,:len(s)] = 1
    _equal(tokens.detach().numpy(),expected)
    _equal(mask.detach().numpy(),expected_mask)
    return tokens,mask

def resolve(ns,name):
    parts = name.split('.')
    value = ns[parts[0]]
    for p in parts[1:]:
        value = getattr(value,p)
    return value

def reference_namespace(id):
    from sklearn.preprocessing import StandardScaler
    ns = {'np':np,'pd':pd,'StandardScaler':StandardScaler}
    import matplotlib.pyplot as plt
    ns['plt'] = plt
    if id.startswith('7.') or id.startswith('B.'):
        import torch
        ns.update(torch=torch,nn=torch.nn,Dataset=torch.utils.data.Dataset,DataLoader=torch.utils.data.DataLoader)
    code = EXERCISES[id]['code']
    filename = f'<reference-{id}>'
    linecache.cache[filename] = (len(code),None,code.splitlines(True),filename)
    exec(compile(code,filename,'exec'),ns)
    return ns

def evaluate(namespace,id,quiet=False):
    """Return True only if every component and every case passes."""
    entry = EXERCISES[id]
    results = []
    ref = reference_namespace(id)
    for name,expressions,*probe_names in entry['tests']:
        try:
            candidate = resolve(namespace,name)
            targets = {name:entry['rules'].get(name,{})}
            # Also check constructor and other explicitly listed methods in this class.
            prefix = name.split('.')[0]+'.'
            for target,rules in entry['rules'].items():
                if target.startswith(prefix):
                    targets[target] = rules
            for target,rules in targets.items():
                bad = check_rules(resolve(namespace,target),**rules)
                if bad:
                    print(f"RULE VIOLATION in {target}: {', '.join(bad)} -> 0 points on the real exam")
                    raise AssertionError('rule check failed')
            env = dict(globals(),rng=np.random.default_rng(2026))
            if id.startswith('7.') or id.startswith('B.'):
                import torch
                env['torch'] = torch
                torch.manual_seed(2026)
            cases = []
            for expression in expressions:
                args = eval(expression,env)
                if probe_names:
                    probe = eval(probe_names[0],globals())
                    expected = probe(ref,args)
                else:
                    expected = resolve(ref,name)(*args)
                cases.append((args,expected))
            if probe_names:
                def wrapped(*args):
                    return probe(namespace,args)
                # Static rules above target the original function, not this adapter.
                passed = grade(wrapped,cases,immutable=id in ('1.6','3.13'))
            else:
                passed = grade(candidate,cases,rules=entry['rules'].get(name,{}),immutable=id in ('1.6','3.13'))
            results.append(passed)
        except Exception as e:
            print(f'  {name}: FAIL - {type(e).__name__}: {e}')
            results.append(False)
    result = bool(results) and all(results)
    if not quiet:
        print(f'Exercise {id}: '+('ALL PASS' if result else 'not yet'))
    return result

def lowrank_probe(ns,args):
    W,r = args
    factor = ns['lowrank_factor'](W,r)
    assert factor.shape==(len(W),r)
    U,s,Vt = np.linalg.svd(W,full_matrices=False)
    target = U[:,:r]*s[:r]
    _equal(factor@factor.T,target@target.T)
    return factor@factor.T

def module_shape_probe(ns,args):
    import torch
    torch.manual_seed(2026)
    model = ns['My_MLP_Model']()
    assert isinstance(model,torch.nn.Module) and isinstance(model.layers,torch.nn.Sequential)
    result = model(args[0])
    assert result.shape==(len(args[0]),1)
    result.sum().backward()
    assert all(p.grad is not None for p in model.parameters())
    return tuple(result.shape),True

def titanic_probe(ns,args):
    N,seed = args
    Xtr,Xte,ytr,yte = ns['prepare_titanic'](titanic_like(N,seed),0.2,seed)
    model = ns['My_Log_Reg'](solver='GD',lr=0.001,num_iter=20)
    model.fit(Xtr,ytr)
    assert model.coef_.shape==(Xtr.shape[1],)
    predictions = model.predict(Xte)
    assert predictions.shape==yte.shape and np.isin(predictions,[0,1]).all()
    _equal(model.score(Xte,yte),np.mean(predictions==yte))
    return predictions,model.score(Xte,yte)

def gqa_torch_probe(ns,args):
    import torch
    seed,H,G = args
    vals = gqa_args(seed,H,G)
    x,y,WQ,WK,WV,WO,H,G = vals
    model = ns['MyGQA'](6,6,H,G,2).double()
    with torch.no_grad():
        for name,W in zip(['WQ','WK','WV','WO'],[WQ,WK,WV,WO]):
            getattr(model,name).copy_(torch.from_numpy(W))
    got = model(torch.from_numpy(x),torch.from_numpy(y))
    target = reference_namespace('5.7')['gqa_numpy'](*vals)
    _equal(got.detach().numpy(),target)
    if G==H:
        mha = ns['MyMHA'](6,6,H,2).double()
        mha.load_state_dict(model.state_dict())
        _equal(mha(torch.from_numpy(x),torch.from_numpy(y)),got,atol=1e-5)
    return got.detach().numpy()
