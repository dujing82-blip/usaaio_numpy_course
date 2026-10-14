"""Instructor reference definitions and test recipes. Student cells never contain answers.

This is an open self-study course: reference code is inspectable, not secret.
"""
EXERCISES = {}

def add(id, code, tests, *, rules=None, prediction=None, hint=None, note='', plot=None):
    EXERCISES[id] = dict(id=id, code=code.strip()+'\n', tests=tests,
        rules=rules or {}, prediction=prediction, hint=hint, note=note, plot=plot)

NL = dict(no_loops=True)
PURE = dict(no_loops=True, no_linalg=True, no_matmul=True, no_transpose=True)
# A test is (function or method to check, input-expression list, optional probe).
# Probes inspect candidate behavior when a direct array comparison is inappropriate.
add('1.1', '''
def describe(a):
    return a.shape, a.ndim, a.size, str(a.dtype)
''', [('describe', ['(np.array([[1,2,3],[4,5,6]], dtype=np.int32),)', '(rng.normal(size=(2,3,4)),)'])], prediction='What are shape, ndim, and size for a (2, 3) array?', hint='Read the four array attributes; a shape is a tuple.')
add('1.2', '''
def build():
    return np.zeros((2,3)), np.ones((3,1)), np.eye(3), np.arange(0,1,0.1), np.linspace(0,1,101)
''', [('build', ['()', '()'])], prediction='How many values does arange(0, 1, 0.1) make? Does it include 1?', hint='Use zeros, ones, eye, arange, and linspace.', note='This fixed-output recipe is checked twice rather than with a random input.')
add('1.3', '''
def seeded(seed, shape):
    np.random.seed(seed)
    return np.random.randn(*shape)
''', [('seeded', ['(4,(2,3))', '(2026,(3,2,4))'])], hint='Set the legacy random seed before calling randn; *shape unpacks dimensions.')
add('1.4', '''
def three_looks(v):
    return v, v.reshape(3,1), v.reshape(1,3)
''', [('three_looks', ['(np.array([1,2,3]),)', '(rng.normal(size=3),)'])], prediction='How does (3,) differ from (3, 1)?', hint='reshape changes grouping, not the order of values.')
add('1.5', '''
def pick(X):
    return X[1], X[:,2], X[0:2,1:3], X[:,-1]
''', [('pick', ['(np.arange(12).reshape(3,4),)', '(rng.normal(size=(3,4)),)'])], hint='The colon selects every index; -1 selects the last index.')
add('1.6', '''
def cap(arr, t, value):
    out = arr.copy()
    out[out > t] = value
    return out
''', [('cap', ['(np.array([0,2,4]),1,100)', '(rng.normal(size=(3,4)),0,9)'])], rules={'cap':NL}, hint='Copy first, then use a boolean mask.', note='The grader also verifies that the input is unchanged.')
add('1.7', '''
def split_indices(N, test_size, seed):
    np.random.seed(seed)
    idx = np.random.permutation(N)
    k = int(N*test_size)
    np.random.seed()
    return idx[k:], idx[:k]
''', [('split_indices', ['(10,0.3,42)', '(37,0.2,2026)'])], hint='Permute indices, take the first k for testing, and reset the seed.')
add('1.8', '''
def regroup(a):
    return a.reshape(2,3), a.reshape(-1,1), a.reshape(2,3).flatten()
''', [('regroup', ['(np.arange(6),)', '(rng.normal(size=6),)'])], hint='-1 asks NumPy to infer one dimension.')
add('1.9', '''
def shape_walk(arr):
    a = np.squeeze(arr)
    b = np.expand_dims(a, axis=2)
    c = np.swapaxes(b,0,1)
    return a.shape, b.shape, c.shape, c.flatten().shape
''', [('shape_walk', ['(np.zeros((5,8,3,1,2)),)', '(rng.normal(size=(2,4,3,1,2)),)'])], prediction='Which dimension disappears under squeeze? Which is inserted at axis 2?', hint='Apply each operation to the result of the previous step.')
add('1.10', '''
def which_moved(M):
    return M.reshape(3,2), M.T
''', [('which_moved', ['(np.arange(6).reshape(2,3),)', '(rng.normal(size=(2,3)),)'])], hint='reshape preserves flattened order; transpose swaps row and column roles.', note='For arange(6), mark the entries that differ between the two outputs.')
add('1.11', '''
def to_chw(img):
    return np.transpose(img,(2,0,1))
''', [('to_chw', ['(np.arange(18).reshape(2,3,3),)', '(rng.integers(0,256,size=(4,5,3),dtype=np.uint8),)'])], rules={'to_chw':NL}, prediction='Where does img[i, j, c] go in the output?', hint='Transpose axes using their old positions: channel, height, width.')
add('1.12', '''
def join(a,b):
    return np.concatenate([a,b]), np.stack([a,b]), np.stack([a,b],axis=1)
''', [('join', ['(np.array([1,2,3]),np.array([4,5,6]))', '(rng.normal(size=3),rng.normal(size=3))'])], hint='concatenate extends an axis; stack creates an axis.')
add('1.13', '''
def add_ones(X):
    return np.concatenate([X,np.ones((X.shape[0],1))],axis=1)
''', [('add_ones', ['(np.arange(6).reshape(2,3),)', '(rng.normal(size=(5,2)),)'])], rules={'add_ones':NL}, hint='Match the row count and concatenate along columns.')
add('1.14', '''
def tile_rows(W,k):
    return np.concatenate([W]*k,axis=0)
''', [('tile_rows', ['(np.arange(6).reshape(2,3),3)', '(rng.normal(size=(4,5)),2)'])], rules={'tile_rows':NL}, hint='A repeated list of arrays can be passed to concatenate.')
add('1.15', '''
def ic_points():
    x = np.linspace(0,1,101)
    return np.stack([np.zeros_like(x),x],axis=1)
def bc_points():
    t = np.linspace(0,1,101)
    left = np.stack([t,np.zeros_like(t)],axis=1)
    right = np.stack([t,np.ones_like(t)],axis=1)
    return np.concatenate([left,right],axis=0)
def grid_points():
    t,x = np.meshgrid(np.linspace(0,1,101),np.linspace(0,1,101),indexing='ij')
    return np.stack([t.reshape(-1),x.reshape(-1)],axis=1)
''', [(n,['()','()']) for n in ['ic_points','bc_points','grid_points']], rules={n:NL for n in ['ic_points','bc_points','grid_points']}, prediction='Why does a 101 by 101 grid become (10201, 2)?', hint='Each row is (t, x). Use meshgrid with indexing="ij".', note='Grid order: t is the outer index, x the inner index. Fixed grids are checked twice.')
add('2.1', '''
def sigmoid(z):
    return 1/(1+np.exp(-z))
def relu(z):
    return np.maximum(0,z)
def tanh_(z):
    return np.tanh(z)
''', [(n,['(np.array([-2.,0.,2.]),)','(rng.normal(size=(2,3)),)']) for n in ['sigmoid','relu','tanh_']], rules={n:NL for n in ['sigmoid','relu','tanh_']}, hint='Apply each function to every entry; the shape stays the same.')
add('2.2', '''
def softmax(z):
    e = np.exp(z-z.max(axis=-1,keepdims=True))
    return e/e.sum(axis=-1,keepdims=True)
''', [('softmax',['(np.array([[1000.,1001.,1002.],[1.,1.,1.]]),)','(rng.normal(size=(2,3,4)),)'])], rules={'softmax':NL}, prediction='What shape must the denominator have to divide each row?', hint='Subtract the row maximum; keep the reduced axis as length one.')
add('2.3', '''
def broadcast_result(shape_a,shape_b):
    out = []
    a,b = list(shape_a),list(shape_b)
    while a or b:
        x = a.pop() if a else 1
        y = b.pop() if b else 1
        if x != y and x != 1 and y != 1:
            return None
        out.append(y if x == 1 else x)
    return tuple(reversed(out))
''', [('broadcast_result',['((2,3),(3,))','((2,3),(2,))','((2,3),(2,1))','((4,1),(1,3))','((2,3,4,5),(4,1))','((0,3),(1,3))','((),(2,3))'])], rules={'broadcast_result':{'no_numpy':True}}, hint='Compare dimensions from the right. Equal or length-one dimensions are compatible.', note='Loops are allowed here because you are implementing the shape rule, not a numerical array calculation.')
add('2.4', '''
def kernel_matrix(x):
    P = x[:,None]*x[None,:]
    return (1+P+2*P**2)**2
''', [('kernel_matrix',['(np.array([-1.,0.,1.]),)','(rng.normal(size=6),)'])], rules={'kernel_matrix':dict(no_loops=True,no_linalg=True)}, hint='Turn x into a column and a row to build all pairs.')
add('2.5', '''
def sums(A):
    return A.sum(0), A.sum(1), A.sum(1,keepdims=True)
''', [('sums',['(np.arange(6).reshape(2,3),)','(rng.normal(size=(4,5)),)'])], rules={'sums':NL}, hint='axis=0 removes rows; axis=1 removes columns.')
add('2.6', '''
def minmax(X):
    return (X-X.min(0))/(X.max(0)-X.min(0))
''', [('minmax',['(np.array([[1.,4.],[3.,8.],[5.,6.]]),)','(rng.normal(size=(7,3)),)'])], rules={'minmax':NL}, hint='Find one minimum and maximum for each column.', note='Assume each column has nonzero range.')
add('2.7', '''
def standardize(X):
    return (X-X.mean(0))/X.std(0)
''', [('standardize',['(np.array([[1.,4.],[3.,8.],[5.,6.]]),)','(rng.normal(size=(7,3)),)'])], rules={'standardize':NL}, hint='Use the population standard deviation (ddof=0).', note='Assume each column has nonzero standard deviation.')
add('2.8', '''
def unit_rows(V):
    return V/np.sqrt(np.sum(V**2,axis=1,keepdims=True))
''', [('unit_rows',['(np.array([[3.,4.],[0.,2.]]),)','(rng.normal(size=(5,3)),)'])], rules={'unit_rows':dict(no_loops=True,no_linalg=True)}, prediction='What is the norm array shape with and without keepdims?', hint='Square, sum across columns, take a square root; keep one column.', note='Assume every row is nonzero.')
add('2.9', '''
def predict(p):
    return np.where(p>=0.5,1,0)
def accuracy(y_pred,y):
    return np.mean(y_pred==y)
''', [('predict',['(np.array([0.2,0.5,0.9]),)','(rng.random(8),)']),('accuracy',['(np.array([0,1,1]),np.array([0,0,1]))','(rng.integers(0,2,8),rng.integers(0,2,8))'])], rules={'predict':NL,'accuracy':NL}, hint='Comparisons produce booleans; their mean is a fraction.')
add('2.10', '''
def count_unknown(df):
    return df.isin(['unknown']).sum()
def odd_married_counts(df):
    return df.loc[(df.marital=='married') & (df.age%2==1),'y'].value_counts()
''', [(n,['(tiny_bank(),)','(tiny_bank(seed=2026,N=20),)']) for n in ['count_unknown','odd_married_counts']], hint='Use parentheses around both comparisons before &.')
add('2.11', '''
def one_hot(labels,categories):
    return (labels[:,None]==np.array(categories)[None,:]).astype(int)
''', [('one_hot',['(np.array(["cat","dog","cat"]),["cat","dog"])','(rng.integers(0,4,9),[0,1,2,3])'])], rules={'one_hot':NL}, hint='Compare an (N, 1) label column with a (1, K) category row.', note='All labels are present in categories, with no duplicate categories.')
add('2.12', '''
def affine_bc(x,W,b):
    return np.sum(x[...,None]*W.T,axis=-2)+b
''', [('affine_bc',['(np.array([1.,2.,3.]),np.arange(6).reshape(2,3),np.array([1.,-1.]))']+[f'(rng.normal(size={s}),rng.normal(size=(2,3)),rng.normal(size=2))' for s in ['(10,3)','(10,20,3)','(10,20,30,3)']])], rules={'affine_bc':dict(no_loops=True,no_matmul=True)}, prediction='What shape is x[..., None], and which axis must be summed?', hint='Insert an output-feature axis; sum the input-feature axis.')
add('3.1', '''
def dot3(x,y):
    return np.dot(x,y),x@y,np.sum(x*y)
''', [('dot3',['(np.array([1,2,3]),np.array([4,5,6]))','(rng.normal(size=5),rng.normal(size=5))'])], hint='A vector dot product returns one number.')
add('3.2', '''
def cosine(x,y):
    return x@y/np.sqrt((x@x)*(y@y))
''', [('cosine',['(np.array([1.,0.]),np.array([0.,1.]))','(rng.normal(size=4),rng.normal(size=4))'])], rules={'cosine':{'no_linalg':True}}, hint='Divide the dot product by both lengths.', note='Both vectors are nonzero.')
add('3.3', '''
def affine(W,x,b):
    return W@x+b
''', [('affine',['(np.array([[1,0,0,0,0],[0,1,0,0,0],[0,0,1,1,1]]),np.array([1,2,3,4,5]),np.array([0,1,8]))','(rng.normal(size=(3,5)),rng.normal(size=5),rng.normal(size=3))'])], hint='Each row of W produces one output.', note='The hand-checkable example yields [1, 3, 20]; these are teaching numbers, not a claim to reproduce the original exam matrix.')
add('3.4', '''
def matmul_shape(sa,sb):
    if len(sa)!=2 or len(sb)!=2 or sa[1]!=sb[0]:
        return None
    return sa[0],sb[1]
''', [('matmul_shape',['((2,3),(3,4))','((2,3),(2,4))','((5,2),(2,7))'])], hint='The inside dimensions must match; the outside dimensions survive.')
add('3.5', '''
def both(B,C):
    return B@C,B*C
''', [('both',['(np.array([[1,2],[3,4]]),np.array([[0,1],[1,0]]))','(rng.normal(size=(3,3)),rng.normal(size=(3,3)))'])], hint='@ combines rows and columns; * multiplies matching entries.')
add('3.6', '''
def grams(X):
    return X.T@X,X@X.T
''', [('grams',['(np.arange(6).reshape(2,3),)','(rng.normal(size=(5,3)),)'])], prediction='Which Gram matrix compares samples, and which compares features?', hint='Write the shapes before multiplying.')
add('3.7', '''
def weighted_rows(X,r):
    return X.T@r
def weighted_rows_pure(X,r):
    return np.sum(r[:,None]*X,axis=0)
''', [(n,['(np.arange(6).reshape(2,3),np.array([1.,2.]))','(rng.normal(size=(5,3)),rng.normal(size=5))']) for n in ['weighted_rows','weighted_rows_pure']], rules={'weighted_rows_pure':PURE}, hint='Multiply each row by one weight, then sum rows.', note='Return the two versions from separate functions so only the pure version is restricted.')
add('3.8', '''
def my_fun_derivative(X,y,theta):
    residual = y-np.sum(X*theta,axis=1)
    return -2/X.shape[0]*np.sum(residual[:,None]*X,axis=0)
''', [('my_fun_derivative',['(np.array([[1.,2.],[3.,4.]]),np.array([2.,5.]),np.array([0.1,0.2]))','(rng.normal(size=(7,3)),rng.normal(size=7),rng.normal(size=3))'])], rules={'my_fun_derivative':PURE}, prediction='The residual is (N,). Why insert a length-one column before multiplying X?', hint='Replace both matrix products with multiply-and-sum operations.')
add('3.9', '''
def sim_matrix(W):
    return np.sum(W[:,None,:]*W[None,:,:],axis=-1)
''', [('sim_matrix',['(np.array([[1.,0.],[0.,1.]]),)','(normalized(rng.normal(size=(5,3))),)'])], rules={'sim_matrix':PURE}, hint='Make (N, 1, d) and (1, N, d); sum over d.')
add('3.10', '''
def attn_scores(Q,K):
    return Q@np.swapaxes(K,-2,-1)/np.sqrt(Q.shape[-1])
''', [('attn_scores',['(np.arange(12).reshape(1,2,3,2).astype(float),np.ones((1,2,4,2)))','(rng.normal(size=(2,3,4,5)),rng.normal(size=(2,3,6,5)))'])], rules={'attn_scores':NL}, hint='Swap only the last two axes of K; keep batch and head axes.')
add('3.11', '''
def perm_matrix(order):
    return np.eye(len(order))[order]
''', [('perm_matrix',['(np.array([2,3,1,0]),)','(rng.permutation(6),)'])], rules={'perm_matrix':NL}, hint='Reorder the rows of an identity matrix.')
add('3.12', '''
def outer_sum(us,vs):
    return np.stack(us).T@np.stack(vs)
''', [('outer_sum',['(list(np.eye(3)),list(np.array([[0,0,1,-1],[0,0,0,1],[1,2,0,0]])))','(list(rng.normal(size=(4,3))),list(rng.normal(size=(4,5))))'])], hint='Stack vector pairs as rows, then transpose the left stack.')
add('3.13', '''
def diag_tools(M,v):
    return np.diag(v),np.diag(M),np.trace(M)
def kill_diag(S):
    out = S.copy()
    np.fill_diagonal(out,-np.inf)
    return out
''', [('diag_tools',['(np.array([[1.,2.],[3.,4.]]),np.array([5.,6.]))','(rng.normal(size=(4,4)),rng.normal(size=4))']),('kill_diag',['(np.ones((2,2)),)','(rng.normal(size=(4,4)),)'])], hint='diag turns vectors into matrices and matrices into vectors.', note='kill_diag must preserve the input; assume S is floating point.')
add('4.1', '''
def project(x,v):
    e = v/np.linalg.norm(v)
    c = x@e
    proj = c*e
    return e,c,proj,x-proj
''', [('project',['(np.array([-1.,-4.,6.]),np.array([1.,-2.,3.]))','(rng.normal(size=4),rng.normal(size=4))'])], hint='Normalize v, find the scalar dot product, then subtract the projection.')
add('4.2', '''
def hyperplane(X):
    w = X[0]-X[1]
    theta = w/np.linalg.norm(w)
    return theta,-theta@(X[0]+X[1])/2
''', [('hyperplane',['(np.array([[-3.,0.],[3.,0.]]),)','(np.array([[5.,7.],[-3.,2.]]),)','(rng.normal(size=(2,4)),)'])], hint='Use the difference as the normal and force the midpoint onto the plane.')
add('4.3', '''
def solve_system(A,b):
    return np.linalg.solve(A,b)
''', [('solve_system',['(np.array([[.91,-.72,0],[-.09,1,-.09],[0,-.09,.91]]),np.array([1.,2.,0.]))','(spd(rng,4),rng.normal(size=4))'])], hint='solve(A, b) directly solves Ax=b.', note='The supplied specification notes a published-key typo: the first entry of this Bellman-system solution is about 2.904, not 2.094. Verify by multiplying A @ x.')
add('4.4', '''
def eig2(a,b,c,d):
    delta = np.sqrt((a+d)**2-4*(a*d-b*c))
    return np.array([(a+d+delta)/2,(a+d-delta)/2])
''', [('eig2',['(2.,1.,1.,2.)','(3.,0.5,0.5,1.)'])], rules={'eig2':{'no_linalg':True}}, hint='Solve the characteristic quadratic, returning the larger root first.', note='Test inputs have real eigenvalues.')
add('4.5', '''
class My_Fib:
    def __init__(self):
        self.lambdas,self.Q = np.linalg.eigh(np.array([[1.,1.],[1.,0.]]))
    def compute_fib(self,f0,f1,indices):
        n = np.array(indices)
        coefficients = self.Q.T@np.array([f1,f0])
        values = (self.lambdas**n[:,None]*coefficients)@self.Q[1]
        # Round before integer conversion; truncation can turn 8.999999 into 8.
        return np.rint(values).astype(np.int32)
    def plot_fib(self,f0,f1,indices):
        values = self.compute_fib(f0,f1,indices)
        plt.plot(indices,values,marker='x',linestyle='--',color='red')
        plt.xlabel('Index'); plt.ylabel('Fibonacci value')
        return values
''', [('My_Fib.compute_fib',['(0,1,np.arange(10))','(2,3,np.array([0,1,4,7,10]))'],'lambda ns,args: ns["My_Fib"]().compute_fib(*args)')], rules={'My_Fib.compute_fib':NL}, hint='Diagonalize the recurrence matrix; multiply its eigenvalues raised to each n.', note='Use eigenvectors as returned by eigh; round before int32 conversion. Keep indices small enough to avoid integer overflow.', plot='obj = My_Fib(); obj.plot_fib(0,1,np.arange(10)); plt.show()')
add('4.6', '''
def spectral(A):
    return np.linalg.eigh(A)
''', [('spectral',['(np.array([[2.,1.],[1.,2.]]),)','(spd(rng,4),)'],'spectral_probe')], hint='eigh is for symmetric matrices; each column of Q is an eigenvector.', note='The grader checks reconstruction and orthogonality, allowing eigenvector sign changes.')
add('4.7', '''
def svd_parts(W):
    U,s,Vt = np.linalg.svd(W,full_matrices=False)
    return U,s,Vt.T
''', [('svd_parts',['(np.array([[1.,0.],[0.,2.],[1.,1.]]),)','(rng.normal(size=(6,3)),)'],'svd_probe')], hint='NumPy returns Vt, so transpose it to return V.', note='Assume N >= d. Eigenvector and singular-vector signs are not uniquely determined.')
add('4.8', '''
def spectral_from_svd(W):
    U,s,Vt = np.linalg.svd(W,full_matrices=False)
    return U,s**2
def fro_sq(W):
    s = np.linalg.svd(W,full_matrices=False)[1]
    return np.sum(s**4)
''', [('spectral_from_svd',['(np.array([[1.,0.],[0.,2.],[1.,1.]]),)','(rng.normal(size=(6,3)),)'],'gram_spectral_probe'),('fro_sq',['(np.array([[1.,0.],[0.,2.]]),)','(rng.normal(size=(6,3)),)'])], hint='The eigenvalues of W W.T are squared singular values of W.', note='fro_sq means the squared Frobenius norm of S = W W.T, not of W.')
add('4.9', '''
def lowrank_error(W,r):
    s = np.linalg.svd(W,full_matrices=False)[1]
    return 1-np.sum(s[:r]**4)/np.sum(s**4)
def error_curve(W):
    s = np.linalg.svd(W,full_matrices=False)[1]
    return 1-np.cumsum(s**4)/np.sum(s**4)
''', [('lowrank_error',['(np.diag([3.,2.,1.]),1)','(rng.normal(size=(8,4)),2)']),('error_curve',['(np.diag([3.,2.,1.]),)','(rng.normal(size=(8,4)),)'])], rules={'lowrank_error':NL}, hint='For S, use fourth powers of singular values; cumsum gives the whole curve.', note='This is relative squared error for S. W itself would use squared singular values.', plot="W = np.random.default_rng(2026).normal(size=(40,10)); curve = error_curve(W)\nplt.plot(np.arange(1,len(curve)+1),curve,'o-'); plt.xlabel('Rank r'); plt.ylabel('Relative squared error of S'); plt.show()")
add('4.10', '''
def GQA_2_MLA(W):
    r,D = W.shape
    tiled = np.concatenate([W]*(D//r),axis=0)
    U,s,Vt = np.linalg.svd(tiled,full_matrices=False)
    return s[:r,None]*Vt[:r],U[:,:r]
''', [('GQA_2_MLA',['(np.array([[1.,2.,3.,4.]]),)','(rng.normal(size=(4,24)),)'],'factor_probe')], rules={'GQA_2_MLA':NL}, hint='Factor the tiled matrix with SVD; put singular values in the right factor.', note='Assume r divides D. The grader compares the product, allowing many valid factor pairs.')
add('4.11', '''
def kernel_rank(x):
    P = x[:,None]*x[None,:]
    K = (1+P+2*P**2)**2
    return np.linalg.matrix_rank(K)
''', [('kernel_rank',['(np.linspace(-1,1,6),)','(rng.uniform(-1,1,size=9),)'])], hint='The polynomial has powers 0 through 4, so the feature space has five dimensions.', note='Use distinct, moderately sized points to avoid ill-conditioning. A singular determinant is approximately zero.')
add('4.12', '''
def pairwise_sq_dist(A,B):
    algebra = np.sum(A**2,1)[:,None]+np.sum(B**2,1)[None,:]-2*A@B.T
    broadcast = np.sum((A[:,None,:]-B[None,:,:])**2,axis=-1)
    return algebra,broadcast
def kmeans_assign(X,C):
    distances = np.sum((X[:,None,:]-C[None,:,:])**2,axis=-1)
    return distances.argmin(axis=1)
def kmeans_update(X,labels,k):
    return np.array([X[labels==j].mean(0) for j in range(k)])
''', [('pairwise_sq_dist',['(np.array([[0.,0.],[1.,0.]]),np.array([[0.,1.],[2.,0.]]))','(rng.normal(size=(5,3)),rng.normal(size=(4,3)))']),('kmeans_assign',['iaio_kmeans_table()','(rng.normal(size=(6,2)),rng.normal(size=(2,2)))']),('kmeans_update',['(iaio_kmeans_table()[0],np.array([0]*5+[1]*5),2)','(rng.normal(size=(9,2)),np.repeat(np.arange(3),3),3)'])], rules={'pairwise_sq_dist':NL}, hint='Build a distance table, choose the smallest column per row, then average each cluster.', note='Loops over clusters are allowed in kmeans_update. Assume every cluster has at least one sample.')
add('5.1', '''
class My_Linear_NumPy:
    def __init__(self,in_features,out_features,random_seed=42):
        self.in_features,self.out_features = in_features,out_features
        np.random.seed(random_seed)
        self.weight = np.random.randn(out_features,in_features)
        self.bias = np.random.randn(out_features)
    def forward(self,x):
        return x@self.weight.T+self.bias
''', [('My_Linear_NumPy.forward',[f'(rng.normal(size={s}),)' for s in ['(3,)','(10,3)','(10,20,3)','(10,20,30,3)']],'linear_probe')], rules={'My_Linear_NumPy.forward':NL}, hint='Multiply on the last axis; leading dimensions remain unchanged.')
add('5.2', '''
def n_params_linear(in_,out,bias=True):
    return out*in_+(out if bias else 0)
def n_params_mlp(sizes):
    return sum(n_params_linear(a,b) for a,b in zip(sizes[:-1],sizes[1:]))
def n_params_conv(in_c,out_c,k,bias=True):
    return out_c*in_c*k*k+(out_c if bias else 0)
def n_params_ffn(d,f):
    return 2*d*f
''', [('n_params_linear',['(3,2,True)','(17,5,False)']),('n_params_mlp',['([2,3,1],)','([7,13,5,2],)']),('n_params_conv',['(3,4,3,True)','(8,16,5,False)']),('n_params_ffn',['(4,8)','(16,32)'])], hint='Count the weight entries, then add biases when present.', note='MLP layers include biases; FFN here is two linear layers without biases.')
add('5.3', '''
def symmetric_forward(W,x):
    return x@W.T@W
''', [('symmetric_forward',['(np.array([[1.,2.,3.],[0.,1.,0.]]),np.array([1.,0.,1.]))','(rng.normal(size=(2,3)),rng.normal(size=(5,3)))'])], rules={'symmetric_forward':NL}, hint='For batches, samples are rows and both projections act on the right.')
add('5.4', '''
def fit_gd(X,y,lr,n_iter):
    theta = np.zeros(X.shape[1])
    history = np.zeros(n_iter)
    for i in range(n_iter):
        grad = -2/X.shape[0]*X.T@(y-X@theta)
        theta -= lr*grad
        history[i] = np.mean((y-X@theta)**2)
    return theta,history
''', [('fit_gd',['(np.eye(2),np.array([2.,-1.]),0.2,80)','regression_args(rng)'])], rules={'fit_gd':{'max_loops':1}}, hint='Initialize theta to zeros; record the MSE after each update.', plot="X,y,lr,n = regression_args(np.random.default_rng(2026)); theta,history = fit_gd(X,y,lr,n)\nplt.plot(history); plt.xlabel('Iteration'); plt.ylabel('MSE'); plt.show()")
add('5.5', '''
def my_sigmoid(z):
    return 1/(1+np.exp(-np.clip(z,-500,500)))
class My_Log_Reg:
    def __init__(self,solver='GD',lr=0.01,num_iter=20):
        if solver not in ('GD','Newton'):
            raise ValueError('Invalid solver')
        self.solver,self.lr,self.num_iter = solver,lr,num_iter
        self.coef_ = None
    def fit(self,X,y):
        self.coef_ = np.zeros(X.shape[1])
        for _ in range(self.num_iter):
            z = my_sigmoid(X@self.coef_)
            grad = X.T@(z-y)
            if self.solver=='Newton':
                H = X.T@np.diag(z*(1-z))@X
                self.coef_ -= np.linalg.inv(H)@grad
            else:
                self.coef_ -= self.lr*grad
        return self
    def predict(self,X):
        return np.where(my_sigmoid(X@self.coef_)>=0.5,1,0)
    def score(self,X,y):
        return np.mean(self.predict(X)==y)
''', [('my_sigmoid',['(np.array([-2.,0.,2.]),)','(rng.normal(size=(3,4)),)']),('My_Log_Reg.fit',['(12,"Newton")','(2026,"GD")'],'logistic_probe')], rules={'My_Log_Reg.fit':{'max_loops':1},'My_Log_Reg.predict':NL,'My_Log_Reg.score':NL}, hint='Use X.T @ (z-y); choose a GD or Newton update inside one iteration loop.', note='The blobs overlap slightly and labels include a few flips to keep the Newton Hessian well-behaved.')
add('5.6', '''
def mha_numpy(x,y,WQ,WK,WV,WO,H):
    B,L1,D = x.shape  # x: (B,L1,D)
    L2 = y.shape[1]  # y: (B,L2,D)
    d = WQ.shape[0]//H  # each head: d
    Q = (x@WQ.T).reshape(B,L1,H,d).transpose(0,2,1,3)  # (B,H,L1,d)
    K = (y@WK.T).reshape(B,L2,H,d).transpose(0,2,1,3)  # (B,H,L2,d)
    V = (y@WV.T).reshape(B,L2,H,d).transpose(0,2,1,3)  # (B,H,L2,d)
    scores = Q@K.transpose(0,1,3,2)/np.sqrt(d)  # (B,H,L1,L2)
    e = np.exp(scores-scores.max(-1,keepdims=True))  # (B,H,L1,L2)
    probs = e/e.sum(-1,keepdims=True)  # (B,H,L1,L2)
    merged = (probs@V).transpose(0,2,1,3).reshape(B,L1,H*d)  # (B,L1,H*d)
    return merged@WO.T  # (B,L1,D1)
''', [('mha_numpy',['attention_args(1,1,2,3,4,2,2)','attention_args(2026,2,3,4,6,3,2)'])], rules={'mha_numpy':NL}, prediction='Which axis does softmax normalize: queries or keys?', hint='Split heads, compute scores, normalize keys, multiply V, and merge.', note='Write a shape comment after every line.')
add('5.7', '''
def gqa_numpy(x,y,WQ,WK,WV,WO,H,G):
    B,L1,D = x.shape
    L2 = y.shape[1]
    d = WQ.shape[0]//H
    Q = (x@WQ.T).reshape(B,L1,H//G,G,d).transpose(0,2,3,1,4)
    K = (y@WK.T).reshape(B,L2,1,G,d).transpose(0,2,3,1,4)
    V = (y@WV.T).reshape(B,L2,1,G,d).transpose(0,2,3,1,4)
    scores = Q@K.swapaxes(-2,-1)/np.sqrt(d)
    e = np.exp(scores-scores.max(-1,keepdims=True))
    probs = e/e.sum(-1,keepdims=True)
    merged = (probs@V).transpose(0,3,1,2,4).reshape(B,L1,H*d)
    return merged@WO.T
''', [('gqa_numpy',['gqa_args(1,2,2)','gqa_args(2026,4,2)'])], rules={'gqa_numpy':dict(no_loops=True,no_kv_copies=True)}, hint='Insert a size-one group-repeat axis for K and V instead of copying.', note='H is divisible by G; the head order is (repeat index, group index).')
add('5.8', '''
def nearest_dict(tokens,W):
    S = W@W.T
    np.fill_diagonal(S,-np.inf)
    indices = np.argmax(S,axis=1)
    return {token:tokens[j] for token,j in zip(tokens,indices)}
''', [('nearest_dict',['(["a","b","c"],normalized(np.array([[1.,0.],[1.,1.],[0.,1.]])))','embedding_args(50,20,2026)'])], hint='Exclude the diagonal before argmax; build the dictionary only at the end.', note='One final dict comprehension is allowed; rows are unit vectors.')
add('5.9', '''
def compress_embeddings(vectors):
    W = vectors/np.sqrt(np.sum(vectors**2,axis=1,keepdims=True))
    S = np.sum(W[:,None,:]*W[None,:,:],axis=-1)
    U,s,Vt = np.linalg.svd(W,full_matrices=False)
    Q,lam = U,s**2
    norm_sq = np.sum(s**4)
    curve = 1-np.cumsum(s**4)/norm_sq
    return W,S,U,s,Vt,Q,lam,norm_sq,curve
''', [('compress_embeddings',['(np.array([[1.,0.],[0.,1.],[1.,1.]]),)','(random_embeddings(N=30,d=10,seed=2026)[1],)'],'compression_probe')], rules={'compress_embeddings':dict(no_loops=True,no_matmul=True)}, hint='Normalize, compute all similarities, then use SVD and cumulative fourth powers.', note='Do not use linalg for row normalization; SVD is allowed for the later factorization.', plot="parts = compress_embeddings(random_embeddings(N=60,d=20,seed=2026)[1]); curve = parts[-1]\nplt.plot(np.arange(1,len(curve)+1),curve); plt.xlabel('Rank r'); plt.ylabel('Relative squared error of S'); plt.show()")
add('5.10', '''
def to_batch(images):
    return np.stack([np.transpose(img,(2,0,1))/255*2-1 for img in images])
''', [('to_batch',['([np.zeros((224,224,3),dtype=np.uint8),np.full((224,224,3),255,dtype=np.uint8)],)','([rng.integers(0,256,size=(224,224,3),dtype=np.uint8) for _ in range(3)],)'])], hint='Transpose and scale each image, then stack the batch.', note='A loop or comprehension over images is allowed.')
add('5.11', '''
def inside_triangle(P,variant=1):
    if variant==1:
        W,b = np.array([[1.,0.],[0.,1.],[-2.,-1.]]),np.array([0.,0.,2.])
    else:
        W,b = np.array([[-1.,3.],[-4.,-2.],[5.,-1.]]),np.array([0.,14.,0.])
    return np.prod((P@W.T+b)>=0,axis=1).astype(int)
''', [('inside_triangle',['(np.array([[0.,0.],[0.2,0.2],[1.,2.]]),1)','(np.array([[0.,0.],[3.,1.],[1.,5.],[1.,1.],[4.,4.]]),2)','(rng.uniform(-1,6,size=(20,2)),2)'])], rules={'inside_triangle':NL}, hint='Derive an inward-pointing inequality for each edge; multiply boolean tests.', note='Boundary points count as inside. variant=2 uses vertices (0,0), (3,1), (1,5).')
add('5.12', '''
def softmax_T(z,tau):
    scaled = z/tau
    e = np.exp(scaled-scaled.max(-1,keepdims=True))
    return e/e.sum(-1,keepdims=True)
''', [('softmax_T',['(np.array([1.,2.,4.]),0.01)','(rng.normal(size=(3,4)),100000.)'])], rules={'softmax_T':NL}, hint='Divide logits by positive temperature before stable softmax.')
add('5.13', '''
def clip_loss(I,T,log_tau):
    I = I/np.linalg.norm(I,axis=1,keepdims=True)
    T = T/np.linalg.norm(T,axis=1,keepdims=True)
    logits = (I@T.T)/np.exp(log_tau)
    row_max = logits.max(1,keepdims=True)
    col_max = logits.max(0,keepdims=True)
    row_log = logits-row_max-np.log(np.exp(logits-row_max).sum(1,keepdims=True))
    col_log = logits-col_max-np.log(np.exp(logits-col_max).sum(0,keepdims=True))
    return -0.5*(np.diag(row_log).mean()+np.diag(col_log).mean())
''', [('clip_loss',['(np.eye(2),np.eye(2),0.)','(rng.normal(size=(4,8)),rng.normal(size=(4,8)),np.log(0.2))'])], rules={'clip_loss':NL}, hint='Compute log-softmax for both directions and take the diagonal losses.', note='Here log_tau is log(temperature), so divide by exp(log_tau). Some implementations use a log scale and multiply instead; these conventions have opposite signs.')
add('7.1', '''
def torch_regroup(a):
    return a.reshape(2,3),a.reshape(-1,1),a.flatten()
def torch_shape_walk(arr):
    a = arr.squeeze()
    b = a.unsqueeze(2)
    c = b.transpose(0,1)
    return tuple(a.shape),tuple(b.shape),tuple(c.shape),tuple(c.flatten().shape)
def torch_join(a,b):
    return torch.cat([a,b]),torch.stack([a,b]),torch.stack([a,b],dim=1)
def torch_unit_rows(V):
    return V/torch.norm(V,dim=-1,keepdim=True)
def torch_sim_matrix(W):
    return torch.sum(W[:,None,:]*W[None,:,:],dim=-1)
def torch_to_chw(img):
    return img.permute(2,0,1)
''', [('torch_regroup',['(torch.arange(6),)','(torch.randn(6),)']),('torch_shape_walk',['(torch.zeros(5,8,3,1,2),)','(torch.randn(2,4,3,1,2),)']),('torch_join',['(torch.tensor([1,2,3]),torch.tensor([4,5,6]))','(torch.randn(3),torch.randn(3))']),('torch_unit_rows',['(torch.tensor([[3.,4.],[0.,2.]]),)','(torch.randn(5,3),)']),('torch_sim_matrix',['(torch.eye(2),)','(torch.randn(5,3),)']),('torch_to_chw',['(torch.arange(18).reshape(2,3,3),)','(torch.randn(4,5,3),)'])], rules={n:NL for n in ['torch_regroup','torch_shape_walk','torch_join','torch_unit_rows','torch_sim_matrix','torch_to_chw']}, hint='PyTorch uses dim and keepdim where NumPy uses axis and keepdims.', note='Add a length-one axis with unsqueeze. from_numpy shares memory; use clone when you need a separate tensor.')
add('7.2', '''
class My_ReLU(nn.Module):
    def forward(self,x):
        return torch.max(torch.zeros_like(x),x)
class My_Threshold(nn.Module):
    def forward(self,x):
        return (x>=0).float()
class My_Linear(nn.Module):
    def __init__(self,in_features,out_features,bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features,in_features)/np.sqrt(in_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
    def forward(self,x):
        out = x@self.weight.T
        return out+self.bias if self.bias is not None else out
''', [('My_ReLU.forward',['(torch.tensor([-1.,0.,2.]),)','(torch.randn(2,3),)'],'lambda ns,args: ns["My_ReLU"]()(*args)'),('My_Threshold.forward',['(torch.tensor([-1.,0.,2.]),)','(torch.randn(2,3),)'],'lambda ns,args: ns["My_Threshold"]()(*args)'),('My_Linear.forward',['(3,2,True)','(64,256,False)'],'torch_linear_probe')], rules={'My_ReLU.forward':NL,'My_Threshold.forward':NL,'My_Linear.forward':NL}, hint='Register trainable tensors with nn.Parameter; initialize weights with std = 1/sqrt(in).')
add('7.3', '''
class My_MLP_Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(1,32),nn.Tanh(),nn.Linear(32,1))
    def forward(self,x):
        return self.layers(x)
def train_mlp(x,y,epochs=80):
    torch.manual_seed(42)
    model = My_MLP_Model()
    optimizer = torch.optim.Adam(model.parameters(),lr=0.03)
    history = np.zeros(epochs)
    for epoch in range(epochs):
        optimizer.zero_grad()
        prediction = model(x)
        loss = torch.mean((prediction-y)**2)
        loss.backward()
        optimizer.step()
        history[epoch] = loss.item()
        if (epoch+1)%10==0:
            print(f'Epoch {epoch+1}: loss {loss.item():.4f}')
    return model,history
''', [('train_mlp',['mlp_args(1,40)','mlp_args(2026,80)'],'mlp_probe')], rules={'train_mlp':{'max_loops':1},'My_MLP_Model.forward':NL}, hint='Clear gradients, compute MSE, backpropagate, then take an Adam step.', note='Whole-batch CPU training; print every 10 epochs to four decimal places.', plot="x,y,epochs = mlp_args(2026,80); model,history = train_mlp(x,y,epochs)\nplt.plot(history); plt.xlabel('Epoch'); plt.ylabel('MSE'); plt.show()")
add('7.4', '''
def scalar_derivatives(p_value,q_value):
    p = torch.tensor(float(p_value),requires_grad=True)
    q = torch.tensor(float(q_value),requires_grad=True)
    f = p**2+q**3+p*q**2
    fp,fq = torch.autograd.grad(f,(p,q),create_graph=True)
    fpp = torch.autograd.grad(fp,p,create_graph=True)[0]
    fqq = torch.autograd.grad(fq,q,create_graph=True)[0]
    fpq = torch.autograd.grad(fp,q,create_graph=True)[0]
    fqqq = torch.autograd.grad(fqq,q)[0]
    return tuple(v.detach().item() for v in (fp,fq,fpp,fqq,fpq,fqqq))
def vector_derivatives(x):
    x = x.clone().detach().requires_grad_(True)
    g = x**2
    first = torch.autograd.grad(g.sum(),x,create_graph=True)[0]
    second = torch.autograd.grad(first.sum(),x)[0]
    return first.detach(),second.detach()
''', [('scalar_derivatives',['(1.,2.)','(-0.5,3.)']),('vector_derivatives',['(torch.tensor([1.,2.,3.]),)','(torch.randn(5),)'])], hint='create_graph=True keeps derivatives differentiable for the next derivative.', note='Correction to the supplied spec: at (1,2), f_qq = 6q+2p = 14. The full tuple is (6,16,2,14,4,6).')
add('7.5', '''
class MyMHA(nn.Module):
    def __init__(self,D,D1,H,d):
        super().__init__()
        self.H,self.d = H,d
        self.WQ = nn.Parameter(torch.randn(H*d,D)*0.1)
        self.WK = nn.Parameter(torch.randn(H*d,D)*0.1)
        self.WV = nn.Parameter(torch.randn(H*d,D)*0.1)
        self.WO = nn.Parameter(torch.randn(D1,H*d)*0.1)
    def forward(self,x,y):
        B,L1,D = x.shape
        L2 = y.shape[1]
        H,d = self.H,self.d
        Q = (x@self.WQ.T).reshape(B,L1,H,d).permute(0,2,1,3)
        K = (y@self.WK.T).reshape(B,L2,H,d).permute(0,2,1,3)
        V = (y@self.WV.T).reshape(B,L2,H,d).permute(0,2,1,3)
        scores = Q@K.transpose(-2,-1)/np.sqrt(d)
        weights = torch.softmax(scores,dim=-1)
        return (weights@V).permute(0,2,1,3).reshape(B,L1,H*d)@self.WO.T
''', [('MyMHA.forward',['(1,1,2,3,4,2,2)','(2026,2,3,4,6,3,2)'],'torch_attention_probe')], rules={'MyMHA.forward':NL}, hint='Match the NumPy attention steps, using permute and dim=-1.')
add('7.6', '''
class TokenDataset(Dataset):
    def __init__(self,sequences):
        self.sequences = sequences
    def __len__(self):
        return len(self.sequences)
    def __getitem__(self,index):
        return torch.tensor(self.sequences[index],dtype=torch.int64)
def collate_fn(batch):
    lengths = torch.tensor([len(seq) for seq in batch],dtype=torch.int64)
    K = lengths.max().item()
    tokens = torch.zeros(len(batch),K,dtype=torch.int64)
    for i,seq in enumerate(batch):
        tokens[i,:len(seq)] = seq
    mask = (torch.arange(K)[None,:]<lengths[:,None]).to(torch.int64)
    return tokens,mask
''', [('collate_fn',['([[1,2,3],[4]],)','([list(rng.integers(1,20,size=n)) for n in [2,5,3]],)'],'loader_probe')], hint='Pad to the longest sequence, then compare positions with original lengths.', note='Padding token is 0; masks use 1 for real tokens and 0 for padding, both int64. Loops over sequences are allowed.')
# Mock exercises adapt the practice tasks. They are not official contest papers.
def mock_copy(id, source):
    original = EXERCISES[source]
    EXERCISES[id] = dict(original,id=id)
mock_copy('A.1','1.9')
mock_copy('A.2','2.4')
mock_copy('A.3','4.2')
mock_copy('A.4','3.8')
add('A.5', '''
def vectors_array(vectors):
    return np.array(vectors,dtype=float)
def unit_rows(V):
    return V/np.sqrt(np.sum(V**2,axis=1,keepdims=True))
def similarity_range(W):
    return (-1.,1.)
def sim_matrix(W):
    return np.sum(W[:,None,:]*W[None,:,:],axis=-1)
def diagonal_of_similarity(S):
    return np.diag(S)
def nearest_dict(tokens,W):
    S = W@W.T
    np.fill_diagonal(S,-np.inf)
    indices = S.argmax(axis=1)
    return {token:tokens[j] for token,j in zip(tokens,indices)}
def svd_parts(W):
    U,s,Vt = np.linalg.svd(W,full_matrices=False)
    return U,s,Vt.T
def spectral_from_svd(W):
    U,s,Vt = np.linalg.svd(W,full_matrices=False)
    return U,s**2
def fro_sq(W):
    return np.sum(np.linalg.svd(W,full_matrices=False)[1]**4)
def lowrank_factor(W,r):
    U,s,Vt = np.linalg.svd(W,full_matrices=False)
    return U[:,:r]*s[:r]
def error_curve(W):
    s = np.linalg.svd(W,full_matrices=False)[1]
    return 1-np.cumsum(s**4)/np.sum(s**4)
''', [('vectors_array',['([[1.,0.],[0.,1.]],)','(random_embeddings(N=300,d=100,seed=2026)[1].tolist(),)']),('unit_rows',['(np.array([[3.,4.],[0.,2.]]),)','(random_embeddings(N=300,d=100,seed=2026)[1],)']),('similarity_range',['(np.eye(2),)','(normalized(random_embeddings(N=300,d=100,seed=2026)[1]),)']),('sim_matrix',['(np.eye(2),)','(normalized(random_embeddings(N=300,d=100,seed=2026)[1]),)']),('diagonal_of_similarity',['(np.eye(2),)','(normalized(random_embeddings(N=300,d=100,seed=2026)[1])@normalized(random_embeddings(N=300,d=100,seed=2026)[1]).T,)']),('nearest_dict',['embedding_args(3,2,1)','embedding_args(300,100,2026)']),('svd_parts',['(np.eye(2),)','(normalized(random_embeddings(N=300,d=100,seed=2026)[1]),)'],'svd_probe'),('spectral_from_svd',['(np.eye(2),)','(normalized(random_embeddings(N=300,d=100,seed=2026)[1]),)'],'gram_spectral_probe'),('fro_sq',['(np.eye(2),)','(normalized(random_embeddings(N=300,d=100,seed=2026)[1]),)']),('lowrank_factor',['(np.diag([2.,1.]),1)','(normalized(random_embeddings(N=300,d=100,seed=2026)[1]),20)'],'lowrank_probe'),('error_curve',['(np.diag([2.,1.]),)','(normalized(random_embeddings(N=300,d=100,seed=2026)[1]),)'])], rules={'unit_rows':dict(no_loops=True,no_linalg=True),'sim_matrix':PURE}, note='Written response: explain why the 300 by 300 similarity matrix has rank at most 100 and cannot be invertible. Also explain when a dot product equals cosine similarity.', plot="W = unit_rows(random_embeddings(N=300,d=100,seed=2026)[1]); curve = error_curve(W)\nplt.plot(np.arange(1,len(curve)+1),curve); plt.xlabel('Rank r'); plt.ylabel('Relative squared error of S'); plt.show()")
mock_copy('A.6','3.12')
mock_copy('B.1','4.5')
mock_copy('B.2','5.1')
# B.3 includes module design but not training: the timed task only asks for definitions.
add('B.3', EXERCISES['7.2']['code'].split('class My_Threshold')[0]+EXERCISES['7.3']['code'].split('def train_mlp')[0], [('My_ReLU.forward',['(torch.tensor([-1.,0.,2.]),)','(torch.randn(2,3),)'],'lambda ns,args: ns["My_ReLU"]()(*args)'),('My_MLP_Model.forward',['(torch.zeros(2,1),)','(torch.randn(5,1),)'],'module_shape_probe')], rules={'My_ReLU.forward':NL,'My_MLP_Model.forward':NL})
add('B.4', EXERCISES['5.5']['code']+'''
def prepare_titanic(df,test_size=0.2,seed=2025):
    clean = df.dropna().copy()
    y = clean['Survived'].to_numpy()
    X = pd.get_dummies(clean.drop(columns=['Survived']),drop_first=True,dtype=np.int8).to_numpy(dtype=float)
    np.random.seed(seed)
    indices = np.random.permutation(len(X))
    np.random.seed()
    k = int(len(X)*test_size)
    test,train = indices[:k],indices[k:]
    scaler = StandardScaler().fit(X[train])
    Xtrain,Xtest = scaler.transform(X[train]),scaler.transform(X[test])
    Xtrain = np.concatenate([Xtrain,np.ones((len(train),1))],axis=1)
    Xtest = np.concatenate([Xtest,np.ones((len(test),1))],axis=1)
    return Xtrain,Xtest,y[train],y[test]
''', [('prepare_titanic',['(titanic_like(N=60,seed=1),0.2,1)','(titanic_like(N=600,seed=2025),0.2,2025)']),('My_Log_Reg.fit',['(60,1)','(600,2025)'],'titanic_probe')], rules={'My_Log_Reg.fit':{'max_loops':1},'My_Log_Reg.predict':NL,'My_Log_Reg.score':NL}, note='This synthetic dataset has independent random labels; chance-level test accuracy is expected. It is a preprocessing exercise, not a realistic prediction benchmark. Fit StandardScaler only on training rows.')
add('B.5', EXERCISES['7.5']['code']+'''
class MyGQA(nn.Module):
    def __init__(self,D,D1,H,G,d):
        super().__init__()
        self.H,self.G,self.d = H,G,d
        self.WQ = nn.Parameter(torch.randn(H*d,D)*0.1)
        self.WK = nn.Parameter(torch.randn(G*d,D)*0.1)
        self.WV = nn.Parameter(torch.randn(G*d,D)*0.1)
        self.WO = nn.Parameter(torch.randn(D1,H*d)*0.1)
    def forward(self,x,y):
        B,L1,D = x.shape
        L2 = y.shape[1]
        H,G,d = self.H,self.G,self.d
        Q = (x@self.WQ.T).reshape(B,L1,H//G,G,d).permute(0,2,3,1,4)
        K = (y@self.WK.T).reshape(B,L2,1,G,d).permute(0,2,3,1,4)
        V = (y@self.WV.T).reshape(B,L2,1,G,d).permute(0,2,3,1,4)
        probs = torch.softmax(Q@K.transpose(-2,-1)/np.sqrt(d),dim=-1)
        return (probs@V).permute(0,3,1,2,4).reshape(B,L1,H*d)@self.WO.T
''', [('MyMHA.forward',['(1,1,2,3,4,2,2)','(2026,2,3,4,6,3,2)'],'torch_attention_probe'),('MyGQA.forward',['(1,2,2)','(2026,4,2)'],'gqa_torch_probe')], rules={'MyMHA.forward':NL,'MyGQA.forward':dict(no_loops=True,no_kv_copies=True)}, note='Compare MyGQA with MyMHA when G == H, using exactly the same weights.')

# Rule refinements shared by ordinary practice and the timed copies.
for id in ('5.8','A.5'):
    EXERCISES[id]['rules']['nearest_dict'] = dict(final_dict_only=True)
EXERCISES['1.7']['rules']['split_indices'] = dict(require_seed_reset=True)
EXERCISES['B.4']['rules']['prepare_titanic'] = dict(require_seed_reset=True)
for id in ('5.1','B.2'):
    EXERCISES[id]['rules']['My_Linear_NumPy.__init__'] = dict(no_loops=True)
EXERCISES['4.4']['tests'][0][1][1] = '(float(rng.uniform(1,3)),0.5,0.5,float(rng.uniform(1,3)))'
EXERCISES['5.2']['tests'][0][1][1] = '(int(rng.integers(2,20)),int(rng.integers(2,10)),False)'
EXERCISES['5.2']['tests'][1][1][1] = '(list(map(int,rng.integers(2,16,size=4))),)'
EXERCISES['5.2']['tests'][2][1][1] = '(int(rng.integers(2,10)),int(rng.integers(2,20)),3,False)'
EXERCISES['5.2']['tests'][3][1][1] = '(int(rng.integers(2,20)),int(rng.integers(2,40)))'
