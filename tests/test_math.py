import numpy as np
import torch
from usaaio_check.exercise_tests import _equal

def test_gradient_against_finite_difference(solution_namespaces):
    rng=np.random.default_rng(88)
    X=rng.normal(size=(9,4)); y=rng.normal(size=9); theta=rng.normal(size=4)
    fn=solution_namespaces['3.8']['my_fun_derivative']
    eps=1e-5
    numerical=[]
    for j in range(4):
        delta=np.eye(4)[j]*eps
        numerical.append((np.mean((y-X@(theta+delta))**2)-np.mean((y-X@(theta-delta))**2))/(2*eps))
    _equal(fn(X,y,theta),np.array(numerical))

def test_fibonacci_recurrence_and_dtype(solution_namespaces):
    fn=solution_namespaces['4.5']['My_Fib']().compute_fib
    expected=np.array([0,1,1,2,3,5,8,13,21,34,55,89,144,233,377],dtype=np.int32)
    _equal(fn(0,1,np.arange(15)),expected)

def test_derivatives_corrected(solution_namespaces):
    _equal(solution_namespaces['7.4']['scalar_derivatives'](1,2),(6.,16.,2.,14.,4.,6.))

def test_svd_error_against_explicit_gram_approximation(solution_namespaces):
    W=np.random.default_rng(4).normal(size=(8,4))
    U,s,Vt=np.linalg.svd(W,full_matrices=False)
    r=2; F=U[:,:r]*s[:r]; S=W@W.T
    explicit=np.sum((S-F@F.T)**2)/np.sum(S**2)
    _equal(solution_namespaces['4.9']['lowrank_error'](W,r),explicit)

def test_attention_equal_head_groups(solution_namespaces):
    from usaaio_check.exercise_tests import attention_args
    args=attention_args(9,2,3,5,4,2,3)
    _equal(solution_namespaces['5.7']['gqa_numpy'](*args,2),solution_namespaces['5.6']['mha_numpy'](*args))

def test_eigenvector_signs_do_not_fail_grader():
    from usaaio_check.exercise_tests import spectral_probe
    A=np.array([[2.,1.],[1.,2.]])
    def negated(A):
        lam,Q=np.linalg.eigh(A)
        return lam,-Q
    _equal(spectral_probe({'spectral':negated},(A,)),A)

def test_kernel_rank_and_bellman_residual(solution_namespaces):
    assert solution_namespaces['4.11']['kernel_rank'](np.linspace(-1,1,8))==5
    A=np.array([[.91,-.72,0],[-.09,1,-.09],[0,-.09,.91]])
    b=np.array([1.,2.,0.])
    x=solution_namespaces['4.3']['solve_system'](A,b)
    _equal(A@x,b)
    assert abs(x[0]-2.90419173)<1e-7
