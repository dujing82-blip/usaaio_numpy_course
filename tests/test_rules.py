import numpy as np
from usaaio_check import check_rules, grade, PURE

def loop_gradient(X,y,theta):
    for i in range(len(X)):
        pass
    return np.zeros(X.shape[1])
def matmul_gradient(X,y,theta):
    return X@theta
def transpose_gradient(X,y,theta):
    return X.T

def test_pure_rejects_prohibited_syntax():
    assert 'loop or comprehension' in check_rules(loop_gradient,**PURE)
    assert 'the @ operator' in check_rules(matmul_gradient,**PURE)
    assert '.T' in check_rules(transpose_gradient,**PURE)

def comprehension(x):
    return [v for v in x]
def imported(x):
    import numpy
    return x
def einsum(x):
    return np.einsum('i,i->',x,x)
def second_loop(x):
    for i in x:
        for j in x:
            pass
    return x

def test_more_rule_bans():
    assert check_rules(comprehension,no_loops=True)
    assert check_rules(imported)
    assert check_rules(einsum,no_matmul=True)
    assert check_rules(second_loop,max_loops=1)

def wrong_dtype(x):
    return x.astype(np.int64)
def wrong_shape(x):
    return x[:,None]
def mutation(x):
    x[:]=0
    return x

def test_dtype_shape_and_mutation_are_rejected():
    x=np.array([1,2,3],dtype=np.int32)
    assert not grade(wrong_dtype,[((x,),x)])
    assert not grade(wrong_shape,[((x,),x)])
    assert not grade(mutation,[((x.copy(),),np.zeros_like(x))],immutable=True)

def metadata(x):
    return (x.shape,x.ndim,x.size,str(x.dtype))
def test_nested_metadata_and_scalar_shape():
    x=np.zeros((2,3),dtype=np.int32)
    assert grade(metadata,[((x,),((2,3),2,6,'int32'))])

def good_dict(tokens):
    return {token:token for token in tokens}
def bad_dict(tokens):
    x=[t for t in tokens]
    return {token:token for token in x}
def test_only_final_dictionary_comprehension():
    assert not check_rules(good_dict,final_dict_only=True)
    assert check_rules(bad_dict,final_dict_only=True)
