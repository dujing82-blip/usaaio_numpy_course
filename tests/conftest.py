import linecache
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import pytest
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import nbformat

torch.set_num_threads(1)
ROOT=Path(__file__).resolve().parents[1]

@pytest.fixture(scope='session')
def solution_namespaces():
    result={}
    for path in sorted((ROOT/'solutions').glob('*.ipynb')):
        ns=dict(np=np,pd=pd,plt=plt,torch=torch,nn=nn,Dataset=Dataset,DataLoader=DataLoader,StandardScaler=StandardScaler)
        nb=nbformat.read(path,as_version=4)
        for index,cell in enumerate(nb.cells):
            if cell.cell_type=='code' and 'solution' in cell.metadata.get('tags',[]):
                name=str(path)+f':cell{index}'
                linecache.cache[name]=(len(cell.source),None,cell.source.splitlines(True),name)
                exec(compile(cell.source,name,'exec'),ns)
                result[cell.metadata['exercise_id']]=ns
    return result
