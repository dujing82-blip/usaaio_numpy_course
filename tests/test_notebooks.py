import ast
import re
from pathlib import Path
import nbformat
from usaaio_check.catalog import EXERCISES
ROOT=Path(__file__).resolve().parents[1]

def test_full_catalog_and_cases():
    required={f'{chapter}.{i}' for chapter,n in [('1',15),('2',12),('3',13),('4',12),('5',13),('7',6)] for i in range(1,n+1)}
    assert required <= EXERCISES.keys()
    for e in EXERCISES.values():
        assert all(len(test[1])>=2 for test in e['tests'])

def test_student_stubs_and_allowed_imports():
    seen=set()
    for path in (ROOT/'notebooks').glob('*.ipynb'):
        nb=nbformat.read(path,as_version=4)
        for cell in nb.cells:
            if cell.cell_type!='code': continue
            source='\n'.join(l for l in cell.source.splitlines() if not l.startswith(('!','%')))
            tree=ast.parse(source)
            for node in ast.walk(tree):
                if isinstance(node,(ast.Import,ast.ImportFrom)):
                    assert 'setup' in cell.metadata.get('tags',[])
                    modules=([node.module] if isinstance(node,ast.ImportFrom) else [a.name for a in node.names])
                    assert all(m.split('.')[0] in {'numpy','torch','matplotlib','pandas','sklearn','usaaio_check'} for m in modules)
            if 'solution' in cell.metadata.get('tags',[]):
                seen.add(cell.metadata['exercise_id'])
                assert 'YOUR CODE HERE' in source
                for node in ast.walk(tree):
                    if isinstance(node,ast.FunctionDef):
                        assert ast.get_docstring(node)
                        assert len(node.body)<=3
    assert seen==set(EXERCISES)

def test_no_execution_outputs_shipped():
    for folder in ['notebooks','solutions']:
        for path in (ROOT/folder).glob('*.ipynb'):
            for cell in nbformat.read(path,as_version=4).cells:
                if cell.cell_type=='code':
                    assert not cell.outputs and cell.execution_count is None
