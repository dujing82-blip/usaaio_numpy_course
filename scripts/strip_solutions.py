"""Regenerate student notebooks from solution notebooks, preserving all other cells."""
import ast
from pathlib import Path
import nbformat
from scripts.build_notebooks import Stubber
ROOT=Path(__file__).resolve().parents[1]
for path in sorted((ROOT/'solutions').glob('*.ipynb')):
    nb=nbformat.read(path,as_version=4)
    for cell in nb.cells:
        if cell.cell_type=='code' and 'solution' in cell.metadata.get('tags',[]):
            tree=Stubber().visit(ast.parse(cell.source))
            cell.source=ast.unparse(ast.fix_missing_locations(tree)).replace("'YOUR CODE HERE'",'### YOUR CODE HERE ###')+'\n'
        if cell.cell_type=='code':
            cell.outputs=[]; cell.execution_count=None
    nbformat.write(nb,ROOT/'notebooks'/path.name)
