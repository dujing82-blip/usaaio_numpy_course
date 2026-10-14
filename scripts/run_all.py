"""Execute shipped notebooks headlessly; skip only network package installation.

Usage: python -m scripts.run_all --folder notebooks
       python -m scripts.run_all --folder solutions --require-pass
"""
import argparse
import json
import os
import time
import contextlib
import io
from pathlib import Path
import nbformat
from nbclient import NotebookClient
ROOT=Path(__file__).resolve().parents[1]

def run(folder,require_pass=False,in_process=False):
    os.environ.update(MPLBACKEND='Agg',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONPATH=str(ROOT))
    records=[]
    for path in sorted((ROOT/folder).glob('*.ipynb')):
        nb=nbformat.read(path,as_version=4)
        for cell in nb.cells:
            if cell.cell_type=='code' and 'setup' in cell.metadata.get('tags',[]):
                cell.source='\n'.join(line for line in cell.source.splitlines() if not line.startswith('!pip') and not line.startswith('%pip'))
        start=time.monotonic()
        if in_process:
            # Restricted hosts may prohibit all sockets, including Jupyter IPC.
            # IPython still executes the exact code cells and caches inspectable source.
            from IPython.core.interactiveshell import InteractiveShell
            shell=InteractiveShell()
            captured=io.StringIO()
            with contextlib.redirect_stdout(captured),contextlib.redirect_stderr(captured):
                for cell in nb.cells:
                    if cell.cell_type=='code':
                        result=shell.run_cell(cell.source,store_history=True)
                        if result.error_before_exec or result.error_in_exec:
                            raise result.error_before_exec or result.error_in_exec
            text=captured.getvalue()
        else:
            executed=NotebookClient(nb,timeout=120,kernel_name='python3',resources={'metadata':{'path':str(ROOT)}}).execute()
            text='\n'.join(output.get('text','') for cell in executed.cells if cell.cell_type=='code' for output in cell.get('outputs',[]))
        elapsed=time.monotonic()-start
        if require_pass:
            assert 'FAIL' not in text and 'RULE VIOLATION' not in text and 'not yet' not in text, f'{path.name}: a solution did not pass'
        assert elapsed<120, f'{path.name}: exceeded CPU runtime target'
        record=dict(notebook=path.name,folder=folder,seconds=round(elapsed,2),status='PASS',execution='IPython in-process' if in_process else 'nbclient')
        records.append(record)
        print(f'{folder}/{path.name}: PASS ({elapsed:.2f}s)',flush=True)
    assert sum(r['seconds'] for r in records)<600, 'total runtime exceeds ten minutes'
    return records

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--folder',choices=['notebooks','solutions','both'],default='notebooks')
    parser.add_argument('--require-pass',action='store_true')
    parser.add_argument('--in-process',action='store_true',help='Run through IPython without Jupyter sockets on restricted hosts')
    args=parser.parse_args()
    records=[]
    for folder in (['notebooks','solutions'] if args.folder=='both' else [args.folder]):
        records.extend(run(folder,require_pass=(folder=='solutions' or args.require_pass),in_process=args.in_process))
    (ROOT/'validation.json').write_text(json.dumps(records,indent=2)+'\n')
