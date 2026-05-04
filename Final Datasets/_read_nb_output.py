"""Read executed notebook and print all cell outputs."""
import json
import sys

# Force stdout to utf-8 just in case, or safely write to avoid charmap errors
sys.stdout.reconfigure(encoding='utf-8')

with open('panel_data_construction_executed.ipynb', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] != 'code':
        continue
    outs = cell.get('outputs', [])
    ecount = cell.get('execution_count', '?')
    
    for o in outs:
        if o.get('name') == 'stdout':
            txt = ''.join(o.get('text', []))
            if txt.strip():
                print(f'--- Cell {ecount} STDOUT ---')
                print(txt[-900:].encode(sys.stdout.encoding, errors='replace').decode(sys.stdout.encoding))
        if o.get('output_type') == 'error':
            print(f'--- Cell {ecount} ERROR ---')
            print(f'  {o["ename"]}: {o["evalue"]}')
            tb = o.get('traceback', [])
            # Strip ANSI codes
            import re
            clean = [re.sub(r'\x1b\[[0-9;]*m', '', line) for line in tb[-6:]]
            print('\n'.join(clean).encode(sys.stdout.encoding, errors='replace').decode(sys.stdout.encoding))
            break
