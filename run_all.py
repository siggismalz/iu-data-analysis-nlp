"""Vollständiger Ablauf; trainierte Kandidaten mit passender Signatur werden wiederverwendet."""
import argparse
import subprocess
import sys
from pathlib import Path

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--archive',type=Path,help='Lokales Original-ZIP anstelle des Downloads')
a=p.parse_args()
root=Path(__file__).resolve().parent
commands=[['analyse.py','environment'],['analyse.py','prepare']+(['--archive',str(a.archive.resolve())] if a.archive else []),
          ['analyse.py','sweep'],['analyse.py','refine'],['report.py'],['validate.py']]
for command in commands:
    print('\nAusführen:', ' '.join(command),flush=True)
    subprocess.run([sys.executable,'-X','utf8',*command],cwd=root,check=True)
