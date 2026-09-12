"""Verify current source or an explicitly archived historical source snapshot."""
import hashlib
from pathlib import Path
import zipfile


def verify_source(name, digest):
    for folder in ('scripts','src','experiments'):
        path = Path(folder)/name
        if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() == digest:
            return 'current'
    snapshot = Path('archive/source_snapshots/pre_layout.zip')
    if snapshot.exists():
        with zipfile.ZipFile(snapshot) as z:
            if name in z.namelist() and hashlib.sha256(z.read(name)).hexdigest() == digest:
                print(f'{name}: verified archived pre-layout source (not current source)')
                return 'archived'
    raise AssertionError('No matching current or archived source: '+name)
