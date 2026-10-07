"""Prüft die erzeugten Daten und Auswahlmetriken ohne erneutes Training."""
import json
import hashlib
import subprocess
import sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

root=Path(__file__).resolve().parent
sample=joblib.load(root/'private/sample.joblib')
b=joblib.load(root/'private/matrices.joblib')
audit=json.loads((root/'results/data_audit.json').read_text())
assert len(sample)==5000
assert len({d['id'] for d in sample})==5000
assert len({' '.join(d['tokens']) for d in sample})==5000
assert len(set(b['train']) & set(b['val']))==0
assert set(b['train']) | set(b['val']) == set(range(5000))
assert len(b['train'])==4000 and len(b['val'])==1000
assert b['counts'].shape==b['tfidf'].shape
assert b['counts'].nnz==b['tfidf'].nnz
assert np.all(np.asarray(b['counts'].sum(1)).ravel()>0)
expected_idf=np.log((1+len(b['train']))/(1+np.asarray((b['counts'][b['train']]>0).sum(0)).ravel()))+1
np.testing.assert_allclose(b['tfidf_model'].idf_,expected_idf)
assert hashlib.sha256('\n'.join(d['id'] for d in sample).encode()).hexdigest()==audit['sample_ids_sha256']
metrics=pd.read_csv(root/'results/metrics_runs.csv')
ordinary=metrics.drop(columns=['cv','cv_min']).select_dtypes('number')
assert np.isfinite(ordinary).all().all()
valid=metrics.undefined_topic_coherences==0
assert np.isfinite(metrics.loc[valid,['cv','cv_min']]).all().all()
assert metrics.loc[~valid,'cv'].isna().all()
selection=pd.read_csv(root/'results/model_selection.csv')
assert selection.loc[selection.undefined_topic_coherences>0,'cv_mean'].isna().all()
assert (metrics.groupby(['method','k']).size()==3).all()
assert metrics['warning_count'].sum()==0, 'Konvergenzwarnungen prüfen'
for path in (root/'private/models').glob('*.joblib'):
    m=joblib.load(path)
    assert np.isfinite(m['h']).all() and np.isfinite(m['w']).all()
    np.testing.assert_allclose(m['h'].sum(1),1,rtol=1e-8)
    np.testing.assert_allclose(m['w'].sum(1),1,rtol=1e-8)
subprocess.run([sys.executable,'-m','unittest','discover','-s','tests','-v'],cwd=root,check=True)
report={'status':'PASS','sample_size':5000,'split':[4000,1000],
        'model_runs':len(metrics),'idf_fitted_on_training_only':True,
        'finite_metrics_or_explicitly_undefined':True,
        'runs_with_undefined_coherence':int((~valid).sum()),'all_factors_normalized':True,'method_tests':4}
(root/'results/validation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report,indent=2))
