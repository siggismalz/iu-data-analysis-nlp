"""Erstellt die endgültigen aggregierten Ergebnisse aus dokumentierten Themenlabels."""
import hashlib
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import adjusted_rand_score
from analyse import save_json, sha256

ROOT=Path(__file__).resolve().parent
RES=ROOT/'results'
INTERPRETATION=ROOT/'interpretation.json'
if not INTERPRETATION.exists():
    raise SystemExit('interpretation.json mit geprüften Themenlabels fehlt. Zuerst Themen inhaltlich beurteilen.')
interpretation=json.loads(INTERPRETATION.read_text(encoding='utf-8'))
audit=json.loads((RES/'data_audit.json').read_text(encoding='utf-8'))
vectors=json.loads((RES/'vectorization.json').read_text(encoding='utf-8'))
assert interpretation['sample_ids_sha256']==audit['sample_ids_sha256'], 'Interpretation gilt für eine andere Stichprobe'
bundle=joblib.load(ROOT/'private/matrices.joblib')
summary=pd.read_csv(RES/'model_selection.csv')
runs=pd.read_csv(RES/'metrics_runs.csv')
topics=[]; choices={}; assignments={}
for method, spec in interpretation['models'].items():
    k=spec['k'];seed=spec['seed']
    model=joblib.load(ROOT/'private/models'/f'{method}_k{k}_s{seed}.joblib')
    expected_signature=sha256(ROOT/'config.json')+':'+sha256(ROOT/'private/matrices.joblib')
    assert model['input_signature']==expected_signature, 'Modell passt nicht zu den aktuellen Eingaben'
    word_hash=hashlib.sha256(json.dumps(model['words'],ensure_ascii=False).encode()).hexdigest()
    assert word_hash==spec['words_sha256'], 'Themenlabels passen nicht zum Modell'
    assert len(spec['labels'])==k
    dominant=model['w'].argmax(1)
    assignments[method]=dominant
    medoid_shares=np.bincount(dominant,minlength=k)/len(dominant)
    aligned_shares=[]
    for other_seed in (17,42,73):
        other=joblib.load(ROOT/'private/models'/f'{method}_k{k}_s{other_seed}.joblib')
        assert other['input_signature']==expected_signature, 'Vergleichsmodell passt nicht zu den aktuellen Eingaben'
        rows,cols=linear_sum_assignment(-cosine_similarity(model['h'],other['h']))
        values=np.bincount(other['w'].argmax(1),minlength=k)/len(dominant)
        matched=np.zeros(k);matched[rows]=values[cols]
        aligned_shares.append(matched)
    aligned_shares=np.array(aligned_shares)
    for t,label in enumerate(spec['labels']):
        topics.append({'method':method,'k':k,'seed':seed,'topic':t+1,'label':label,
            'top_words':', '.join(model['words'][t]),'documents':int(np.sum(dominant==t)),
            'share':float(medoid_shares[t]),'share_min_across_seeds':float(aligned_shares[:,t].min()),
            'share_max_across_seeds':float(aligned_shares[:,t].max()),
            'train_documents':int(np.sum(dominant[bundle['train']]==t)),
            'validation_documents':int(np.sum(dominant[bundle['val']]==t)),
            'interpretation':spec['notes'][t]})
    metrics=summary[(summary.method==method)&(summary.k==k)].iloc[0].to_dict()
    choices[method]={'k':k,'seed':seed,'reason':spec['reason'],'metrics':metrics}
    frame=pd.DataFrame([t for t in topics if t['method']==method]).sort_values('share')
    fig,ax=plt.subplots(figsize=(11,max(4,k*.48)),layout='constrained')
    lo=(frame.share-frame.share_min_across_seeds).clip(lower=0)*100
    hi=(frame.share_max_across_seeds-frame.share).clip(lower=0)*100
    ax.barh(frame.label,frame.share*100,xerr=[lo,hi],color='#385e79',capsize=3)
    ax.set(xlabel='Anteil der 5.000 Dokumente nach dominantem Thema (%)',
           title=f'{method.upper()}: {k} Themen, repräsentativer Startwert {seed}')
    ax.grid(axis='x',alpha=.2)
    for i,(_,row) in enumerate(frame.iterrows()):
        ax.text(row.share_max_across_seeds*100+.5,i,f'{row.share*100:.1f} %',va='center',fontsize=9)
    ax.set_xlim(0,max(frame.share_max_across_seeds)*100+6)
    fig.text(.5,-.035,'Quelle: Eigene Berechnung auf Basis von Reyes (2019), Version 1. Linien: Spannweite über drei Startwerte.',ha='center',fontsize=9)
    fig.savefig(RES/f'themen_{method}.png',dpi=180,bbox_inches='tight');plt.close(fig)

table=pd.DataFrame(topics)
table.to_csv(RES/'topics.csv',index=False)
contingency=pd.crosstab(pd.Series(assignments['lda']+1,name='LDA-Thema'),pd.Series(assignments['nmf']+1,name='NMF-Thema'))
contingency.to_csv(RES/'method_overlap.csv')
choices['comparison']={'dominant_assignment_ari_all':float(adjusted_rand_score(assignments['lda'],assignments['nmf'])),
                       'note':'ARI vergleicht Zuordnungen, ist kein Qualitätsmaß ohne Ground Truth.'}
choices['primary_method']=interpretation['primary_method']
choices['scope']='Gilt für die untersuchte Stichprobe und den geprüften Suchbereich; keine universell optimale Themenzahl.'
save_json(RES/'final_selection.json',choices)

fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
for method,ax,ylabel in [('lda',axes[0],'Validierungsperplexität'),('nmf',axes[1],'Relativer Rekonstruktionsfehler')]:
    g=summary[summary.method==method]
    ax.errorbar(g.k,g.loss_mean,yerr=g.loss_std,marker='o',capsize=3)
    ax.set(xlabel='Themenanzahl k',ylabel=ylabel,title=method.upper());ax.grid(alpha=.2)
fig.text(.5,-.04,'Quelle: Eigene Berechnung. Die zwei Skalen sind nicht zwischen den Verfahren vergleichbar.',ha='center',fontsize=9)
fig.savefig(RES/'modelldiagnostik.png',dpi=180,bbox_inches='tight');plt.close(fig)

out=['# Ergebnisse der Themenanalyse','',
     'Die Kennzahlen wurden lokal berechnet. Die Themenbezeichnungen und Inhaltsnotizen beruhen auf der Sichtung ausgewählter Textausschnitte.','',
     f"Aus {audit['total_rows']:,} Datensätzen enthalten {audit['nonempty_narratives']:,} einen Beschwerdetext. "
     f"Nach Grundbereinigung und Dublettenentfernung verbleiben {audit['unique_clean_candidates']:,} Kandidaten. "
     f"Für die Auswahl von 5.000 geeigneten Texten wurden {audit['candidates_examined']:,} zufällig geordnete Kandidaten geprüft. "
     f"Die ausgewählten Beschwerden reichen vom {audit['sample_date_min']} bis {audit['sample_date_max']}.",'',
     '## Vergleich der Vektorisierungen','',
     f"Count und TF-IDF verwenden dieselben {vectors['shape'][1]:,} Merkmale und besitzen "
     f"eine Besetzungsdichte von {vectors['density']*100:.3f} %. Beide dünnbesetzten Matrizen "
     f"benötigen jeweils {vectors['count_sparse_bytes']:,} Bytes für Werte und Indexarrays. "
     'Die Gewichtung ändert also die Geometrie, nicht die Zahl der belegten Einträge.','',
     '| Diagnose im gemeinsamen Pool von 300 Trainingsdokumenten | Count | TF-IDF |',
     '|---|---:|---:|',
     f"| Mittlere Kosinusähnlichkeit über {vectors['pair_count']:,} Paare | {vectors['cosine_count_mean']:.4f} | {vectors['cosine_tfidf_mean']:.4f} |",'',
     f"Der nächste Nachbar stimmt nur für {vectors['nearest_neighbor_agreement']*100:.2f} % der Dokumente überein. "
     'TF-IDF schwächt verbreitete Begriffe relativ ab und verändert dadurch die Nachbarschaften. '
     'Die geringere mittlere Ähnlichkeit ist für sich genommen kein Qualitätsnachweis. '
     'Count liefert diskrete Häufigkeiten für LDA; TF-IDF wird für NMF verwendet. '
     'Damit vergleicht die Untersuchung zwei komplette Kombinationen aus Darstellung und Modell. '
     'Ein Unterschied lässt sich nicht allein auf das Modellverfahren zurückführen.','',
     f"Die zusätzliche Diagnose im gesamten Korpus findet {vectors['near_duplicate_pairs_at_0_95']} "
     f"Textpaare mit TF-IDF-Kosinusähnlichkeit ab 0,95; davon liegt {vectors['near_duplicate_cross_split_pairs']} "
     'Paar über der Trainings-/Validierungsgrenze. Solche ähnlichen Texte können die Bewertung begünstigen; '
     'die exakte Dublettenbereinigung entfernt nicht jede sinngleiche Beschwerde.','',
     '## Wahl der Themenanzahl','',
     '| Verfahren | k | c_v, Mittelwert ± SD | Stabilität | Startwert |',
     '|---|---:|---:|---:|---:|']
for method in ('lda','nmf'):
    s=choices[method];m=s['metrics']
    out.append(f"| {method.upper()} | {s['k']} | {m['cv_mean']:.3f} ± {m['cv_std']:.3f} | {m['stability']:.3f} | {s['seed']} |")
out+=['','![Themenzahlvergleich](../results/themenwahl.png)','']
for method in ('lda','nmf'):
    out+=[f"**{method.upper()}:** {choices[method]['reason']}",'']
out+=['Der vollständige Vergleich steht in [model_selection.csv](../results/model_selection.csv). '
      'Die Validierung wurde für die Auswahl genutzt und ist kein unabhängiger Test. '
      'Fehlerbalken beschreiben drei Initialisierungen, keine Unsicherheit über die Bevölkerung.','',
      '## Häufigste Themen','']
for method in ('lda','nmf'):
    out += [f'### {method.upper()}','',f'![Themenanteile {method.upper()}](../results/themen_{method}.png)','',
            '| Thema | Dokumente | Anteil | Topwörter |','|---|---:|---:|---|']
    for _,r in table[table.method==method].sort_values('share',ascending=False).iterrows():
        out.append(f"| {r.label} | {r.documents} | {r.share*100:.2f} % | {r.top_words} |")
    out+=['']
out += ['## Inhaltliche Einordnung','']
for r in topics:
    out += [f"- **{r['method'].upper()} {r['topic']} - {r['label']}:** {r['interpretation']}"]
out += ['','## Vergleich und Grenzen','',interpretation['discussion'],'',
        f"Die Übereinstimmung der dominanten Zuordnungen zwischen beiden Verfahren beträgt ARI = {choices['comparison']['dominant_assignment_ari_all']:.3f}. "
        'Dies beschreibt die Übereinstimmung zweier Modelle und ist kein Genauigkeitsnachweis.','',
        '![Zusatzdiagnostik](../results/modelldiagnostik.png)','',
        'Alle Dokumentanteile beziehen sich auf genau 5.000 eindeutige ausgewählte Texte. '
        'Historische Finanzbeschwerden sind keine repräsentative Stichprobe kommunaler Anliegen. '
        'Häufigkeit misst weder Dringlichkeit noch den Wahrheitsgehalt einzelner Beschwerden.','',
        'Datenquelle: Reyes, S. (13. Mai 2019). *Consumer complaint database* (Version 1) [Datensatz]. Kaggle. '
        'https://www.kaggle.com/datasets/selener/consumer-complaint-database/versions/1','',
        'Methodenquellen und Definitionen: [methodik.md](methodik.md).']
(ROOT/'docs/ergebnisse.md').write_text('\n'.join(out)+'\n',encoding='utf-8')
print(json.dumps(choices,ensure_ascii=False,indent=2))
