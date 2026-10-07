"""Reproduzierbare Themenanalyse. Aufruf: python analyse.py --help.

Öffentliche Ergebnisse sind aggregiert; Texte, Modelle und Dokumentzuordnungen
werden nur unter den von Git ausgeschlossenen Ordnern data/ und private/ erzeugt.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import itertools
import json
import os
import re
import time
import unicodedata
import urllib.request
import warnings
import zipfile
from pathlib import Path

# Drei unabhängige Modellfits; keine zusätzliche BLAS-Überbelegung.
for env in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(env, "1")

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy import sparse
from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parent
CFG = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
PRIVATE = ROOT / "private"
RESULTS = ROOT / "results"
SOURCE_URL = "https://www.kaggle.com/api/v1/datasets/download/selener/consumer-complaint-database?datasetVersionNumber=1"
SOURCE_SHA = "a2ee5d14aae1dba483e70ae94a96af7f401a09fb1011028f44bc59a32138f453"
NEGATIONS = {"no", "not", "never", "nor"}
URL = re.compile(r"(?:https?://|www\.)\S+", re.I)
EMAIL = re.compile(r"\b[^\s@]+@[^\s@]+\.[^\s@]+\b")
MASK = re.compile(r"\bx{2,}\b", re.I)
NUMBERS = re.compile(r"\d+")


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def basic_clean(text):
    text = unicodedata.normalize("NFKC", str(text)).replace("\u2019", "'")
    text = URL.sub(" ", text)
    text = EMAIL.sub(" ", text)
    text = MASK.sub(" ", text)
    text = NUMBERS.sub(" ", text)
    return " ".join(text.split())


def normalized_factors(w, h):
    """L1-Norm der Themen = 1; das Produkt W@H bleibt dabei unverändert."""
    scales = h.sum(axis=1)
    if np.any(scales <= 0):
        raise ValueError("Leeres Thema in der Faktorisierung")
    return w * scales[None, :], h / scales[:, None]


def topic_match(a, b):
    similarity = cosine_similarity(a, b)
    rows, cols = linear_sum_assignment(-similarity)
    return float(similarity[rows, cols].mean())


def relative_error(x, w, h):
    """Frobeniusfehler ohne die große dünn besetzte Matrix zu verdichten."""
    x2 = float(x.multiply(x).sum())
    prediction2 = float(np.sum((w.T @ w) * (h @ h.T)))
    cross = float(np.sum((x @ h.T) * w))
    return float(np.sqrt(max(0.0, x2 + prediction2 - 2 * cross) / x2))


def prepare(archive):
    import spacy
    from langdetect import DetectorFactory, detect_langs
    DetectorFactory.seed = CFG["sample_seed"]
    PRIVATE.mkdir(exist_ok=True)
    RESULTS.mkdir(exist_ok=True)
    start = time.perf_counter()
    if archive is None:
        archive = ROOT / "data" / "consumer-complaint-database-v1.zip"
        archive.parent.mkdir(exist_ok=True)
        if not archive.exists():
            print("Lade historische Kaggle-Version 1 ...", flush=True)
            urllib.request.urlretrieve(SOURCE_URL, archive)
    archive = Path(archive)
    actual = sha256(archive)
    if actual != SOURCE_SHA:
        raise ValueError(f"Datenversion stimmt nicht: SHA256 {actual}")
    with zipfile.ZipFile(archive) as z:
        with z.open("rows.csv") as f:
            frame = pd.read_csv(f, usecols=["Complaint ID", "Consumer complaint narrative", "Date received", "Product"], dtype=str, keep_default_na=False)
    audit = {"archive_sha256": actual, "total_rows": len(frame)}
    text_col = "Consumer complaint narrative"
    present = frame[text_col].str.strip().ne("")
    audit["empty_narratives"] = int((~present).sum())
    frame = frame.loc[present].copy()
    audit["nonempty_narratives"] = len(frame)
    audit["duplicate_raw_texts"] = int(frame[text_col].duplicated().sum())
    print(f"Archiv: {audit['total_rows']:,} Zeilen; {len(frame):,} mit Text", flush=True)
    frame["clean"] = frame[text_col].map(basic_clean)
    frame["key"] = frame["clean"].str.casefold()
    usable = frame["key"].str.contains(r"[a-z]", regex=True)
    audit["empty_after_basic_clean"] = int((~usable).sum())
    frame = frame.loc[usable]
    audit["duplicate_normalized_texts"] = int(frame["key"].duplicated().sum())
    frame = frame.drop_duplicates("key").reset_index(drop=True)
    audit["unique_clean_candidates"] = len(frame)
    nlp = spacy.load("en_core_web_sm", disable=["parser"])
    selected, seen_tokens = [], set()
    audit.update({"candidates_examined": 0, "language_rejected": 0,
                  "short_after_nlp": 0, "duplicate_lemmas": 0})
    order = np.random.RandomState(CFG["sample_seed"]).permutation(len(frame))
    entity_types = {"PERSON", "GPE", "LOC", "FAC"}
    # Reihenfolge bleibt erhalten; nur die NLP-Verarbeitung erfolgt in Batches.
    for begin in range(0, len(order), 100):
        batch = frame.iloc[order[begin:begin + 100]]
        candidates = []
        for row in batch.itertuples(index=False, name=None):
            d = dict(zip(frame.columns, row))
            try:
                lang = detect_langs(d["clean"][:10000])[0]
                is_english = lang.lang == "en" and lang.prob >= CFG["language_min_probability"]
            except Exception:
                is_english = False
            candidates.append((d, is_english))
        docs = iter(nlp.pipe([d["clean"] for d, ok in candidates if ok], batch_size=50))
        for d, ok in candidates:
            audit["candidates_examined"] += 1
            if not ok:
                audit["language_rejected"] += 1
                continue
            doc = next(docs)
            tokens = []
            for tok in doc:
                lemma = tok.lemma_.lower()
                if tok.ent_type_ in entity_types:
                    continue
                if (tok.is_alpha or lemma in NEGATIONS) and len(lemma) > 1 and (lemma in NEGATIONS or not tok.is_stop):
                    tokens.append(lemma)
            if len(tokens) < CFG["min_tokens"]:
                audit["short_after_nlp"] += 1
                continue
            key = " ".join(tokens)
            if key in seen_tokens:
                audit["duplicate_lemmas"] += 1
                continue
            seen_tokens.add(key)
            redacted = d["clean"]
            for ent in reversed(doc.ents):
                if ent.label_ in entity_types:
                    redacted = redacted[:ent.start_char] + "[ENTFERNT]" + redacted[ent.end_char:]
            selected.append({"id": d["Complaint ID"], "date": d["Date received"],
                             "product": d["Product"], "tokens": tokens, "redacted": redacted})
            if len(selected) == CFG["sample_size"]:
                break
        if len(selected) % 500 < 100:
            print(f"Geeignete Texte: {len(selected)}/{CFG['sample_size']}", flush=True)
        if len(selected) == CFG["sample_size"]:
            break
    if len(selected) != CFG["sample_size"]:
        raise ValueError("Nicht genügend geeignete Texte")
    joblib.dump(selected, PRIVATE / "sample.joblib", compress=3)
    audit["sample_size"] = len(selected)
    audit["sample_ids_sha256"] = hashlib.sha256("\n".join(d["id"] for d in selected).encode()).hexdigest()
    lengths = np.array([len(d["tokens"]) for d in selected])
    audit["tokens_min_median_max"] = [int(lengths.min()), float(np.median(lengths)), int(lengths.max())]
    dates = pd.to_datetime([d["date"] for d in selected], format="%m/%d/%Y")
    audit["sample_date_min"] = str(dates.min().date())
    audit["sample_date_max"] = str(dates.max().date())
    audit["seconds"] = round(time.perf_counter() - start, 2)
    save_json(RESULTS / "data_audit.json", audit)
    print(json.dumps(audit, indent=2), flush=True)
    vectorize()


def vectorize():
    sample = joblib.load(PRIVATE / "sample.joblib")
    strings = [" ".join(d["tokens"]) for d in sample]
    train, val = train_test_split(np.arange(len(sample)), test_size=CFG["validation_fraction"], random_state=CFG["split_seed"])
    vectorizer = CountVectorizer(lowercase=False, tokenizer=str.split, token_pattern=None,
                                min_df=CFG["min_df"], max_df=CFG["max_df"], max_features=CFG["max_features"])
    vectorizer.fit([strings[i] for i in train])
    counts = vectorizer.transform(strings)
    tfidf_model = TfidfTransformer(norm="l2", smooth_idf=True, sublinear_tf=False)
    tfidf_model.fit(counts[train])
    tfidf = tfidf_model.transform(counts)
    features = vectorizer.get_feature_names_out()
    bundle = dict(train=train, val=val, counts=counts, tfidf=tfidf, features=features,
                  vectorizer=vectorizer, tfidf_model=tfidf_model)
    joblib.dump(bundle, PRIVATE / "matrices.joblib", compress=3)
    assert np.all(np.asarray(counts.sum(axis=1)).ravel() > 0), "Leere Modellzeile"
    assert counts.shape == tfidf.shape
    assert counts.nnz == tfidf.nnz
    # Gleiche 300 Dokumentpaare für den Darstellungsvergleich, diagonal ausgeschlossen.
    idx = np.random.RandomState(42).choice(train, min(300, len(train)), replace=False)
    a = cosine_similarity(counts[idx]); b = cosine_similarity(tfidf[idx])
    mask = np.triu_indices(len(idx), 1)
    np.fill_diagonal(a, -1); np.fill_diagonal(b, -1)
    # Nahe Dubletten: maximale paarweise Kosinusähnlichkeit; Chunking vermeidet n²-RAM.
    near_pairs = 0; cross_pairs = 0
    train_set = set(train.tolist())
    xn = normalize(tfidf)
    for start in range(0, len(sample), 250):
        sim = (xn[start:start+250] @ xn.T).tocoo()
        for row, col, value in zip(sim.row, sim.col, sim.data):
            i = start + int(row); j = int(col)
            if j > i and value >= .95:
                near_pairs += 1
                cross_pairs += int((i in train_set) != (j in train_set))
    report = {"train_documents": len(train), "validation_documents": len(val),
              "shape": list(counts.shape), "nonzero_entries": counts.nnz,
              "density": counts.nnz / np.prod(counts.shape),
              "count_sparse_bytes": int(counts.data.nbytes+counts.indices.nbytes+counts.indptr.nbytes),
              "tfidf_sparse_bytes": int(tfidf.data.nbytes+tfidf.indices.nbytes+tfidf.indptr.nbytes),
              "pair_count": len(mask[0]), "cosine_count_mean": float(a[mask].mean()),
              "cosine_tfidf_mean": float(b[mask].mean()),
              "nearest_neighbor_agreement": float(np.mean(a.argmax(1)==b.argmax(1))),
              "near_duplicate_pairs_at_0_95": near_pairs,
              "near_duplicate_cross_split_pairs": cross_pairs,
              "top_count_terms": features[np.asarray(counts[train].sum(0)).ravel().argsort()[-15:][::-1]].tolist(),
              "top_tfidf_terms": features[np.asarray(tfidf[train].sum(0)).ravel().argsort()[-15:][::-1]].tolist()}
    save_json(RESULTS / "vectorization.json", report)
    print("Vektorisierung abgeschlossen", flush=True)


def fit_one(method, k, seed):
    path = PRIVATE / "models" / f"{method}_k{k}_s{seed}.joblib"
    signature = sha256(ROOT / "config.json") + ":" + sha256(PRIVATE / "matrices.joblib")
    if path.exists():
        if joblib.load(path).get("input_signature") == signature:
            return str(path)
    bundle = joblib.load(PRIVATE / "matrices.joblib")
    train, val = bundle["train"], bundle["val"]
    x = bundle["counts"] if method == "lda" else bundle["tfidf"]
    start = time.perf_counter()
    with threadpool_limits(limits=1), warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        if method == "lda":
            model = LatentDirichletAllocation(n_components=k, random_state=seed,
                learning_method="batch", max_iter=CFG["lda_max_iter"], n_jobs=1,
                doc_topic_prior=CFG["lda_doc_topic_prior"], topic_word_prior=CFG["lda_topic_word_prior"],
                evaluate_every=5, perp_tol=.1)
            model.fit(x[train])
            w = model.transform(x)
            h = normalize(model.components_, norm="l1", axis=1)
            loss = float(model.perplexity(x[val]))
        else:
            model = NMF(n_components=k, init="random", random_state=seed,
                        max_iter=CFG["nmf_max_iter"], tol=1e-4, solver="cd")
            model.fit(x[train])
            original_w = model.transform(x)
            loss = relative_error(x[val], original_w[val], model.components_)
            w, h = normalized_factors(original_w, model.components_)
            w = normalize(w, norm="l1", axis=1)
        top_idx = np.argsort(h, axis=1)[:, -CFG["top_words"]:][:, ::-1]
        words = bundle["features"][top_idx].tolist()
        shares = np.bincount(w[val].argmax(1), minlength=k) / len(val)
        similarities = cosine_similarity(h)
        np.fill_diagonal(similarities, -1)
        result = dict(method=method, k=k, seed=seed, model=model, w=w, h=h, words=words,
                      input_signature=signature,
                      loss=loss, seconds=time.perf_counter()-start,
                      n_iter=int(model.n_iter_), warnings=[str(s.message) for s in captured],
                      min_share=float(shares.min()), max_topic_cosine=float(similarities.max()),
                      diversity=len(set(itertools.chain.from_iterable(words))) / (k*CFG["top_words"]))
    path.parent.mkdir(exist_ok=True)
    joblib.dump(result, path, compress=3)
    print(f"{method.upper()} k={k} seed={seed}: {result['seconds']:.1f}s ({result['n_iter']} Iterationen)", flush=True)
    return str(path)


def sweep(ks, methods=("lda", "nmf")):
    jobs = [(m,k,s) for m in methods for k in ks for s in CFG["model_seeds"]]
    joblib.Parallel(n_jobs=CFG["workers"], backend="loky")(
        joblib.delayed(fit_one)(*args) for args in jobs)
    evaluate()


def evaluate():
    from gensim.corpora import Dictionary
    from gensim.models import CoherenceModel
    paths = sorted((PRIVATE / "models").glob("*.joblib"))
    models = [joblib.load(p) for p in paths]
    expected_signature = sha256(ROOT / "config.json") + ":" + sha256(PRIVATE / "matrices.joblib")
    if not models or any(m.get("input_signature") != expected_signature for m in models):
        raise ValueError("Modellcache fehlt oder enthält andere Eingaben. Neue Konfiguration in einem eigenen Arbeitsordner ausführen.")
    bundle = joblib.load(PRIVATE / "matrices.joblib")
    sample = joblib.load(PRIVATE / "sample.joblib")
    features = bundle["features"].tolist(); vocabulary = set(features)
    texts = [[t for t in sample[i]["tokens"] if t in vocabulary] for i in bundle["val"]]
    dictionary = Dictionary([features])
    all_topics = [topic for m in models for topic in m["words"]]
    print(f"Berechne c_v für {len(models)} Modelle auf {len(texts)} Validierungstexten ...", flush=True)
    cm = CoherenceModel(topics=all_topics, texts=texts, dictionary=dictionary,
                        coherence="c_v", topn=CFG["top_words"], window_size=CFG["coherence_window"], processes=1)
    with warnings.catch_warnings(record=True) as coherence_warnings:
        warnings.simplefilter('always')
        coherences = cm.get_coherence_per_topic()
    seen_validation = set(itertools.chain.from_iterable(texts))
    offset=0; rows=[]
    for m in models:
        per_topic = np.asarray(coherences[offset:offset+m["k"]]); offset+=m["k"]
        row = {key:m[key] for key in ["method","k","seed","loss","seconds","n_iter","min_share","max_topic_cosine","diversity"]}
        missing_words=sorted(set(itertools.chain.from_iterable(m['words']))-seen_validation)
        row.update(cv=float(per_topic.mean()), cv_min=float(per_topic.min()), warning_count=len(m["warnings"]),
                   undefined_topic_coherences=int((~np.isfinite(per_topic)).sum()),
                   missing_validation_topwords=';'.join(missing_words))
        rows.append(row)
    metrics = pd.DataFrame(rows)
    metrics.to_csv(RESULTS / "metrics_runs.csv", index=False)
    groups=[]
    for (method,k), group in metrics.groupby(["method","k"]):
        ms = [m for m in models if m["method"]==method and m["k"]==k]
        scores={m["seed"]:[] for m in ms}; pair_scores=[]; aris=[]
        for a,b in itertools.combinations(ms,2):
            value=topic_match(a["h"],b["h"])
            pair_scores.append(value); scores[a["seed"]].append(value); scores[b["seed"]].append(value)
            aris.append(adjusted_rand_score(a["w"][bundle['val']].argmax(1),b["w"][bundle['val']].argmax(1)))
        medoid=sorted(scores,key=lambda s:(-np.mean(scores[s]),s))[0]
        groups.append(dict(method=method,k=int(k),cv_mean=float(np.mean(group.cv.to_numpy())),cv_std=float(np.std(group.cv.to_numpy(),ddof=1)),
            stability=float(np.mean(pair_scores)),stability_min=float(min(pair_scores)),ari=float(np.mean(aris)),
            loss_mean=float(group.loss.mean()),loss_std=float(group.loss.std(ddof=1)),
            diversity=float(group.diversity.mean()),min_share=float(group.min_share.min()),
            max_topic_cosine=float(group.max_topic_cosine.max()),medoid_seed=int(medoid),
            warnings=int(group.warning_count.sum()),undefined_topic_coherences=int(group.undefined_topic_coherences.sum())))
    summary=pd.DataFrame(groups)
    summary.to_csv(RESULTS / "model_selection.csv",index=False)
    choices={}
    for method, group in summary.groupby("method"):
        best=float(group.cv_mean.max())
        eligible=group[(group.cv_mean>=best-CFG["coherence_tolerance"]) &
                       (group.stability>=CFG["minimum_stability"]) &
                       (group.min_share>=CFG["minimum_topic_share"]) &
                       (group.max_topic_cosine<=CFG["maximum_topic_cosine"])]
        passed=len(eligible)>0
        chosen=(eligible.sort_values("k") if passed else group.sort_values(["cv_mean","stability"],ascending=False)).iloc[0]
        choices[method]={"suggested_k":int(chosen.k),"medoid_seed":int(chosen.medoid_seed),
                         "all_heuristic_criteria_met":passed,"best_cv_k":int(group.loc[group.cv_mean.idxmax(),"k"]),
                         "status":"Vorauswahl; inhaltliche Prüfung erforderlich"}
    save_json(RESULTS / "suggestions.json",choices)
    save_json(RESULTS / "coherence_diagnostics.json", {
        'policy':'Nicht berechenbare Themenkohärenz bleibt undefiniert; betroffene k-Kandidaten werden nicht in die metrische Vorauswahl aufgenommen.',
        'warnings':sorted(set(str(w.message) for w in coherence_warnings)),
        'affected_runs':[{'method':r['method'],'k':r['k'],'seed':r['seed'],
                          'undefined_topics':r['undefined_topic_coherences'],
                          'missing_validation_topwords':r['missing_validation_topwords']}
                         for r in rows if r['undefined_topic_coherences']>0]})
    public_topics=[{key:m[key] for key in ("method","k","seed","words","warnings")} for m in models]
    save_json(RESULTS / "candidate_topics.json",public_topics)
    print(summary.round(4).to_string(index=False),flush=True)
    print(json.dumps(choices,indent=2),flush=True)
    plot_selection(summary)
    make_review_examples(choices)


def plot_selection(df):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes=plt.subplots(2,2,figsize=(11,8),layout="constrained")
    for method,g in df.groupby("method"):
        axes[0,0].errorbar(g.k,g.cv_mean,yerr=g.cv_std,marker="o",capsize=3,label=method.upper())
        axes[0,1].plot(g.k,g.stability,marker="o",label=method.upper())
        axes[1,0].plot(g.k,g.diversity,marker="o",label=method.upper())
        axes[1,1].plot(g.k,g.min_share*100,marker="o",label=method.upper())
    for ax,label in zip(axes.flat,["c_v: Mittelwert +/- Standardabweichung","Themenstabilität (Kosinus, zugeordnet)","Vielfalt der Top-10-Wörter","Kleinster dominanter Themenanteil (%)"]):
        ax.set(xlabel="Themenanzahl k",ylabel=label);ax.grid(alpha=.2);ax.legend()
    fig.suptitle("Themenzahlvergleich: 4.000 Training / 1.000 Validierung, drei Startwerte")
    fig.text(.5,-.025,"Quelle: Eigene Berechnung auf Basis von Reyes (2019), Version 1; CFPB-Beschwerden.",ha="center",fontsize=9)
    fig.savefig(RESULTS / "themenwahl.png",dpi=180,bbox_inches="tight");plt.close(fig)


def make_review_examples(choices):
    sample=joblib.load(PRIVATE / "sample.joblib")
    rows=[]
    for method,ch in choices.items():
        m=joblib.load(PRIVATE / "models" / f"{method}_k{ch['suggested_k']}_s{ch['medoid_seed']}.joblib")
        dom=m["w"].argmax(1)
        for t,words in enumerate(m["words"]):
            ids=np.flatnonzero(dom==t)
            strongest=ids[np.argsort(m["w"][ids,t])[-3:][::-1]]
            remainder=np.setdiff1d(ids,strongest)
            random=np.random.RandomState(42+t).choice(remainder,min(2,len(remainder)),replace=False)
            for typ,indices in [("stark",strongest),("zufällig",random)]:
                for i in indices:
                    rows.append(dict(method=method,k=ch["suggested_k"],seed=ch["medoid_seed"],topic=t+1,
                                     words=words,example_type=typ,index=int(i),weight=float(m["w"][i,t]),
                                     text=sample[i]["redacted"]))
    save_json(PRIVATE / "review_examples.json",rows)


def refine():
    choices=json.loads((RESULTS / "suggestions.json").read_text())
    for method,ch in choices.items():
        ks=set()
        for k in [ch["suggested_k"],ch["best_cv_k"]]:
            ks.update([max(2,k-1),k+1])
        if ch["best_cv_k"]==max(CFG["initial_k"]):
            ks.update([25,30])
        sweep(sorted(ks),methods=[method])
    # Falls sich das Maximum durch die Verfeinerung verschiebt, werden auch
    # seine unmittelbaren Nachbarn ergänzt (begrenzt auf den Bereich 2..30).
    for _ in range(3):
        choices=json.loads((RESULTS/'suggestions.json').read_text(encoding='utf-8'))
        summary=pd.read_csv(RESULTS/'model_selection.csv')
        pending=[]
        for method,ch in choices.items():
            existing=set(summary.loc[summary.method==method,'k'])
            wanted={k+d for k in (ch['suggested_k'],ch['best_cv_k']) for d in (-1,1)}
            missing=sorted(k for k in wanted-existing if 2<=k<=30)
            if missing: pending.append((method,missing))
        if not pending: break
        for method,ks in pending: sweep(ks,methods=[method])


def environment():
    import platform
    names=["numpy","scipy","pandas","scikit-learn","spacy","en-core-web-sm","gensim","langdetect","matplotlib","joblib","threadpoolctl"]
    save_json(RESULTS/"environment.json",{"python":platform.python_version(),"platform":platform.platform(),
        "packages":{n:importlib.metadata.version(n) for n in names},"config_sha256":sha256(ROOT/"config.json")})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",choices=["prepare","sweep","refine","evaluate","environment"])
    parser.add_argument("--archive",type=Path,help="Bereits vorhandenes ZIP der exakt gleichen Datenversion")
    parser.add_argument("--k",type=int,nargs="+",help="Optionales Themenzahlgitter")
    args=parser.parse_args()
    PRIVATE.mkdir(exist_ok=True);RESULTS.mkdir(exist_ok=True)
    if args.command=="prepare": prepare(args.archive)
    elif args.command=="sweep": sweep(args.k or CFG["initial_k"])
    elif args.command=="refine": refine()
    elif args.command=="evaluate": evaluate()
    elif args.command=="environment": environment()


if __name__=="__main__":
    main()
