"""Approved experiment. Selection uses clean development data only.

No test predictions are made until both configurations have been selected.
Text grouping is a label-independent leakage-control operation, not a model feature.
"""
from __future__ import annotations

import hashlib
import importlib.metadata as metadata
import io
import json
import os
from pathlib import Path
import platform
import random
import re
import time
import urllib.request
import zipfile

_CACHE_ROOT = Path(__file__).resolve().parents[1] / ".cache"
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE_ROOT / "matplotlib"))
os.environ.setdefault("HF_HOME", str(_CACHE_ROOT / "huggingface"))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import triu
from scipy.stats import binomtest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from wordfreq import zipf_frequency

SEED = 20261116
MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
DATA_URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
EXPECTED_ARCHIVE_SHA256 = "1587ea43e58e82b14ff1f5425c88e17f8496bfcdb67a583dbff9eefaf9963ce3"
DATA_PAGE = "https://archive.ics.uci.edu/dataset/228/sms+spam+collection"
LABELS = {0: "Not spam", 1: "Spam"}
ROOT = Path(__file__).resolve().parents[1]


def save_json(value, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare_dirs(root=ROOT):
    for d in ["data/raw", "data/processed", "models", "results/figures", "results/predictions"]:
        (root / d).mkdir(parents=True, exist_ok=True)


def acquire_data(root=ROOT):
    """Use original bytes, not pandas' quote parser, to preserve all SMS records."""
    prepare_dirs(root)
    archive = root / "data/raw/uci_sms_spam_collection.zip"
    if not archive.exists():
        with urllib.request.urlopen(DATA_URL, timeout=90) as response:
            archive.write_bytes(response.read())
    if sha256(archive) != EXPECTED_ARCHIVE_SHA256:
        raise ValueError("UCI archive checksum differs from the experiment. Check the source.")
    with zipfile.ZipFile(archive) as z:
        names = z.namelist()
        raw = z.read(next(n for n in names if n.endswith("SMSSpamCollection")))
        (root / "data/raw/SMSSpamCollection").write_bytes(raw)
        for n in names:
            if n.lower().endswith("readme"):
                (root / "data/raw/original_readme.txt").write_bytes(z.read(n))
    records, malformed = [], []
    for line_no, line in enumerate(raw.decode("utf-8").splitlines(), 1):
        if "\t" not in line:
            malformed.append({"line": line_no, "reason": "no tab"})
            continue
        label, text = line.split("\t", 1)
        if label not in ("ham", "spam") or not text.strip():
            malformed.append({"line": line_no, "reason": "invalid label or empty text"})
            continue
        records.append({"row_id": line_no, "label": int(label == "spam"), "text": text})
    df = pd.DataFrame(records)
    conflict = df.groupby("text").label.nunique()
    conflicting = set(conflict[conflict > 1].index)
    excluded = df[df.text.isin(conflicting)]
    excluded.to_csv(root / "data/processed/conflicting_labels.csv", index=False)
    # Conflicting labels would be quarantined before splitting, never resolved from predictions.
    clean = df[~df.text.isin(conflicting)].drop_duplicates("text", keep="first").copy()
    stats = {
        "source": DATA_PAGE, "download_url": DATA_URL,
        "archive_sha256": sha256(archive), "raw_sha256": hashlib.sha256(raw).hexdigest(),
        "raw_records": len(df), "physical_lines": len(raw.decode("utf-8").splitlines()),
        "raw_class_counts": {str(k): int(v) for k,v in df.label.value_counts().sort_index().items()},
        "malformed_records": malformed, "missing_text": int(df.text.isna().sum()),
        "exact_duplicate_rows": int(df.duplicated("text").sum()),
        "conflicting_unique_texts": len(conflicting), "quarantined_rows": len(excluded),
        "clean_records": len(clean),
        "clean_class_counts": {str(k): int(v) for k,v in clean.label.value_counts().sort_index().items()},
    }
    save_json(stats, root / "results/data_audit.json")
    return clean.reset_index(drop=True), stats


def canonical_text(text):
    """Grouping only: case/space, URLs, email addresses, and digit sequences."""
    s = re.sub(r"(?:https?://\S+|www\.\S+)", " URL ", text.lower())
    s = re.sub(r"\b[\w.+-]+@[\w.-]+\.[a-z]+\b", " EMAIL ", s)
    s = re.sub(r"\d+", " NUMBER ", s)
    return " ".join(s.split())


def group_and_split(df, root=ROOT):
    """Group duplicate templates and near duplicates before selecting 14/3/3 folds.

    Complete graph edges have cosine >= .92 in char 3-5gram TF-IDF space.
    Connected components stay together. This uses text but never target labels.
    The 20-fold stratified group assignment uses labels only to balance partitions.
    """
    df = df.copy()
    canonical = df.text.map(canonical_text)
    vec = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), min_df=1,
                          dtype=np.float32, lowercase=False)
    x = vec.fit_transform(canonical)
    sim = triu(x @ x.T, k=1).tocoo()
    pairs = np.flatnonzero(sim.data >= .92)
    parent = np.arange(len(df))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a,b):
        a,b = find(int(a)), find(int(b))
        if a != b:
            parent[max(a,b)] = min(a,b)

    seen = {}
    for i, s in enumerate(canonical):
        if s in seen:
            union(i,seen[s])
        else:
            seen[s] = i
    for p in pairs:
        union(sim.row[p], sim.col[p])
    df["group_id"] = [int(df.iloc[find(i)].row_id) for i in range(len(df))]
    fold = np.full(len(df), -1)
    splitter = StratifiedGroupKFold(n_splits=20, shuffle=True, random_state=SEED)
    for f, (_,idx) in enumerate(splitter.split(df.text, df.label, df.group_id)):
        fold[idx] = f
    df["split"] = np.where(fold < 14, "train", np.where(fold < 17, "development", "test"))
    assert not df.text.duplicated().any()
    assert df.groupby("group_id").split.nunique().max() == 1
    cross = [(int(sim.row[p]), int(sim.col[p])) for p in pairs
             if df.iloc[sim.row[p]]["split"] != df.iloc[sim.col[p]]["split"]]
    assert len(cross) == 0
    audit = {
        "seed": SEED, "target_ratios": {"train": .70,"development": .15,"test": .15},
        "strategy": "20 stratified group folds; folds 0-13 train, 14-16 development, 17-19 test",
        "groups": int(df.group_id.nunique()), "near_duplicate_edges_ge_0_92": len(pairs),
        "largest_group": int(df.group_id.value_counts().max()),
        "exact_text_overlap": 0, "group_overlap": 0, "near_duplicate_cross_split_edges": 0,
        "counts": {split: {"total": len(g), "ham": int((g.label==0).sum()),
                           "spam": int((g.label==1).sum())} for split,g in df.groupby("split")},
        "limitations": "No campaign or sender identifiers exist. Similarity grouping reduces, but cannot rule out, all semantic relationships."
    }
    df.to_csv(root / "data/processed/split_manifest.csv", index=False)
    save_json(audit,root / "results/split_audit.json")
    return df, audit


def measure(y,p):
    y,p = np.asarray(y,dtype=int), np.asarray(p,dtype=int)
    tn,fp,fn,tp = confusion_matrix(y,p,labels=[0,1]).ravel()
    return {"n": len(y), "accuracy": float(accuracy_score(y,p)),
            "macro_f1": float(f1_score(y,p,average="macro",zero_division=0)),
            "spam_precision": float(precision_score(y,p,zero_division=0)),
            "spam_recall": float(recall_score(y,p,zero_division=0)),
            "spam_f1": float(f1_score(y,p,zero_division=0)),
            "false_positives": int(fp), "false_positive_rate": float(fp/(tn+fp)) if tn+fp else 0.,
            "false_negatives": int(fn), "tn": int(tn), "tp": int(tp)}


def choose_best(rows):
    # Highest dev macro F1, then recall, then fewer false positives; stable grid order breaks remaining ties.
    return max(range(len(rows)),key=lambda i:(rows[i]["macro_f1"],rows[i]["spam_recall"],-rows[i]["false_positives"]))


def train_tfidf(df, root=ROOT):
    train,dev = (df[df.split==s] for s in ("train","development"))
    candidates,models = [],[]
    for ngrams in [(1,1),(1,2)]:
        for c in [.5,2.,8.]:
            for weight in [None,"balanced"]:
                model = Pipeline([
                    ("tfidf",TfidfVectorizer(ngram_range=ngrams,sublinear_tf=True,min_df=1,
                                            lowercase=True,strip_accents=None)),
                    ("classifier",LogisticRegression(C=c,class_weight=weight,solver="liblinear",
                                                     max_iter=2000,random_state=SEED))])
                model.fit(train.text,train.label)
                score = measure(dev.label,model.predict(dev.text))
                candidates.append({"ngrams":str(ngrams),"C":c,"class_weight":str(weight),**score})
                models.append(model)
    best = choose_best(candidates)
    model=models[best]
    joblib.dump(model,root / "models/tfidf_logreg.joblib")
    pd.DataFrame(candidates).to_csv(root / "results/tfidf_development_grid.csv",index=False)
    result = {"chosen_index":best,"selected":candidates[best],
              "vocabulary_size":len(model.named_steps["tfidf"].vocabulary_),
              "threshold":.5,"refit_on_development":False}
    save_json(result,root / "results/tfidf_selection.json")
    return model,result


def load_encoder(root=ROOT):
    import torch
    from huggingface_hub import model_info
    from sentence_transformers import SentenceTransformer
    torch.manual_seed(SEED)
    torch.set_num_threads(4)
    os.environ.setdefault("TOKENIZERS_PARALLELISM","false")
    local = root / "models/minilm_encoder"
    meta_path = root / "models/encoder_metadata.json"
    if local.exists() and meta_path.exists():
        info = json.loads(meta_path.read_text(encoding="utf-8"))
        encoder = SentenceTransformer(str(local),device="cpu",local_files_only=True)
    else:
        revision = (json.loads(meta_path.read_text(encoding="utf-8"))["revision"]
                    if meta_path.exists() else model_info(MODEL_ID).sha)
        encoder = SentenceTransformer(MODEL_ID,revision=revision,device="cpu")
        encoder.save(str(local))
        info = {"model_id":MODEL_ID,"revision":revision,"embedding_dimension":384,
                "device":"cpu","normalize_embeddings":True,"max_seq_length":encoder.max_seq_length,
                "frozen":True,"source":"https://huggingface.co/"+MODEL_ID}
        save_json(info,meta_path)
    encoder.eval()
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    assert all(not p.requires_grad for p in encoder.parameters())
    assert encoder.get_sentence_embedding_dimension() == 384
    info["parameter_count"] = sum(p.numel() for p in encoder.parameters())
    for name in ["model.safetensors", "pytorch_model.bin"]:
        weight_file = local / name
        if weight_file.exists():
            info["weight_file"] = name
            info["weights_sha256"] = sha256(weight_file)
            break
    save_json(info,meta_path)
    return encoder,info


def embed(encoder,texts):
    return encoder.encode(list(texts),batch_size=64,normalize_embeddings=True,
                          show_progress_bar=False,convert_to_numpy=True)


def train_minilm(df,encoder,root=ROOT):
    train,dev = (df[df.split==s] for s in ("train","development"))
    xt,xd=embed(encoder,train.text),embed(encoder,dev.text)
    candidates,models=[],[]
    for c in [.5,2.,8.,32.]:
        for weight in [None,"balanced"]:
            model=LogisticRegression(C=c,class_weight=weight,solver="liblinear",max_iter=2000,random_state=SEED)
            model.fit(xt,train.label)
            candidates.append({"C":c,"class_weight":str(weight),**measure(dev.label,model.predict(xd))})
            models.append(model)
    best=choose_best(candidates)
    joblib.dump(models[best],root / "models/minilm_logreg.joblib")
    pd.DataFrame(candidates).to_csv(root / "results/minilm_development_grid.csv",index=False)
    result={"chosen_index":best,"selected":candidates[best],"threshold":.5,"refit_on_development":False}
    save_json(result,root / "results/minilm_selection.json")
    return models[best],result


PROTECTED = re.compile(r"(?:https?://\S+|www\.\S+|\b\S+@\S+\b|\b(?:[a-z0-9-]+\.)+[a-z]{2,}(?:/\S*)?|\b\w*\d\w*\b)",re.I)
WORDS = re.compile(r"(?<![\w@])[A-Za-z]{5,}(?![\w@])")


def typo_variant(text,row_id,kind):
    """One internal edit per message; exclude common real-word outputs.

    The frozen wordfreq English lexicon is used only as a safety filter. It is
    not learned from this corpus or selected using predictions. No spell correction.
    """
    rng = random.Random(f"{SEED}:{row_id}:{kind}")
    protected = [m.span() for m in PROTECTED.finditer(text)]
    candidates=[]
    for m in WORDS.finditer(text):
        if any(m.start()<b and m.end()>a for a,b in protected):
            continue
        word=m.group()
        if zipf_frequency(word.lower(),"en") < 3.0:
            continue
        variants=[]
        if kind=="deletion":
            for i in range(1,len(word)-1):
                v=word[:i]+word[i+1:]
                if zipf_frequency(v.lower(),"en") < 2.5:
                    variants.append((v,i))
        elif kind=="transposition":
            for i in range(1,len(word)-2):
                if word[i].lower()==word[i+1].lower():
                    continue
                v=word[:i]+word[i+1]+word[i]+word[i+2:]
                if zipf_frequency(v.lower(),"en") < 2.5:
                    variants.append((v,i))
        else:
            raise ValueError(kind)
        if variants:
            candidates.append((m,variants))
    if not candidates:
        return {"text_variant":text,"modified":False,"operation":"none","original_word":"",
                "replacement_word":"","word_start":-1,"character_index":-1}
    match, variants=rng.choice(candidates)
    replacement,index=rng.choice(variants)
    return {"text_variant":text[:match.start()]+replacement+text[match.end():],"modified":True,
            "operation":kind,"original_word":match.group(),"replacement_word":replacement,
            "word_start":match.start(),"character_index":index}


def make_variants(test,root=ROOT):
    variants={}
    for kind in ["deletion","transposition","mixed"]:
        records=[]
        for row in test.itertuples():
            operation=kind
            if kind=="mixed":
                operation=random.Random(f"{SEED}:{row.row_id}:mixed").choice(["deletion","transposition"])
            records.append({"row_id":row.row_id,"label":row.label,"text":row.text,
                            **typo_variant(row.text,row.row_id,operation)})
        variant=pd.DataFrame(records)
        variant.to_csv(root / f"data/processed/test_{kind}.csv",index=False)
        variants[kind]=variant
    return variants


def evaluate(df,tfidf,encoder,minilm,variants,root=ROOT):
    test=df[df.split=="test"].reset_index(drop=True)
    conditions={"clean":test.assign(text_variant=test.text,modified=False),**variants}
    metrics=[]
    for condition,frame in conditions.items():
        text=frame.text_variant
        embeddings=embed(encoder,text)
        probabilities={"TF-IDF":tfidf.predict_proba(text)[:,1],"MiniLM":minilm.predict_proba(embeddings)[:,1]}
        for model,prob in probabilities.items():
            pred=(prob>=.5).astype(int)
            output=frame.copy()
            output["prediction"]=pred
            output["spam_score_uncalibrated"]=prob
            output["correct"]=output.label==pred
            output.to_csv(root / f"results/predictions/{model}_{condition}.csv",index=False)
            metrics.append({"model":model,"condition":condition,"cohort":"all",**measure(frame.label,pred)})
            if condition!="clean":
                eligible=frame.modified.to_numpy(bool)
                metrics.append({"model":model,"condition":condition,"cohort":"modified_only",
                                **measure(frame.label[eligible],pred[eligible])})
                clean_pred=pd.read_csv(root / f"results/predictions/{model}_clean.csv").prediction.to_numpy()
                metrics.append({"model":model,"condition":condition+"_clean_counterpart","cohort":"modified_only",
                                **measure(frame.label[eligible],clean_pred[eligible])})
    table=pd.DataFrame(metrics)
    table.to_csv(root / "results/metrics.csv",index=False)
    return table


def paired_analysis(root=ROOT):
    rng=np.random.default_rng(SEED)
    result=[]
    for condition in ["clean","deletion","transposition","mixed"]:
        a=pd.read_csv(root / f"results/predictions/TF-IDF_{condition}.csv")
        b=pd.read_csv(root / f"results/predictions/MiniLM_{condition}.csv")
        assert a.row_id.equals(b.row_id) and a.label.equals(b.label)
        mask=(a.label==1).to_numpy()
        if condition!="clean":
            mask &= a.modified.to_numpy(bool)
        aa=(a.prediction.to_numpy()[mask]==1).astype(int)
        bb=(b.prediction.to_numpy()[mask]==1).astype(int)
        diff=bb-aa
        bootstrap=diff[rng.integers(0,len(diff),size=(5000,len(diff)))].mean(axis=1)
        only_a=int(((aa==1)&(bb==0)).sum())
        only_b=int(((aa==0)&(bb==1)).sum())
        result.append({"condition":condition,"spam_n":len(diff),"tfidf_only_correct":only_a,
                       "minilm_only_correct":only_b,"recall_difference_minilm_minus_tfidf":float(diff.mean()),
                       "paired_bootstrap_95_low":float(np.quantile(bootstrap,.025)),
                       "paired_bootstrap_95_high":float(np.quantile(bootstrap,.975)),
                       "mcnemar_exact_p_exploratory":float(binomtest(only_b,only_a+only_b).pvalue) if only_a+only_b else 1.})
    pd.DataFrame(result).to_csv(root / "results/paired_spam_analysis.csv",index=False)
    return pd.DataFrame(result)


def make_figures(df,table,root=ROOT):
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"axes.spines.top":False,
                         "axes.spines.right":False,"figure.dpi":130,"savefig.bbox":"tight"})
    colors=["#176B87","#D06439"]
    fig,axs=plt.subplots(1,2,figsize=(9,3.2))
    counts=df.label.value_counts().sort_index()
    axs[0].bar(["Not spam","Spam"],counts,color=colors)
    axs[0].set(title="Unique SMS messages",ylabel="Messages")
    for i,v in enumerate(counts): axs[0].text(i,v+30,str(v),ha="center")
    df.assign(length=df.text.str.len()).boxplot(column="length",by="label",ax=axs[1],showfliers=False,grid=False)
    axs[1].set(title="Message lengths",xlabel="Class",ylabel="Characters",xticklabels=["Not spam","Spam"])
    fig.suptitle(""); fig.tight_layout(); fig.savefig(root / "results/figures/data_overview.png"); plt.close(fig)
    all_rows=table[table.cohort=="all"]
    fig,axs=plt.subplots(1,3,figsize=(11,3.3))
    for ax,metric,title in zip(axs,["accuracy","macro_f1","spam_recall"],["Accuracy","Macro F1","Spam recall"]):
        for i,model in enumerate(["TF-IDF","MiniLM"]):
            vals=all_rows[all_rows.model==model].set_index("condition").loc[["clean","deletion","transposition","mixed"],metric]
            ax.plot(range(4),vals,marker="o",color=colors[i],label=model)
        ax.set(title=title,ylim=(max(0,all_rows[metric].min()-.035),1.005),xticks=range(4),
               xticklabels=["Clean","Delete","Swap","Mixed"])
        ax.grid(axis="y",alpha=.2)
    axs[0].legend(frameon=False); fig.tight_layout()
    fig.savefig(root / "results/figures/performance_comparison.png"); plt.close(fig)
    fig,axs=plt.subplots(2,2,figsize=(7.6,6.1))
    for ax,(model,condition) in zip(axs.flat,[(m,c) for m in ["TF-IDF","MiniLM"] for c in ["clean","mixed"]]):
        r=all_rows[(all_rows.model==model)&(all_rows.condition==condition)].iloc[0]
        matrix=np.array([[r.tn,r.false_positives],[r.false_negatives,r.tp]],dtype=int)
        ax.imshow(matrix,cmap="Blues")
        for (i,j),v in np.ndenumerate(matrix): ax.text(j,i,str(v),ha="center",va="center",color="white" if v>matrix.max()/2 else "#123044",fontsize=13)
        ax.set(title=f"{model} | {condition}",xticks=[0,1],yticks=[0,1],
               xticklabels=["Not spam","Spam"],yticklabels=["Not spam","Spam"],xlabel="Predicted",ylabel="Actual")
    fig.tight_layout(); fig.savefig(root / "results/figures/confusion_matrices.png"); plt.close(fig)


class SpamPredictor:
    """Reload saved models. The score is an uncalibrated classifier output."""
    def __init__(self,root=ROOT,encoder=None):
        self.root=Path(root)
        self.tfidf=joblib.load(self.root / "models/tfidf_logreg.joblib")
        self.minilm=joblib.load(self.root / "models/minilm_logreg.joblib")
        self.encoder=encoder

    def predict(self,text,model="both"):
        if not isinstance(text,str) or not text.strip():
            return {"message":"Please enter a non-empty English SMS message."}
        if model not in ["TF-IDF","MiniLM","both"]:
            raise ValueError("Choose TF-IDF, MiniLM, or both.")
        output={}
        if model in ["TF-IDF","both"]:
            p=float(self.tfidf.predict_proba([text])[0,1])
            output["TF-IDF"]={"prediction":LABELS[int(p>=.5)],"uncalibrated_spam_score":p}
        if model in ["MiniLM","both"]:
            if self.encoder is None:
                self.encoder,_=load_encoder(self.root)
            p=float(self.minilm.predict_proba(embed(self.encoder,[text]))[0,1])
            output["MiniLM"]={"prediction":LABELS[int(p>=.5)],"uncalibrated_spam_score":p}
        return output


def demo_checks(encoder=None,root=ROOT):
    predictor=SpamPredictor(root,encoder)
    examples={"legitimate":"Please bring your notebook to tomorrow's lab.",
              "spam":"Congratulations! You have won a cash prize. Call now to claim your free reward!",
              "misspelled_spam":"Congratulatinos! You have won a cash prize. Call now to claim your free reward!",
              "ambiguous":"I have a free ticket for you. Call me after class.","empty":"","whitespace":"   "}
    output={key:{"input":text,"result":predictor.predict(text)} for key,text in examples.items()}
    assert "message" in output["empty"]["result"] and "message" in output["whitespace"]["result"]
    for key in ["legitimate","spam","misspelled_spam","ambiguous"]:
        assert set(output[key]["result"])=={"TF-IDF","MiniLM"}
    save_json(output,root / "results/try_it_examples.json")
    return output


def collect_errors(root=ROOT):
    errors=[]
    for model in ["TF-IDF","MiniLM"]:
        for condition in ["clean","mixed","deletion","transposition"]:
            frame=pd.read_csv(root / f"results/predictions/{model}_{condition}.csv")
            wrong=frame[frame.label!=frame.prediction].copy()
            wrong["model"]=model; wrong["condition"]=condition
            errors.append(wrong)
    result=pd.concat(errors,ignore_index=True)
    result.to_csv(root / "results/all_errors.csv",index=False)
    return result


def run_all(root=ROOT):
    start=time.perf_counter()
    df,data_stats=acquire_data(root)
    df,split_stats=group_and_split(df,root)
    print("Data audit",json.dumps(data_stats),flush=True)
    print("Split audit",json.dumps(split_stats),flush=True)
    tfidf,a=train_tfidf(df,root)
    print("TF-IDF selected",a,flush=True)
    encoder,info=load_encoder(root)
    minilm,b=train_minilm(df,encoder,root)
    print("MiniLM selected",b,flush=True)
    # Both configurations are now fixed. Create variants and open the test set.
    test=df[df.split=="test"].reset_index(drop=True)
    variants=make_variants(test,root)
    table=evaluate(df,tfidf,encoder,minilm,variants,root)
    paired=paired_analysis(root)
    make_figures(df,table,root)
    collect_errors(root)
    demo_checks(encoder,root)
    versions={p:metadata.version(p) for p in ["numpy","pandas","scipy","scikit-learn","matplotlib",
              "torch","transformers","sentence-transformers","huggingface-hub","joblib","wordfreq","nbformat","nbclient","ipykernel","ipywidgets"]}
    run={"seed":SEED,"python":platform.python_version(),"platform":platform.platform(),
         "versions":versions,"elapsed_seconds":time.perf_counter()-start,"model":info,
         "completed_utc":pd.Timestamp.now(tz="UTC").isoformat(),
         "test_manifest_sha256":sha256(root / "data/processed/split_manifest.csv")}
    save_json(run,root / "results/run_metadata.json")
    print(table[table.cohort=="all"].to_string(index=False),flush=True)
    return df,table,paired


if __name__=="__main__":
    run_all()
