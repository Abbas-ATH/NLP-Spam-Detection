"""Independent numerical and data-integrity checks against saved evidence."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from src.project import ROOT, SEED, PROTECTED, SpamPredictor, typo_variant, sha256, save_json


def manual_metrics(y,p):
    y,p=np.asarray(y),np.asarray(p)
    tn=int(((y==0)&(p==0)).sum()); fp=int(((y==0)&(p==1)).sum())
    fn=int(((y==1)&(p==0)).sum()); tp=int(((y==1)&(p==1)).sum())
    def divide(a,b): return a/b if b else 0.
    spam_f1=divide(2*tp,2*tp+fp+fn); ham_f1=divide(2*tn,2*tn+fp+fn)
    return {'accuracy':divide(tn+tp,len(y)), 'macro_f1':(spam_f1+ham_f1)/2,
            'spam_precision':divide(tp,tp+fp),'spam_recall':divide(tp,tp+fn),'spam_f1':spam_f1,
            'false_positive_rate':divide(fp,fp+tn),'false_positives':fp,'false_negatives':fn,
            'tn':tn,'tp':tp,'n':len(y)}


def verify(root=ROOT):
    df=pd.read_csv(root/'data/processed/split_manifest.csv')
    assert len(df)==df.row_id.nunique()==df.text.nunique()
    assert df.groupby('group_id').split.nunique().max()==1
    split_sets={s:set(g.text) for s,g in df.groupby('split')}
    assert all(not (split_sets[a]&split_sets[b]) for a,b in [('train','development'),('train','test'),('development','test')])
    test=df[df.split=='test'].reset_index(drop=True)
    metrics=pd.read_csv(root/'results/metrics.csv')
    checks=0
    for model in ['TF-IDF','MiniLM']:
        for condition in ['clean','deletion','transposition','mixed']:
            pred=pd.read_csv(root/f'results/predictions/{model}_{condition}.csv')
            assert pred.row_id.equals(test.row_id) and pred.label.equals(test.label)
            row=metrics[(metrics.model==model)&(metrics.condition==condition)&(metrics.cohort=='all')].iloc[0]
            for k,v in manual_metrics(pred.label,pred.prediction).items():
                assert np.isclose(row[k],v,atol=1e-12), (model,condition,k)
                checks+=1
            if condition!='clean':
                eligible=pred.modified.to_numpy(bool)
                row=metrics[(metrics.model==model)&(metrics.condition==condition)&(metrics.cohort=='modified_only')].iloc[0]
                for k,v in manual_metrics(pred.label[eligible],pred.prediction[eligible]).items():
                    assert np.isclose(row[k],v,atol=1e-12)
                    checks+=1
    variant_count=0
    for kind in ['deletion','transposition','mixed']:
        v=pd.read_csv(root/f'data/processed/test_{kind}.csv',keep_default_na=False)
        assert v.row_id.equals(test.row_id) and v.label.equals(test.label)
        assert not set(v.text_variant)&split_sets['train']
        assert not set(v.text_variant)&split_sets['development']
        for r in v.itertuples():
            assert r.modified==(r.text!=r.text_variant)
            if not r.modified:
                assert r.operation=='none'
                continue
            variant_count+=1
            a,b=r.original_word,r.replacement_word
            i=r.character_index
            expected=a[:i]+a[i+1:] if r.operation=='deletion' else a[:i]+a[i+1]+a[i]+a[i+2:]
            assert expected==b and a[0]==b[0] and a[-1]==b[-1]
            assert r.text_variant==r.text[:r.word_start]+b+r.text[r.word_start+len(a):]
            assert typo_variant(r.text,r.row_id,r.operation)['text_variant']==r.text_variant
            original_protected=[m.group() for m in PROTECTED.finditer(r.text)]
            changed_protected=[m.group() for m in PROTECTED.finditer(r.text_variant)]
            assert original_protected==changed_protected
    a=json.loads((root/'results/tfidf_selection.json').read_text())
    b=json.loads((root/'results/minilm_selection.json').read_text())
    assert a['refit_on_development']==b['refit_on_development']==False
    encoder=json.loads((root/'models/encoder_metadata.json').read_text())
    assert encoder['model_id']=='sentence-transformers/all-MiniLM-L6-v2' and encoder['frozen']
    assert (root/'models/minilm_encoder/model.safetensors').exists() or (root/'models/minilm_encoder/pytorch_model.bin').exists()
    predictor=SpamPredictor(root)
    assert 'message' in predictor.predict('') and 'message' in predictor.predict('  ')
    assert predictor.predict('Please bring the notebook to class.','TF-IDF')['TF-IDF']['prediction'] in ['Spam','Not spam']
    result={'passed':True,'independent_metric_checks':checks,'verified_applied_edits':variant_count,
            'exact_overlap':0,'linked_group_overlap':0,'protected_token_changes':0,
            'variants_in_training_or_development':0,'saved_model_inference':True,'empty_input_handled':True,
            'scope':'Manual confusion-count arithmetic, split integrity, variant replay, saved artifacts and inference.'}
    save_json(result,root/'results/quality_verification.json')
    print(json.dumps(result,indent=2))
    return result


if __name__=='__main__':
    verify()
