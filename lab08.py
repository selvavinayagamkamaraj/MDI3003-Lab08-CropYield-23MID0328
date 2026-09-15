"""Run: python lab08.py --config config.json
Use instructor-verified canonical CSV; see manual for source adaptation.
Stages: validate (no final test evaluation), then test (one-time lock).
"""
import argparse, hashlib, json, platform, time
from pathlib import Path
import numpy as np
import pandas as pd
import sklearn
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.dummy import DummyRegressor, DummyClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (mean_absolute_error, mean_squared_error,
    r2_score, accuracy_score, precision_recall_fscore_support,
    classification_report, ConfusionMatrixDisplay)
from sklearn.model_selection import train_test_split

SEED = 42
REG = ['state', 'district', 'season', 'year']
CLS = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def dump(obj, path):
    Path(path).write_text(json.dumps(obj, indent=2, default=str))

def load_data(cfg):
    df = pd.read_csv(cfg['data'])
    task = cfg['task']
    features = REG if task == 'regression' else CLS
    target = 'yield_t_ha' if task == 'regression' else 'label'
    required = features + [target, 'row_id']
    if not set(required).issubset(df):
        raise ValueError('Missing canonical columns: ' + str(set(required)-set(df)))
    if df.row_id.isna().any() or df.row_id.duplicated().any():
        raise ValueError('row_id must be complete and unique')
    if df[target].isna().any():
        raise ValueError('Missing targets: resolve and log exclusions upstream')
    if task == 'regression':
        if cfg.get('crop') != 'Rice' or cfg.get('yield_unit') != 't/ha':
            raise ValueError('Core requires Rice and verified t/ha units')
        if 'crop' not in df or set(df.crop) != {'Rice'}:
            raise ValueError('Supply a Rice-only canonical file')
        if not np.isfinite(df[target]).all() or (df[target] < 0).any():
            raise ValueError('Invalid yield')
        if not np.isfinite(df.year).all() or (df.year % 1 != 0).any():
            raise ValueError('year must be an integer harvest-year index')
        if df[['state','district','season']].isna().any().any():
            raise ValueError('Missing grouping identifiers')
        if df.duplicated(REG).any():
            raise ValueError('Repeated district-season-year keys require source review')
        years = sorted(df.year.unique())
        if len(years) < 7:
            raise ValueError('Need at least seven years for development/validation/test')
        test_years, val_years = years[-2:], years[-4:-2]
        split = np.where(df.year.isin(test_years), 'test',
                np.where(df.year.isin(val_years), 'validation', 'train'))
    else:
        if not cfg.get('iid_justified', False):
            raise ValueError('Instructor must justify independence before random splitting')
        for col in CLS:
            df[col] = pd.to_numeric(df[col], errors='raise')
        if np.isinf(df[CLS].to_numpy()).any():
            raise ValueError('Infinite input')
        if df.duplicated(CLS).any():
            raise ValueError('Duplicate feature vectors: resolve grouping before splitting')
        if df.label.nunique() < 2 or df.label.value_counts().min() < 10:
            raise ValueError('Need >=2 classes and >=10 records per class')
        ix = np.arange(len(df))
        tr, rest = train_test_split(ix, test_size=.4, stratify=df.label,
            random_state=SEED)
        va, te = train_test_split(rest, test_size=.5,
            stratify=df.iloc[rest].label, random_state=SEED)
        split = np.full(len(df), 'train', dtype=object)
        split[va], split[te] = 'validation', 'test'
    df['split'] = split
    if task == 'regression':
        assert df.loc[df.split=='train','year'].max() < df.loc[df.split=='validation','year'].min()
        assert df.loc[df.split=='validation','year'].max() < df.loc[df.split=='test','year'].min()
        assert all((df.split == s).sum() >= 2 for s in ['train','validation','test'])
    return df, features, target

def candidates(task, advanced=False):
    if task == 'regression':
        pre = ColumnTransformer([
            ('cat', OneHotEncoder(handle_unknown='ignore'), REG[:3]),
            ('num', Pipeline([('impute', SimpleImputer(strategy='median')),
                ('scale', StandardScaler())]), ['year'])])
        models = {
            'median': DummyRegressor(strategy='median'),
            'ridge_trend': Ridge(alpha=1.0, solver='lsqr'),
            'tree': DecisionTreeRegressor(max_depth=6, min_samples_leaf=10,
                random_state=SEED),
            'forest': RandomForestRegressor(n_estimators=60, max_depth=12,
                min_samples_leaf=5, n_jobs=2, random_state=SEED)}
        if advanced:
            models['forest_depth6'] = RandomForestRegressor(n_estimators=60,
                max_depth=6, min_samples_leaf=5, n_jobs=2, random_state=SEED)
    else:
        pre = Pipeline([('impute', SimpleImputer(strategy='median')),
            ('scale', StandardScaler())])
        models = {'majority': DummyClassifier(strategy='most_frequent'),
            'logistic': LogisticRegression(max_iter=2000),
            'forest': RandomForestClassifier(n_estimators=60,
                min_samples_leaf=2, random_state=SEED, n_jobs=2)}
    return {k: Pipeline([('pre', clone(pre)), ('model', v)])
            for k,v in models.items()}

def metrics(y, p, task):
    if task == 'regression':
        return {'MAE': mean_absolute_error(y,p),
            'RMSE': float(np.sqrt(mean_squared_error(y,p))),
            'R2': r2_score(y,p) if len(y)>1 and np.ptp(np.asarray(y))>0 else float('nan')}
    pr,re,f,_ = precision_recall_fscore_support(y,p,average='macro',zero_division=0)
    return {'accuracy':accuracy_score(y,p),'macro_precision':pr,
        'macro_recall':re,'macro_F1':f}

def plot_save(out, name, title, x, y):
    plt.title(title); plt.xlabel(x); plt.ylabel(y)
    plt.tight_layout(); plt.savefig(out/'figures'/name, dpi=150); plt.close()

def validate(cfg, out, df, features, target):
    if (out/'selection.json').exists():
        raise FileExistsError('Development already saved. Use a new run directory.')
    tr,va = [df[df.split==s] for s in ['train','validation']]
    rows=[]
    for name,pipe in candidates(cfg['task'],cfg.get('advanced',False)).items():
        t=time.perf_counter(); pipe.fit(tr[features],tr[target])
        p=pipe.predict(va[features])
        rows.append({'model':name,'seconds':time.perf_counter()-t,
            **metrics(va[target],p,cfg['task'])})
        joblib.dump(pipe,out/'models'/f'{name}.joblib')
    if cfg['task'] == 'regression':
        development = df[df.split != 'test']
        origins = sorted(development.year.unique())[-3:]
        rolling = []
        for origin in origins:
            past = development[development.year < origin]
            future = development[development.year == origin]
            for name, model in candidates('regression', cfg.get('advanced', False)).items():
                model.fit(past[features], past[target])
                prediction = model.predict(future[features])
                rolling.append({'origin': int(origin), 'model': name,
                    'training_max_year': int(past.year.max()),
                    'n': len(future), **metrics(future[target], prediction, 'regression')})
        pd.DataFrame(rolling).to_csv(out/'artifacts'/'rolling_origins.csv', index=False)
    if cfg['task'] == 'regression':
        fitted = joblib.load(out/'models'/'tree.joblib')
        encoder = fitted.named_steps['pre'].named_transformers_['cat']
        for col, categories in zip(REG[:3], encoder.categories_):
            assert set(categories) == set(tr[col].unique())
    result=pd.DataFrame(rows)
    key='MAE' if cfg['task']=='regression' else 'macro_F1'
    result=result.sort_values(key,ascending=cfg['task']=='regression',kind='stable')
    result.to_csv(out/'Lab08_Validation_Results.csv',index=False)
    chosen=result.iloc[0]['model']
    dump({'model':chosen,'data_sha256':digest(cfg['data']),
        'features':features,'target':target,'task':cfg['task'],
        'config_sha256':hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest(),
        'note':'Selected training-only pipeline; no train+validation refit'},
        out/'selection.json')
    df[['row_id','split']].to_csv(out/'artifacts'/'split_manifest.csv',index=False)
    dump(cfg,out/'artifacts'/'config.json')
    dump({'python':platform.python_version(),'sklearn':sklearn.__version__,
        'numpy':np.__version__,'pandas':pd.__version__,'seed':SEED},
        out/'artifacts'/'versions.json')
    if cfg['task']=='regression':
        plt.hist(tr[target],bins=25)
        plot_save(out,'target.png','Training yield distribution','Yield (t/ha)','Count')
        plt.hist(tr.year,bins=len(tr.year.unique()))
        plot_save(out,'feature.png','Training year coverage','Harvest year','Count')
    else:
        tr.label.value_counts().plot.bar()
        plot_save(out,'target.png','Training crop labels','Dataset label','Count')
        plt.hist(tr.ph.dropna(),bins=20)
        plot_save(out,'feature.png','Training soil pH','pH','Count')
    plt.bar(result.model,result[key])
    plot_save(out,'comparison.png','Validation comparison','Model',key)
    print(result.to_string(index=False)); print('Selected:',chosen)

def test_once(cfg,out,df,features,target):
    selection=json.loads((out/'selection.json').read_text())
    assert selection['data_sha256']==digest(cfg['data']), 'Dataset changed'
    assert selection['config_sha256']==hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest(), 'Config changed'
    old=pd.read_csv(out/'artifacts'/'split_manifest.csv',dtype={'row_id':str})
    now=df[['row_id','split']].astype({'row_id':str})
    pd.testing.assert_frame_equal(old,now)
    pipe=joblib.load(out/'models'/f"{selection['model']}.joblib")
    te=df[df.split=='test']; tr=df[df.split=='train']
    with (out/'TEST_LOCK').open('x') as f: f.write('Final evaluation started')
    start=time.perf_counter(); pred=pipe.predict(te[features]); latency=time.perf_counter()-start
    if cfg['task']=='regression': assert np.isfinite(pred).all()
    else: assert set(pred).issubset(set(tr[target]))
    test_rows = [{'model':selection['model'],
        **metrics(te[target],pred,cfg['task']),
        'batch_inference_seconds':latency}]
    reference = 'median' if cfg['task']=='regression' else 'majority'
    if selection['model'] != reference:
        baseline = joblib.load(out/'models'/f'{reference}.joblib')
        baseline_pred = baseline.predict(te[features])
        test_rows.append({'model':reference,
            **metrics(te[target],baseline_pred,cfg['task'])})
    pd.DataFrame(test_rows).to_csv(
        out/'Lab08_Test_Results.csv',index=False)
    errs=te[['row_id']+features+[target]].copy(); errs['prediction']=pred
    errs['error']=np.abs(te[target].to_numpy()-pred) if cfg['task']=='regression' else (te[target].to_numpy()!=pred).astype(int)
    errs.to_csv(out/'artifacts'/'test_predictions.csv',index=False)
    cases=errs[errs.error>0].sort_values('error',ascending=False).head(5).copy()
    for c in ['likely_cause','agricultural_consequence','mitigation']: cases[c]=''
    cases.to_csv(out/'Lab08_Error_Analysis.csv',index=False)
    if cfg['task']=='regression':
        plt.scatter(te[target],pred,s=9,alpha=.6)
        bounds=[min(te[target].min(),pred.min()),max(te[target].max(),pred.max())]
        plt.plot(bounds,bounds,'k--',label='Perfect prediction'); plt.legend()
        plot_save(out,'actual_predicted.png','Locked-test yield','Actual (t/ha)','Predicted (t/ha)')
        plt.scatter(pred,te[target].to_numpy()-pred,s=9)
        plt.axhline(0,color='black')
        plot_save(out,'residuals.png','Locked-test residuals','Predicted (t/ha)','Actual minus predicted (t/ha)')
        rows=[]
        for yr,g in errs.groupby('year'):
            rows.append({'year':yr,'n':len(g),**metrics(g[target],g.prediction,cfg['task'])})
        pd.DataFrame(rows).to_csv(out/'artifacts'/'year_robustness.csv',index=False)
    else:
        dump(classification_report(te[target],pred,output_dict=True,zero_division=0),
            out/'artifacts'/'per_class.json')
        ConfusionMatrixDisplay.from_predictions(te[target],pred,xticks_rotation=90)
        plt.gcf().set_size_inches(10,9)
        plot_save(out,'confusion.png','Locked-test crop labels','Predicted label','True label')
    bundle={'pipeline':pipe,'features':features,'task':cfg['task'],
        'year_range':[int(df.year.min()),int(df.year.max())] if cfg['task']=='regression' else None,
        'categories':{c:sorted(tr[c].unique().tolist()) for c in REG[:3]} if cfg['task']=='regression' else {}}
    joblib.dump(bundle,out/'models'/'selected_bundle.joblib')
    loaded=joblib.load(out/'models'/'selected_bundle.joblib')
    reload_pred = loaded['pipeline'].predict(te[features])
    if cfg['task']=='regression':
        np.testing.assert_allclose(pred,reload_pred,rtol=1e-8,atol=1e-8)
    else:
        np.testing.assert_array_equal(pred,reload_pred)
    dump({'split_disjoint':True,'reload_consistent':True,'valid_predictions':True,
        'data_hash_matched':True,'task':cfg['task'],
        'manual_failure_rows_available':len(cases)},out/'artifacts'/'acceptance.json')
    print('Final evaluation saved. Inspect failures; do not tune on these results.')

def run(cfg,stage):
    if cfg['task'] not in ['regression','classification']: raise ValueError('Invalid task')
    out=Path(cfg['output'])
    for p in [out,out/'models',out/'figures',out/'artifacts']: p.mkdir(parents=True,exist_ok=True)
    df,features,target=load_data(cfg)
    (validate if stage=='validate' else test_once)(cfg,out,df,features,target)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='config.json')
    parser.add_argument('--stage',choices=['validate','test'],default='validate')
    args=parser.parse_args(); run(json.loads(Path(args.config).read_text()),args.stage)
