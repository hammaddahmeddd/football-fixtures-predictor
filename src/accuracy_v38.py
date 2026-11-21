"""Phase 8: reproducible computational benchmarking (isolated from production).
Install psutil; run python -m src.accuracy_v38. --fast is a smoke test only.
Historical 2025-26 benchmark is previously examined, not untouched.
"""
import os
# Apply before importing numerical libraries. One thread per candidate.
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[_key]='1'
import argparse
import importlib.util
import json
import platform
import subprocess
import sys
import tempfile
import threading
from pathlib import Path
from time import perf_counter, process_time
import joblib
import numpy as np
import pandas as pd
import psutil
from sklearn import __version__ as sklearn_version
from sklearn.calibration import CalibratedClassifierCV
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LinearRegression, Ridge, PoissonRegressor, LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, log_loss,
    mean_absolute_error, mean_squared_error, r2_score, silhouette_score, davies_bouldin_score)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from threadpoolctl import threadpool_limits
from src import accuracy_v3 as v3
from src import accuracy_v37 as v37

OUT=v3.ROOT/'outputs'/'v38_experiments'
CLASSES=np.array(['A','D','H'])
SEED=42
REQUIRED=['KNN','SVM','DecisionTree','RandomForest','LinearRegression','Ridge','Poisson','KMeans','QLearning','SoftVoting']
EXTRA=['Logistic','ExtraTrees','HistGB','XGBoost','LightGBM','CatBoost','DixonColes']


def models(name, fast=False):
    trees=30 if fast else 250
    if name=='KNN': return make_pipeline(StandardScaler(),KNeighborsClassifier(n_neighbors=60,weights='distance',n_jobs=1))
    if name=='SVM':
        # Explicitly avoid deprecated SVC(probability=True). Calibration uses a
        # chronological holdout within the training sample (see fit_classifier).
        return make_pipeline(StandardScaler(),SVC(C=.5,random_state=SEED))
    if name=='DecisionTree': return DecisionTreeClassifier(max_depth=4,min_samples_leaf=70,random_state=SEED)
    if name=='RandomForest': return RandomForestClassifier(n_estimators=trees,max_depth=7,min_samples_leaf=20,n_jobs=1,random_state=SEED)
    if name=='ExtraTrees': return ExtraTreesClassifier(n_estimators=trees,max_depth=8,min_samples_leaf=20,n_jobs=1,random_state=SEED)
    if name=='Logistic': return make_pipeline(StandardScaler(),LogisticRegression(C=.1,max_iter=2500))
    if name=='HistGB': return HistGradientBoostingClassifier(max_iter=30 if fast else 120,max_leaf_nodes=12,min_samples_leaf=45,l2_regularization=8,random_state=SEED,early_stopping=False)
    if name=='LinearRegression': return make_pipeline(StandardScaler(),LinearRegression())
    if name=='Ridge': return make_pipeline(StandardScaler(),Ridge(alpha=100))
    if name=='Poisson': return make_pipeline(StandardScaler(),PoissonRegressor(alpha=1,max_iter=600))
    if name=='KMeans': return make_pipeline(StandardScaler(),KMeans(n_clusters=2,n_init=10,random_state=SEED))
    if name=='XGBoost':
        from xgboost import XGBClassifier
        return XGBClassifier(n_estimators=30 if fast else 250,max_depth=3,learning_rate=.035,subsample=.85,colsample_bytree=.8,reg_lambda=8,n_jobs=1,random_state=SEED)
    if name=='LightGBM':
        from lightgbm import LGBMClassifier
        return LGBMClassifier(n_estimators=30 if fast else 220,num_leaves=12,learning_rate=.035,min_child_samples=50,reg_lambda=8,n_jobs=1,verbosity=-1,random_state=SEED)
    if name=='CatBoost':
        from catboost import CatBoostClassifier
        return CatBoostClassifier(iterations=30 if fast else 300,depth=4,learning_rate=.035,l2_leaf_reg=8,verbose=False,thread_count=1,random_seed=SEED,allow_writing_files=False)
    raise ValueError(name)


def chronological_cut(dates, fraction=.75):
    # Keep all same-date matches on the same side of the split.
    idx=int(len(dates)*fraction)
    cut=int(np.searchsorted(dates,dates[idx],side='left'))
    if not 65<=cut<len(dates): raise ValueError('Insufficient data for chronological inner split')
    return cut


def fit_classifier(name,x,y,dates,fast):
    m=models(name,fast)
    if name=='SVM':
        from sklearn.frozen import FrozenEstimator
        k=chronological_cut(dates,.8)
        m.fit(x[:k],y[:k])
        m=CalibratedClassifierCV(FrozenEstimator(m),method='sigmoid')
        m.fit(x[k:],y[k:])
    else: m.fit(x,y)
    return m


def probs(m,x):
    p=m.predict_proba(x)
    return np.asarray(p,float)[:,[list(m.classes_).index(i) for i in range(3)]]


class Selector:
    """Representative V3.6 compact Q-learning, gamma=0, 4 base actions.
    Base fits use an earlier partition; policy sees only later forecasts.
    No refit on policy labels, preserving the evaluated architecture.
    """
    def __init__(self,fast=False): self.fast=fast
    @staticmethod
    def context(p):
        mean=p.mean(axis=1); margin=np.abs(mean[:,2]-mean[:,0]); votes=p.argmax(2)
        agreement=np.max(np.stack([(votes==i).mean(1) for i in range(3)],1),1)
        return np.digitize(margin,[.10,.25])*3+np.digitize(agreement,[.60,.85])
    def forecasts(self,x): return np.stack([probs(m,x) for m in self.base],axis=1)
    def fit(self,x,y,dates):
        k=chronological_cut(dates); self.base_rows=k; self.policy_rows=len(y)-k
        self.base=[fit_classifier(n,x[:k],y[:k],dates[:k],self.fast) for n in ('KNN','SVM','DecisionTree','RandomForest')]
        p=self.forecasts(x[k:]); states=self.context(p); rewards=(p.argmax(2)==y[k:,None]).astype(float)
        self.prior=rewards.mean(0); self.q=np.tile(self.prior,(9,1)); self.counts=np.zeros((9,4),int)
        rng=np.random.default_rng(SEED); self.passes=5 if self.fast else 100
        start=perf_counter()
        for epoch in range(self.passes):
            eps=max(.02,.35*.995**epoch)
            for i in rng.permutation(len(states)):
                s=states[i]; a=int(rng.integers(4) if rng.random()<eps else np.argmax((self.counts[s]*self.q[s]+12*self.prior)/(self.counts[s]+12)))
                self.q[s,a]+=.15*(rewards[i,a]-self.q[s,a]); self.counts[s,a]+=1
        self.policy_fit_seconds=perf_counter()-start
        return self
    def predict_proba(self,x):
        p=self.forecasts(x); s=self.context(p)
        a=((self.counts[s]*self.q[s]+12*self.prior)/(self.counts[s]+12)).argmax(1)
        return p[np.arange(len(s)),a]


class SoftVote:
    def fit(self,x,y,dates,fast):
        self.base=[fit_classifier(n,x,y,dates,fast) for n in ('KNN','SVM','DecisionTree','RandomForest')]
        return self
    def predict_proba(self,x): return np.mean([probs(m,x) for m in self.base],axis=0)


class MemorySampler:
    """Sample process RSS during an operation; native allocations included.
    Peak is a sampled approximation, not exact OS high-water mark.
    """
    def __enter__(self):
        self.proc=psutil.Process(); self.base=self.proc.memory_info().rss; self.peak=self.base
        self.stop=threading.Event()
        def poll():
            while not self.stop.wait(.005): self.peak=max(self.peak,self.proc.memory_info().rss)
        self.thread=threading.Thread(target=poll,daemon=True); self.thread.start(); return self
    def __exit__(self,*args):
        self.stop.set(); self.thread.join(); self.peak=max(self.peak,self.proc.memory_info().rss)
    def result(self): return self.peak/2**20,max(0,self.peak-self.base)/2**20


def worker(name, data, n, fast, out):
    a=np.load(data,allow_pickle=False); x=a['train_x'][-n:]; y=a['train_y'][-n:]; dates=a['train_dates'][-n:]
    test=a['test_x']; actual=a['test_y']; totals=a['train_total'][-n:]; testtotal=a['test_total']
    task='Classification'; extra={}; fit_rows=len(x); infer_rows=len(test); features=x.shape[1]
    if name=='KMeans':
        task='Clustering'; x=a['train_profiles'][-2*n:]; test=a['test_profiles']; fit_rows=len(x); infer_rows=len(test); features=x.shape[1]
    elif name in ('LinearRegression','Ridge','Poisson'): task='Regression'
    elif name=='QLearning': task='ReinforcementLearning'
    elif name=='SoftVoting': task='Ensemble'
    elif name=='DixonColes':
        task='StatisticalClassification'; features=2 # team identifiers; date used for time decay
        from src.accuracy_v33 import DixonColes, forecast
        hist=pd.DataFrame(dict(Date=pd.to_datetime(dates,unit='D'),HomeTeam=a['train_home'][-n:],AwayTeam=a['train_away'][-n:],FTHG=a['train_hg'][-n:],FTAG=a['train_ag'][-n:]))
        test=pd.DataFrame(dict(HomeTeam=a['test_home'],AwayTeam=a['test_away']))
    with threadpool_limits(limits=1):
        with MemorySampler() as mem:
            wall=perf_counter(); cpu=process_time()
            if name=='QLearning': m=Selector(fast).fit(x,y,dates)
            elif name=='SoftVoting': m=SoftVote().fit(x,y,dates,fast)
            elif name=='DixonColes': m=DixonColes(365,20,True).fit(hist,pd.to_datetime(int(a['test_dates'][0]),unit='D'))
            elif task in ('Regression','Clustering'):
                m=models(name,fast); m.fit(x,totals) if task=='Regression' else m.fit(x)
            else: m=fit_classifier(name,x,y,dates,fast)
            fit_wall=perf_counter()-wall; fit_cpu=process_time()-cpu
        peak,delta=mem.result()
        if name=='QLearning': extra=dict(PolicyFitSeconds=m.policy_fit_seconds,PolicyEpisodes=m.passes*m.policy_rows,BaseTrainingRows=m.base_rows,PolicyTrainingRows=m.policy_rows)
        if name=='SVM': extra['BaseTrainingRows']=chronological_cut(dates,.8)
        if name=='DixonColes':
            def infer(z):
                h,aw,r=m.rates(z); return forecast(h,aw,r)['p']
        elif name in ('QLearning','SoftVoting'): infer=m.predict_proba
        elif task in ('Regression','Clustering'): infer=m.predict
        else: infer=lambda z:probs(m,z)
        # Warm-up outside timed measurements; single calls use different fixtures.
        infer(test[:min(5,len(test))]); single=[]; batch=[]; batch_cpu=[]
        with MemorySampler() as pm:
            for i in range(10 if fast else 50):
                j=i%len(test); t=perf_counter(); infer(test[j:j+1]); single.append((perf_counter()-t)*1000)
            for _ in range(3 if fast else 7):
                t=perf_counter(); c=process_time(); pred=infer(test); batch.append((perf_counter()-t)*1000); batch_cpu.append(process_time()-c)
        ppeak,pdelta=pm.result()
        # Save uncompressed model to measure serialization size, not disk compression.
        with tempfile.TemporaryDirectory() as td:
            f=Path(td)/'model.joblib'; joblib.dump(m,f,compress=0); size=f.stat().st_size/1024
        row=dict(Model=name,Task=task,TrainingMatches=n,FitRows=fit_rows,EvaluationRows=infer_rows,Features=features,
            FitWallSeconds=fit_wall,FitCPUSeconds=fit_cpu,FitPeakRSSMiB=peak,FitRSSIncreaseMiB=delta,
            InferencePeakRSSMiB=ppeak,InferenceRSSIncreaseMiB=pdelta,SingleMedianMs=float(np.median(single)),
            SingleP95Ms=float(np.percentile(single,95)),BatchMedianMs=float(np.median(batch)),BatchCPUSeconds=float(np.median(batch_cpu)),
            BatchPerRowMs=float(np.median(batch)/infer_rows),ModelKiB=size,**extra)
        if task=='Regression': row.update(MAE=mean_absolute_error(testtotal,pred),RMSE=mean_squared_error(testtotal,pred)**.5,R2=r2_score(testtotal,pred))
        elif task=='Clustering':
            z=m[0].transform(test); labels=pred
            row.update(Silhouette=silhouette_score(z,labels) if 1<len(np.unique(labels))<len(labels) else np.nan,DaviesBouldin=davies_bouldin_score(z,labels) if 1<len(np.unique(labels))<len(labels) else np.nan)
        else:
            labels=pred.argmax(1)
            row.update(Accuracy=accuracy_score(actual,labels),MacroPrecision=precision_score(actual,labels,average='macro',labels=[0,1,2],zero_division=0),MacroRecall=recall_score(actual,labels,average='macro',labels=[0,1,2],zero_division=0),MacroF1=f1_score(actual,labels,average='macro',labels=[0,1,2],zero_division=0),LogLoss=log_loss(actual,pred,labels=[0,1,2]))
    Path(out).write_text(json.dumps(row,allow_nan=True),encoding='utf-8')


def prepare_data(df,path):
    df=df.sort_values(['Date','HomeTeam','AwayTeam']).reset_index(drop=True); X=v3.make_features(df); p=v37.team_profiles(df,X)
    tr=(df.Season>='2012-13')&(df.Season<'2025-26'); te=df.Season=='2025-26'
    arrays={}
    for label,mask in [('train',tr),('test',te)]:
        z=df.loc[mask]; profiles=p[p.FixtureRow.isin(z.index)].sort_values(['FixtureRow','Side'])
        arrays.update({label+'_x':X.loc[mask].to_numpy(float),label+'_y':np.searchsorted(CLASSES,z.FTR),label+'_total':(z.FTHG+z.FTAG).to_numpy(float),label+'_dates':z.Date.to_numpy(dtype='datetime64[D]').astype('int64'),label+'_profiles':profiles[v37.PROFILE].to_numpy(float),label+'_home':z.HomeTeam.to_numpy(dtype=str),label+'_away':z.AwayTeam.to_numpy(dtype=str),label+'_hg':z.FTHG.to_numpy(float),label+'_ag':z.FTAG.to_numpy(float)})
    np.savez(path,**arrays); return arrays


def theoretical():
    # Conventional approximate bounds; constants, library optimizations and
    # calibration overhead matter. d=features, n=rows, m=queries, T=trees,
    # h=depth, k=clusters, I=iterations, S=states, A=actions, E=episodes.
    rows=[('KNN','O(nd)','O(mnd)','O(nd)','brute-force distance; scaling included'),('SVM','typically O(n²d) to O(n³d)','O(m s d)','up to O(n²) kernel workspace; O(sd) fitted','s=support vectors; solver dependent; calibration extra'),('DecisionTree','approx O(d n log n) balanced','O(mh)','O(nodes)','split search depends on implementation and depth'),('RandomForest','approx O(T d_sub n log n)','O(mTh)','O(T nodes)','d_sub=features searched per split'),('ExtraTrees','approx O(T d_sub n log n)','O(mTh)','O(T nodes)','randomized split candidates'),('HistGB','approx O(nd + T d B nodes)','O(mTh)','O(nd + T nodes)','B=histogram bins; implementation dependent'),('XGBoost','approx O(T n d h)','O(mTh)','O(nd + T nodes)','histogram/quantile methods and sparsity affect cost'),('LightGBM','approx O(T n d h)','O(mTh)','O(nd + T nodes)','leaf-wise histogram method; approximation'),('CatBoost','approx O(T n d h)','O(mTh)','O(nd + T nodes)','ordered boosting and symmetric trees alter constants'),('LinearRegression','O(nd² + d³) for n>=d','O(md)','O(nd + d²)','dense least squares'),('Ridge','O(nd² + d³) for direct solver','O(md)','O(nd + d²)','solver dependent'),('Poisson','approx O(I n d)','O(md)','O(nd + d)','L-BFGS iterations'),('Logistic','approx O(I n d c)','O(mdc)','O(nd + dc)','c=classes; solver dependent'),('KMeans','O(I n k d)','O(mkd)','O(nd + kd)','multiply training by n_init'),('SoftVoting','sum of base fits','sum of base inference','sum of base models','four base models'),('QLearning','base fits + O(E A)','base inference + O(mA)','base models + O(SA)','one-step policy; E=passes × policy rows; forecast generation extra'),('DixonColes','approx O(I(n+u))','O(m G²)','O(n+u + mG²)','u=teams; G=score grid width; optimizer dependent')]
    return pd.DataFrame(rows,columns=['Model','TrainingTimeComplexity','InferenceTimeComplexity','SpaceComplexity','Notes'])


def save_outputs(raw,errors,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out.mkdir(parents=True,exist_ok=True); d=pd.DataFrame(raw); d.to_csv(out/'benchmark_runs.csv',index=False)
    pd.DataFrame(errors,columns=['Model','TrainingMatches','Repeat','Error']).to_csv(out/'benchmark_errors.csv',index=False)
    theoretical().to_csv(out/'theoretical_complexity.csv',index=False)
    if d.empty: raise RuntimeError('No successful benchmarks; inspect benchmark_errors.csv')
    keys=['Model','Task','TrainingMatches']; numeric=d.select_dtypes(include='number').columns.difference(['Repeat','TrainingMatches']); summary=d.groupby(keys)[list(numeric)].median().reset_index(); counts=d.groupby(keys).size().rename('SuccessfulRepeats').reset_index(); summary=summary.merge(counts,on=keys); summary.to_csv(out/'benchmark_summary.csv',index=False)
    full=summary[summary.TrainingMatches==summary.TrainingMatches.max()].copy(); full.to_csv(out/'full_dataset_comparison.csv',index=False)
    for metric in ('FitWallSeconds','SingleMedianMs','ModelKiB','FitPeakRSSMiB'):
        fig,ax=plt.subplots(figsize=(10,6)); z=full.sort_values(metric); ax.barh(z.Model,z[metric]); ax.set_xlabel(metric); ax.set_title('Full training workload: '+metric); ax.grid(axis='x',alpha=.25); fig.tight_layout(); fig.savefig(out/(metric+'.png'),dpi=150); plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,6))
    for name,z in summary.groupby('Model'): z=z.sort_values('TrainingMatches'); ax.plot(z.TrainingMatches,z.FitWallSeconds,marker='o',label=name)
    ax.set(xlabel='Training fixtures',ylabel='Median fit seconds',title='Training scalability'); ax.set_yscale('log'); ax.legend(fontsize=7,ncol=2); fig.tight_layout(); fig.savefig(out/'Scalability.png',dpi=150); plt.close(fig)
    print('\nFull workload (median across repeats):'); print(full[['Model','Task','FitRows','FitWallSeconds','SingleMedianMs','BatchMedianMs','FitPeakRSSMiB','ModelKiB']].to_string(index=False)); print('Saved:',out)


def main():
    p=argparse.ArgumentParser(); p.add_argument('--fast',action='store_true'); p.add_argument('--repeats',type=int,default=2); p.add_argument('--required-only',action='store_true'); p.add_argument('--worker',nargs=4,metavar=('MODEL','DATA','N','OUT')); args=p.parse_args()
    if args.worker:
        name,data,n,out=args.worker; worker(name,data,int(n),args.fast,out); return
    if args.repeats<1: p.error('--repeats must be positive')
    out=OUT/('smoke' if args.fast else 'full'); out.mkdir(parents=True,exist_ok=True)
    df=v3.load_history(); raw=[]; errors=[]
    with tempfile.TemporaryDirectory() as td:
        data=Path(td)/'benchmark_data.npz'; arrays=prepare_data(df,data); n=len(arrays['train_y']); dates=arrays['train_dates']
        desired=[380,n] if args.fast else [760,2280,n]; sizes=sorted({n-int(np.searchsorted(dates,dates[max(0,n-min(k,n))],side='left')) for k in desired})
        names=REQUIRED.copy(); skips=[]
        if not args.required_only and not args.fast:
            for name in EXTRA:
                package={'XGBoost':'xgboost','LightGBM':'lightgbm','CatBoost':'catboost'}.get(name)
                if package and importlib.util.find_spec(package) is None: skips.append(name)
                else: names.append(name)
        metadata=dict(Python=sys.version,Platform=platform.platform(),Processor=platform.processor(),LogicalCPUs=psutil.cpu_count(),PhysicalCPUs=psutil.cpu_count(logical=False),RAMGiB=psutil.virtual_memory().total/2**30,Sklearn=sklearn_version,NumPy=np.__version__,Pandas=pd.__version__,Threads=1,Repeats=1 if args.fast else args.repeats,Fast=args.fast,TrainingFixtures=sizes,EvaluationSeason='2025-26',EvaluationFixtures=len(arrays['test_y']),SkippedOptionalModels=skips,Notes='Representative configurations, not exact V3.6 policy or deployed artifacts. Feature construction and model tuning excluded. RSS sampled every 5 ms. Same-date boundaries preserved. Q-learning inner policy holdout; KMeans uses 2 profiles per match; DixonColes fixed fit, not rolling refits.')
        (out/'environment.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
        for size in sizes:
            for name in names:
                for rep in range(1,metadata['Repeats']+1):
                    f=Path(td)/'result.json'; f.unlink(missing_ok=True)
                    cmd=[sys.executable,'-m','src.accuracy_v38','--worker',name,str(data),str(size),str(f)]+(['--fast'] if args.fast else [])
                    print(f'{name}: {size} fixtures, repeat {rep}',flush=True)
                    try:
                        completed=subprocess.run(cmd,capture_output=True,text=True,check=True)
                        row=json.loads(f.read_text(encoding='utf-8')); row['Repeat']=rep; raw.append(row)
                    except (subprocess.CalledProcessError,OSError,ValueError) as e:
                        errors.append(dict(Model=name,TrainingMatches=size,Repeat=rep,Error=(getattr(e,'stderr',None) or str(e))[-4000:])); print('FAILED:',name,errors[-1]['Error'],flush=True)
                    pd.DataFrame(raw).to_csv(out/'benchmark_runs.csv',index=False)
                    pd.DataFrame(errors,columns=['Model','TrainingMatches','Repeat','Error']).to_csv(out/'benchmark_errors.csv',index=False)
    save_outputs(raw,errors,out)
    if errors: print('WARNING:',len(errors),'failed runs; see benchmark_errors.csv')

if __name__=='__main__': main()
