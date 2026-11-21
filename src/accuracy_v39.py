"""Phase 9: paired retrospective statistical audit. Run python -m src.accuracy_v39.
Reads saved forecasts and historical labels only. No model fitting or production changes.
"""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import roc_auc_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs' / 'v39_experiments'
KEYS = ['Season', 'HomeTeam', 'AwayTeam']
SEASONS = ['2024-25', '2025-26']
CLASSES = ['A', 'D', 'H']
EPS = 1e-12


def reference(root):
    h = pd.read_csv(root / 'data/v3_experiments/history.csv', parse_dates=['Date'])
    if h.duplicated(KEYS).any():
        raise ValueError('Duplicate historical fixture keys')
    h['Total'] = h.FTHG + h.FTAG
    h['Over'] = (h.Total >= 3).astype(int)
    r = h[h.Season.isin(SEASONS)].sort_values(['Season','Date','HomeTeam','AwayTeam']).reset_index(drop=True)
    if r.groupby('Season').size().to_dict() != dict.fromkeys(SEASONS,380):
        raise ValueError('Expected 380 fixtures in each benchmark season')
    if not r.FTR.isin(CLASSES).all(): raise ValueError('Invalid historical outcome')
    return h, r


def align(df, ref, seasons=SEASONS):
    df = df[df.Season.isin(seasons)].copy()
    r = ref[ref.Season.isin(seasons)].reset_index(drop=True)
    if df.duplicated(KEYS).any(): raise ValueError('Duplicate prediction keys')
    cols = KEYS + [c for c in ['FTR','Total','Over','Date'] if c in df]
    m = r[KEYS + ['FTR','Total','Over','Date']].merge(df, on=KEYS, how='left', validate='one_to_one', indicator=True, suffixes=('_ref',''), sort=False)
    if len(df) != len(r) or not m._merge.eq('both').all(): raise ValueError('Fixture coverage mismatch')
    for c in cols[3:]:
        lhs,rhs=m[c+'_ref'],m[c]
        if c=='Date': lhs,rhs=pd.to_datetime(lhs).dt.normalize(),pd.to_datetime(rhs).dt.normalize()
        if not lhs.eq(rhs).all(): raise ValueError('Actual/date mismatch: '+c)
    return m


def prob(df, cols):
    if not all(c in df for c in cols): return None
    p=df[cols].to_numpy(float)
    if not np.isfinite(p).all() or np.any(p < -1e-8) or np.any(p > 1+1e-8) or not np.allclose(p.sum(1),1,atol=1e-6): raise ValueError('Invalid probability matrix: '+str(cols))
    return np.clip(p,0,1)


def load_candidates(root, h, ref):
    tasks={'Outcome':{}, 'OverUnder':{}, 'Goals':{}}
    audit=[]
    def add(task,name,dec,p=None,source=''):
        dec=np.asarray(dec)
        if len(dec)!=len(ref): raise ValueError('Length mismatch '+name)
        if task=='Outcome' and not np.isin(dec,CLASSES).all(): raise ValueError('Invalid outcome '+name)
        if task=='OverUnder' and not np.isin(dec,[0,1]).all(): raise ValueError('Invalid binary label '+name)
        if task=='Goals' and (not np.isfinite(dec.astype(float)).all() or np.any(dec.astype(float)<0)): raise ValueError('Invalid goals '+name)
        if p is not None:
            p=np.asarray(p,float)
            if not np.isfinite(p).all() or np.any(p<0) or np.any(p>1): raise ValueError('Invalid probability '+name)
        tasks[task][name]=(dec,p)
        audit.append(dict(Source=source,Task=task,Model=name,Status='Loaded',Matches=len(dec),ProbabilityAvailable=p is not None))
    prior=[]; mean=[]; median=[]; over=[]
    for s in ref.Season:
        hist=h[h.Season<s]
        prior.append([(hist.FTR==c).mean() for c in CLASSES]);mean.append(hist.Total.mean());median.append(hist.Total.median());over.append(hist.Over.mean())
    prior=np.array(prior);over=np.array(over)
    add('Outcome','AlwaysHome',np.repeat('H',len(ref)),source='baseline')
    add('Outcome','SeasonPrior',np.array(CLASSES)[prior.argmax(1)],prior,'baseline')
    add('OverUnder','SeasonPrior',(over>=.5).astype(int),over,'baseline')
    add('Goals','SeasonMean',mean,source='baseline');add('Goals','SeasonMedian',median,source='baseline')
    for ver in ['v3','v31','v32','v33','v34','v35','v36','v37']:
        path=root/'outputs'/f'{ver}_experiments'/('historical_predictions.csv' if ver=='v3' else 'predictions.csv')
        if not path.exists():
            audit.append(dict(Source=ver,Status='Missing file',Matches=0));continue
        df=pd.read_csv(path)
        if ver=='v36':
            for name,g in df.groupby('Strategy'):
                z=align(g,ref);add('Outcome',ver+'_'+name,z.Prediction,prob(z,['Prob'+c for c in CLASSES]),ver)
            continue
        z=align(df,ref)
        if ver=='v37':
            for name in ['RF_ScoreOnly','RF_ClusterEnhanced','Logistic_ScoreOnly','Logistic_ClusterEnhanced']:
                if name in z: add('Outcome',ver+'_'+name,z[name],prob(z,[name+'_P'+c for c in CLASSES]),ver)
            continue
        pc=['DecisionProb'+c for c in CLASSES] if ver=='v34' else ['Prob'+c for c in CLASSES]
        if ver=='v3':
            sel=pd.read_csv(path.parent/'validation_selection.csv').iloc[0];pc=[str(sel.OutcomeModel)+'_P_'+c for c in CLASSES]
            for name in ['KNN','SVM','Decision Tree','Random Forest','RandomForest','Logistic','ExtraTrees','HistGradientBoosting']:
                if name+'_Prediction' in z: add('Outcome',ver+'_'+name,z[name+'_Prediction'],prob(z,[name+'_P_'+c for c in CLASSES]),ver)
        add('Outcome',ver+'_SelectedDecision',z.DrawAdjusted,prob(z,pc),ver)
        if ver=='v34':
            p=prob(z,['Prob'+c for c in CLASSES]);add('Outcome',ver+'_ProbabilityOnly',z.Outcome,p,ver)
        if ver=='v35':
            p=prob(z,['ProbabilityOnly'+c for c in CLASSES]);
            if p is not None: add('Outcome',ver+'_ProbabilityOnly',np.array(CLASSES)[p.argmax(1)],p,ver)
        bcol='DecisionProbOver' if 'DecisionProbOver' in z else 'ProbOver'
        if bcol in z:
            p=z[bcol].to_numpy(float); d=z.OverPrediction if 'OverPrediction' in z else (p>=.5).astype(int)
            add('OverUnder',ver+'_SelectedDecision',d,p,ver)
        pcol='ProbOver' if ver=='v34' else 'ProbabilityOnlyOver'
        if ver in ['v34','v35'] and pcol in z:
            p=z[pcol].to_numpy(float);add('OverUnder',ver+'_ProbabilityOnly',(p>=.5).astype(int),p,ver)
        for col in ['Goals','GoalsRMSE','PredictedGoals']:
            if col in z: add('Goals',ver+'_'+col,z[col],source=ver)
    return tasks,pd.DataFrame(audit)


def draws(ref,B,days,seed):
    # Nonoverlapping calendar blocks, resampled within season. All same-day matches stay together.
    rng=np.random.default_rng(seed); season_blocks=[]
    for s in SEASONS:
        d=ref[ref.Season==s]; b=((d.Date-d.Date.min()).dt.days//days).to_numpy()
        season_blocks.append([d.index.to_numpy()[b==j] for j in np.unique(b)])
    return [np.concatenate([blocks[i] for blocks in season_blocks for i in rng.integers(0,len(blocks),len(blocks))]) for _ in range(B)]


def scalar_metrics(task,y,d,p):
    if task=='Goals':
        e=y-d; mse=np.mean(e**2);v=np.var(y)
        return {'MAE':np.mean(abs(e)),'RMSE':np.sqrt(mse),'R2':1-mse/v if v>0 else np.nan}
    labels=CLASSES if task=='Outcome' else [0,1]; f=[]; rec=[]; prec=[]
    for c in labels:
        tp=np.sum((y==c)&(d==c)); actual=np.sum(y==c); pred=np.sum(d==c)
        f.append(2*tp/(actual+pred) if actual+pred else 0); rec.append(tp/actual if actual else 0);prec.append(tp/pred if pred else 0)
    m={'Accuracy':np.mean(y==d),'MacroF1':np.mean(f),'MacroPrecision':np.mean(prec),'MacroRecall':np.mean(rec)}
    i=1 if task=='Outcome' else 0; prefix='Draw' if task=='Outcome' else 'Under'
    m.update({prefix+'Recall':rec[i],prefix+'Precision':prec[i],prefix+'F1':f[i]})
    if p is not None:
        if task=='Outcome':
            one=(y[:,None]==np.array(CLASSES)[None,:]).astype(float); m['LogLoss']=np.mean(-np.log(np.clip(p[one.astype(bool)],EPS,1)));m['Brier']=np.mean(np.sum((p-one)**2,axis=1))
        else:
            m['LogLoss']=np.mean(-np.log(np.clip(np.where(y==1,p,1-p),EPS,1)));m['Brier']=np.mean((p-y)**2)
    return m


def holm(p):
    p=np.asarray(p,float);out=np.full(len(p),np.nan);idx=np.where(np.isfinite(p))[0];order=idx[np.argsort(p[idx])]
    if len(order):out[order]=np.minimum(1,np.maximum.accumulate(p[order]*(len(order)-np.arange(len(order)))))
    return out


def evaluate(tasks,ref,B,days):
    boots=draws(ref,B,days,42); metrics=[]; intervals=[]; tests=[]; cal=[]; auc=[]; season=[]
    truths={'Outcome':ref.FTR.to_numpy(),'OverUnder':ref.Over.to_numpy(),'Goals':ref.Total.to_numpy(float)}
    contrasts={'Outcome':[('v36_QAccuracy','v36_FixedAccuracy'),('v36_QBalanced','v36_FixedBalanced'),('v37_RF_ClusterEnhanced','v37_RF_ScoreOnly'),('v37_Logistic_ClusterEnhanced','v37_Logistic_ScoreOnly')], 'OverUnder':[], 'Goals':[]}
    for task,models in tasks.items():
        y=truths[task]; cache={}
        for name,(d,p) in models.items():
            point=scalar_metrics(task,y,d,p)
            metrics.append(dict(Task=task,Model=name,Matches=len(y),**point))
            # Resample identical block indices for every model, preserving paired comparisons.
            vals={k:[] for k in point}
            for ix in boots:
                m=scalar_metrics(task,y[ix],d[ix],None if p is None else p[ix]);
                for k,v in m.items():vals[k].append(v)
            cache[name]=(point,{k:np.asarray(v) for k,v in vals.items()})
            for metric,v in vals.items():
                lo,hi=np.nanpercentile(v,[2.5,97.5]);intervals.append(dict(Task=task,Model=name,Metric=metric,Estimate=point[metric],Lower95=lo,Upper95=hi,Replicates=B,BlockDays=days))
            for s in SEASONS:
                ix=np.flatnonzero(ref.Season.eq(s));season.append(dict(Task=task,Model=name,Season=s,Matches=len(ix),**scalar_metrics(task,y[ix],d[ix],None if p is None else p[ix])))
            if p is not None:
                ps=[(c,p[:,i],(y==c).astype(int)) for i,c in enumerate(CLASSES)] if task=='Outcome' else [('Over',p,y)]
                for label,q,target in ps:
                    auc.append(dict(Task=task,Model=name,Class=label,ROC_AUC=roc_auc_score(target,q)))
                    ids=np.minimum((q*10).astype(int),9)
                    for k in range(10):
                        ok=ids==k
                        if ok.any():cal.append(dict(Task=task,Model=name,Class=label,Bin=k,Count=ok.sum(),MeanProbability=q[ok].mean(),ObservedRate=target[ok].mean()))
        baseline={'Outcome':'AlwaysHome','OverUnder':'SeasonPrior','Goals':'SeasonMedian'}[task]
        pairs=[(n,baseline) for n in models if n!=baseline]+contrasts[task]
        if task=='Goals':pairs += [(n,'SeasonMean') for n in models if n not in ['SeasonMean','SeasonMedian']]
        if task in ['Outcome','OverUnder']:
            pairs += [(n,'SeasonPrior') for n in models if task=='Outcome' and n not in ['SeasonPrior','AlwaysHome'] and models[n][1] is not None]
        for a,b in dict.fromkeys(pairs):
            if a not in cache or b not in cache:continue
            ap,av=cache[a];bp,bv=cache[b]
            for metric in (['MAE','RMSE'] if task=='Goals' else ['Accuracy','MacroF1','LogLoss','Brier']):
                if metric not in av or metric not in bv:continue
                sign=-1 if metric in ['MAE','RMSE','LogLoss','Brier'] else 1
                delta=sign*(ap[metric]-bp[metric]); ds=sign*(av[metric]-bv[metric]);lo,hi=np.percentile(ds,[2.5,97.5]);pval=(1+np.sum(abs(ds-delta)>=abs(delta)))/(B+1)
                row=dict(Task=task,Candidate=a,Reference=b,Metric=metric,Improvement=delta,Lower95=lo,Upper95=hi,BlockBootstrapP=pval)
                if metric=='Accuracy':
                    ca=models[a][0]==y;cb=models[b][0]==y;n10=int(np.sum(ca&~cb));n01=int(np.sum(~ca&cb));row.update(CandidateOnlyCorrect=n10,ReferenceOnlyCorrect=n01,McNemarP=binomtest(n10,n10+n01,.5).pvalue if n10+n01 else 1.)
                tests.append(row)
    t=pd.DataFrame(tests);t['HolmBootstrapP']=holm(t.BlockBootstrapP);t['HolmMcNemarP']=holm(t.McNemarP);t['ExploratorySignificant']=t.HolmBootstrapP<.05
    return dict(metrics=pd.DataFrame(metrics),confidence_intervals=pd.DataFrame(intervals),paired_tests=t,calibration=pd.DataFrame(cal),roc_auc=pd.DataFrame(auc),by_season=pd.DataFrame(season))


def plots(result,out):
    ci=result['confidence_intervals'];df=ci[(ci.Task=='Outcome')&(ci.Metric=='Accuracy')].sort_values('Estimate')
    fig,ax=plt.subplots(figsize=(10,max(5,len(df)*.27)));ax.errorbar(df.Estimate,np.arange(len(df)),xerr=np.vstack([np.maximum(0,df.Estimate-df.Lower95),np.maximum(0,df.Upper95-df.Estimate)]),fmt='o');ax.set_yticks(np.arange(len(df)),df.Model);ax.set_xlabel('Accuracy (95% block bootstrap interval)');fig.tight_layout();fig.savefig(out/'accuracy_intervals.png',dpi=150);plt.close(fig)
    c=result['calibration'];
    for task,labels in [('Outcome',['A','D','H']),('OverUnder',['Over'])]:
        fig,axes=plt.subplots(1,len(labels),figsize=(5*len(labels),5),squeeze=False)
        for ax,label in zip(axes[0],labels):
            ax.plot([0,1],[0,1],':',color='gray');q=c[(c.Task==task)&(c.Class==label)]
            for name,g in q.groupby('Model'):
                if name in ['SeasonPrior','v32_SelectedDecision','v33_SelectedDecision','v34_SelectedDecision','v35_SelectedDecision','v36_QAccuracy']:
                    ax.plot(g.MeanProbability,g.ObservedRate,'o-',label=name)
            ax.set(xlim=(0,1),ylim=(0,1),title=label,xlabel='Predicted probability',ylabel='Observed frequency');ax.legend(fontsize=7)
        fig.tight_layout();fig.savefig(out/f'{task}_calibration.png',dpi=150);plt.close(fig)


def benchmark_curve(root,out):
    path=root/'outputs/v38_experiments/full/benchmark_summary.csv'
    if not path.exists():return
    d=pd.read_csv(path);d.to_csv(out/'training_size_metrics.csv',index=False)
    for metric in ['Accuracy','MAE']:
        if metric not in d:continue
        fig,ax=plt.subplots(figsize=(10,6))
        for name,g in d.dropna(subset=[metric]).groupby('Model'):
            g=g.sort_values('TrainingMatches');ax.plot(g.TrainingMatches,g[metric],'o-',label=name)
        ax.set(xlabel='Training matches (different historical windows)',ylabel=metric);ax.legend(fontsize=7,ncol=2);fig.tight_layout();fig.savefig(out/f'training_size_{metric}.png',dpi=150);plt.close(fig)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--bootstrap',type=int,default=2000);parser.add_argument('--block-days',type=int,default=28);parser.add_argument('--fast',action='store_true');args=parser.parse_args()
    B=100 if args.fast else args.bootstrap
    if B<100 or args.block_days<1:raise ValueError('Need at least 100 bootstrap replicates and positive block days')
    out=OUT/('fast' if args.fast else 'full');out.mkdir(parents=True,exist_ok=True)
    h,ref=reference(ROOT);tasks,audit=load_candidates(ROOT,h,ref);audit.to_csv(out/'source_audit.csv',index=False)
    print('Aligned fixtures:',len(ref),'models:',{k:len(v) for k,v in tasks.items()},flush=True)
    result=evaluate(tasks,ref,B,args.block_days)
    for name,df in result.items():df.to_csv(out/f'{name}.csv',index=False)
    plots(result,out);benchmark_curve(ROOT,out)
    (out/'methodology.json').write_text(json.dumps({'seasons':SEASONS,'bootstrap':B,'block_days':args.block_days,'seed':42,'multiple_testing':'Holm across all reported tests; exploratory','limitations':'Repeatedly inspected historical seasons; retrospective only. Block bootstrap approximate; McNemar assumes independent pairs. Calibration bins sparse at extremes. No production selection from these results.'},indent=2))
    print('\nMetrics:\n',result['metrics'].to_string(index=False));print('\nPaired tests (first 20 by adjusted p-value):\n',result['paired_tests'].sort_values('HolmBootstrapP').head(20).to_string(index=False));print('Saved:',out)

if __name__=='__main__':main()
