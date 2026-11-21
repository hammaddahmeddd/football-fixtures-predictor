"""Phase 4: chronological calibration and decision optimisation.
Run python -m src.accuracy_v34 from the project root. No production changes.
2022-23 fits calibrators; 2023-24 selects probability models and thresholds;
2024-26 is a previously examined historical benchmark, not an untouched test.
"""
import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression, PoissonRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, f1_score, log_loss, precision_score,
                             recall_score, roc_auc_score, confusion_matrix)
from src import accuracy_v3 as v3
from src import accuracy_v31 as v31
from src import accuracy_v33 as v33

OUT = v3.ROOT / 'outputs' / 'v34_experiments'
KEYS = ['Season', 'HomeTeam', 'AwayTeam']
C = v3.CLASSES
EPS = 1e-7


def fit_fold(df, X, season):
    tr = (df.Season < season) & (df.Season != '2011-12')
    te = df.Season == season
    a = df.loc[te].copy().reset_index(drop=True)
    a['Total'] = a.FTHG + a.FTAG
    a['Over'] = (a.Total >= 3).astype(int)
    x, z = X.loc[tr], X.loc[te]
    y = df.loc[tr, 'FTR']; total = df.loc[tr, 'FTHG'] + df.loc[tr, 'FTAG']
    w = np.exp(-(a.Date.min()-df.loc[tr,'Date']).dt.days.to_numpy()/365.25/5)
    w /= w.mean()
    p, o = {}, {}
    for alpha in (1.,5.):
        h = make_pipeline(StandardScaler(),PoissonRegressor(alpha=alpha,max_iter=600)); b=clone(h)
        h.fit(x,df.loc[tr,'FTHG'],poissonregressor__sample_weight=w)
        b.fit(x,df.loc[tr,'FTAG'],poissonregressor__sample_weight=w)
        for rho in (0.,-.1):
            name=f'ShotPoisson{alpha}_{rho}'
            p[name],o[name],_=v3.score_probs(h.predict(z),b.predict(z),rho)
    binary=make_pipeline(StandardScaler(),LogisticRegression(C=.03,max_iter=2000))
    binary.fit(x,(total>=3).astype(int));o['DirectLogistic']=binary.predict_proba(z)[:,1]
    dm=clone(binary);wm=clone(binary)
    dm.fit(x,(y=='D').astype(int));dec=y.ne('D').to_numpy()
    wm.fit(x.iloc[dec],(y.iloc[dec]=='H').astype(int))
    d=dm.predict_proba(z)[:,1];h=wm.predict_proba(z)[:,1]
    p['DrawHurdle']=np.c_[(1-d)*(1-h),d,(1-d)*h]
    # Refit the score-only statistical models every 28 days, strictly before cutoff.
    blocks=((a.Date-a.Date.min()).dt.days//28).to_numpy()
    for half in (180,365,730):
        for dc in (False,True):
            name=f'{"DC" if dc else "IP"}_{half}'
            ps=np.empty((len(a),3));os=np.empty(len(a))
            for block in np.unique(blocks):
                ix=np.flatnonzero(blocks==block);zz=a.iloc[ix]
                m=v33.DixonColes(half,20.,dc).fit(df,zz.Date.min())
                h,b,r=m.rates(zz);f=v33.forecast(h,b,r)
                ps[ix]=f['p'];os[ix]=f['over']
            p[name]=ps;o[name]=os
    p['TrainPrior']=np.tile([(y==c).mean() for c in C],(len(a),1))
    o['TrainPrior']=np.full(len(a),(total>=3).mean())
    print(season,'base forecasts complete',flush=True)
    return a,p,o


def calibrate(fit, selection, tests):
    """Fit calibration exclusively on first validation season, not threshold selection rows."""
    a,p,o=fit
    result=[]
    for z,q,r in [selection,*tests]:
        result.append((z,dict(q),dict(r)))
    for name in p:
        if name=='TrainPrior':continue
        # Multiclass log-probability calibration, regularised to avoid extreme fits.
        m=LogisticRegression(C=.1,max_iter=2000).fit(np.log(np.clip(p[name],EPS,1)),a.FTR)
        for (_,q,_),(_,raw,_) in zip(result,[selection,*tests]):
            q[name+'_Cal']=m.predict_proba(np.log(np.clip(raw[name],EPS,1)))[:,[list(m.classes_).index(c) for c in C]]
    for name in o:
        if name=='TrainPrior':continue
        m=LogisticRegression(C=.1,max_iter=2000).fit(logit(np.clip(o[name],EPS,1-EPS)).reshape(-1,1),a.Over)
        for (_,_,r),(_,_,raw) in zip(result,[selection,*tests]):
            r[name+'_Cal']=m.predict_proba(logit(np.clip(raw[name],EPS,1-EPS)).reshape(-1,1))[:,1]
    # Predefined blends; weights are not fitted to historical test outcomes.
    for _,q,r in result:
        for w in (.25,.5,.75):
            q[f'Blend{w}']=w*q['ShotPoisson1.0_0.0']+(1-w)*q['DC_365']
            r[f'Blend{w}']=w*r['ShotPoisson1.0_0.0']+(1-w)*r['IP_730']
        q['BlendHurdle']=.5*q['DrawHurdle']+.5*q['DC_365']
        r['BlendDirect']=.5*r['DirectLogistic']+.5*r['IP_730']
    return result


def binary_metrics(y,p,threshold=.5):
    pred=(p>=threshold).astype(int);y=np.asarray(y)
    return dict(Accuracy=accuracy_score(y,pred),MacroF1=f1_score(y,pred,average='macro',labels=[0,1],zero_division=0),LogLoss=log_loss(y,p,labels=[0,1]),Brier=np.mean((p-y)**2),ROC_AUC=roc_auc_score(y,p) if len(np.unique(y))==2 else np.nan,PredictedUnder=int((pred==0).sum()),UnderRecall=recall_score(y,pred,pos_label=0,zero_division=0),UnderPrecision=precision_score(y,pred,pos_label=0,zero_division=0),UnderF1=f1_score(y,pred,pos_label=0,zero_division=0))


def choose(z,p,o):
    # Probability model selection is independent of the decision thresholds.
    pn=min(p,key=lambda k:log_loss(z.FTR,p[k],labels=C))
    on=min(o,key=lambda k:log_loss(z.Over,o[k],labels=[0,1]))
    cr=[]; br=[]
    for k,prob in p.items():
        for boost in np.arange(1.,2.51,.1):
            cr.append(dict(Model=k,DrawBoost=float(round(boost,2)),**v31.outcome_metrics(z.FTR.to_numpy(),prob,v3.labels(prob,boost))))
    for k,prob in o.items():
        for threshold in np.arange(.35,.701,.01):
            br.append(dict(Model=k,Threshold=float(round(threshold,2)),**binary_metrics(z.Over,prob,threshold)))
    cr=pd.DataFrame(cr);br=pd.DataFrame(br)
    # Allow at most 2 percentage points accuracy loss relative to the best standard decision.
    base=max(accuracy_score(z.FTR,v3.labels(x)) for x in p.values())
    eligible=cr[cr.Accuracy>=base-.02-1e-12]
    draw=eligible.sort_values(['MacroF1','Accuracy','LogLoss'],ascending=[False,False,True]).iloc[0]
    # Binary accuracy must stay within one point of majority baseline.
    base=max(z.Over.mean(),1-z.Over.mean())
    eligible=br[br.Accuracy>=base-.01-1e-12]
    over=eligible.sort_values(['MacroF1','Accuracy','LogLoss'],ascending=[False,False,True]).iloc[0]
    return dict(OutcomeProbability=pn,OutcomeDecision=draw.Model,DrawBoost=draw.DrawBoost,OverProbability=on,OverDecision=over.Model,OverThreshold=over.Threshold),cr,br


def evaluate(fit,selection,tests):
    val,*test=calibrate(fit,selection,tests)
    z,p,o=val;chosen,cs,bs=choose(z,p,o)
    a=pd.concat([f[0] for f in test],ignore_index=True)
    pp={k:np.concatenate([f[1][k] for f in test]) for k in p}
    oo={k:np.concatenate([f[2][k] for f in test]) for k in o}
    rows=[];binary=[];cms=[]
    for split,b,q,r in [('Selection',z,p,o),('HistoricalTest',a,pp,oo)]+[(str(f[0].Season.iloc[0]),*f) for f in test]:
        for k,prob in q.items():rows.append(dict(Split=split,Model=k,**v31.outcome_metrics(b.FTR.to_numpy(),prob)))
        for k,prob in r.items():binary.append(dict(Split=split,Model=k,**binary_metrics(b.Over,prob)))
        k=chosen['OutcomeDecision'];pred=v3.labels(q[k],chosen['DrawBoost'])
        rows.append(dict(Split=split,Model='SelectedDrawDecision',**v31.outcome_metrics(b.FTR.to_numpy(),q[k],pred)))
        k=chosen['OverDecision'];threshold=chosen['OverThreshold']
        binary.append(dict(Split=split,Model='SelectedOverDecision',**binary_metrics(b.Over,r[k],threshold)))
        for task,truth,guess,labels in [('Outcome',b.FTR,pred,C),('OverUnder',b.Over,(r[k]>=threshold).astype(int),[0,1])]:
            mat=confusion_matrix(truth,guess,labels=labels)
            for i,u in enumerate(labels):
                for j,v in enumerate(labels):cms.append(dict(Split=split,Task=task,Actual=u,Predicted=v,Count=mat[i,j]))
    rec=a[KEYS+['Date','FTR','Total','Over']].copy()
    cp=pp[chosen['OutcomeProbability']];dp=pp[chosen['OutcomeDecision']]
    for i,c in enumerate(C):rec['Prob'+c]=cp[:,i];rec['DecisionProb'+c]=dp[:,i]
    rec['DrawAdjusted']=v3.labels(dp,chosen['DrawBoost']);rec['Outcome']=v3.labels(cp)
    rec['ProbOver']=oo[chosen['OverProbability']];rec['DecisionProbOver']=oo[chosen['OverDecision']];rec['OverPrediction']=(rec.DecisionProbOver>=chosen['OverThreshold']).astype(int)
    return dict(classification=pd.DataFrame(rows),over_under=pd.DataFrame(binary),selection=pd.DataFrame([chosen]),draw_threshold_search=cs,over_threshold_search=bs,predictions=rec,confusion_matrices=pd.DataFrame(cms))


def compare(rec):
    rows=[]
    for ver in ('v31','v32','v33','v34'):
        if ver=='v34':a=rec
        else:
            path=v3.ROOT/'outputs'/f'{ver}_experiments'/'predictions.csv'
            if not path.exists():print('Missing comparison',path);continue
            a=pd.read_csv(path).merge(rec[KEYS+['FTR','Over','Total']],on=KEYS,validate='one_to_one',suffixes=('','_check'))
            if len(a)!=len(rec) or any(not a[k].eq(a[k+'_check']).all() for k in ('FTR','Over','Total')):raise ValueError('Fixture mismatch '+ver)
        for split,z in [('HistoricalTest',a),*list(a.groupby('Season'))]:
            pred=z.OverPrediction if ver=='v34' else (z.ProbOver>=.5).astype(int)
            rows.append(dict(Version=ver,Split=split,Matches=len(z),Accuracy=accuracy_score(z.FTR,z.DrawAdjusted),MacroF1=f1_score(z.FTR,z.DrawAdjusted,labels=C,average='macro',zero_division=0),DrawRecall=recall_score(z.FTR,z.DrawAdjusted,labels=['D'],average='macro',zero_division=0),OverAccuracy=accuracy_score(z.Over,pred),OverMacroF1=f1_score(z.Over,pred,average='macro',zero_division=0),OverLogLoss=log_loss(z.Over,z.ProbOver,labels=[0,1])))
    return pd.DataFrame(rows)


def main():
    df=v3.load_history().sort_values(['Date','HomeTeam','AwayTeam']).reset_index(drop=True)
    df['Date']=pd.to_datetime(df.Date).dt.normalize()
    X=v31.features(df)
    folds=[fit_fold(df,X,s) for s in v3.VALID+v3.TEST]
    result=evaluate(folds[0],folds[1],folds[2:]);result['paired_comparison']=compare(result['predictions'])
    OUT.mkdir(parents=True,exist_ok=True)
    for name,d in result.items():d.to_csv(OUT/(name+'.csv'),index=False)
    print('\nSelection (2023-24 only):\n',result['selection'].to_string(index=False))
    for name in ('classification','over_under'):
        t=result[name].query("Split == 'HistoricalTest'")
        print('\n',name,'historical benchmark:');print(t.sort_values('MacroF1',ascending=False).head(12).to_string(index=False))
        print(t[t.Model.str.startswith('Selected')].to_string(index=False))
    print('\nSame-fixture comparison:\n',result['paired_comparison'].to_string(index=False));print('Saved:',OUT)

if __name__=='__main__':main()
