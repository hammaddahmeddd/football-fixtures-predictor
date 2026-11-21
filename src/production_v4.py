"""V4 shadow production pipeline. Run python -m src.production_v4 train/predict/backtest.
Keep V2 production files untouched. Only load trusted locally generated joblib files.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse, json, os, uuid
import numpy as np
import pandas as pd
import joblib
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import PoissonRegressor, Ridge
from sklearn.metrics import accuracy_score, f1_score, log_loss, mean_absolute_error, mean_squared_error
from sklearn.cluster import KMeans
from src import accuracy_v3 as v3, accuracy_v33 as v33, accuracy_v37 as v37, accuracy_v36 as v36

ROOT=v3.ROOT; REG=ROOT/'models'/'v4'; OUT=ROOT/'outputs'/'v4_production'
C=v3.CLASSES; META=['2021-22','2022-23']; VALID='2023-24'; TEST=['2024-25','2025-26']
KEY=['Date','Season','HomeTeam','AwayTeam','FTHG','FTAG','FTR']
NAMES=['KNN','SVM','DecisionTree','RandomForest','Logistic','ScorePoisson5','DC365','IP730','EqualVote']

def atomic_csv(df,path):
    path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_name(path.name+'.tmp'); df.to_csv(tmp,index=False); os.replace(tmp,path)

def history():
    d=v3.load_history()[KEY].copy(); d.Date=pd.to_datetime(d.Date).dt.normalize(); return d.sort_values(['Date','HomeTeam','AwayTeam']).reset_index(drop=True)

def fit(d,X,cutoff):
    d=d.copy(); mask=(d.Date<cutoff)&(d.Season!='2011-12'); x=X.loc[mask]; y=d.loc[mask,'FTR']; t=d.loc[mask]; total=t.FTHG+t.FTAG
    if len(t)<100 or set(y)!=set(C): raise ValueError('Insufficient training data')
    models={k:m for k,m in v3.classifiers().items() if k in NAMES}
    # Chronological calibration: fit SVC on earlier rows, calibrate on later rows.
    # FrozenEstimator is available in sklearn >=1.6 (project uses 1.9).
    from sklearn.frozen import FrozenEstimator
    s=make_pipeline(StandardScaler(),SVC(C=.5,random_state=42)); boundary=t.Date.quantile(.8); early=t.Date<boundary
    s.fit(x.loc[early],y.loc[early]); cal=CalibratedClassifierCV(FrozenEstimator(s),method='sigmoid'); cal.fit(x.loc[~early],y.loc[~early]); models['SVM']=cal
    for name,m in models.items():
        if name!='SVM': m.fit(x,y)
    rates=make_pipeline(StandardScaler(),PoissonRegressor(alpha=5,max_iter=600)); h=clone(rates).fit(x,t.FTHG); a=clone(rates).fit(x,t.FTAG)
    dc=v33.DixonColes(365,20,True).fit(d,cutoff); ip=v33.DixonColes(730,20,False).fit(d,cutoff)
    goal=make_pipeline(StandardScaler(),Ridge(alpha=50)).fit(x,total)
    profiles=v37.team_profiles(t,x); scaler=StandardScaler().fit(profiles[v37.PROFILE]); km=KMeans(n_clusters=2,n_init=10,random_state=42).fit(scaler.transform(profiles[v37.PROFILE]))
    return dict(models=models,h=h,a=a,dc=dc,ip=ip,goal=goal,median=float(total.median()),mean=float(total.mean()),overprior=float((total>=3).mean()),scaler=scaler,km=km,columns=list(X.columns),cutoff=str(pd.Timestamp(cutoff).date()),rows=len(t))

def forecasts(b,z,x):
    x=x[b['columns']]; p={name:m.predict_proba(x)[:,[list(m.classes_).index(c) for c in C]] for name,m in b['models'].items()}
    stats={'ScorePoisson5':v33.forecast(b['h'].predict(x),b['a'].predict(x),0), 'DC365':v33.forecast(*b['dc'].rates(z)), 'IP730':v33.forecast(*b['ip'].rates(z))}
    p.update({k:v['p'] for k,v in stats.items()}); p['EqualVote']=np.mean([p[k] for k in ['RandomForest','ScorePoisson5','DC365','IP730']],axis=0)
    over={k:v['over'] for k,v in stats.items()}; over['EqualStats']=np.mean(list(over.values()),axis=0); over['Prior']=np.full(len(z),b['overprior'])
    goals={'TrainMedian':np.full(len(z),b['median']),'TrainMean':np.full(len(z),b['mean']),'Ridge':np.maximum(b['goal'].predict(x),0)}
    for k,v in stats.items(): goals[k+'_Mean']=v['mean']; goals[k+'_Median']=v['median']
    return p,over,goals,stats

def fold(d,X,s):
    z=d[d.Season==s]; b=fit(d,X,z.Date.min()); p,o,g,_=forecasts(b,z,X.loc[z.index]); print(s,'forecasts complete',flush=True); return z.reset_index(drop=True),p,o,g

def join(fs):
    return (pd.concat([f[0] for f in fs],ignore_index=True),*[{k:np.concatenate([f[i][k] for f in fs]) for k in fs[0][i]} for i in (1,2,3)])

def context(z,p):
    return v36.prepare((z,p,{},{}),NAMES)

def choose(meta,val):
    z,p,o,g=val; y=z.FTR.to_numpy(); total=(z.FTHG+z.FTAG).to_numpy(); over=total>=3
    m=context(meta[0],meta[1]); v=context(z,p)
    policy=v36.train(m,v,'compact','accuracy',.15,.35,42,(100,),[])[0]
    q={k:policy[k] for k in ['Q','Counts','Prior']}; picks=v36.greedy_policy(q['Q'],q['Counts'],q['Prior'],v['states']['compact'],12)
    candidates=[]
    for name,probs in p.items():
        for boost in (1.,1.2,1.4,1.6,1.8):
            pred=v3.labels(probs,boost); candidates.append(dict(Model=name,Boost=boost,Accuracy=accuracy_score(y,pred),MacroF1=f1_score(y,pred,labels=C,average='macro',zero_division=0),LogLoss=log_loss(y,probs,labels=C)))
    ix=np.arange(len(z)); candidates.append(dict(Model='QAccuracy',Boost=1.,Accuracy=accuracy_score(y,v['pred'][ix,picks]),MacroF1=f1_score(y,v['pred'][ix,picks],labels=C,average='macro',zero_division=0),LogLoss=log_loss(y,v['q'][ix,picks],labels=C)))
    audit=pd.DataFrame(candidates); best=audit.sort_values(['Accuracy','MacroF1','LogLoss'],ascending=[False,False,True]).iloc[0]; balanced=audit[audit.Accuracy>=best.Accuracy-.03].sort_values(['MacroF1','Accuracy'],ascending=False).iloc[0]
    prob=min(p,key=lambda k:log_loss(y,p[k],labels=C)); oo=[]
    for k,a in o.items():
        for th in (.45,.5,.55): oo.append((accuracy_score(over,a>=th),f1_score(over,a>=th,average='macro',zero_division=0),k,th))
    _,_,om,th=max(oo); gm=min(g,key=lambda k:mean_absolute_error(total,g[k])); gr=min(g,key=lambda k:mean_squared_error(total,g[k]))
    sel=dict(primary=str(best.Model),primary_boost=float(best.Boost),balanced=str(balanced.Model),balanced_boost=float(balanced.Boost),probability=prob,over=om,threshold=th,goals_mae=gm,goals_rmse=gr,scorelines='DC365',validation=VALID)
    return sel,q,audit

def predict(b,z,x):
    p,o,g,stats=forecasts(b,z,x); sel=b['selection']; dummy=z.copy(); dummy['FTR']='H' # prepare computes unused rewards; states use forecasts alone
    c=context(dummy,p); q=b['policy']; picks=v36.greedy_policy(q['Q'],q['Counts'],q['Prior'],c['states']['compact'],12); ix=np.arange(len(z)); qp=c['q'][ix,picks]; qpred=c['pred'][ix,picks]
    def decision(name,boost): return (qpred,qp) if name=='QAccuracy' else (v3.labels(p[name],boost),p[name])
    primary,pp=decision(sel['primary'],sel['primary_boost']); balanced,bp=decision(sel['balanced'],sel['balanced_boost']); r=z.copy().reset_index(drop=True)
    r['ModelVersion']=b['version']; r['PrimaryModel']=sel['primary']; r['PredictedResult']=primary; r['BalancedModel']=sel['balanced']; r['BalancedResult']=balanced; r['ProbabilityModel']=sel['probability']; r['QSelectedAction']=[str(c['actions'][j]) for j in picks]; r['QPrediction']=qpred
    for name,a in {**p,'Primary':pp,'Balanced':bp,'Outcome':p[sel['probability']],'Q':qp}.items():
        for i,label in enumerate(C): r[f'{name}_P_{label}']=a[:,i]
        if name in p: r[f'{name}_Prediction']=v3.labels(a)
    r['SelectedConfidence']=pp[ix,np.searchsorted(C,primary)]; r['PredictedTotalGoals']=g[sel['goals_mae']]; r['ExpectedTotalGoals']=g[sel['goals_rmse']]; r['GoalsMAEModel']=sel['goals_mae']; r['GoalsRMSEModel']=sel['goals_rmse']; r['OverModel']=sel['over']; r['ProbOver2_5']=o[sel['over']]; r['ProbUnder2_5']=1-r.ProbOver2_5; r['OverUnder2_5']=np.where(r.ProbOver2_5>=sel['threshold'],'Over','Under')
    grid=stats['DC365']['grid']; r['ExpectedHomeGoalsDC']=stats['DC365']['h']; r['ExpectedAwayGoalsDC']=stats['DC365']['a']; r['ScorelineModel']='DC365'
    for rank in range(3):
        j=np.argsort(-grid.reshape(len(z),-1),axis=1)[:,rank]; h,a=np.divmod(j,grid.shape[1]); r[f'Score{rank+1}']=[f'{i}-{j}' for i,j in zip(h,a)]; r[f'Score{rank+1}Probability']=grid[ix,h,a]
    profiles=v37.team_profiles(z,x); assigned,_=v37.assignments(profiles,b['scaler'],b['km']);
    for side in ('H','A'): r[side+'Cluster']=assigned[assigned.Side==side].set_index('FixtureRow').reindex(z.index).Cluster.to_numpy()
    if not np.isfinite(r.select_dtypes(include='number').drop(columns=['Matchday'],errors='ignore').to_numpy()).all(): raise ValueError('Non-finite numeric predictions')
    return r

def train(backtest=False):
    d=history(); X=v3.make_features(d); fs=[fold(d,X,s) for s in META+[VALID]]; sel,q,audit=choose(join(fs[:2]),fs[2]); print(json.dumps(sel,indent=2)); version=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')+'-'+uuid.uuid4().hex[:8]; folder=REG/version; folder.mkdir(parents=True)
    if backtest:
        rows=[]
        for s in TEST:
            z=d[d.Season==s]; b=fit(d,X,z.Date.min()); b.update(selection=sel,policy=q,version=version); rows.append(predict(b,z,X.loc[z.index]))
        result=pd.concat(rows,ignore_index=True); atomic_csv(result,OUT/version/'historical_predictions.csv'); print('Historical accuracy:',accuracy_score(result.FTR,result.PredictedResult),'MacroF1:',f1_score(result.FTR,result.PredictedResult,average='macro'))
    b=fit(d,X,d.Date.max()+pd.Timedelta(days=1)); b.update(selection=sel,policy=q,version=version,history_last=str(d.Date.max().date())); joblib.dump(b,folder/'bundle.joblib'); atomic_csv(audit,folder/'validation.csv'); (folder/'manifest.json').write_text(json.dumps(dict(version=version,selection=sel,training_rows=b['rows'],cutoff=b['cutoff'],features=b['columns'],historical_benchmark_not_untouched=True),indent=2)); tmp=REG/'active.tmp'; tmp.write_text(version); os.replace(tmp,REG/'active.txt'); print('V4 registry:',folder)

def live():
    from src import predictions as old
    version=(REG/'active.txt').read_text().strip(); b=joblib.load(REG/version/'bundle.joblib'); d=history(); current=old.download_current_results()[KEY].copy(); current.Date=pd.to_datetime(current.Date).dt.normalize(); d=pd.concat([d,current],ignore_index=True).sort_values(['Date','HomeTeam','AwayTeam']).reset_index(drop=True)
    if d.duplicated(['Season','HomeTeam','AwayTeam']).any(): raise ValueError('Duplicate historical fixtures')
    z=old.fetch_upcoming_fixtures().copy().reset_index(drop=True); z['Season']=old.CURRENT_SEASON; z['Date']=pd.to_datetime(z.KickoffUTC,utc=True).dt.tz_convert(None).dt.normalize(); now=pd.Timestamp.now(tz='UTC');
    if z.MatchID.duplicated().any() or z.MatchID.isna().any() or (pd.to_datetime(z.KickoffUTC,utc=True)<=now).any(): raise ValueError('Invalid upcoming fixtures')
    if (z.Date<=d.Date.max()).any(): raise ValueError('History overlaps forecast date; remove same-day results')
    hist=v3.History()
    for _,day in d.groupby('Date',sort=True): hist.update_day(day)
    x=pd.DataFrame([hist.features(m.Date,m.Season,m.HomeTeam,m.AwayTeam) for m in z.itertuples(index=False)],index=z.index); r=predict(b,z,x); stamp=pd.Timestamp.now(tz='UTC');
    if (pd.to_datetime(r.KickoffUTC,utc=True)<=stamp).any(): raise ValueError('Fixture started during prediction')
    r['PredictionGeneratedUTC']=stamp.isoformat(); r['HistoryLastDate']=str(d.Date.max().date()); atomic_csv(r,OUT/'weekly_predictions_v4.csv'); archive=OUT/'prediction_archive'; archive.mkdir(parents=True,exist_ok=True); atomic_csv(r,archive/(stamp.strftime('%Y%m%dT%H%M%S%f')+'-'+uuid.uuid4().hex[:8]+'.csv')); print(r[['HomeTeam','AwayTeam','PredictedResult','BalancedResult','PredictedTotalGoals','OverUnder2_5','Score1']].to_string(index=False)); print('Saved:',OUT)

def main():
    p=argparse.ArgumentParser(); p.add_argument('command',choices=['train','backtest','predict']); a=p.parse_args(); live() if a.command=='predict' else train(a.command=='backtest')
if __name__=='__main__': main()
