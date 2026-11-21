"""Phase 5: validation-selected ensembles. Run python -m src.accuracy_v35.
Keeps production untouched. Meta-training: 2021/22 and 2022/23;
selection: 2023/24; historical benchmark: 2024/25 and 2025/26 (not untouched).
Use --skip-xg for a separate score/shot-only benchmark.
"""
import argparse
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, PoissonRegressor, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, log_loss, mean_absolute_error, mean_squared_error, recall_score
from src import accuracy_v3 as v3, accuracy_v31 as v31, accuracy_v32 as v32, accuracy_v33 as v33, accuracy_v34 as v34

C = v3.CLASSES
KEYS = ['Season','HomeTeam','AwayTeam']
META = ['2021-22','2022-23']
SELECT = '2023-24'
TEST = ['2024-25','2025-26']
EPS = 1e-9


def fold(df, groups, season, fast=False):
    """Return fixture-level out-of-season base forecasts, with rolling statistical fits."""
    tr = (df.Season < season) & (df.Season >= '2012-13')
    te = df.Season == season
    z = df.loc[te].copy().reset_index(drop=True)
    z['Total'] = z.FTHG + z.FTAG
    z['Over'] = (z.Total >= 3).astype(int)
    hist = df.loc[tr]; total = hist.FTHG + hist.FTAG
    y = hist.FTR
    P = {'TrainPrior': np.tile([(y==c).mean() for c in C], (len(z),1))}
    O = {'TrainPrior': np.full(len(z), (total>=3).mean())}
    G = {'TrainMean': np.full(len(z), total.mean()), 'TrainMedian':np.full(len(z),total.median())}
    for name in ('Score85','Shot129','XG'):
        if name not in groups: continue
        t = tr & (df.Season >= '2018-19') if name=='XG' else tr
        x, xx = groups[name].loc[t], groups[name].loc[te]
        ys = df.loc[t,'FTR']; totals = df.loc[t,'FTHG'] + df.loc[t,'FTAG']
        w = np.exp(-(z.Date.min()-df.loc[t,'Date']).dt.days.to_numpy()/365.25/5); w/=w.mean()
        rf = RandomForestClassifier(n_estimators=30 if fast else 250,max_depth=7,min_samples_leaf=20,random_state=42,n_jobs=2)
        rf.fit(x,ys,sample_weight=w)
        P[name+'_RF'] = rf.predict_proba(xx)[:,[list(rf.classes_).index(c) for c in C]]
        for alpha in (1.,5.,20.):
            h=make_pipeline(StandardScaler(),PoissonRegressor(alpha=alpha,max_iter=600)); a=clone(h)
            h.fit(x,df.loc[t,'FTHG'],poissonregressor__sample_weight=w); a.fit(x,df.loc[t,'FTAG'],poissonregressor__sample_weight=w)
            k=f'{name}_Pois{alpha:g}'
            P[k],O[k],G[k]=v3.score_probs(h.predict(xx),a.predict(xx),rho=-.08)
        lr=make_pipeline(StandardScaler(),LogisticRegression(C=.03,max_iter=2000))
        b=clone(lr); b.fit(x,(totals>=3).astype(int)); O[name+'_Direct']=b.predict_proba(xx)[:,1]
        dm=clone(lr); hm=clone(lr); dm.fit(x,(ys=='D').astype(int)); decisive=ys.ne('D').to_numpy()
        hm.fit(x.iloc[decisive],(ys.iloc[decisive]=='H').astype(int)); d=dm.predict_proba(xx)[:,1]; h=hm.predict_proba(xx)[:,1]
        P[name+'_Hurdle']=np.c_[(1-d)*(1-h),d,(1-d)*h]
        ridge=make_pipeline(StandardScaler(),Ridge(alpha=500)); ridge.fit(x,totals);G[name+'_Ridge']=np.maximum(ridge.predict(xx),0)
    blocks=((z.Date-z.Date.min()).dt.days//28).to_numpy()
    for half in (180,365,730):
        for dc in (False,True):
            k=f'{"DC" if dc else "IP"}_{half}'
            p=np.empty((len(z),3)); o=np.empty(len(z)); g=np.empty(len(z)); median=np.empty(len(z)); base=np.empty(len(z))
            for block in np.unique(blocks):
                ix=np.flatnonzero(blocks==block); fixtures=z.iloc[ix]; cutoff=fixtures.Date.min()
                model=v33.DixonColes(half,20.,dc).fit(df,cutoff)
                h,a,rho=model.rates(fixtures); f=v33.forecast(h,a,rho)
                p[ix]=f['p'];o[ix]=f['over'];g[ix]=f['mean'];median[ix]=f['median'];base[ix]=model.baseline
            P[k]=p;O[k]=o;G[k+'_Mean']=g;G[k+'_Median']=median;G[k+'_Shrink']=.5*g+.5*base
    print(season,'base forecasts complete',flush=True)
    return z,P,O,G


def join(folds):
    z=pd.concat([f[0] for f in folds],ignore_index=True)
    return (z,*[{k:np.concatenate([f[i][k] for f in folds],axis=0) for k in folds[0][i]} for i in (1,2,3)])


def check(fold_data):
    z,P,O,G=fold_data
    if z.duplicated(KEYS).any(): raise ValueError('Duplicate fixtures')
    for k,p in P.items():
        if p.shape!=(len(z),3) or not np.isfinite(p).all() or (p<0).any() or not np.allclose(p.sum(1),1): raise ValueError('Invalid outcome probabilities: '+k)
    for k,p in O.items():
        if p.shape!=(len(z),) or not np.isfinite(p).all() or (p<0).any() or (p>1).any(): raise ValueError('Invalid binary probabilities: '+k)
    for k,p in G.items():
        if p.shape!=(len(z),) or not np.isfinite(p).all() or (p<0).any(): raise ValueError('Invalid goal forecasts: '+k)


def loss(task,z,p):
    if task=='Outcome':return log_loss(z.FTR,p,labels=C)
    if task=='Over':return log_loss(z.Over,p,labels=[0,1])
    if task=='GoalsMAE':return mean_absolute_error(z.Total,p)
    return mean_squared_error(z.Total,p)


def stack_input(task,d,names):
    if task=='Outcome':return np.hstack([np.log(np.clip(d[k],EPS,1)) for k in names])
    if task=='Over':
        p=np.column_stack([d[k] for k in names]);p=np.clip(p,EPS,1-EPS);return np.log(p/(1-p))
    return np.column_stack([d[k] for k in names])


def ensemble(task,meta,sets,weight_rows):
    """Fit all ensemble parameters using meta-season forecasts only."""
    z,d=meta; names=[k for k in d if not k.startswith('Train')]
    ranked=sorted(names,key=lambda k:loss(task,z,d[k]))
    out=[dict(q) for q in sets]
    for n in (3,6,len(ranked)):
        keys=ranked[:n]; tag='All' if n==len(ranked) else str(n)
        # Equal weights and regularised non-negative simplex weights.
        base=np.stack([d[k] for k in keys]); init=np.full(len(keys),1/len(keys))
        def mix(w,a):return np.tensordot(w,a,axes=(0,0))
        for reg in (.01,.1):
            name=f'Weighted{tag}_L{reg:g}'
            def objective(w):return loss(task,z,mix(w,base)) + reg*np.sum((w-init)**2)
            fit=minimize(objective,init,method='SLSQP',bounds=[(0,1)]*len(keys),constraints=[{'type':'eq','fun':lambda w:w.sum()-1}],options={'maxiter':200,'ftol':1e-10})
            if not fit.success or not np.isfinite(fit.x).all():
                print(task,name,'optimisation failed; skipped:',fit.message,flush=True);continue
            w=np.maximum(fit.x,0);w/=w.sum()
            for k,x in zip(keys,w):weight_rows.append(dict(Task=task,Ensemble=name,BaseModel=k,Weight=x))
            for q,raw in zip(out,sets):q[name]=mix(w,np.stack([raw[k] for k in keys]))
        for q,raw in zip(out,sets):q['Equal'+tag]=mix(init,np.stack([raw[k] for k in keys]))
        for k,x in zip(keys,init):weight_rows.append(dict(Task=task,Ensemble='Equal'+tag,BaseModel=k,Weight=x))
    # Regularised stacking uses the same out-of-season forecasts, never test labels.
    for n in (6,len(ranked)):
        keys=ranked[:n];tag='All' if n==len(ranked) else str(n);x=stack_input(task,d,keys)
        for reg in (.03,.1):
            name=f'Stack{tag}_C{reg:g}'
            if task in ('Outcome','Over'):
                model=LogisticRegression(C=reg,max_iter=2000).fit(x,z.FTR if task=='Outcome' else z.Over)
                for q,raw in zip(out,sets):
                    pred=model.predict_proba(stack_input(task,raw,keys));q[name]=pred[:,[list(model.classes_).index(c) for c in C]] if task=='Outcome' else pred[:,list(model.classes_).index(1)]
            elif task=='GoalsRMSE':
                model=Ridge(alpha=1/reg).fit(x,z.Total)
                for q,raw in zip(out,sets):q[name]=np.maximum(model.predict(stack_input(task,raw,keys)),0)
            else:continue
            # Persist stack coefficients and feature ordering for auditing.
            co=np.atleast_2d(model.coef_); intercept=np.atleast_1d(model.intercept_)
            for i,row in enumerate(co):
                label=str(model.classes_[i if len(co)>1 else 1]) if task in ('Outcome','Over') else 'Total'
                for j,value in enumerate(row):
                    feature=keys[j//3]+':'+str(C[j%3]) if task=='Outcome' else keys[j]
                    weight_rows.append(dict(Task=task,Ensemble=name,BaseModel=feature,Weight=value,CoefficientClass=label))
                weight_rows.append(dict(Task=task,Ensemble=name,BaseModel='Intercept',Weight=intercept[i],CoefficientClass=label))
    return out


def evaluate(folds):
    for f in folds:check(f)
    meta=join(folds[:2]); val=folds[2]; tests=folds[3:]; sets=[val,*tests];weights=[]
    results={}
    for task,i in [('Outcome',1),('Over',2),('GoalsMAE',3),('GoalsRMSE',3)]:
        results[task]=ensemble(task,(meta[0],meta[i]),[f[i] for f in sets],weights)
    p,o,g,r=[results[k][0] for k in ('Outcome','Over','GoalsMAE','GoalsRMSE')];z=val[0]
    # Select decision rules and their associated probabilities as one unit.
    # No separate probability model is silently used for log loss.
    csearch=[];bsearch=[]
    for k,q in p.items():
        for boost in np.round(np.arange(1,2.51,.1),2):
            csearch.append(dict(Model=k,DrawBoost=boost,**v31.outcome_metrics(z.FTR.to_numpy(),q,v3.labels(q,boost))))
    for k,q in o.items():
        for threshold in np.round(np.arange(.35,.701,.01),2):
            bsearch.append(dict(Model=k,Threshold=threshold,**v34.binary_metrics(z.Over,q,threshold)))
    cs=pd.DataFrame(csearch);bs=pd.DataFrame(bsearch)
    # Compare two objectives; choose balanced version under a two-point accuracy tolerance.
    cbest=max(accuracy_score(z.FTR,v3.labels(q)) for q in p.values())
    obest=max(accuracy_score(z.Over,q>=.5) for q in o.values())
    cr=cs.loc[cs.Accuracy>=cbest-.02-1e-12].sort_values(['MacroF1','LogLoss','Accuracy'],ascending=[False,True,False]).iloc[0]
    br=bs.loc[bs.Accuracy>=obest-.02-1e-12].sort_values(['MacroF1','LogLoss','Accuracy'],ascending=[False,True,False]).iloc[0]
    cp=min(p,key=lambda k:loss('Outcome',z,p[k]));op=min(o,key=lambda k:loss('Over',z,o[k]));gm=min(g,key=lambda k:loss('GoalsMAE',z,g[k]));gr=min(r,key=lambda k:loss('GoalsRMSE',z,r[k]))
    chosen=dict(OutcomeDecision=cr.Model,DrawBoost=cr.DrawBoost,OutcomeProbability=cp,OverDecision=br.Model,OverThreshold=br.Threshold,OverProbability=op,GoalsMAE=gm,GoalsRMSE=gr)
    records=[];cls=[];binary=[];goals=[]
    for j,f in enumerate(sets):
        a=f[0];q,b,m,n=[results[k][j] for k in ('Outcome','Over','GoalsMAE','GoalsRMSE')]
        rec=a[KEYS+['Date','FTR','Total','Over']].copy();rec['Split']='Selection' if j==0 else 'HistoricalTest'
        for i,c in enumerate(C):rec['Prob'+c]=q[cr.Model][:,i];rec['ProbabilityOnly'+c]=q[cp][:,i]
        rec['DecisionProbOver']=b[br.Model];rec['ProbOver']=b[br.Model];rec['ProbabilityOnlyOver']=b[op]
        rec['DrawAdjusted']=v3.labels(q[cr.Model],cr.DrawBoost);rec['Outcome']=v3.labels(q[cr.Model]);rec['OverPrediction']=(b[br.Model]>=br.Threshold).astype(int)
        rec['Goals']=m[gm];rec['GoalsRMSE']=n[gr];rec['OutcomeModel']=cr.Model;rec['OverModel']=br.Model
        records.append(rec)
    # Report individual candidates and selected models on identical season rows.
    for split,ids in [('Selection',[0]),('HistoricalTest',[1,2]),(TEST[0],[1]),(TEST[1],[2])]:
        a=pd.concat([sets[i][0] for i in ids],ignore_index=True)
        def combine(task):
            return {k:np.concatenate([results[task][i][k] for i in ids]) for k in results[task][0]}
        q,b,m,n=[combine(k) for k in ('Outcome','Over','GoalsMAE','GoalsRMSE')]
        for k,v in q.items():cls.append(dict(Split=split,Model=k,**v31.outcome_metrics(a.FTR.to_numpy(),v)))
        for label,k,boost in [('SelectedDecision',cr.Model,cr.DrawBoost),('SelectedProbability',cp,1.)]:cls.append(dict(Split=split,Model=label,SourceModel=k,DrawBoost=boost,**v31.outcome_metrics(a.FTR.to_numpy(),q[k],v3.labels(q[k],boost))))
        for k,v in b.items():binary.append(dict(Split=split,Model=k,**v34.binary_metrics(a.Over,v)))
        for label,k,t in [('SelectedDecision',br.Model,br.Threshold),('SelectedProbability',op,.5)]:binary.append(dict(Split=split,Model=label,SourceModel=k,Threshold=t,**v34.binary_metrics(a.Over,b[k],t)))
        for task,d in [('MAE',m),('RMSE',n)]:
            for k,v in d.items():goals.append(dict(Split=split,Objective=task,Model=k,**v3.goal_metrics(a.Total,v)))
        for task,d,k in [('MAE',m,gm),('RMSE',n,gr)]:goals.append(dict(Split=split,Objective=task,Model='Selected',SourceModel=k,**v3.goal_metrics(a.Total,d[k])))
    return dict(classification=pd.DataFrame(cls),over_under=pd.DataFrame(binary),goals=pd.DataFrame(goals),selection=pd.DataFrame([chosen]),ensemble_weights=pd.DataFrame(weights),draw_threshold_search=cs,over_threshold_search=bs,predictions=pd.concat(records[1:],ignore_index=True),selection_predictions=records[0])


def compare(rec):
    rows=[]
    for ver in ('v31','v32','v33','v34','v35'):
        if ver=='v35':a=rec
        else:
            path=v3.ROOT/'outputs'/f'{ver}_experiments'/'predictions.csv'
            if not path.exists():print('Missing comparison:',path);continue
            a=pd.read_csv(path).merge(rec[KEYS+['FTR','Total','Over']],on=KEYS,validate='one_to_one',suffixes=('','_check'))
            if len(a)!=len(rec) or any(not a[k].eq(a[k+'_check']).all() for k in ('FTR','Total','Over')):raise ValueError('Fixture mismatch '+ver)
        for split,z in [('HistoricalTest',a),*list(a.groupby('Season'))]:
            bp=z.DecisionProbOver if 'DecisionProbOver' in z else z.ProbOver
            pred=z.OverPrediction if 'OverPrediction' in z else (bp>=.5).astype(int)
            d=dict(Version=ver,Split=split,Matches=len(z),Accuracy=accuracy_score(z.FTR,z.DrawAdjusted),MacroF1=f1_score(z.FTR,z.DrawAdjusted,labels=C,average='macro',zero_division=0),DrawRecall=recall_score(z.FTR,z.DrawAdjusted,labels=['D'],average='macro',zero_division=0),OverAccuracy=accuracy_score(z.Over,pred),OverMacroF1=f1_score(z.Over,pred,average='macro',zero_division=0),OverDecisionLogLoss=log_loss(z.Over,bp,labels=[0,1]),GoalMAE=mean_absolute_error(z.Total,z.Goals) if 'Goals' in z else np.nan)
            # Older exports (notably V3.1) do not store outcome probabilities.
            # Never reconstruct probabilities from hard labels or substitute a different model.
            pc = next((cols for cols in (['DecisionProb'+c for c in C],
                                         ['Prob'+c for c in C])
                       if all(c in z.columns for c in cols)), None)
            d['OutcomeDecisionLogLoss'] = (log_loss(z.FTR,z[pc].to_numpy(),labels=C)
                                            if pc is not None else np.nan)
            d['OutcomeProbabilityAvailable'] = pc is not None
            rows.append(d)
    return pd.DataFrame(rows)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--skip-xg',action='store_true');args=parser.parse_args()
    df=v3.load_history();df['Date']=pd.to_datetime(df.Date).dt.normalize();df=df.sort_values(['Date','HomeTeam','AwayTeam']).reset_index(drop=True)
    if args.skip_xg:groups={'Shot129':v31.features(df)};groups['Score85']=groups['Shot129'].iloc[:,:85]
    else:
        df,_=v32.merge_xg(df);df=df.sort_values(['Date','HomeTeam','AwayTeam']).reset_index(drop=True)
        groups=v32.build_features(df,v32.load_championship())
        if 'XG' not in groups:raise ValueError('Expected XG feature group; use --skip-xg for separate experiment')
    folds=[fold(df,groups,s) for s in META+[SELECT]+TEST]
    result=evaluate(folds)
    out=v3.ROOT/'outputs'/('v35_no_xg_experiments' if args.skip_xg else 'v35_experiments');out.mkdir(parents=True,exist_ok=True)
    # Persist completed training results before attempting legacy comparisons.
    for k,d in result.items():d.to_csv(out/(k+'.csv'),index=False)
    result['paired_comparison']=compare(result['predictions'])
    result['paired_comparison'].to_csv(out/'paired_comparison.csv',index=False)
    print('\nSelection (2023-24 only):\n',result['selection'].to_string(index=False))
    for k in ('classification','over_under','goals'):
        t=result[k].query("Split == 'HistoricalTest'");print('\n',k,'historical benchmark:');print(t[t.Model.str.startswith('Selected')].to_string(index=False));print(t.sort_values('MAE' if k=='goals' else 'MacroF1',ascending=k=='goals').head(8).to_string(index=False))
    print('\nSame-fixture comparison:\n',result['paired_comparison'].to_string(index=False));print('Saved:',out)

if __name__=='__main__':main()
