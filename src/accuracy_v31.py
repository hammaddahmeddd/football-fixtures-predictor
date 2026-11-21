"""V3.1 isolated chronological experiments. Run python -m src.accuracy_v31.
Historical shots are lagged; missing live statistics must not be silently imputed at deployment.
"""
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, PoissonRegressor, Ridge, LinearRegression
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor, RandomForestClassifier
from sklearn.metrics import log_loss, f1_score, mean_absolute_error, precision_score, recall_score, roc_auc_score, confusion_matrix
from src import accuracy_v3 as v3

OUT=v3.ROOT/'outputs'/'v31_experiments'
STAT=['HS','AS','HST','AST','HC','AC']

def features(df):
    base=v3.make_features(df)
    history=defaultdict(list); rows=[]
    for _,day in df.groupby('Date',sort=True):
        for m in day.itertuples(index=False):
            r={}
            for side,team in [('H',m.HomeTeam),('A',m.AwayTeam)]:
                for n in (5,15,30):
                    a=np.array(history[team][-n:],float).reshape(-1,6)
                    prior=np.array([13.,13.,4.5,4.5,5.,5.])
                    valid=np.isfinite(a); count=valid.sum(0)
                    means=(np.nansum(a,0)+5*prior)/(count+5)
                    for k,x in zip(['shots','shots_allowed','sot','sot_allowed','corners','corners_allowed'],means):r[f'{side}_{n}_{k}']=x
                    r[f'{side}_{n}_coverage']=float(count.mean()/n)
            r['sot_diff']=r['H_15_sot']-r['A_15_sot'];r['shot_total']=r['H_15_shots']+r['A_15_shots']
            rows.append(r)
        for m in day.itertuples(index=False):
            h,a,ht,at,hc,ac=[getattr(m,k,np.nan) for k in STAT]
            history[m.HomeTeam].append([h,a,ht,at,hc,ac]);history[m.AwayTeam].append([a,h,at,ht,ac,hc])
    return pd.concat([base,pd.DataFrame(rows,index=df.index)],axis=1)

def external_models():
    models={}
    try:
        from xgboost import XGBClassifier
        models['XGBoost']=XGBClassifier(n_estimators=250,max_depth=3,learning_rate=.035,subsample=.85,colsample_bytree=.8,reg_lambda=8,n_jobs=2,random_state=42)
    except ImportError: print('XGBoost unavailable; skipped')
    try:
        from lightgbm import LGBMClassifier
        models['LightGBM']=LGBMClassifier(n_estimators=220,num_leaves=12,learning_rate=.035,min_child_samples=50,reg_lambda=8,n_jobs=2,verbosity=-1,random_state=42)
    except ImportError: print('LightGBM unavailable; skipped')
    try:
        from catboost import CatBoostClassifier
        models['CatBoost']=CatBoostClassifier(iterations=300,depth=4,learning_rate=.035,l2_leaf_reg=8,verbose=False,thread_count=2,random_seed=42,allow_writing_files=False)
    except ImportError: print('CatBoost unavailable; skipped')
    return models

def fold(df,X,B,season):
    tr=(df.Season<season)&(df.Season!='2011-12'); te=df.Season==season
    x,z=X.loc[tr],X.loc[te]; b,c=B.loc[tr],B.loc[te]
    y=df.loc[tr,'FTR']; actual=df.loc[te].copy(); actual['Total']=actual.FTHG+actual.FTAG;actual['Over']=(actual.Total>=3).astype(int)
    yi=np.searchsorted(v3.CLASSES,y); total=df.loc[tr,'FTHG']+df.loc[tr,'FTAG']
    age=(actual.Date.min()-df.loc[tr,'Date']).dt.days.to_numpy()/365.25; w=np.exp(-age/5);w/=w.mean()
    P={}; G={}; O={}
    for name,xx,zz in [('ScoreOnly',b,c),('ShotEnhanced',x,z)]:
        # Controlled feature ablation with identical model settings.
        rf=RandomForestClassifier(n_estimators=250,max_depth=7,min_samples_leaf=20,n_jobs=2,random_state=42)
        rf.fit(xx,yi,sample_weight=w);P[name+'RF']=rf.predict_proba(zz)
        for alpha in (.1,1.,5.):
            hm=make_pipeline(StandardScaler(),PoissonRegressor(alpha=alpha,max_iter=600));am=clone(hm)
            hm.fit(xx,df.loc[tr,'FTHG'],poissonregressor__sample_weight=w);am.fit(xx,df.loc[tr,'FTAG'],poissonregressor__sample_weight=w)
            for rho in (0.,-.05,-.1):
                k=f'{name}Poisson{alpha}_{rho}';P[k],O[k],G[k]=v3.score_probs(hm.predict(zz),am.predict(zz),rho)
    models={'Logistic':make_pipeline(StandardScaler(),LogisticRegression(C=.05,max_iter=2000)), 'HistGB':HistGradientBoostingClassifier(max_iter=150,max_leaf_nodes=12,min_samples_leaf=50,l2_regularization=10,early_stopping=False,random_state=42),**external_models()}
    for name,m in models.items():
        print(season,name,flush=True);m.fit(x,yi);P[name]=m.predict_proba(z)[:,[list(m.classes_).index(i) for i in range(3)]]
    # Factor outcome into draw vs decisive, then home vs away conditional on decisive.
    for name,model in [('Logistic',make_pipeline(StandardScaler(),LogisticRegression(C=.05,max_iter=2000))),('HistGB',HistGradientBoostingClassifier(max_iter=120,max_leaf_nodes=10,min_samples_leaf=60,l2_regularization=10,early_stopping=False,random_state=42))]:
        dm=clone(model);wm=clone(model);dm.fit(x,(y=='D').astype(int)); decisive=(y!='D').to_numpy();wm.fit(x.iloc[decisive],(y.iloc[decisive]=='H').astype(int));d=dm.predict_proba(z)[:,1];h=wm.predict_proba(z)[:,1];P['DrawHurdle'+name]=np.column_stack([(1-d)*(1-h),d,(1-d)*h])
    G['TrainMean']=np.full(len(z),total.mean());O['TrainPrior']=np.full(len(z),(total>=3).mean())
    for name,m in [('Linear',make_pipeline(StandardScaler(),LinearRegression())),('Ridge',make_pipeline(StandardScaler(),Ridge(alpha=100))),('Poisson',make_pipeline(StandardScaler(),PoissonRegressor(alpha=1,max_iter=600))),('HistGB',HistGradientBoostingRegressor(max_iter=100,max_leaf_nodes=10,min_samples_leaf=60,l2_regularization=12,early_stopping=False,random_state=42))]:
        m.fit(x,total);G[name]=np.maximum(m.predict(z),0)
    # Boosting regression candidates use the same pre-match features.
    from xgboost import XGBRegressor
    from lightgbm import LGBMRegressor
    from catboost import CatBoostRegressor
    for name,m in [
        ('XGBoostReg',XGBRegressor(n_estimators=200,max_depth=2,learning_rate=.035,reg_lambda=10,n_jobs=2,random_state=42)),
        ('LightGBMReg',LGBMRegressor(n_estimators=200,num_leaves=8,learning_rate=.035,min_child_samples=60,reg_lambda=10,n_jobs=2,verbosity=-1,random_state=42)),
        ('CatBoostReg',CatBoostRegressor(iterations=220,depth=4,learning_rate=.035,l2_leaf_reg=10,verbose=False,thread_count=2,random_seed=42,allow_writing_files=False))]:
        m.fit(x,total);G[name]=np.maximum(m.predict(z),0)
    binary=make_pipeline(StandardScaler(),LogisticRegression(C=.03,max_iter=2000));binary.fit(x,(total>=3).astype(int));O['Logistic']=binary.predict_proba(z)[:,1]
    return actual,P,G,O

def outcome_metrics(y,p,pred=None):
    pred=v3.labels(p) if pred is None else pred
    metrics=v3.class_metrics(y,p,pred)
    metrics['MacroPrecision']=precision_score(y,pred,labels=v3.CLASSES,average='macro',zero_division=0)
    metrics['MacroRecall']=recall_score(y,pred,labels=v3.CLASSES,average='macro',zero_division=0)
    for c in v3.CLASSES:
        metrics[f'{c}_Precision']=precision_score(y,pred,labels=[c],average='macro',zero_division=0)
        metrics[f'{c}_Recall']=recall_score(y,pred,labels=[c],average='macro',zero_division=0)
        metrics[f'{c}_F1']=f1_score(y,pred,labels=[c],average='macro',zero_division=0)
    return metrics


def binary_metrics(y,p):
    metrics=v3.over_metrics(y,p)
    metrics['ROC_AUC']=roc_auc_score(y,p) if len(np.unique(y))==2 else np.nan
    metrics['UnderRecall']=recall_score(y,p>=.5,pos_label=0,zero_division=0)
    metrics['UnderPrecision']=precision_score(y,p>=.5,pos_label=0,zero_division=0)
    return metrics


def main():
    df=v3.load_history(); missing=[k for k in STAT if k not in df or df[k].notna().sum()==0]
    if missing:raise ValueError(f'Missing historical statistics: {missing}. Run V3 with --refresh-data first.')
    X=features(df);B=X.iloc[:,:85]; assert B.shape[1]==85
    print('Matches:',len(df),'Features:',X.shape[1],flush=True)
    print(df[STAT].notna().mean().to_string())
    folds=[fold(df,X,B,s) for s in v3.VALID+v3.TEST]; v,t=folds[:2],folds[2:];va=pd.concat([f[0] for f in v],ignore_index=True);ta=pd.concat([f[0] for f in t],ignore_index=True)
    vy,ty=va.FTR.to_numpy(),ta.FTR.to_numpy();vp=v3.combine(v,v[0][1],1);tp=v3.combine(t,t[0][1],1);vg=v3.combine(v,v[0][2],2);tg=v3.combine(t,t[0][2],2);vo=v3.combine(v,v[0][3],3);to=v3.combine(t,t[0][3],3)
    # Train a regularized probability stack on out-of-training-season forecasts, evaluate only later seasons.
    names=sorted(vp,key=lambda n:log_loss(vy,vp[n],labels=v3.CLASSES))[:6]
    def design(p):return np.column_stack([np.log(np.clip(p[n],1e-6,1)) for n in names])
    stack=LogisticRegression(C=.05,max_iter=2000).fit(design(vp),vy);tp['Stack']=stack.predict_proba(design(tp))[:,[list(stack.classes_).index(c) for c in v3.CLASSES]]
    # Do not select stack or estimate its validation performance using its own fitted rows.
    best=min(vp,key=lambda n:log_loss(vy,vp[n],labels=v3.CLASSES)); boost=max(np.arange(1,2.51,.1),key=lambda b:f1_score(vy,v3.labels(vp[best],b),average='macro'))
    gbest=min(vg,key=lambda n:mean_absolute_error(va.Total,vg[n])); obest=min(vo,key=lambda n:log_loss(va.Over,np.column_stack([1-vo[n],vo[n]]),labels=[0,1]))
    cr=[];gr=[];br=[]
    for n,p in vp.items():cr.append(dict(Model=n,Split='Validation',**outcome_metrics(vy,p)))
    cr.append(dict(Model=best+'_DrawDecision',Split='Validation',**outcome_metrics(vy,vp[best],v3.labels(vp[best],boost))))
    for n,p in vg.items():gr.append(dict(Model=n,Split='Validation',**v3.goal_metrics(va.Total,p)))
    for n,p in vo.items():br.append(dict(Model=n,Split='Validation',**binary_metrics(va.Over,p)))
    for split,mask in [('HistoricalTest',np.ones(len(ta),bool)),*((s,(ta.Season==s).to_numpy()) for s in v3.TEST)]:
        for n,p in tp.items():cr.append(dict(Model=n,Split=split,**outcome_metrics(ty[mask],p[mask])))
        cr.append(dict(Model=best+'_DrawDecision',Split=split,**outcome_metrics(ty[mask],tp[best][mask],v3.labels(tp[best][mask],boost))))
        for n,p in tg.items():gr.append(dict(Model=n,Split=split,**v3.goal_metrics(ta.Total[mask],p[mask])))
        for n,p in to.items():br.append(dict(Model=n,Split=split,**binary_metrics(ta.Over[mask],p[mask])))
    OUT.mkdir(parents=True,exist_ok=True)
    for name,rows in [('classification',cr),('goals',gr),('over_under',br)]:pd.DataFrame(rows).to_csv(OUT/(name+'.csv'),index=False)
    pd.DataFrame([dict(Outcome=best,DrawBoost=boost,Goals=gbest,OverUnder=obest,StackMembers=';'.join(names))]).to_csv(OUT/'selection.csv',index=False)
    records=ta[['Date','Season','HomeTeam','AwayTeam','FTR','Total','Over']].copy()
    records['Outcome']=v3.labels(tp[best]);records['DrawAdjusted']=v3.labels(tp[best],boost);records['Stack']=v3.labels(tp['Stack']);records['Goals']=tg[gbest];records['ProbOver']=to[obest];records.to_csv(OUT/'predictions.csv',index=False)
    # Compare V3 and V3.1 on exactly the same saved match identities.
    old=v3.OUT/'historical_predictions.csv'
    if old.exists():
        prior=pd.read_csv(old); keys=['Season','HomeTeam','AwayTeam']; merged=records.merge(prior,on=keys,validate='one_to_one',suffixes=('','_v3'))
        if len(merged)!=len(records) or not (merged.FTR==merged.FTR_v3).all():raise ValueError('V3 comparison fixture mismatch')
        compare=[]
        for season,g in merged.groupby('Season'):
            for name,col in [('V3 RandomForest','RandomForest_Prediction'),('V3 DrawAdjusted','DrawAdjusted_v3'),('V3.1 Selected','Outcome'),('V3.1 DrawAdjusted','DrawAdjusted'),('V3.1 Stack','Stack')]:
                compare.append(dict(Season=season,Model=name,Accuracy=(g[col]==g.FTR).mean(),MacroF1=f1_score(g.FTR,g[col],average='macro')))
        pd.DataFrame(compare).to_csv(OUT/'paired_comparison.csv',index=False)
    # V2 comparison is restricted to the identical 2025/26 fixtures.
    v2=v3.ROOT/'outputs'/'metrics'/'rl_test_predictions.csv'
    if v2.exists():
        prior=pd.read_csv(v2); keys=['Season','HomeTeam','AwayTeam']
        this=records[records.Season=='2025-26']
        merged=this.merge(prior,on=keys,validate='one_to_one',suffixes=('','_v2'))
        if len(merged)!=len(this) or not (merged.FTR==merged.FTR_v2).all():raise ValueError('V2 comparison fixture mismatch')
        compare=[]
        for name,col in [('V2 KNN','KNN_Prediction'),('V2 SVM','SVM_Prediction'),('V2 DecisionTree','Decision_Tree_Prediction'),('V2 RandomForest','Random_Forest_Prediction'),('V3.1 Selected','Outcome'),('V3.1 DrawAdjusted','DrawAdjusted'),('V3.1 Stack','Stack')]:
            y=merged.FTR;p=merged[col]
            compare.append(dict(Season='2025-26',Model=name,Accuracy=(y==p).mean(),MacroF1=f1_score(y,p,labels=v3.CLASSES,average='macro',zero_division=0),DrawRecall=recall_score(y,p,labels=['D'],average='macro',zero_division=0)))
        pd.DataFrame(compare).to_csv(OUT/'v2_paired_comparison.csv',index=False)
    else: print('V2 predictions unavailable; skipped V2 paired comparison.')
    print('Validation selections:',best,boost,gbest,obest)
    for rows in (cr,gr,br):print(pd.DataFrame(rows).query("Split == 'HistoricalTest'").to_string(index=False))
    print('Saved:',OUT)

if __name__=='__main__':main()
