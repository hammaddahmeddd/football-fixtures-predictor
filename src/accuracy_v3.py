"""V3 candidate benchmark. Copy to src/accuracy_v3.py and run python -m src.accuracy_v3.
No production files are overwritten. Score-only features are available from the live API.
"""
from pathlib import Path
from collections import defaultdict
import argparse
import numpy as np
import pandas as pd
from scipy.stats import poisson
from sklearn.base import clone
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, PoissonRegressor, Ridge
from sklearn.metrics import accuracy_score, f1_score, log_loss, balanced_accuracy_score, mean_absolute_error, r2_score, silhouette_score, mean_squared_error, recall_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs' / 'v3_experiments'
CACHE = ROOT / 'data' / 'v3_experiments' / 'history.csv'
CLASSES = np.array(['A','D','H'])
VALID = ['2022-23','2023-24']
TEST = ['2024-25','2025-26']
EPS = 1e-9


def season_name(y):
    return f'{y}-{str(y+1)[-2:]}'


def load_history(refresh=False):
    if CACHE.exists() and not refresh:
        df = pd.read_csv(CACHE, parse_dates=['Date'])
    else:
        from src.data_collection import download_season
        chunks = []
        for y in range(2011, 2026):
            s = season_name(y)
            part = download_season(s)
            part['Date'] = pd.to_datetime(part['Date'], dayfirst=True, format='mixed', errors='coerce')
            part = part.dropna(subset=['Date','HomeTeam','AwayTeam','FTHG','FTAG','FTR']).copy()
            if len(part) != 380 or part.duplicated(['HomeTeam','AwayTeam']).any():
                raise ValueError(f'Incomplete or duplicate season: {s}, {len(part)} rows')
            chunks.append(part)
        df = pd.concat(chunks, ignore_index=True)
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(CACHE,index=False)
    if len(df) != 5700 or set(df['Season']) != {season_name(y) for y in range(2011,2026)}:
        raise ValueError('Expected exactly 15 complete seasons and 5700 matches')
    if df.duplicated(['Season','HomeTeam','AwayTeam']).any():
        raise ValueError('Duplicate fixtures')
    if not df.groupby('Season').size().eq(380).all():
        raise ValueError('Incomplete season')
    scores = df[['FTHG','FTAG']].to_numpy(dtype=float)
    if not np.isfinite(scores).all() or (scores<0).any() or (scores%1!=0).any():
        raise ValueError('Invalid scores')
    expected = np.where(scores[:,0]>scores[:,1],'H',np.where(scores[:,0]<scores[:,1],'A','D'))
    if not np.array_equal(expected,df.FTR.to_numpy()):
        raise ValueError('Inconsistent match result')
    return df.sort_values(['Date','HomeTeam','AwayTeam']).reset_index(drop=True)


class History:
    def __init__(self):
        self.games = defaultdict(list)
        self.elo = defaultdict(lambda:1500.)
        self.season = defaultdict(lambda:np.zeros(4))
        self.league = []

    def features(self, date, season, home, away):
        result = {}
        prior = np.array(self.league[-380:] or [[1.5,1.2]])
        league_home, league_away = prior.mean(axis=0)
        for prefix, team, venue in [('H',home,1),('A',away,0)]:
            matches = self.games[team]
            result[prefix+'_elo'] = self.elo[team]
            result[prefix+'_experience'] = min(len(matches),38)
            result[prefix+'_rest'] = min(30,(date-matches[-1][0]).days) if matches else 7
            for window in (5,10,20):
                recent = matches[-window:]
                arr = np.array([m[2:] for m in recent],dtype=float).reshape(-1,5)
                # GF, GA, points, opponent Elo, Elo-adjusted points
                avg = (arr.sum(axis=0)+3*np.array([1.35,1.35,1.25,1500.,0.]))/(len(arr)+3)
                for k,val in zip(['gf','ga','pts','opp','adj'],avg):
                    result[f'{prefix}_{window}_{k}']=val
                result[f'{prefix}_{window}_draw']=(sum(m[4]==1 for m in recent)+.75)/(len(recent)+3)
                result[f'{prefix}_{window}_clean']=(sum(m[3]==0 for m in recent)+.75)/(len(recent)+3)
                result[f'{prefix}_{window}_blank']=(sum(m[2]==0 for m in recent)+.75)/(len(recent)+3)
                result[f'{prefix}_{window}_over']=(sum(m[2]+m[3]>=3 for m in recent)+1.65)/(len(recent)+3)
                result[f'{prefix}_{window}_goalvar']=float(np.var(arr[:,0]+arr[:,1])) if len(arr)>2 else 2.8
            v=[m for m in matches if m[1]==venue][-12:]
            gf0,ga0=(league_home,league_away) if venue else (league_away,league_home)
            result[prefix+'_venue_gf']=(sum(m[2] for m in v)+5*gf0)/(len(v)+5)
            result[prefix+'_venue_ga']=(sum(m[3] for m in v)+5*ga0)/(len(v)+5)
            played,pts,gf,ga=self.season[(season,team)]
            result[prefix+'_season_ppg']=(pts+5*1.25)/(played+5)
            result[prefix+'_season_gd']=(gf-ga)/(played+5)
            result[prefix+'_season_played']=played
        result['elo_diff']=result['H_elo']-result['A_elo']
        result['abs_elo_diff']=abs(result['elo_diff']+65)
        result['form_diff']=result['H_10_adj']-result['A_10_adj']
        result['season_form_diff']=result['H_season_ppg']-result['A_season_ppg']
        result['home_rate']=result['H_venue_gf']*result['A_venue_ga']/max(league_home,.1)
        result['away_rate']=result['A_venue_gf']*result['H_venue_ga']/max(league_away,.1)
        result['total_rate']=result['home_rate']+result['away_rate']
        result['rate_diff']=result['home_rate']-result['away_rate']
        result['abs_rate_diff']=abs(result['rate_diff'])
        return result

    def update_day(self, day):
        updates=[]
        for m in day.itertuples(index=False):
            h,a=m.HomeTeam,m.AwayTeam
            hg,ag=int(m.FTHG),int(m.FTAG)
            he,ae=self.elo[h],self.elo[a]
            p=1/(1+10**((ae-he-65)/400))
            outcome=1 if hg>ag else .5 if hg==ag else 0
            delta=20*(1+.25*min(abs(hg-ag),4))*(outcome-p)
            hp,ap=(3,0) if hg>ag else (0,3) if hg<ag else (1,1)
            updates.append((h,a,delta))
            self.games[h].append((m.Date,1,hg,ag,hp,ae,hp-3*p))
            self.games[a].append((m.Date,0,ag,hg,ap,he,ap-3*(1-p)))
            self.season[(m.Season,h)]+=np.array([1,hp,hg,ag])
            self.season[(m.Season,a)]+=np.array([1,ap,ag,hg])
            self.league.append((hg,ag))
        for h,a,delta in updates:
            self.elo[h]+=delta
            self.elo[a]-=delta


def make_features(df):
    history=History(); rows=[]
    for date,day in df.groupby('Date',sort=True):
        for m in day.itertuples(index=False):
            rows.append(history.features(date,m.Season,m.HomeTeam,m.AwayTeam))
        history.update_day(day)
    X=pd.DataFrame(rows,index=df.index)
    if not np.isfinite(X.to_numpy()).all():
        raise ValueError('Non-finite features')
    return X


def classifiers():
    return {
        'KNN':make_pipeline(StandardScaler(),KNeighborsClassifier(n_neighbors=60,weights='distance')),
        'SVM':make_pipeline(StandardScaler(),SVC(C=.5,probability=True,random_state=42)),
        'DecisionTree':DecisionTreeClassifier(max_depth=4,min_samples_leaf=70,random_state=42),
        'RandomForest':RandomForestClassifier(n_estimators=250,max_depth=7,min_samples_leaf=20,n_jobs=-1,random_state=42),
        'ExtraTrees':ExtraTreesClassifier(n_estimators=250,max_depth=8,min_samples_leaf=20,n_jobs=-1,random_state=42),
        'Logistic':make_pipeline(StandardScaler(),LogisticRegression(C=.1,max_iter=2500)),
        'HistGradientBoosting':HistGradientBoostingClassifier(max_iter=120,max_leaf_nodes=12,min_samples_leaf=45,l2_regularization=8,random_state=42,early_stopping=False),
    }


def score_probs(hg,ag,rho=0):
    # Independent Poisson rates, with a Dixon-Coles-style low-score adjustment.
    h=np.clip(np.asarray(hg),.05,6); a=np.clip(np.asarray(ag),.05,6)
    grid=np.arange(20)
    joint=poisson.pmf(grid[None,:],h[:,None])[:,:,None]*poisson.pmf(grid[None,:],a[:,None])[:,None,:]
    for i,j,tau in [(0,0,1-h*a*rho),(0,1,1+h*rho),(1,0,1+a*rho),(1,1,np.full(len(h),1-rho))]:
        joint[:,i,j]*=np.maximum(tau,.001)
    joint/=joint.sum(axis=(1,2),keepdims=True)
    i,j=np.meshgrid(grid,grid,indexing='ij')
    p=np.column_stack([joint[:,i<j].sum(1),joint[:,i==j].sum(1),joint[:,i>j].sum(1)])
    over=joint[:,i+j>=3].sum(1)
    mean=(joint*(i+j)).sum(axis=(1,2))
    return p,over,mean


def fold(df,X,season):
    train=(df.Season<season)&(df.Season!='2011-12'); test=df.Season==season
    x,z=X.loc[train],X.loc[test]; y=df.loc[train,'FTR']; actual=df.loc[test].copy()
    actual['Total']=actual.FTHG+actual.FTAG
    actual['Over']=(actual.Total>=3).astype(int)
    age=(actual.Date.min()-df.loc[train,'Date']).dt.days.to_numpy()/365.25
    w=np.exp(-age/5); w/=w.mean()
    p={}; goals={}; over={}
    historical_prior=np.array([(y==c).mean() for c in CLASSES]); p['TrainPrior']=np.tile(historical_prior,(len(z),1))
    for name,m in classifiers().items():
        print(season,name,flush=True)
        if name in ['RandomForest','ExtraTrees','DecisionTree','HistGradientBoosting']:
            m.fit(x,y,sample_weight=w)
        elif name=='Logistic':
            m.fit(x,y,logisticregression__sample_weight=w)
        else:
            m.fit(x,y)
        p[name]=m.predict_proba(z)[:,[list(m.classes_).index(c) for c in CLASSES]]
    h_train=df.loc[train,'FTHG']; a_train=df.loc[train,'FTAG']; total=h_train+a_train
    goals['TrainMean']=np.full(len(z),total.mean()); over['TrainPrior']=np.full(len(z),(total>=3).mean())
    for name,m in [('Ridge',make_pipeline(StandardScaler(),Ridge(alpha=50))),('Poisson',make_pipeline(StandardScaler(),PoissonRegressor(alpha=1,max_iter=600))),('HistGradientBoosting',HistGradientBoostingRegressor(max_iter=100,max_leaf_nodes=10,l2_regularization=10,early_stopping=False,random_state=42))]:
        m.fit(x,total); goals[name]=np.maximum(m.predict(z),0)
    rates={'VenueRate':(z.home_rate.to_numpy(),z.away_rate.to_numpy())}
    for alpha in (.1,1.,5.):
        hm=make_pipeline(StandardScaler(),PoissonRegressor(alpha=alpha,max_iter=600)); am=clone(hm)
        hm.fit(x,h_train,poissonregressor__sample_weight=w); am.fit(x,a_train,poissonregressor__sample_weight=w)
        rates[f'PairedPoisson{alpha}']=(hm.predict(z),am.predict(z))
    for name,(h,a) in rates.items():
        for rho in (0.,-.05,-.10):
            key=f'{name}_rho{rho}'
            p[key],over[key],goals[key]=score_probs(h,a,rho)
    binary=make_pipeline(StandardScaler(),LogisticRegression(C=.05,max_iter=1500))
    binary.fit(x,(total>=3).astype(int)); over['Logistic']=binary.predict_proba(z)[:,1]
    return actual,p,goals,over


def combine(folds,keys,slot):
    return {k:np.concatenate([f[slot][k] for f in folds]) for k in keys}


def labels(p,boost=1.):
    adjusted=p.copy(); adjusted[:,1]*=boost
    return CLASSES[adjusted.argmax(1)]


def class_metrics(y,p,pred=None):
    pred=labels(p) if pred is None else pred
    return dict(Accuracy=accuracy_score(y,pred),MacroF1=f1_score(y,pred,average='macro',labels=CLASSES,zero_division=0),BalancedAccuracy=balanced_accuracy_score(y,pred),DrawRecall=recall_score(y,pred,labels=['D'],average='macro',zero_division=0),PredictedDraws=int((pred=='D').sum()),LogLoss=log_loss(y,p,labels=CLASSES),Brier=np.mean(np.sum((p-(np.asarray(y)[:,None]==CLASSES))**2,axis=1)))


def goal_metrics(y,p):
    return dict(MAE=mean_absolute_error(y,p),RMSE=mean_squared_error(y,p)**.5,R2=r2_score(y,p))


def over_metrics(y,p):
    return dict(Accuracy=accuracy_score(y,p>=.5),MacroF1=f1_score(y,p>=.5,average='macro',zero_division=0),LogLoss=log_loss(y,np.column_stack([1-p,p]),labels=[0,1]),Brier=np.mean((p-y)**2),PredictedUnder=int((p<.5).sum()))


def states(probs,names):
    stack=np.array([probs[n] for n in names]); c=stack.mean(0); votes=stack.argmax(2)
    return [(int(len(set(votes[:,i]))),int(np.digitize(c[i].max(),[.45,.60])),int(np.digitize(c[i,1],[.22,.30]))) for i in range(len(c))]


def select_policy(vp,y,tp,names):
    # One-action contextual bandit: smoothed empirical reward (gamma=0).
    # Each validation forecast is out of its training season.
    rewards=defaultdict(lambda:np.zeros(len(names))); counts=defaultdict(int)
    prior=np.array([accuracy_score(y,labels(vp[n])) for n in names])
    for i,s in enumerate(states(vp,names)):
        rewards[s]+=np.array([labels(vp[n][i:i+1])[0]==y[i] for n in names]); counts[s]+=1
    def apply(probs):
        ids=[int(np.argmax((rewards[s]+20*prior)/(counts[s]+20))) for s in states(probs,names)]
        return np.array([probs[names[j]][i] for i,j in enumerate(ids)]),[names[j] for j in ids]
    return apply(tp)


def cluster_check(X,df):
    train=df.Season<'2024-25'; parts=[]
    for side,team in [('H','HomeTeam'),('A','AwayTeam')]:
        columns=[side+'_'+x for x in ['10_pts','10_gf','10_ga','10_draw','elo']]
        p=X.loc[train,columns].copy(); p.columns=['pts','gf','ga','draw','elo'];p['Season']=df.loc[train,'Season'];p['Team']=df.loc[train,team];parts.append(p)
    profiles=pd.concat(parts).groupby(['Season','Team']).mean(); z=StandardScaler().fit_transform(profiles)
    return pd.DataFrame([{'Clusters':k,'Silhouette':silhouette_score(z,KMeans(n_clusters=k,n_init=20,random_state=42).fit_predict(z))} for k in range(2,7)])


def season_backtest(df, seasons=TEST, simulations=1000):
    """Compare frozen and dynamic score-rate simulations at a historical cutoff.

    Uses only the first ~100 matches to initialise each season, plus prior seasons.
    Dynamic rates update from simulated scores, never future actual scores.
    This is a separate prototype, not a backtest of the deployed V2 simulator.
    """
    rows=[]; team_rows=[]
    for season in seasons:
        season_df=df[df.Season==season].sort_values('Date').reset_index(drop=True)
        cutoff=season_df.loc[99,'Date']
        observed=season_df[season_df.Date<=cutoff]
        future=season_df[season_df.Date>cutoff]
        teams=sorted(set(season_df.HomeTeam)|set(season_df.AwayTeam)); ids={t:i for i,t in enumerate(teams)}
        n=len(teams)
        prior=df[df.Date<=cutoff].sort_values('Date')
        league=prior.tail(380)[['FTHG','FTAG']].mean().to_numpy()
        # Home GF, home GA, away GF, away GA; use 12 matches at each venue.
        rates=np.zeros((n,4))
        for team,i in ids.items():
            h=prior[prior.HomeTeam==team].tail(12); a=prior[prior.AwayTeam==team].tail(12)
            rates[i]=[(h.FTHG.sum()+5*league[0])/(len(h)+5),(h.FTAG.sum()+5*league[1])/(len(h)+5),(a.FTAG.sum()+5*league[1])/(len(a)+5),(a.FTHG.sum()+5*league[0])/(len(a)+5)]
        def table(matches):
            pts=np.zeros(n); gf=np.zeros(n); ga=np.zeros(n)
            for m in matches.itertuples(index=False):
                h,a=ids[m.HomeTeam],ids[m.AwayTeam]; x,y=m.FTHG,m.FTAG
                pts[h]+=3*(x>y)+(x==y); pts[a]+=3*(y>x)+(x==y)
                gf[h]+=x;gf[a]+=y;ga[h]+=y;ga[a]+=x
            return pts,gf,ga
        actual,gf,ga=table(season_df); order=np.lexsort((np.arange(n),-gf,-(gf-ga),-actual)); actual_rank=np.empty(n,int);actual_rank[order]=np.arange(1,n+1)
        initial,igf,iga=table(observed)
        for mode in ['FrozenRates','DynamicRates']:
            rng=np.random.default_rng(42); r=np.broadcast_to(rates,(simulations,n,4)).copy()
            pts=np.broadcast_to(initial,(simulations,n)).copy(); gf=np.broadcast_to(igf,(simulations,n)).copy();ga=np.broadcast_to(iga,(simulations,n)).copy()
            for m in future.itertuples(index=False):
                h,a=ids[m.HomeTeam],ids[m.AwayTeam]
                hr=np.clip(r[:,h,0]*r[:,a,3]/league[0],.1,5);ar=np.clip(r[:,a,2]*r[:,h,1]/league[1],.1,5)
                x,y=rng.poisson(hr),rng.poisson(ar)
                pts[:,h]+=3*(x>y)+(x==y);pts[:,a]+=3*(y>x)+(x==y)
                gf[:,h]+=x;gf[:,a]+=y;ga[:,h]+=y;ga[:,a]+=x
                if mode=='DynamicRates':
                    for t,k,g in [(h,0,x),(h,1,y),(a,2,y),(a,3,x)]:r[:,t,k]=.92*r[:,t,k]+.08*g
            ranks=np.empty((simulations,n),int)
            for j in range(simulations):
                order=np.lexsort((np.arange(n),-gf[j],-(gf[j]-ga[j]),-pts[j])); ranks[j,order]=np.arange(1,n+1)
            title=(ranks==1).mean(0); top4=(ranks<=4).mean(0); releg=(ranks>n-3).mean(0)
            rows.append(dict(Season=season,Mode=mode,ObservedMatches=len(observed),Simulations=simulations,PointsMAE=mean_absolute_error(actual,pts.mean(0)),RankMAE=mean_absolute_error(actual_rank,ranks.mean(0)),TitleBrier=np.mean((title-(actual_rank==1))**2),Top4Brier=np.mean((top4-(actual_rank<=4))**2),RelegationBrier=np.mean((releg-(actual_rank>n-3))**2)))
            for i,team in enumerate(teams):team_rows.append(dict(Season=season,Mode=mode,Team=team,ActualPoints=actual[i],ExpectedPoints=pts[:,i].mean(),ActualRank=actual_rank[i],ExpectedRank=ranks[:,i].mean(),TitleProbability=title[i],Top4Probability=top4[i],RelegationProbability=releg[i]))
    return pd.DataFrame(rows),pd.DataFrame(team_rows)


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--refresh-data',action='store_true'); args=parser.parse_args()
    df=load_history(args.refresh_data); X=make_features(df); OUT.mkdir(parents=True,exist_ok=True)
    print('Matches',len(df),'features',X.shape[1],flush=True)
    folds=[fold(df,X,s) for s in VALID+TEST]; v,t=folds[:2],folds[2:]
    va=pd.concat([f[0] for f in v],ignore_index=True); ta=pd.concat([f[0] for f in t],ignore_index=True)
    vy,ty=va.FTR.to_numpy(),ta.FTR.to_numpy()
    vp=combine(v,v[0][1],1); tp=combine(t,t[0][1],1)
    vg=combine(v,v[0][2],2); tg=combine(t,t[0][2],2)
    vo=combine(v,v[0][3],3); to=combine(t,t[0][3],3)
    # Choose ensemble members/weights only on validation forecasts.
    names=[n for n in vp if n!='TrainPrior']; ranked=sorted(names,key=lambda n:log_loss(vy,vp[n],labels=CLASSES))
    for k in (3,5,8):
        members=ranked[:k]; losses=np.array([log_loss(vy,vp[n],labels=CLASSES) for n in members]); weights=np.exp(-20*(losses-losses.min()));weights/=weights.sum()
        vp[f'WeightedVote{k}']=sum(w*vp[n] for n,w in zip(members,weights));tp[f'WeightedVote{k}']=sum(w*tp[n] for n,w in zip(members,weights))
    best=min(vp,key=lambda n:log_loss(vy,vp[n],labels=CLASSES)); boost=max([1.,1.1,1.2,1.3,1.45,1.6],key=lambda b:f1_score(vy,labels(vp[best],b),average='macro',zero_division=0))
    rows=[]
    for name in vp:
        for split,y,p in [('Validation',vy,vp[name]),('HistoricalTest',ty,tp[name])]: rows.append(dict(Model=name,Split=split,**class_metrics(y,p)))
    rows.append(dict(Model=best+'_DrawDecision',Split='Validation',**class_metrics(vy,vp[best],labels(vp[best],boost))))
    rows.append(dict(Model=best+'_DrawDecision',Split='HistoricalTest',**class_metrics(ty,tp[best],labels(tp[best],boost))))
    bandit,selected=select_policy(vp,vy,tp,['KNN','SVM','DecisionTree','RandomForest','ExtraTrees','Logistic','HistGradientBoosting',best] if best not in ['KNN','SVM','DecisionTree','RandomForest','ExtraTrees','Logistic','HistGradientBoosting'] else ['KNN','SVM','DecisionTree','RandomForest','ExtraTrees','Logistic','HistGradientBoosting'])
    rows.append(dict(Model='SmoothedBandit',Split='HistoricalTest',**class_metrics(ty,bandit)))
    always=np.tile([.0,.0,1.],(len(ty),1)); rows.append(dict(Model='AlwaysHome',Split='HistoricalTest',**class_metrics(ty,always)))
    # Separate regression selection by MAE and binary selection by log loss.
    gr=[]; br=[]
    for name in vg:
        for split,y,p in [('Validation',va.Total,vg[name]),('HistoricalTest',ta.Total,tg[name])]:gr.append(dict(Model=name,Split=split,**goal_metrics(y,p)))
    for name in vo:
        for split,y,p in [('Validation',va.Over,vo[name]),('HistoricalTest',ta.Over,to[name])]:br.append(dict(Model=name,Split=split,**over_metrics(y,p)))
    goal_best=min(vg,key=lambda n:mean_absolute_error(va.Total,vg[n])); over_best=min(vo,key=lambda n:log_loss(va.Over,np.column_stack([1-vo[n],vo[n]]),labels=[0,1]))
    classifications=pd.DataFrame(rows); regression=pd.DataFrame(gr); binary=pd.DataFrame(br)
    classifications.to_csv(OUT/'classification.csv',index=False); regression.to_csv(OUT/'goals.csv',index=False); binary.to_csv(OUT/'over_under.csv',index=False); cluster_check(X,df).to_csv(OUT/'clustering.csv',index=False)
    records=ta[['Date','Season','HomeTeam','AwayTeam','FTR','Total','Over']].copy()
    for name,p in tp.items():
        records[name+'_Prediction']=labels(p)
        for j,c in enumerate(CLASSES): records[name+'_P_'+c]=p[:,j]
    records['DrawAdjusted']=labels(tp[best],boost);records['BanditModel']=selected;records['BanditPrediction']=labels(bandit);records['PredictedGoals']=tg[goal_best];records['ProbOver']=to[over_best];records.to_csv(OUT/'historical_predictions.csv',index=False)
    pd.DataFrame([{'OutcomeModel':best,'DrawMultiplier':boost,'GoalModel':goal_best,'OverUnderModel':over_best,'Features':X.shape[1],'Matches':len(df)}]).to_csv(OUT/'validation_selection.csv',index=False)
    print('\nHistorical benchmark (not an untouched holdout):\n',classifications[classifications.Split=='HistoricalTest'].sort_values('LogLoss').to_string(index=False))
    print('\nSelected on validation:',best,goal_best,over_best,'draw multiplier',boost)
    print(regression[(regression.Split=='HistoricalTest') & regression.Model.isin([goal_best,'TrainMean'])].to_string(index=False));print(binary[(binary.Split=='HistoricalTest') & binary.Model.isin([over_best,'TrainPrior'])].to_string(index=False)); print('Saved:',OUT)
    sim,sim_teams=season_backtest(df);sim.to_csv(OUT/'season_backtest.csv',index=False);sim_teams.to_csv(OUT/'season_backtest_teams.csv',index=False);print('Season backtest:\n',sim.to_string(index=False))
    per_season=[]
    for season in TEST:
        mask=ta.Season.to_numpy()==season
        for name,p in tp.items(): per_season.append(dict(Season=season,Model=name,**class_metrics(ty[mask],p[mask])))
        per_season.append(dict(Season=season,Model=best+'_DrawDecision',**class_metrics(ty[mask],tp[best][mask],labels(tp[best][mask],boost))))
        per_season.append(dict(Season=season,Model='SmoothedBandit',**class_metrics(ty[mask],bandit[mask])))
    pd.DataFrame(per_season).to_csv(OUT/'classification_by_season.csv',index=False)

if __name__=='__main__': main()
