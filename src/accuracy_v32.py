"""Phase 2: data integration and controlled feature ablation.
Run python -m src.accuracy_v32. Does not change production files.
Historical xG: Understat via xGTables archive, 2017/18–2025/26.
Championship: Football-Data.co.uk E1 regular-season results.
"""
import argparse
from collections import defaultdict
from io import StringIO
from pathlib import Path
import re
import unicodedata
import numpy as np
import pandas as pd
import requests
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, PoissonRegressor, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss, mean_absolute_error, f1_score
from src import accuracy_v3 as v3
from src import accuracy_v31 as v31

OUT = v3.ROOT / 'outputs' / 'v32_experiments'
CACHE = v3.ROOT / 'data' / 'v32_experiments'
KEYS = ['Season', 'HomeTeam', 'AwayTeam']


def cached_csv(url, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return pd.read_csv(path, low_memory=False)
    print('Downloading:', url, flush=True)
    response = requests.get(url, timeout=45)
    response.raise_for_status()
    df = pd.read_csv(StringIO(response.text), low_memory=False)
    # The caller validates the parsed schema before saving the cache.
    return df


def club_key(name):
    name = unicodedata.normalize('NFKD', str(name).lower())
    name = ''.join(c for c in name if not unicodedata.combining(c))
    name = re.sub(r'[^a-z0-9 ]', '', name)
    name = re.sub(r'\b(fc|afc|football club)\b', '', name)
    name = ' '.join(name.split())
    aliases = {
        'manchester city':'man city', 'manchester united':'man united',
        'man utd':'man united', 'newcastle united':'newcastle',
        'tottenham hotspur':'tottenham', 'spurs':'tottenham',
        'brighton and hove albion':'brighton', 'brighton hove albion':'brighton',
        'wolverhampton wanderers':'wolves', 'wolverhampton':'wolves',
        'west ham united':'west ham', 'leeds united':'leeds',
        'leicester city':'leicester', 'norwich city':'norwich',
        'cardiff city':'cardiff', 'swansea city':'swansea',
        'stoke city':'stoke', 'luton town':'luton', 'ipswich town':'ipswich',
        'huddersfield town':'huddersfield', 'sheffield utd':'sheffield united',
        'nottingham forest':'nottm forest', 'nottingham':'nottm forest',
        'west bromwich albion':'west brom', 'west bromwich':'west brom',
        'queens park rangers':'qpr', 'bournemouth':'bournemouth',
    }
    return aliases.get(name, name)


def load_championship(first=2011, last=2024):
    frames = []
    for year in range(first, last+1):
        season = v3.season_name(year)
        code = f'{year%100:02d}{(year+1)%100:02d}'
        path = CACHE / f'E1_{season}.csv'
        raw = cached_csv(f'https://www.football-data.co.uk/mmz4281/{code}/E1.csv', path)
        cols = ['Date','HomeTeam','AwayTeam','FTHG','FTAG']
        if not set(cols).issubset(raw): raise ValueError(f'Invalid Championship schema: {season}')
        d = raw[cols].copy()
        d['Date'] = pd.to_datetime(d.Date, dayfirst=True, format='mixed', errors='coerce')
        for c in ['FTHG','FTAG']: d[c] = pd.to_numeric(d[c], errors='coerce')
        d = d.dropna().copy()
        d['Season'] = season
        if len(d)!=552 or d.duplicated(['HomeTeam','AwayTeam']).any():
            raise ValueError(f'Expected 552 regular-season Championship matches in {season}; got {len(d)}')
        g = d[['FTHG','FTAG']].to_numpy(float)
        if not np.isfinite(g).all() or (g<0).any() or (g%1!=0).any(): raise ValueError('Invalid Championship scores')
        if not path.exists(): raw.to_csv(path,index=False)
        frames.append(d)
    return pd.concat(frames, ignore_index=True)


def merge_xg(df):
    """Strict fixture identity and actual score verification; no fuzzy matching."""
    result = df.copy()
    result['XGH'] = np.nan; result['XGA'] = np.nan
    audit=[]
    for year in range(2017,2026):
        season = v3.season_name(year)
        path = CACHE / f'xg_{season}.csv'
        raw = cached_csv(f'https://www.xgtables.io/history/epl/{year}/fixtures.csv',path)
        needed={'date','home','away','score','home_xg','away_xg'}
        if not needed.issubset(raw.columns): raise ValueError(f'Invalid xG schema for {season}: {list(raw.columns)}')
        u=raw.copy()
        u['hk']=u.home.map(club_key); u['ak']=u.away.map(club_key)
        if u.duplicated(['hk','ak']).any(): raise ValueError(f'Duplicate xG fixture keys {season}')
        u['xdate']=pd.to_datetime(u.date,format='mixed',errors='coerce',utc=True).dt.tz_convert(None).dt.normalize()
        score=u.score.astype(str).str.extract(r'^\s*(\d+)\s*[-–—:]\s*(\d+)\s*$').apply(pd.to_numeric,errors='coerce')
        u['hg']=score[0];u['ag']=score[1]
        # Prefer published xG when both values exist. Otherwise use shot totals.
        for side in ['home','away']:
            u[side+'_xg']=pd.to_numeric(u[side+'_xg'],errors='coerce')
        if {'home_published_xg','away_published_xg'}.issubset(u.columns):
            published=u[['home_published_xg','away_published_xg']].apply(pd.to_numeric,errors='coerce')
            valid=published.notna().all(axis=1) & published.ge(0).all(axis=1)
            u.loc[valid,['home_xg','away_xg']]=published.loc[valid].to_numpy()
        if 'totals_unreliable' in u:
            bad=u.totals_unreliable.astype(str).str.lower().isin(['true','1','yes','1.0'])
            u.loc[bad,['home_xg','away_xg']]=np.nan
        m=result[result.Season==season].copy();m['rowid']=m.index
        m['hk']=m.HomeTeam.map(club_key);m['ak']=m.AwayTeam.map(club_key)
        z=m.merge(u[['hk','ak','xdate','hg','ag','home_xg','away_xg']],on=['hk','ak'],how='left',validate='one_to_one')
        matched=(z.FTHG==z.hg)&(z.FTAG==z.ag)&((z.Date.dt.normalize()-z.xdate).abs().dt.days<=1)
        good=matched & z[['home_xg','away_xg']].notna().all(axis=1) & z[['home_xg','away_xg']].ge(0).all(axis=1) & z[['home_xg','away_xg']].le(20).all(axis=1)
        coverage=float(good.mean())
        print(f'{season} xG verified: {good.sum()}/{len(z)}',flush=True)
        audit.append(dict(Season=season,Fixtures=len(z),IdentityScoreDateMatches=int(matched.sum()),UsableXG=int(good.sum()),Coverage=coverage))
        if coverage<.95:
            print(z.loc[~good,['HomeTeam','AwayTeam','Date','FTHG','FTAG','xdate','hg','ag']].head(12).to_string(index=False))
            raise ValueError(f'xG matching coverage below 95% for {season}; inspect aliases/data before proceeding')
        result.loc[z.loc[good,'rowid'],'XGH']=z.loc[good,'home_xg'].to_numpy()
        result.loc[z.loc[good,'rowid'],'XGA']=z.loc[good,'away_xg'].to_numpy()
        if not path.exists(): raw.to_csv(path,index=False)
    return result,pd.DataFrame(audit)


def championship_profiles(champ):
    profiles={}
    if champ is None:return profiles
    for season,d in champ.groupby('Season'):
        for t in set(d.HomeTeam)|set(d.AwayTeam):
            h=d[d.HomeTeam==t];a=d[d.AwayTeam==t]
            gf=np.r_[h.FTHG,a.FTAG];ga=np.r_[h.FTAG,a.FTHG]
            if len(gf)!=46:raise ValueError(f'Incomplete Championship team: {t} {season}')
            profiles[(season,club_key(t))]=(d.Date.max(),float(np.mean(np.where(gf>ga,3,np.where(gf==ga,1,0)))),float(gf.mean()),float(ga.mean()))
    return profiles


def build_features(df, champ=None):
    """Same-day fixtures are all featurised before any same-day results enter history."""
    df=df.sort_values(['Date','HomeTeam','AwayTeam']).reset_index(drop=True)
    shots=v31.features(df); scores=shots.iloc[:,:85].copy()
    hist=defaultdict(list); prior_seasons=defaultdict(set); head=defaultdict(list)
    profiles=championship_profiles(champ); hstate=v3.History(); strength=[]; promoted=[]; xg=[]
    for date,day in df.groupby('Date',sort=True):
        pending=[]
        for m in day.itertuples(index=False):
            r={}; c={}; x={}; current=int(m.Season[:4]); prev=v3.season_name(current-1)
            for side,team,opp,venue in [('H',m.HomeTeam,m.AwayTeam,1),('A',m.AwayTeam,m.HomeTeam,0)]:
                games=hist[team]; recent=games[-30:]; pts_prev=prior_seasons[prev]
                for days in (7,14,30):
                    r[f'{side}_matches_{days}d']=sum((date-g[0]).days<=days for g in recent)
                r[f'{side}_days_since_pl']=min(730,(date-games[-1][0]).days) if games else 730
                for n in (10,30):
                    z=recent[-n:]; weights=np.array([.5**(max(0,(date-g[0]).days)/180) for g in z]); den=weights.sum()+5
                    for j,key,base in [(1,'gf',1.35),(2,'ga',1.35),(3,'attack_resid',0),(4,'defence_resid',0)]:
                        r[f'{side}_{n}_weighted_{key}']=(sum(w*g[j] for w,g in zip(weights,z))+5*base)/den
                    valid=[(g,w) for g,w in zip(z,weights) if np.isfinite(g[5]) and np.isfinite(g[6])]
                    x[f'{side}_{n}_xg_for']=(sum(w*g[5] for g,w in valid)+5*1.35)/(sum(w for g,w in valid)+5)
                    x[f'{side}_{n}_xg_against']=(sum(w*g[6] for g,w in valid)+5*1.35)/(sum(w for g,w in valid)+5)
                    x[f'{side}_{n}_xg_coverage']=len(valid)/n
                prior=profiles.get((prev,club_key(team)))
                newcomer=bool(pts_prev and team not in pts_prev)
                usable=bool(newcomer and prior and prior[0]<date)
                c[f'{side}_newly_promoted']=int(newcomer)
                c[f'{side}_champ_available']=int(usable)
                for k,val,default in zip(['ppg','gf','ga'],prior[1:] if usable else [1.4,1.35,1.35],[1.4,1.35,1.35]):c[f'{side}_champ_{k}']=val
                direct=head[(team,opp)][-5:]
                r[f'{side}_h2h_ppg']=(sum(direct)+5*1.25)/(len(direct)+5)
                r[f'{side}_h2h_count']=len(direct)
            r['congestion_diff']=r['H_matches_14d']-r['A_matches_14d']
            r['adjusted_attack_diff']=r['H_30_weighted_attack_resid']-r['A_30_weighted_attack_resid']
            x['xg_rate_home']=x['H_30_xg_for']*x['A_30_xg_against']/1.35
            x['xg_rate_away']=x['A_30_xg_for']*x['H_30_xg_against']/1.35
            x['xg_total_rate']=x['xg_rate_home']+x['xg_rate_away']
            strength.append(r);promoted.append(c);xg.append(x)
            # Opponent-adjustment is against the pre-match expected scoring rates, never final season averages.
            pre=hstate.features(date,m.Season,m.HomeTeam,m.AwayTeam)
            hg,ag=int(m.FTHG),int(m.FTAG); hx=getattr(m,'XGH',np.nan);ax=getattr(m,'XGA',np.nan)
            pending.append((m,hg,ag,hx,ax,pre['home_rate'],pre['away_rate']))
        for m,hg,ag,hx,ax,hr,ar in pending:
            hist[m.HomeTeam].append((date,hg,ag,hg-hr,ag-ar,hx,ax))
            hist[m.AwayTeam].append((date,ag,hg,ag-ar,hg-hr,ax,hx))
            hp,ap=(3,0) if hg>ag else (0,3) if hg<ag else (1,1)
            head[(m.HomeTeam,m.AwayTeam)].append(hp);head[(m.AwayTeam,m.HomeTeam)].append(ap)
            prior_seasons[m.Season].update([m.HomeTeam,m.AwayTeam])
        hstate.update_day(day)
    s=pd.DataFrame(strength);c=pd.DataFrame(promoted);x=pd.DataFrame(xg)
    groups={'Score85':scores,'Shot129':shots,'Strength':pd.concat([shots,s],axis=1),'Promotion':pd.concat([shots,s,c],axis=1)}
    if df[['XGH','XGA']].notna().all(axis=1).any():groups['XG']=pd.concat([shots,s,c,x],axis=1)
    for k,z in groups.items():
        if not np.isfinite(z.to_numpy()).all() or not z.columns.is_unique:raise ValueError(f'Invalid features {k}')
    return groups


def fit_fold(df,groups,season,start,fast=False):
    tr=(df.Season<season)&(df.Season>=start);te=df.Season==season
    y=df.loc[tr,'FTR']; a=df.loc[te].copy();a['Total']=a.FTHG+a.FTAG;a['Over']=(a.Total>=3).astype(int)
    total=df.loc[tr,'FTHG']+df.loc[tr,'FTAG'];age=(a.Date.min()-df.loc[tr,'Date']).dt.days.to_numpy()/365.25
    w=np.exp(-age/5);w/=w.mean();p={};g={};o={}
    p['TrainPrior']=np.tile([(y==k).mean() for k in v3.CLASSES],(len(a),1));g['TrainMean']=np.full(len(a),total.mean());o['TrainPrior']=np.full(len(a),(total>=3).mean())
    for group,X in groups.items():
        print(season,group,len(X.columns),'features',flush=True)
        x,z=X.loc[tr],X.loc[te]
        rf=RandomForestClassifier(n_estimators=60 if fast else 250,max_depth=7,min_samples_leaf=20,n_jobs=2,random_state=42)
        rf.fit(x,y,sample_weight=w);p[group+'_RF']=rf.predict_proba(z)[:,[list(rf.classes_).index(k) for k in v3.CLASSES]]
        for reg in (.03,.1):
            clf=make_pipeline(StandardScaler(),LogisticRegression(C=reg,max_iter=1500));clf.fit(x,y,logisticregression__sample_weight=w)
            p[f'{group}_Logistic{reg}']=clf.predict_proba(z)[:,[list(clf.classes_).index(k) for k in v3.CLASSES]]
        for alpha in (1.,5.,20.):
            h=make_pipeline(StandardScaler(),PoissonRegressor(alpha=alpha,max_iter=600));b=clone(h)
            h.fit(x,df.loc[tr,'FTHG'],poissonregressor__sample_weight=w);b.fit(x,df.loc[tr,'FTAG'],poissonregressor__sample_weight=w)
            k=f'{group}_Poisson{alpha}';p[k],o[k],g[k]=v3.score_probs(h.predict(z),b.predict(z),rho=-.08)
        for alpha in (100.,500.):
            model=make_pipeline(StandardScaler(),Ridge(alpha=alpha));model.fit(x,total,ridge__sample_weight=w);g[f'{group}_Ridge{alpha}']=np.maximum(model.predict(z),0)
        binary=make_pipeline(StandardScaler(),LogisticRegression(C=.03,max_iter=1500));binary.fit(x,(total>=3).astype(int),logisticregression__sample_weight=w);o[group+'_Logistic']=binary.predict_proba(z)[:,1]
    return a,p,g,o


def evaluate(df,groups,start,fast=False):
    folds=[fit_fold(df,groups,s,start,fast) for s in v3.VALID+v3.TEST]
    val=folds[:2];test=folds[2:];va=pd.concat([f[0] for f in val],ignore_index=True);ta=pd.concat([f[0] for f in test],ignore_index=True)
    vp,vg,vo=[v3.combine(val,val[0][i],i) for i in (1,2,3)];tp,tg,to=[v3.combine(test,test[0][i],i) for i in (1,2,3)]
    best=min(vp,key=lambda k:log_loss(va.FTR,vp[k],labels=v3.CLASSES));gbest=min(vg,key=lambda k:mean_absolute_error(va.Total,vg[k]));obest=min(vo,key=lambda k:log_loss(va.Over,np.c_[1-vo[k],vo[k]],labels=[0,1]))
    boost=max(np.arange(1,2.51,.1),key=lambda b:f1_score(va.FTR,v3.labels(vp[best],b),labels=v3.CLASSES,average='macro'))
    cr=[];gr=[];br=[]
    for split,a,ps,gs,os in [('Validation',va,vp,vg,vo),('HistoricalTest',ta,tp,tg,to)]+[(s,f[0],f[1],f[2],f[3]) for s,f in zip(v3.TEST,test)]:
        for k,p in ps.items():cr.append(dict(Model=k,Split=split,**v31.outcome_metrics(a.FTR.to_numpy(),p)))
        cr.append(dict(Model=best+'_DrawDecision',Split=split,**v31.outcome_metrics(a.FTR.to_numpy(),ps[best],v3.labels(ps[best],boost))))
        for k,g in gs.items():gr.append(dict(Model=k,Split=split,**v3.goal_metrics(a.Total,g)))
        for k,o in os.items():br.append(dict(Model=k,Split=split,**v31.binary_metrics(a.Over,o)))
    selection=pd.DataFrame([dict(TrainingStart=start,Outcome=best,DrawBoost=boost,Goals=gbest,OverUnder=obest)])
    records=ta[KEYS+['Date','FTR','Total','Over']].copy();records['Outcome']=v3.labels(tp[best]);records['DrawAdjusted']=v3.labels(tp[best],boost);records['Goals']=tg[gbest];records['ProbOver']=to[obest]
    for k,i in zip(v3.CLASSES,range(3)):records['Prob'+k]=tp[best][:,i]
    return {'classification':pd.DataFrame(cr),'goals':pd.DataFrame(gr),'over_under':pd.DataFrame(br),'selection':selection,'predictions':records}


def paired_comparison(records):
    path=v31.OUT/'predictions.csv'
    if not path.exists():return None
    prior=pd.read_csv(path);m=records.merge(prior,on=KEYS,validate='one_to_one',suffixes=('_v32','_v31'))
    if len(m)!=len(records) or len(m)!=len(prior) or not (m.FTR_v32==m.FTR_v31).all() or not (m.Total_v32==m.Total_v31).all():raise ValueError('V3.1 paired comparison mismatch')
    rows=[]
    for s,d in [('HistoricalTest',m),*list(m.groupby('Season'))]:
        for version in ['v31','v32']:
            y=d.FTR_v32;pred=d['DrawAdjusted_'+version];o=d['ProbOver_'+version];g=d['Goals_'+version]
            rows.append(dict(Season=s,Version=version,Accuracy=float((y==pred).mean()),MacroF1=f1_score(y,pred,labels=v3.CLASSES,average='macro'),GoalMAE=mean_absolute_error(d.Total_v32,g),OverAccuracy=float(((o>=.5)==d.Over_v32).mean()),OverLogLoss=log_loss(d.Over_v32,np.c_[1-o,o],labels=[0,1])))
    return pd.DataFrame(rows)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--skip-xg',action='store_true');parser.add_argument('--skip-championship',action='store_true');args=parser.parse_args()
    df=v3.load_history();champ=None if args.skip_championship else load_championship()
    if args.skip_xg:
        df['XGH']=np.nan;df['XGA']=np.nan;audit=pd.DataFrame(columns=['Season','Fixtures','IdentityScoreDateMatches','UsableXG','Coverage'])
    else:df,audit=merge_xg(df)
    groups=build_features(df,champ);start='2018-19' if 'XG' in groups else '2012-13'
    print('EPL matches:',len(df),'Championship matches:',0 if champ is None else len(champ),'Common training start:',start,flush=True)
    print({k:len(z.columns) for k,z in groups.items()},flush=True)
    result=evaluate(df,groups,start);OUT.mkdir(parents=True,exist_ok=True)
    for k,d in result.items():d.to_csv(OUT/(k+'.csv'),index=False)
    audit.to_csv(OUT/'xg_coverage.csv',index=False)
    # Fixed-hyperparameter ablation: report like-for-like models on identical fixtures.
    ablation=[]
    for group in groups:
        for metric_file,suffixes in [('classification',['RF','Logistic0.03','Poisson5.0']),('goals',['Poisson5.0','Ridge500.0']),('over_under',['Poisson5.0','Logistic'])]:
            for suffix in suffixes:
                rows=result[metric_file].loc[result[metric_file].Model==group+'_'+suffix].copy()
                rows.insert(0,'FeatureGroup',group);rows.insert(1,'Task',metric_file);rows.insert(2,'FixedModel',suffix)
                ablation.append(rows)
    pd.concat(ablation,ignore_index=True).to_csv(OUT/'feature_ablation.csv',index=False)
    coverage=[]
    p=groups['Promotion']
    for season,idx in df.groupby('Season').groups.items():
        promoted=int(p.loc[idx,['H_newly_promoted','A_newly_promoted']].to_numpy().sum())
        available=int(p.loc[idx,['H_champ_available','A_champ_available']].to_numpy().sum())
        coverage.append(dict(Season=season,PromotedTeamFixtureAppearances=promoted,ChampionshipProfilesAvailable=available))
    pd.DataFrame(coverage).to_csv(OUT/'promotion_coverage.csv',index=False)
    pd.DataFrame([dict(Group=k,Features=z.shape[1],TrainingStart=start) for k,z in groups.items()]).to_csv(OUT/'feature_groups.csv',index=False)
    pd.DataFrame([dict(Group=k,Feature=c) for k,z in groups.items() for c in z.columns]).to_csv(OUT/'feature_dictionary.csv',index=False)
    pair=paired_comparison(result['predictions'])
    if pair is not None:pair.to_csv(OUT/'v31_paired_comparison.csv',index=False);print(pair.to_string(index=False))
    else:print('No saved V3.1 predictions available; paired comparison skipped.')
    print('Validation selections:',result['selection'].to_string(index=False))
    for k,metric in [('classification','LogLoss'),('goals','MAE'),('over_under','LogLoss')]:
        print('\n'+k+' historical benchmark, not an untouched holdout:');print(result[k].query("Split == 'HistoricalTest'").sort_values(metric).head(12).to_string(index=False))
    print('Saved:',OUT)

if __name__=='__main__':main()
