"""V4 shadow refresh, settlement and season simulation. Run python -m src.refresh_v4."""
from pathlib import Path
import argparse, json, uuid
import numpy as np
import pandas as pd
import joblib
import requests
from sklearn.metrics import accuracy_score, f1_score, log_loss, mean_absolute_error


def parse_schedule(matches, normalize, season):
    rows=[]
    for m in matches:
        h=normalize(m['homeTeam']['name']); a=normalize(m['awayTeam']['name'])
        status=m['status']; score=m.get('score',{}).get('fullTime',{})
        rows.append(dict(MatchID=m['id'],Matchday=m.get('matchday'),KickoffUTC=m['utcDate'],HomeTeam=h,AwayTeam=a,Season=season,Status=status,FTHG=score.get('home') if status=='FINISHED' else None,FTAG=score.get('away') if status=='FINISHED' else None))
    d=pd.DataFrame(rows); d.KickoffUTC=pd.to_datetime(d.KickoffUTC,utc=True,errors='raise'); d['Date']=d.KickoffUTC.dt.tz_convert(None).dt.normalize()
    teams=sorted(set(d.HomeTeam)|set(d.AwayTeam)); expected={(h,a) for h in teams for a in teams if h!=a}
    if len(teams)!=20 or len(d)!=380 or d.KickoffUTC.isna().any() or d.MatchID.isna().any() or d.MatchID.duplicated().any() or set(zip(d.HomeTeam,d.AwayTeam))!=expected: raise ValueError('Expected complete 20-team double round-robin schedule')
    for c in ['FTHG','FTAG']: d[c]=pd.to_numeric(d[c],errors='coerce')
    f=d.Status.eq('FINISHED'); scores=d.loc[f,['FTHG','FTAG']].to_numpy()
    if not np.isfinite(scores).all() or (scores<0).any() or (scores%1!=0).any(): raise ValueError('Invalid finished scores')
    d['FTR']=pd.Series(np.where(d.FTHG>d.FTAG,'H',np.where(d.FTHG<d.FTAG,'A','D')),index=d.index).where(f)
    return d.sort_values(['KickoffUTC','MatchID']).reset_index(drop=True)


def simulate(schedule, remaining, grids, n=5000, seed=42):
    teams=sorted(set(schedule.HomeTeam)); idx={t:i for i,t in enumerate(teams)}; rng=np.random.default_rng(seed)
    if n<1 or len(grids)!=len(remaining): raise ValueError('Invalid simulation inputs')
    pts=np.zeros((n,len(teams)),dtype=int); gf=pts.copy(); ga=pts.copy(); played=np.zeros(len(teams),int)
    def add(h,a,x,y):
        pts[:,h]+=3*(x>y)+(x==y); pts[:,a]+=3*(y>x)+(x==y); gf[:,h]+=x; ga[:,h]+=y; gf[:,a]+=y; ga[:,a]+=x
    for m in schedule[schedule.Status=='FINISHED'].itertuples():
        h,a=idx[m.HomeTeam],idx[m.AwayTeam]; add(h,a,int(m.FTHG),int(m.FTAG)); played[[h,a]]+=1
    actual=pts[0].copy()
    for m,g in zip(remaining.itertuples(),grids):
        if not np.isfinite(g).all() or (g<0).any() or not np.isclose(g.sum(),1): raise ValueError('Invalid score distribution')
        x,y=np.divmod(rng.choice(g.size,size=n,p=g.ravel()/g.sum()),g.shape[1]); add(idx[m.HomeTeam],idx[m.AwayTeam],x,y)
    # Points, goal difference, goals scored; random tie-break for remaining exact ties.
    order=np.lexsort((rng.random(pts.shape),-gf,-(gf-ga),-pts),axis=1); ranks=np.empty_like(order); np.put_along_axis(ranks,order,np.broadcast_to(np.arange(1,len(teams)+1),order.shape),axis=1)
    return pd.DataFrame(dict(Team=teams,Played=played,CurrentPoints=actual,ExpectedPoints=pts.mean(0),PointsP05=np.quantile(pts,.05,axis=0),PointsP95=np.quantile(pts,.95,axis=0),ExpectedRank=ranks.mean(0),TitleProbability=(ranks==1).mean(0),Top4Probability=(ranks<=4).mean(0),RelegationProbability=(ranks>=18).mean(0))).sort_values('ExpectedRank')


def settle(archive,schedule,now):
    if archive.empty: return pd.DataFrame(),pd.DataFrame()
    keys=['MatchID','HomeTeam','AwayTeam','Season','KickoffUTC','PredictionGeneratedUTC']; a=archive.copy()
    if not set(keys).issubset(a): raise ValueError('Archive schema mismatch')
    a.MatchID=pd.to_numeric(a.MatchID,errors='raise').astype('int64'); a['Generated']=pd.to_datetime(a.PredictionGeneratedUTC,utc=True,errors='coerce',format='mixed'); a['ArchivedKickoff']=pd.to_datetime(a.KickoffUTC,utc=True,errors='coerce',format='mixed')
    s=schedule[['MatchID','HomeTeam','AwayTeam','Season','KickoffUTC','Status','FTHG','FTAG','FTR']].rename(columns={c:'Actual'+c for c in ['HomeTeam','AwayTeam','Season','KickoffUTC','FTHG','FTAG','FTR','Status']}); a=a.merge(s,on='MatchID',how='left',validate='many_to_one'); a.ActualKickoffUTC=pd.to_datetime(a.ActualKickoffUTC,utc=True)
    good=a.Generated.notna()&a.ArchivedKickoff.notna()&(a.Generated<a.ArchivedKickoff)&(a.Generated<a.ActualKickoffUTC)&(a.Generated<=now)
    for c in ['HomeTeam','AwayTeam','Season']: good &= a[c].eq(a['Actual'+c])
    a=a[good].sort_values('Generated',kind='stable').drop_duplicates('MatchID').copy(); a['Settlement']=np.where(a.ActualStatus.eq('FINISHED')&(a.ActualKickoffUTC<=now),'Settled','Pending'); done=a[a.Settlement=='Settled']; metrics=[]
    for version,z in [('All',done),*list(done.groupby('ModelVersion'))] if 'ModelVersion' in done else [('All',done)]:
        if z.empty: continue
        for c in ['PredictedResult','BalancedResult','QPrediction']+[c for c in z if c.endswith('_Prediction')]:
            if c not in z: continue
            v=z[z[c].isin(['A','D','H'])];
            if v.empty: continue
            metrics.append(dict(ModelVersion=version,Task='Outcome',Model=c,Matches=len(v),Accuracy=accuracy_score(v.ActualFTR,v[c]),MacroF1=f1_score(v.ActualFTR,v[c],labels=['A','D','H'],average='macro',zero_division=0)))
        for prefix in ['Primary','Balanced','Outcome','Q']:
            cols=[f'{prefix}_P_{c}' for c in ['A','D','H']]
            if not set(cols).issubset(z): continue
            p=z[cols].to_numpy(float); ok=np.isfinite(p).all(1)&(p>=0).all(1)&np.isclose(p.sum(1),1)
            if ok.any(): metrics.append(dict(ModelVersion=version,Task='Probability',Model=prefix,Matches=int(ok.sum()),LogLoss=log_loss(z.ActualFTR[ok],p[ok],labels=['A','D','H'])))
        total=z.ActualFTHG+z.ActualFTAG
        if 'OverUnder2_5' in z: metrics.append(dict(ModelVersion=version,Task='OverUnder',Model='Selected',Matches=len(z),Accuracy=accuracy_score(total>=3,z.OverUnder2_5.eq('Over'))))
        for c in ['PredictedTotalGoals','ExpectedTotalGoals']:
            if c in z and z[c].notna().all(): metrics.append(dict(ModelVersion=version,Task='Goals',Model=c,Matches=len(z),MAE=mean_absolute_error(total,z[c])))
        if 'Score1' in z: metrics.append(dict(ModelVersion=version,Task='ExactScore',Model='DC365',Matches=len(z),Accuracy=float((z.Score1==z.ActualFTHG.astype(int).astype(str)+'-'+z.ActualFTAG.astype(int).astype(str)).mean())))
    return a,pd.DataFrame(metrics)


def main():
    from src import production_v4 as v4, predictions as old
    p=argparse.ArgumentParser(); p.add_argument('--simulations',type=int,default=5000); args=p.parse_args(); out=v4.OUT
    if not old.FOOTBALL_API_KEY: raise ValueError('FOOTBALL_API_KEY missing')
    response=requests.get(old.FOOTBALL_API_BASE_URL+'/competitions/PL/matches',headers={'X-Auth-Token':old.FOOTBALL_API_KEY},params={'season':old.CURRENT_SEASON_START_YEAR},timeout=old.REQUEST_TIMEOUT)
    response.raise_for_status(); schedule=parse_schedule(response.json()['matches'],old.normalize_team_name,old.CURRENT_SEASON); now=pd.Timestamp.now(tz='UTC')
    finished=schedule[schedule.Status=='FINISHED']; remaining=schedule[schedule.Status!='FINISHED'].copy(); future=remaining[remaining.Status.isin(['SCHEDULED','TIMED'])&(remaining.KickoffUTC>now)].copy(); print('Finished:',len(finished),'Remaining:',len(remaining),'Future timed:',len(future))
    version=(v4.REG/'active.txt').read_text().strip();
    if not version or Path(version).name!=version or version in ('.','..'): raise ValueError('Invalid registry ID')
    b=joblib.load(v4.REG/version/'bundle.joblib');
    if b['version']!=version: raise ValueError('Registry bundle version mismatch')
    if (finished.KickoffUTC>now).any(): raise ValueError('Finished fixture has future kickoff')
    d=v4.history(); current=finished[finished.Date<now.tz_convert(None).normalize()][v4.KEY]; d=pd.concat([d,current],ignore_index=True).sort_values(['Date','HomeTeam','AwayTeam']); hist=v4.v3.History()
    for _,day in d.groupby('Date',sort=True): hist.update_day(day)
    if len(remaining):
        # Remaining games, even postponed/in-progress, are simulated from unconditional frozen pre-match DC rates.
        dc=v4.v33.forecast(*b['dc'].rates(remaining)); grids=dc['grid']
        remaining['DCExpectedHomeGoals']=dc['h']; remaining['DCExpectedAwayGoals']=dc['a']; remaining['DCProbOver2_5']=dc['over']
    else: grids=[]
    standings=simulate(schedule,remaining,grids,args.simulations); standings['ModelVersion']=version; standings['GeneratedUTC']=now.isoformat(); standings['Simulations']=args.simulations
    v4.atomic_csv(standings,out/'season_simulation_v4.csv'); v4.atomic_csv(remaining,out/'remaining_fixtures_v4.csv')
    if not future.empty:
        # Forecast only the next scheduled matchday, without using same-day completed results.
        first=future.iloc[0]; z=future[future.Matchday==first.Matchday].copy() if pd.notna(first.Matchday) else future[future.KickoffUTC<=first.KickoffUTC+pd.Timedelta(days=7)].copy(); z=z.reset_index(drop=True)
        x=pd.DataFrame([hist.features(m.Date,m.Season,m.HomeTeam,m.AwayTeam) for m in z.itertuples(index=False)],index=z.index)
        r=v4.predict(b,z.drop(columns=['FTHG','FTAG','FTR']),x); stamp=pd.Timestamp.now(tz='UTC'); r=r[pd.to_datetime(r.KickoffUTC,utc=True)>stamp].copy(); r['PredictionGeneratedUTC']=stamp.isoformat(); r['HistoryLastDate']=str(d.Date.max().date()); v4.atomic_csv(r,out/'weekly_predictions_v4.csv')
        if len(r): v4.atomic_csv(r,out/'prediction_archive'/(stamp.strftime('%Y%m%dT%H%M%S%f')+'-'+uuid.uuid4().hex[:8]+'.csv'))
    else: v4.atomic_csv(pd.DataFrame(columns=['MatchID','KickoffUTC','HomeTeam','AwayTeam']),out/'weekly_predictions_v4.csv')
    files=sorted((out/'prediction_archive').glob('*.csv')); a=pd.concat([pd.read_csv(f) for f in files],ignore_index=True) if files else pd.DataFrame(); settled,metrics=settle(a,schedule,pd.Timestamp.now(tz='UTC'))
    if settled.empty: settled=pd.DataFrame(columns=['MatchID','Settlement'])
    if metrics.empty: metrics=pd.DataFrame(columns=['ModelVersion','Task','Model','Matches','Accuracy','MacroF1','LogLoss','MAE'])
    v4.atomic_csv(settled,out/'live_history_v4.csv'); v4.atomic_csv(metrics,out/'live_metrics_v4.csv'); print('Simulations:',args.simulations,'Archived fixtures:',len(settled),'Settled:',int(settled.Settlement.eq('Settled').sum())); print(standings[['Team','ExpectedPoints','ExpectedRank']].to_string(index=False)); print('Saved:',out)

if __name__=='__main__': main()
