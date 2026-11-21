"""Phase 3: fitted Dixon-Coles and rolling goal forecasts.
Run: python -m src.accuracy_v33. Uses the validated V3 historical cache.
No production artifacts are changed. Historical benchmarks are not untouched tests.
"""
import argparse
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson
from sklearn.metrics import log_loss, f1_score, mean_absolute_error, mean_squared_error
from src import accuracy_v3 as v3
from src import accuracy_v31 as v31

OUT = v3.ROOT / 'outputs' / 'v33_experiments'
KEYS = ['Season', 'HomeTeam', 'AwayTeam']
LOW, HIGH = np.log(.05), np.log(6.)
GRID = np.arange(21)
TOTAL = GRID[:, None] + GRID[None, :]
CLASSES = v3.CLASSES


def tau(hg, ag, h, a, rho):
    """Dixon-Coles low-score likelihood correction and derivatives."""
    t = np.ones(len(h)); dh = np.zeros(len(h)); da = dh.copy(); dr = dh.copy()
    m = (hg == 0) & (ag == 0)
    t[m] = 1 - h[m]*a[m]*rho; dh[m] = da[m] = -h[m]*a[m]*rho; dr[m] = -h[m]*a[m]
    m = (hg == 0) & (ag == 1)
    t[m] = 1 + h[m]*rho; dh[m] = h[m]*rho; dr[m] = h[m]
    m = (hg == 1) & (ag == 0)
    t[m] = 1 + a[m]*rho; da[m] = a[m]*rho; dr[m] = a[m]
    m = (hg == 1) & (ag == 1)
    t[m] = 1-rho; dr[m] = -1
    return t, dh, da, dr


class DixonColes:
    """Penalised time-weighted maximum likelihood. Positive defence = more conceded.
    Zero-centred L2 penalty identifies team coefficients and shrinks sparse teams.
    Unseen teams default to the fitted league-average attack and defence.
    """
    def __init__(self, half_life=365, penalty=5., correction=True):
        self.half_life = half_life; self.penalty = penalty; self.correction = correction

    def fit(self, history, cutoff):
        cutoff = pd.Timestamp(cutoff)
        d = history.loc[history.Date < cutoff].copy()
        if d.empty: raise ValueError('No matches strictly before cutoff')
        self.teams = {t:i for i,t in enumerate(sorted(set(d.HomeTeam)|set(d.AwayTeam)))}
        n = len(self.teams)
        hi = d.HomeTeam.map(self.teams).to_numpy(int); ai = d.AwayTeam.map(self.teams).to_numpy(int)
        hg = d.FTHG.to_numpy(float); ag = d.FTAG.to_numpy(float)
        w = np.exp2(-(cutoff-d.Date).dt.total_seconds().to_numpy()/86400/self.half_life)
        # Normalise weights to preserve a comparable regularisation scale across half-lives.
        w *= len(w)/w.sum()
        self.baseline = float(np.average(hg+ag, weights=w))
        self.cutoff = cutoff
        start = np.zeros(2*n+3); start[-3] = np.log(max(.1,np.average(ag,weights=w)))
        start[-2] = np.log(max(.1,np.average(hg,weights=w))/max(.1,np.average(ag,weights=w)))
        penalty = self.penalty
        def objective(x):
            attack, defence = x[:n], x[n:2*n]; base, home, rho = x[-3:]
            rawh = base+home+attack[hi]+defence[ai]; rawa = base+attack[ai]+defence[hi]
            lh = np.clip(rawh,LOW,HIGH); la = np.clip(rawa,LOW,HIGH); h=np.exp(lh); a=np.exp(la)
            t,dh,da,dr = tau(hg,ag,h,a,rho)
            loss = -np.sum(w*(hg*lh-h+ag*la-a+np.log(t))) + .5*penalty*np.dot(x[:2*n],x[:2*n])
            gh = w*(h-hg-dh/t)*((rawh>LOW)&(rawh<HIGH))
            ga = w*(a-ag-da/t)*((rawa>LOW)&(rawa<HIGH))
            g = np.zeros_like(x)
            g[:n] = np.bincount(hi,weights=gh,minlength=n)+np.bincount(ai,weights=ga,minlength=n)+penalty*attack
            g[n:2*n] = np.bincount(ai,weights=gh,minlength=n)+np.bincount(hi,weights=ga,minlength=n)+penalty*defence
            g[-3] = (gh+ga).sum(); g[-2] = gh.sum(); g[-1] = -np.sum(w*dr/t)
            return loss,g
        # Bounds ensure the four tau factors stay positive for every rate in [.05,6].
        bounds = [(-2,2)]*(2*n)+[(-2,2),(-1,1),(-.15,.025) if self.correction else (0,0)]
        result = minimize(objective,start,jac=True,bounds=bounds,method='L-BFGS-B',options={'maxiter':500,'ftol':1e-10,'gtol':1e-5})
        if not result.success or not np.isfinite(result.x).all():
            raise RuntimeError(f'Optimisation failed at {cutoff}: {result.message}')
        self.params=result.x; self.n=n; self.loss=float(result.fun); self.iterations=int(result.nit)
        return self

    def rates(self, fixtures):
        def strength(team,offset):
            return self.params[offset+self.teams[team]] if team in self.teams else 0.
        ah=np.array([strength(t,0) for t in fixtures.HomeTeam]); aa=np.array([strength(t,0) for t in fixtures.AwayTeam])
        dh=np.array([strength(t,self.n) for t in fixtures.HomeTeam]); da=np.array([strength(t,self.n) for t in fixtures.AwayTeam])
        base,home,rho=self.params[-3:]
        return np.exp(np.clip(base+home+ah+da,LOW,HIGH)),np.exp(np.clip(base+aa+dh,LOW,HIGH)),rho


def distributions(h,a,rho):
    h=np.asarray(h,float); a=np.asarray(a,float)
    p=poisson.pmf(GRID[None,:],h[:,None])[:,:,None]*poisson.pmf(GRID[None,:],a[:,None])[:,None,:]
    p[:,0,0]*=1-h*a*rho; p[:,0,1]*=1+h*rho; p[:,1,0]*=1+a*rho; p[:,1,1]*=1-rho
    if (p<0).any() or not np.isfinite(p).all(): raise ValueError('Invalid scoreline probabilities')
    p/=p.sum(axis=(1,2),keepdims=True)
    return p


def forecast(h,a,rho):
    p=distributions(h,a,rho)
    outcome=np.c_[p[:,GRID[:,None]<GRID[None,:]].sum(1),p[:,GRID[:,None]==GRID[None,:]].sum(1),p[:,GRID[:,None]>GRID[None,:]].sum(1)]
    over=p[:,TOTAL>=3].sum(1)
    mean=(p*TOTAL).sum((1,2))
    median=(np.stack([p[:,TOTAL<=i].sum(1) for i in range(41)],1)>=.5).argmax(1)
    mode=p.reshape(len(p),-1).argmax(1); sh,sa=np.divmod(mode,len(GRID))
    return dict(p=outcome,over=over,mean=mean,median=median,h=(p*GRID[None,:,None]).sum((1,2)),a=(p*GRID[None,None,:]).sum((1,2)),sh=sh,sa=sa,grid=p)


def run_fold(df,season,fast=False):
    actual=df.loc[df.Season==season].copy().reset_index(drop=True)
    actual['Total']=actual.FTHG+actual.FTAG; actual['Over']=(actual.Total>=3).astype(int)
    # Refit every 28 days using only scores strictly before the block starts.
    blocks=((actual.Date-actual.Date.min()).dt.days//28).to_numpy()
    candidates=[(365,5.,True),(365,5.,False)] if fast else [(half,pen,dc) for half in (180,365,730) for pen in (5.,20.) for dc in (False,True)]
    preds={}; audit=[]
    for half,pen,dc in candidates:
        name=f'{"DC" if dc else "IP"}_HL{half}_L2{pen:g}'
        chunks=[]
        for block in np.unique(blocks):
            z=actual.loc[blocks==block]; cutoff=z.Date.min()
            model=DixonColes(half,pen,dc).fit(df,cutoff)
            h,a,rho=model.rates(z); f=forecast(h,a,rho)
            f['base']=np.full(len(z),model.baseline)
            # Retain only the actual-score probability, not the whole grid.
            hg=z.FTHG.to_numpy(int); ag=z.FTAG.to_numpy(int); inside=(hg<=20)&(ag<=20)
            prob=np.full(len(z),1e-15); prob[inside]=f['grid'][np.arange(len(z))[inside],hg[inside],ag[inside]]
            f['actual_score_prob']=prob; del f['grid']; chunks.append(f)
            earlier=df.loc[df.Date<cutoff]
            audit.append(dict(Season=season,Model=name,Cutoff=cutoff,TrainingMatches=len(earlier),LatestTrainingDate=earlier.Date.max(),Rho=rho,HomeLogAdvantage=model.params[-2],Iterations=model.iterations))
        preds[name]={k:np.concatenate([f[k] for f in chunks]) for k in chunks[0]}
        print(season,name,'complete',flush=True)
    baseline=[]
    for block in np.unique(blocks):
        z=actual.loc[blocks==block]; hist=df.loc[df.Date<z.Date.min()]; goals=hist.FTHG+hist.FTAG
        p=np.tile([(hist.FTR==c).mean() for c in CLASSES],(len(z),1))
        baseline.append(dict(p=p,over=np.full(len(z),(goals>=3).mean()),mean=np.full(len(z),goals.mean()),median=np.full(len(z),goals.median())))
    preds['TrainPrior']={k:np.concatenate([f[k] for f in baseline]) for k in baseline[0]}
    return actual,preds,pd.DataFrame(audit)


def combine(folds):
    a=pd.concat([f[0] for f in folds],ignore_index=True)
    p={name:{k:np.concatenate([f[1][name][k] for f in folds]) for k in values} for name,values in folds[0][1].items()}
    return a,p


def goal_candidates(p):
    g={}
    for name,f in p.items():
        g[name+'_Mean']=f['mean']; g[name+'_Median']=f['median']
        if name!='TrainPrior':
            for weight in (.25,.5,.75):
                g[f'{name}_Shrink{weight}']=weight*f['mean']+(1-weight)*f['base']
    return g


def evaluate(folds):
    va,vp=combine(folds[:2]); ta,tp=combine(folds[2:]); vg=goal_candidates(vp)
    best=min(vp,key=lambda k:log_loss(va.FTR,vp[k]['p'],labels=CLASSES))
    gb=min(vg,key=lambda k:mean_absolute_error(va.Total,vg[k])); rm=min(vg,key=lambda k:mean_squared_error(va.Total,vg[k]))
    ob=min(vp,key=lambda k:log_loss(va.Over,vp[k]['over'],labels=[0,1]))
    boost=max(np.arange(1.,2.51,.1),key=lambda b:f1_score(va.FTR,v3.labels(vp[best]['p'],b),labels=CLASSES,average='macro'))
    cr=[];gr=[];br=[];sr=[]
    sets=[('Validation',va,vp),('HistoricalTest',ta,tp)]+[(s,f[0],f[1]) for s,f in zip(v3.TEST,folds[2:])]
    for split,a,p in sets:
        for name,f in p.items():
            cr.append(dict(Model=name,Split=split,**v31.outcome_metrics(a.FTR.to_numpy(),f['p'])))
            br.append(dict(Model=name,Split=split,**v31.binary_metrics(a.Over,f['over'])))
            if 'sh' in f:
                sr.append(dict(Model=name,Split=split,HomeMAE=mean_absolute_error(a.FTHG,f['h']),AwayMAE=mean_absolute_error(a.FTAG,f['a']),ExactScoreAccuracy=np.mean((a.FTHG==f['sh'])&(a.FTAG==f['sa'])),ScoreLogLoss=-np.log(np.clip(f['actual_score_prob'],1e-15,1)).mean()))
        cr.append(dict(Model=best+'_DrawDecision',Split=split,**v31.outcome_metrics(a.FTR.to_numpy(),p[best]['p'],v3.labels(p[best]['p'],boost))))
        for name,g in goal_candidates(p).items():gr.append(dict(Model=name,Split=split,**v3.goal_metrics(a.Total,g)))
    selection=pd.DataFrame([dict(Outcome=best,DrawBoost=boost,GoalsMAE=gb,GoalsRMSE=rm,OverUnder=ob)])
    records=ta[KEYS+['Date','FTR','FTHG','FTAG','Total','Over']].copy()
    records['Outcome']=v3.labels(tp[best]['p']);records['DrawAdjusted']=v3.labels(tp[best]['p'],boost)
    records['Goals']=goal_candidates(tp)[gb];records['GoalsRMSE']=goal_candidates(tp)[rm];records['ProbOver']=tp[ob]['over']
    for i,c in enumerate(CLASSES):records['Prob'+c]=tp[best]['p'][:,i]
    # Scoreline distribution is from the outcome model; if baseline wins, use validated best fitted score model.
    score_model=min([k for k in vp if 'sh' in vp[k]],key=lambda k:log_loss(va.FTR,vp[k]['p'],labels=CLASSES))
    f=tp[score_model];records['ScoreModel']=score_model;records['ExpectedHomeGoals']=f['h'];records['ExpectedAwayGoals']=f['a'];records['ModalHomeGoals']=f['sh'];records['ModalAwayGoals']=f['sa']
    return dict(classification=pd.DataFrame(cr),goals=pd.DataFrame(gr),over_under=pd.DataFrame(br),scorelines=pd.DataFrame(sr),selection=selection,predictions=records)


def compare(records):
    rows=[]
    versions={'v33':records}
    for version in ('v31','v32'):
        path=v3.ROOT/'outputs'/f'{version}_experiments'/'predictions.csv'
        if not path.exists():print('Missing comparison:',path);continue
        prior=pd.read_csv(path)
        m=records[KEYS+['FTR','Total','Over']].merge(prior,on=KEYS,validate='one_to_one',suffixes=('_new',''))
        if len(m)!=len(records) or len(m)!=len(prior) or not all((m[c]==m[c+'_new']).all() for c in ('FTR','Total','Over')):raise ValueError(f'{version} fixture comparison mismatch')
        versions[version]=m
    for version,a in versions.items():
        for split,z in [('HistoricalTest',a),*list(a.groupby('Season'))]:
            rows.append(dict(Version=version,Season=split,Matches=len(z),Accuracy=np.mean(z.FTR==z.DrawAdjusted),MacroF1=f1_score(z.FTR,z.DrawAdjusted,labels=CLASSES,average='macro',zero_division=0),GoalMAE=mean_absolute_error(z.Total,z.Goals),OverAccuracy=np.mean((z.ProbOver>=.5)==z.Over),OverLogLoss=log_loss(z.Over,z.ProbOver,labels=[0,1])))
    return pd.DataFrame(rows)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--fast',action='store_true',help='Reduced candidate grid, separate output directory');args=parser.parse_args()
    df=v3.load_history();df['Date']=pd.to_datetime(df.Date).dt.normalize()
    folds=[run_fold(df,s,args.fast) for s in v3.VALID+v3.TEST]
    result=evaluate(folds);result['fit_audit']=pd.concat([f[2] for f in folds],ignore_index=True);result['paired_comparison']=compare(result['predictions'])
    out=OUT.with_name('v33_fast_experiments') if args.fast else OUT;out.mkdir(parents=True,exist_ok=True)
    for name,d in result.items():d.to_csv(out/(name+'.csv'),index=False)
    print('\nValidation selections:',result['selection'].to_string(index=False))
    for name,metric in [('classification','LogLoss'),('goals','MAE'),('over_under','LogLoss'),('scorelines','ScoreLogLoss')]:
        print('\n'+name+' historical benchmark:');print(result[name].query("Split == 'HistoricalTest'").sort_values(metric).head(12).to_string(index=False))
    print('\nSame-fixture comparison:');print(result['paired_comparison'].to_string(index=False));print('Saved:',out)

if __name__=='__main__':main()
