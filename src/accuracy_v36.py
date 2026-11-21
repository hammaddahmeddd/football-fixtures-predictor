"""Phase 6: contextual, one-step Q-learning model selection.
Run python -m src.accuracy_v36 in the project root. Production untouched.
Train policy on 2021-22 and 2022-23 out-of-season forecasts; select on
2023-24; report 2024-25 and 2025-26 historical benchmarks (not untouched).
Gamma=0 because selecting a forecast cannot change the football result or next state.
"""
import argparse
import json
from time import perf_counter
import numpy as np
import pandas as pd
from src import accuracy_v3 as v3, accuracy_v31 as v31, accuracy_v32 as v32, accuracy_v35 as v35

C = np.asarray(v3.CLASSES)
CHECKPOINTS = (50, 100, 250, 500, 1000)
BOOST = 1.6  # fixed alternative decision action; not interpreted as calibrated probability


def build_folds(skip_xg=False, fast=False):
    df = v3.load_history()
    df['Date'] = pd.to_datetime(df.Date).dt.normalize()
    df = df.sort_values(['Date','HomeTeam','AwayTeam']).reset_index(drop=True)
    if skip_xg:
        groups = {'Shot129': v31.features(df)}
        groups['Score85'] = groups['Shot129'].iloc[:, :85]
    else:
        df, _ = v32.merge_xg(df)
        df = df.sort_values(['Date','HomeTeam','AwayTeam']).reset_index(drop=True)
        groups = v32.build_features(df, v32.load_championship())
        if 'XG' not in groups:
            raise ValueError('XG group missing; use --skip-xg for a separate experiment')
    folds = []
    for season in v35.META + [v35.SELECT] + v35.TEST:
        z, p, o, g = v35.fold(df, groups, season, fast=fast)
        tr = (df.Season < season) & (df.Season >= '2012-13')
        te = df.Season == season
        x, xx = groups['Score85'].loc[tr], groups['Score85'].loc[te]
        for name, model in v3.classifiers().items():
            if name not in ('KNN','SVM','DecisionTree'):
                continue
            model.fit(x, df.loc[tr,'FTR'])
            p[name] = model.predict_proba(xx)[:, [list(model.classes_).index(c) for c in C]]
        v35.check((z,p,o,g))
        folds.append((z,p,o,g))
    meta = v35.join(folds[:2])
    # Fit voting weights/stacking on meta seasons only. For policy-training rows,
    # use equal-weight ensembles with a fixed predeclared base model list instead
    # of in-sample fitted stacking forecasts. This avoids circular meta-learning.
    keys = ['Score85_RF','Shot129_Pois20','DC_365','IP_180']
    if not skip_xg: keys.append('XG_Pois5')
    all_sets = [meta, *folds[2:]]
    for z,p,o,g in all_sets:
        p['EqualVote'] = np.mean([p[k] for k in keys], axis=0)
    return all_sets


def prepare(f, names):
    z,p,_,_ = f
    q = np.stack([p[n] for n in names], axis=1)
    # Include a fixed draw-sensitive option for three statistical models.
    actions = [(n,1.) for n in names]
    for n in ('Shot129_Pois5','DC_365','EqualVote'):
        if n in names:
            q = np.concatenate([q,p[n][:,None,:]],axis=1)
            actions.append((n,BOOST))
    scores = q.copy()
    scores[:,:,1] *= np.asarray([b for _,b in actions])[None,:]
    pred = C[scores.argmax(2)]
    y = z.FTR.to_numpy()
    correct = (pred == y[:,None]).astype(float)
    brier = ((q - (y[:,None,None]==C[None,None,:]))**2).sum(2)
    # Brier reward in [0,1], combined with decision accuracy for tie-breaking.
    rewards = {'accuracy':correct, 'hybrid':.5*correct+.5*(1-brier/2)}
    # Context uses only pre-match model forecasts, not observed labels.
    base = q[:,:len(names),:]
    mean = base.mean(1)
    entropy = -(mean*np.log(np.clip(mean,1e-12,1))).sum(1)/np.log(3)
    votes = base.argmax(2)
    agreement = np.max(np.stack([(votes==i).mean(1) for i in range(3)],axis=1),axis=1)
    margin = np.abs(mean[:,2]-mean[:,0])
    favored = np.where(mean[:,2]>=mean[:,0],1,0)
    # Fixed bins across all seasons; no test-derived quantiles.
    compact = np.digitize(margin,[.10,.25])*3 + np.digitize(agreement,[.60,.85])
    rich = ((compact*3 + np.digitize(entropy,[.85,.95]))*2 + favored)*2 + (mean[:,1]>=.26)
    return dict(z=z,q=q,pred=pred,rewards=rewards,states={'compact':compact.astype(int),'rich':rich.astype(int)},actions=actions)


def greedy_policy(Q, counts, prior, states, strength):
    # Shrink sparse states toward globally observed training action rewards.
    adjusted = (counts*Q + strength*prior[None,:])/(counts+strength)
    return adjusted[states].argmax(1)


def evaluate_policy(data, picks):
    rows = np.arange(len(picks)); p = data['q'][rows,picks]; y = data['z'].FTR.to_numpy()
    pred = data['pred'][rows,picks]
    return v31.outcome_metrics(y,p,pred)


def train(meta, val, mode, reward, alpha, eps0, seed, checkpoints, history):
    s=meta['states'][mode]; vs=val['states'][mode]; r=meta['rewards'][reward]
    nstates=9 if mode=='compact' else 108
    na=r.shape[1]; prior=r.mean(0); Q=np.tile(prior,(nstates,1)); counts=np.zeros((nstates,na),dtype=int)
    rng=np.random.default_rng(seed); cumulative=0.; snapshots=[]; start=perf_counter()
    name=f'{mode}_{reward}_a{alpha:g}_e{eps0:g}'
    for epoch in range(1,max(checkpoints)+1):
        eps=max(.02,eps0*.995**(epoch-1)); total=0.
        for i in rng.permutation(len(s)):
            state=s[i]
            # Epsilon-greedy, gamma=0: each fixture terminates independently.
            if rng.random()<eps: action=rng.integers(na)
            else:
                action=int(np.argmax((counts[state]*Q[state]+12*prior)/(counts[state]+12)))
            value=r[i,action]; Q[state,action]+=alpha*(value-Q[state,action]);counts[state,action]+=1;total+=value
        cumulative+=total
        if epoch==1 or epoch%10==0 or epoch in checkpoints:
            greedy=greedy_policy(Q,counts,prior,s,12)
            history.append(dict(Config=name,Seed=seed,Pass=epoch,Epsilon=eps,MeanExplorationReward=total/len(s),CumulativeReward=cumulative,GreedyTrainReward=r[np.arange(len(s)),greedy].mean(),ElapsedSeconds=perf_counter()-start))
        if epoch in checkpoints:
            picks=greedy_policy(Q,counts,prior,vs,12)
            snapshots.append(dict(Config=name,Seed=seed,Passes=epoch,Mode=mode,Reward=reward,Alpha=alpha,EpsilonStart=eps0,Q=Q.copy(),Counts=counts.copy(),Prior=prior.copy(),Metrics=evaluate_policy(val,picks)))
    return snapshots


def experiment(folds, fast=False):
    # Fixed candidate list, not ranked on historical test outcomes.
    preferred=['KNN','SVM','DecisionTree','Score85_RF','Shot129_RF','Shot129_Pois5','Shot129_Pois20','DC_365','IP_180','IP_730','Shot129_Hurdle','XG_Pois5','XG_RF','EqualVote']
    names=[n for n in preferred if all(n in f[1] for f in folds)]
    if not names: raise ValueError('No compatible base forecasts')
    for f in folds: v35.check(f)
    data=[prepare(f,names) for f in folds]; meta,val,*test=data
    history=[]; candidates=[]
    checkpoints=(2,4) if fast else CHECKPOINTS
    seeds=(42,) if fast else (42,123)
    for mode in ('compact','rich'):
        for reward in ('accuracy','hybrid'):
            for alpha,eps in ((.05,.2),(.15,.35)):
                for seed in seeds:
                    candidates.extend(train(meta,val,mode,reward,alpha,eps,seed,checkpoints,history))
    search=pd.DataFrame([{**{k:x[k] for k in ('Config','Seed','Passes','Mode','Reward','Alpha','EpsilonStart')},**x['Metrics']} for x in candidates])
    # Select configuration on mean validation accuracy across seeds; balanced
    # policy permits at most two percentage points loss relative to fixed actions.
    summary=search.groupby(['Config','Passes'],as_index=False)[['Accuracy','MacroF1','LogLoss']].mean()
    fixed_val=[evaluate_policy(val,np.full(len(val['z']),i)) for i in range(len(meta['actions']))]
    fixed_best=max(x['Accuracy'] for x in fixed_val)
    best=summary.sort_values(['Accuracy','LogLoss','MacroF1','Passes'],ascending=[False,True,False,True]).iloc[0]
    eligible=summary[summary.Accuracy>=fixed_best-.02-1e-12]
    balanced=(eligible if len(eligible) else summary).sort_values(['MacroF1','Accuracy','LogLoss','Passes'],ascending=[False,False,True,True]).iloc[0]
    # Aggregate Q estimates across fixed seeds, without choosing best seed on test.
    policies={}
    for label,row in [('QAccuracy',best),('QBalanced',balanced)]:
        ss=[x for x in candidates if x['Config']==row.Config and x['Passes']==row.Passes]
        counts=np.sum([x['Counts'] for x in ss],axis=0)
        q=np.mean([x['Q'] for x in ss],axis=0)
        policies[label]=dict(mode=ss[0]['Mode'],q=q,counts=counts,prior=ss[0]['Prior'],config=row.Config,passes=int(row.Passes),reward=ss[0]['Reward'])
    fixed_accuracy=int(np.argmax([m['Accuracy'] for m in fixed_val]));fixed_balanced=int(np.argmax([m['MacroF1'] for m in fixed_val]));fixed_prob=int(np.argmin([m['LogLoss'] for m in fixed_val]))
    records=[]; metrics=[]; usages=[]; trajectories=[]
    all_test=prepare(v35.join([folds[2],folds[3]]),names)
    for split,d in [('Selection',val),('HistoricalTest',all_test),*[(str(t['z'].Season.iloc[0]),t) for t in test]]:
        n=len(d['z']); choices={f'{k}@{b:g}':np.full(n,i) for i,(k,b) in enumerate(d['actions'])}
        choices.update({label:np.full(n,i) for label,i in [('FixedAccuracy',fixed_accuracy),('FixedBalanced',fixed_balanced),('FixedProbability',fixed_prob)]})
        for label,pol in policies.items():
            choices[label]=greedy_policy(pol['q'],pol['counts'],pol['prior'],d['states'][pol['mode']],12)
        for label,picks in choices.items():
            m=evaluate_policy(d,picks); hits=d['rewards']['accuracy'][np.arange(n),picks]
            metrics.append(dict(Split=split,Strategy=label,Matches=n,CumulativeReward=hits.sum(),RegretVsFixedAccuracy=float(d['rewards']['accuracy'][:,fixed_accuracy].sum()-hits.sum()),**m))
            if label not in ('QAccuracy','QBalanced','FixedAccuracy','FixedBalanced','FixedProbability'):continue
            for i,count in enumerate(np.bincount(picks,minlength=len(d['actions']))):
                usages.append(dict(Split=split,Strategy=label,Action=str(d['actions'][i]),Count=count))
            if split!='HistoricalTest':continue
            rec=d['z'][v35.KEYS+['Date','FTR']].copy();rec['Strategy']=label;rec['Action']=[str(d['actions'][i]) for i in picks];rec['Prediction']=d['pred'][np.arange(n),picks]
            for j,c in enumerate(C):rec['Prob'+c]=d['q'][np.arange(n),picks,j]
            rec['Correct']=hits;rec['CumulativeReward']=np.cumsum(hits);rec['CumulativeRegretVsFixed']=np.cumsum(d['rewards']['accuracy'][:,fixed_accuracy]-hits);records.append(rec)
            trajectories.append(rec[['Date','Strategy','CumulativeReward','CumulativeRegretVsFixed']])
    selection=pd.DataFrame([dict(Strategy=k,Config=p['config'],Passes=p['passes'],StateMode=p['mode'],Reward=p['reward'],VisitedStates=int((p['counts'].sum(1)>0).sum()),UnseenHistoricalStates=int((p['counts'].sum(1)[all_test['states'][p['mode']]]==0).sum()),Gamma=0) for k,p in policies.items()])
    qrows=[]
    for label,p in policies.items():
        for state in range(len(p['q'])):
            for i,(name,b) in enumerate(meta['actions']):
                qrows.append(dict(Strategy=label,State=state,Action=name,DrawBoost=b,Q=p['q'][state,i],Visits=p['counts'][state,i],GlobalReward=p['prior'][i]))
    return dict(comparison=pd.DataFrame(metrics),selection=selection,validation_search=search,validation_summary=summary,training_history=pd.DataFrame(history),action_usage=pd.DataFrame(usages),predictions=pd.concat(records,ignore_index=True),test_reward_history=pd.concat(trajectories,ignore_index=True),q_table=pd.DataFrame(qrows))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--skip-xg',action='store_true');parser.add_argument('--fast',action='store_true',help='Smoke run only; not full benchmark');args=parser.parse_args()
    result=experiment(build_folds(args.skip_xg,args.fast),args.fast)
    tag=('v36_no_xg' if args.skip_xg else 'v36')+('_fast' if args.fast else '')+'_experiments'
    out=v3.ROOT/'outputs'/tag;out.mkdir(parents=True,exist_ok=True)
    for k,v in result.items():v.to_csv(out/(k+'.csv'),index=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for label,row in result['selection'].set_index('Strategy').iterrows():
        h=result['training_history'];h=h[(h.Config==row.Config)&(h.Pass<=row.Passes)]
        avg=h.groupby('Pass')[['MeanExplorationReward','GreedyTrainReward']].mean()
        ax=avg.plot(title=f'{label}: training convergence');ax.set_ylabel('Mean reward');ax.set_ylim(0,1);ax.figure.tight_layout();ax.figure.savefig(out/(label+'_convergence.png'),dpi=150);plt.close(ax.figure)
    for col in ('CumulativeReward','CumulativeRegretVsFixed'):
        fig,ax=plt.subplots();
        for k,z in result['test_reward_history'].groupby('Strategy'):ax.plot(np.arange(1,len(z)+1),z[col],label=k)
        ax.set_xlabel('Historical fixtures');ax.set_ylabel(col);ax.legend();fig.tight_layout();fig.savefig(out/(col+'.png'),dpi=150);plt.close(fig)
    print('\nValidation selections:\n',result['selection'].to_string(index=False))
    print('\nHistorical benchmark (not untouched):\n',result['comparison'].query("Split=='HistoricalTest'").sort_values('Accuracy',ascending=False)[['Strategy','Accuracy','MacroF1','DrawRecall','LogLoss','CumulativeReward','RegretVsFixedAccuracy']].to_string(index=False))
    print('Saved:',out)

if __name__=='__main__':main()
