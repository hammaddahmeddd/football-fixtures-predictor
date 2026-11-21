"""Phase 7: chronological K-Means clustering and match-prediction ablation.
Copy to src/accuracy_v37.py; run python -m src.accuracy_v37.
No production artifacts are modified. Requires existing accuracy_v3 and v31.
"""
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import (silhouette_score, davies_bouldin_score,
    calinski_harabasz_score, adjusted_rand_score, log_loss)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from src import accuracy_v3 as v3
from src import accuracy_v31 as v31

OUT = v3.ROOT / 'outputs' / 'v37_experiments'
PROFILE = ['elo','20_gf','20_ga','20_pts','20_adj','20_draw','20_clean',
           '20_blank','20_over','20_goalvar','season_ppg','season_gd']
KEYS = ['Season','Date','HomeTeam','AwayTeam']


def team_profiles(df, X):
    """One pre-match team observation for each side, aligned to fixture row."""
    frames = []
    for side, team in [('H','HomeTeam'),('A','AwayTeam')]:
        p = X[[side+'_'+c for c in PROFILE]].copy()
        p.columns = PROFILE
        p['FixtureRow'] = df.index
        p['Team'] = df[team].to_numpy()
        p['Season'] = df.Season.to_numpy()
        p['Date'] = df.Date.to_numpy()
        p['Side'] = side
        frames.append(p)
    return pd.concat(frames, ignore_index=True)


def fit_clusters(train, candidates, seed=42, fast=False):
    scaler = StandardScaler().fit(train[PROFILE])
    a = scaler.transform(train[PROFILE])
    rows, models = [], {}
    rng = np.random.default_rng(seed)
    sample = rng.choice(len(a), min(2000,len(a)), replace=False)
    for k in candidates:
        if k >= len(a): continue
        m = KMeans(n_clusters=k,n_init=10,random_state=seed).fit(a)
        lab = m.labels_
        if len(np.unique(lab[sample])) < 2: continue
        stability = []
        for j in range(2 if fast else 5):
            # Fit on a resampled training set; compare assignments on same reference set.
            idx = rng.choice(len(a),len(a),replace=True)
            other = KMeans(n_clusters=k,n_init=10,random_state=seed+j+1).fit(a[idx])
            stability.append(adjusted_rand_score(lab,other.predict(a)))
        rows.append(dict(K=k,Silhouette=silhouette_score(a[sample],lab[sample]),
                         DaviesBouldin=davies_bouldin_score(a,lab),
                         CalinskiHarabasz=calinski_harabasz_score(a,lab),
                         Inertia=m.inertia_,BootstrapARI=np.mean(stability),
                         BootstrapARI_SD=np.std(stability),MinClusterShare=np.bincount(lab,minlength=k).min()/len(a)))
        models[k] = m
    if not rows: raise ValueError('No valid clustering candidate')
    audit = pd.DataFrame(rows)
    eligible = audit[audit.MinClusterShare>=.05]
    if eligible.empty: eligible = audit
    k = int(eligible.sort_values(['Silhouette','BootstrapARI'],ascending=False).iloc[0].K)
    return scaler,models[k],audit


def assignments(profiles, scaler, model):
    p = profiles.copy(); a=scaler.transform(p[PROFILE]); d=model.transform(a)
    p['Cluster']=model.predict(a)
    # Stable descriptive ordering by attacking strength minus defensive vulnerability.
    centers=pd.DataFrame(scaler.inverse_transform(model.cluster_centers_),columns=PROFILE)
    order=np.argsort((centers['20_gf']-centers['20_ga']).to_numpy())
    mapping={int(old):int(new) for new,old in enumerate(order)}
    p['Cluster']=p.Cluster.map(mapping)
    for j,old in enumerate(order): p[f'Distance{j}']=d[:,old]
    centers['Cluster']=[mapping[i] for i in range(len(centers))]
    return p,centers.sort_values('Cluster')


def fixture_features(p,k,index):
    frames=[]
    for side in ['H','A']:
        a=p[p.Side==side].set_index('FixtureRow').reindex(index)
        if a.Cluster.isna().any(): raise ValueError('Missing team profile')
        cols={side+f'_Cluster{i}':(a.Cluster==i).astype(float) for i in range(k)}
        cols.update({side+f'_Distance{i}':a[f'Distance{i}'] for i in range(k)})
        frames.append(pd.DataFrame(cols,index=index))
    z=pd.concat(frames,axis=1)
    h=p[p.Side=='H'].set_index('FixtureRow').Cluster.reindex(index)
    a=p[p.Side=='A'].set_index('FixtureRow').Cluster.reindex(index)
    z['SameCluster']=(h==a).astype(float)
    return z


def evaluate(df, fast=False, seasons=None):
    df=df.sort_values(['Date','HomeTeam','AwayTeam']).reset_index(drop=True)
    X=v3.make_features(df); p=team_profiles(df,X)
    rows=[]; audits=[]; centers=[]; counts=[]; predictions=[]; snapshots=[]; plot_data=None
    for season in (seasons or v3.VALID+v3.TEST):
        tr=(df.Season<season)&(df.Season!='2011-12'); te=df.Season==season
        train=p[p.FixtureRow.isin(df.index[tr])]; test=p[p.FixtureRow.isin(df.index[te])]
        if train.empty or test.empty: raise ValueError(f'Empty fold {season}')
        scaler,km,audit=fit_clusters(train,range(2,5 if fast else 9),fast=fast)
        audit.insert(0,'Season',season);audit['Selected']=audit.K==km.n_clusters;audits.append(audit)
        assigned,center=assignments(pd.concat([train,test]),scaler,km)
        center.insert(0,'Season',season);centers.append(center)
        held=assigned[assigned.FixtureRow.isin(df.index[te])].copy()
        # Latest available pre-match profile for each team, not a full-season average.
        latest=held.sort_values(['Date','FixtureRow']).groupby('Team').tail(1)
        snapshots.append(latest[['Season','Date','Team','Cluster']+PROFILE])
        for split,part in [('Train',assigned[assigned.FixtureRow.isin(df.index[tr])]),('Evaluation',held)]:
            for c,g in part.groupby('Cluster'): counts.append(dict(Season=season,Split=split,Cluster=c,Observations=len(g),Share=len(g)/len(part)))
        # Evaluate separation on new profiles using training scaler/centroids.
        a=scaler.transform(held[PROFILE]); lab=held.Cluster.to_numpy()
        if 1<len(np.unique(lab))<len(lab):
            audit.loc[audit.Selected,'EvaluationSilhouette']=silhouette_score(a,lab)
            audit.loc[audit.Selected,'EvaluationDaviesBouldin']=davies_bouldin_score(a,lab)
        Z=fixture_features(assigned,km.n_clusters,df.index[tr|te])
        y=df.loc[tr,'FTR']; actual=df.loc[te,'FTR'].to_numpy()
        rec=df.loc[te,KEYS+['FTR']].copy()
        for kind in ['RF','Logistic']:
            for mode in ['ScoreOnly','ClusterEnhanced']:
                xx=X.loc[tr|te].copy()
                if mode=='ClusterEnhanced': xx=pd.concat([xx,Z],axis=1)
                model=(RandomForestClassifier(n_estimators=60 if fast else 250,max_depth=7,min_samples_leaf=20,n_jobs=2,random_state=42) if kind=='RF' else make_pipeline(StandardScaler(),LogisticRegression(C=.03,max_iter=1500)))
                model.fit(xx.loc[tr[tr|te]],y)
                prob=model.predict_proba(xx.loc[te[tr|te]])[:,[list(model.classes_).index(c) for c in v3.CLASSES]]
                name=kind+'_'+mode
                rows.append(dict(Season=season,Model=name,K=km.n_clusters,**v31.outcome_metrics(actual,prob)))
                rec[name]=v3.labels(prob)
                for i,c in enumerate(v3.CLASSES):rec[name+'_P'+c]=prob[:,i]
        predictions.append(rec)
        pca=PCA(n_components=2).fit(scaler.transform(train[PROFILE])); coords=pca.transform(scaler.transform(held[PROFILE]))
        plot_data=(coords,lab,season,pca.explained_variance_ratio_)
        print(season,'selected K:',km.n_clusters,'training profiles:',len(train),'evaluation profiles:',len(test),flush=True)
    pred=pd.concat(predictions,ignore_index=True); metrics=pd.DataFrame(rows)
    summary=[]
    for split,ss in [('Validation',v3.VALID),('HistoricalTest',v3.TEST)]:
        d=pred[pred.Season.isin(ss)]
        if d.empty: continue
        for name in ['RF_ScoreOnly','RF_ClusterEnhanced','Logistic_ScoreOnly','Logistic_ClusterEnhanced']:
            prob=d[[name+'_P'+c for c in v3.CLASSES]].to_numpy()
            summary.append(dict(Split=split,Model=name,Matches=len(d),**v31.outcome_metrics(d.FTR.to_numpy(),prob)))
    summary=pd.DataFrame(summary)
    return {'clustering_metrics':pd.concat(audits,ignore_index=True),'cluster_profiles':pd.concat(centers,ignore_index=True),'cluster_counts':pd.DataFrame(counts),'latest_team_profiles':pd.concat(snapshots,ignore_index=True),'classification_by_season':metrics,'classification':summary,'predictions':pred},plot_data


def save(result,plot_data,out=OUT):
    out.mkdir(parents=True,exist_ok=True)
    for name,d in result.items():d.to_csv(out/(name+'.csv'),index=False)
    a=result['clustering_metrics']; last=a[a.Season==a.Season.iloc[-1]]
    for metric in ['Silhouette','Inertia','BootstrapARI']:
        fig,ax=plt.subplots();ax.plot(last.K,last[metric],marker='o');ax.set(xlabel='Number of clusters',ylabel=metric,title=f'{metric} ({last.Season.iloc[0]} training)');fig.tight_layout();fig.savefig(out/(metric+'.png'),dpi=150);plt.close(fig)
    if plot_data is not None:
        coords,lab,season,var=plot_data;fig,ax=plt.subplots()
        for c in np.unique(lab):ax.scatter(*coords[lab==c].T,s=12,alpha=.55,label=f'Cluster {c}')
        ax.set(xlabel=f'PC1 ({var[0]:.1%})',ylabel=f'PC2 ({var[1]:.1%})',title=f'Pre-match profiles: {season}');ax.legend();fig.tight_layout();fig.savefig(out/'PCA_clusters.png',dpi=150);plt.close(fig)
    print('\nClustering selection:');print(a[a.Selected].to_string(index=False))
    print('\nClassification ablation:');print(result['classification'][['Split','Model','Matches','Accuracy','MacroF1','LogLoss']].to_string(index=False));print('Saved:',out)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--fast',action='store_true');args=parser.parse_args()
    result,plot=evaluate(v3.load_history(),fast=args.fast);save(result,plot)

if __name__=='__main__': main()
