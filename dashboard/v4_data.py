"""V4 data adapters and supplementary research sections for the original UI."""
from pathlib import Path
import json
import pandas as pd
import numpy as np
import streamlit as st
from config import OUTPUTS_DIR
ROOT=Path(__file__).resolve().parents[1]
PROD=OUTPUTS_DIR/'v4_production'
REG=ROOT/'models'/'v4'
LABEL={'H':'Home Win','D':'Draw','A':'Away Win'}
@st.cache_data(show_spinner=False)
def _csv(path,mtime): return pd.read_csv(path)
def read(path):
    try: return _csv(str(path),path.stat().st_mtime_ns)
    except (OSError,pd.errors.EmptyDataError,pd.errors.ParserError): return pd.DataFrame()
def active():
    try:
        v=(REG/'active.txt').read_text().strip()
        if not v or Path(v).name!=v or v in ('.','..'): return '',{}
        return v,json.loads((REG/v/'manifest.json').read_text())
    except (OSError,ValueError): return '',{}
def weekly():
    d=read(PROD/'weekly_predictions_v4.csv').copy()
    if d.empty:return None
    required={'KickoffUTC','HomeTeam','AwayTeam','PredictedResult','PrimaryModel','SelectedConfidence','PredictedTotalGoals','OverUnder2_5'}
    if not required.issubset(d):return None
    d=d[pd.to_datetime(d.KickoffUTC,utc=True,errors='coerce')>pd.Timestamp.now(tz='UTC')].copy()
    d['ResultLabel']=d.PredictedResult.map(LABEL); d['SelectedModel']=d.PrimaryModel
    for source,target in [('KNN','KNN'),('SVM','SVM'),('DecisionTree','Decision_Tree'),('RandomForest','Random_Forest')]:
        cols=[f'{source}_P_{c}' for c in 'ADH']
        if f'{source}_Prediction' in d and set(cols).issubset(d):
            d[f'{target}_Prediction']=d[f'{source}_Prediction']; d[f'{target}_Confidence']=d[cols].max(axis=1)
    return d

def season():
    d=read(PROD/'season_simulation_v4.csv').copy()
    if d.empty:return None
    required={'Team','ExpectedRank','ExpectedPoints','TitleProbability','Top4Probability','RelegationProbability','PointsP05','PointsP95'}
    if not required.issubset(d):return None
    d=d.sort_values('ExpectedRank').reset_index(drop=True);d['ProjectedPosition']=d.index+1;d['AveragePosition']=d.ExpectedRank
    return d

def metrics(task):
    d=read(OUTPUTS_DIR/'v39_experiments'/'full'/'metrics.csv')
    if 'Task' not in d:return None
    z=d[d.Task==task].dropna(axis=1,how='all');return z if not z.empty else None

def table(d):
    if d is not None and not d.empty:st.dataframe(d,hide_index=True,width='stretch')
    else:st.info('Saved results are not available yet.')
def graph(path):
    if path.is_file():st.image(str(path),width='stretch')
def match_extra(r):
    st.divider();st.subheader('V4 statistical analysis')
    a,b,c=st.columns(3)
    a.metric('Balanced prediction',LABEL.get(r.get('BalancedResult'),'N/A'))
    b.metric('Q-Learning prediction',LABEL.get(r.get('QPrediction'),'N/A'))
    g=r.get('ExpectedHomeGoalsDC',np.nan)+r.get('ExpectedAwayGoalsDC',np.nan)
    c.metric('Dixon-Coles goal rate',f'{g:.2f}' if pd.notna(g) else 'N/A')
    p=pd.DataFrame({group:{LABEL[k]:r.get(f'{group}_P_{k}',np.nan) for k in 'HDA'} for group in ['Primary','Outcome','Q']})
    st.bar_chart(p);st.caption('Primary decision weights can differ from original probabilities. Outcome is an independently selected probability model.')
    table(pd.DataFrame([{'Rank':i,'Score':r.get(f'Score{i}'),'Probability':r.get(f'Score{i}Probability')} for i in range(1,4)]))
    st.caption('Modal exact score and aggregate outcome need not agree. Goal median and Over/Under prior can be constant across fixtures.')

def research():
    with st.expander('Advanced experimental analysis'):
        t1,t2,t3,t4=st.tabs(['Computational','Statistical','Team clustering','Dataset'])
        with t1:
            d=read(OUTPUTS_DIR/'v38_experiments'/'full'/'full_dataset_comparison.csv');table(d)
            table(read(OUTPUTS_DIR/'v38_experiments'/'full'/'theoretical_complexity.csv'))
        with t2:
            st.caption('Retrospective results, not independent live accuracy. Confidence intervals are not multiplicity-adjusted.')
            table(read(OUTPUTS_DIR/'v39_experiments'/'full'/'confidence_intervals.csv')); table(read(OUTPUTS_DIR/'v39_experiments'/'full'/'paired_tests.csv'))
        with t3:
            table(read(OUTPUTS_DIR/'v37_experiments'/'cluster_profiles.csv'));graph(OUTPUTS_DIR/'v37_experiments'/'PCA_clusters.png')
            d=weekly()
            if d is not None:table(d[[c for c in ['HomeTeam','AwayTeam','HCluster','ACluster'] if c in d]])
        with t4:
            v,m=active();st.caption('Active registry: '+(v or 'Unavailable'));table(pd.DataFrame({'Feature':m.get('features',[])}));table(read(OUTPUTS_DIR/'v32_experiments'/'feature_dictionary.csv'))
def goal_charts():
    v,_=active(); d=read(PROD/v/'historical_predictions.csv') if v else pd.DataFrame()
    if {'FTHG','FTAG','PredictedTotalGoals'}.issubset(d):
        z=pd.DataFrame({'Actual':d.FTHG+d.FTAG,'Predicted':d.PredictedTotalGoals});st.scatter_chart(z,x='Actual',y='Predicted');st.bar_chart((z.Actual-z.Predicted).value_counts().sort_index().rename('Matches'))
    else:st.info('V4 historical goal predictions unavailable.')
def history_page():
    st.title('📊 Prediction History');st.write('Track original pre-match predictions and compare them against actual results.')
    d=read(PROD/'live_history_v4.csv');m=read(PROD/'live_metrics_v4.csv')
    if 'Settlement' not in d:st.info('Run python -m src.refresh_v4 to initialise V4 history.');return
    f=d[d.Settlement=='Settled'].copy();p=d[d.Settlement=='Pending']; a,b,c=st.columns(3)
    a.metric('Registered Predictions',len(d));b.metric('Completed Matches',len(f));c.metric('Pending Matches',len(p))
    def fmt(x):return f'{x:.1%}' if pd.notna(x) else 'N/A'
    a,b,c=st.columns(3)
    acc=(f.PredictedResult==f.ActualFTR).mean() if not f.empty else np.nan
    ou=(f.OverUnder2_5.eq('Over')==(f.ActualFTHG+f.ActualFTAG>=3)).mean() if not f.empty else np.nan
    mae=(f.PredictedTotalGoals-f.ActualFTHG-f.ActualFTAG).abs().mean() if not f.empty else np.nan
    a.metric('Match Result Accuracy',fmt(acc));b.metric('Over / Under Accuracy',fmt(ou));c.metric('Goal Prediction MAE',f'{mae:.2f}' if pd.notna(mae) else 'N/A')
    st.divider();st.subheader('Prediction Records');s=st.selectbox('Filter by status',['All','Pending','Settled']);z=d if s=='All' else d[d.Settlement==s]; cols=['Matchday','HomeTeam','AwayTeam','PredictedResult','ActualFTR','PrimaryModel','QSelectedAction','PredictedTotalGoals','Settlement','ModelVersion'];table(z[[c for c in cols if c in z]])
    st.download_button('Download Prediction History',d.to_csv(index=False),file_name='live_history_v4.csv',mime='text/csv')
    if not f.empty:
        st.subheader('Cumulative Prediction Accuracy'); f=f.sort_values('ActualKickoffUTC');st.line_chart((f.PredictedResult==f.ActualFTR).astype(float).expanding().mean().reset_index(drop=True).rename('Accuracy'))
    st.subheader('Live model performance');table(m);st.caption('First valid pre-kickoff forecasts only; metrics are separated by model version.')
