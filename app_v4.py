"""Read-only V4 dashboard. Run: python -m streamlit run app_v4.py
No network calls, training, model deserialization or modifications to V2.
"""
from pathlib import Path
import json
import pandas as pd
import numpy as np
import streamlit as st

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'outputs'
PROD = OUT / 'v4_production'
REG = ROOT / 'models' / 'v4'
LABEL = {'H':'Home win','D':'Draw','A':'Away win'}
PAGES = ['Overview','Match Predictions','Model Comparison','Goal Analysis','Team Analysis','Strategy Maker','Season Simulation','Computational Analysis','Statistical Analysis','Prediction History','Dataset Explorer']
st.set_page_config(page_title='Football Predictor V4',page_icon='⚽',layout='wide')

@st.cache_data(show_spinner=False)
def _read(path, stamp, size):
    return pd.read_csv(path)

def read(path, warn=True):
    try:
        s=path.stat(); return _read(str(path),s.st_mtime_ns,s.st_size)
    except (OSError, ValueError, pd.errors.ParserError) as e:
        if warn: st.info(f'Unavailable: {path.relative_to(ROOT)} ({type(e).__name__})')
        return pd.DataFrame()

def active():
    try:
        v=(REG/'active.txt').read_text().strip()
        if not v or Path(v).name!=v or v in ('.','..'): raise ValueError('Invalid registry ID')
        m=json.loads((REG/v/'manifest.json').read_text()); return v,m
    except (OSError,ValueError) as e:
        st.warning(f'Active V4 registry unavailable: {e}'); return '',{}

def table(d, name=None):
    if d.empty: return
    st.dataframe(d, width='stretch', hide_index=True)
    if name: st.download_button('Download '+name, d.to_csv(index=False).encode('utf-8'), file_name=name+'.csv', mime='text/csv',key='download_'+name)

def chart(d,x,y):
    if not d.empty and {x,y}.issubset(d.columns):
        z=d[[x,y]].dropna(); z[y]=pd.to_numeric(z[y],errors='coerce'); z=z.dropna()
        if not z.empty: st.bar_chart(z.set_index(x)[y])

def image(path):
    if path.is_file(): st.image(str(path),width='stretch')

def exp(v,file,warn=True): return read(OUT/f'{v}_experiments'/file,warn)
def stats(file,warn=True): return exp('v39','full/'+file,warn)
def bench(file,warn=True): return exp('v38','full/'+file,warn)
def historical(): return read(PROD/version/'historical_predictions.csv') if version else pd.DataFrame()
def live(): return read(PROD/'weekly_predictions_v4.csv')
def number(v,fmt='.3f'):
    try:
        x=float(v); return format(x,fmt) if np.isfinite(x) else 'N/A'
    except (TypeError,ValueError): return 'N/A'
def pct(v): return number(v,'.1%')
def metricrow(items):
    for col,(k,v) in zip(st.columns(len(items)),items): col.metric(k,v)
def accuracy(d,p='PredictedResult'):
    if not {'FTR',p}.issubset(d): return np.nan
    z=d[d.FTR.isin(LABEL)&d[p].isin(LABEL)]; return (z.FTR==z[p]).mean() if len(z) else np.nan

def overview():
    st.title('Football Intelligence | V4'); st.caption('Premier League prediction and research dashboard')
    h=historical(); l=live(); l=l if 'KickoffUTC' not in l else l[pd.to_datetime(l.KickoffUTC,utc=True,errors='coerce')>pd.Timestamp.now(tz='UTC')]
    metricrow([('Training rows',str(manifest.get('training_rows','N/A'))),('Live fixtures',len(l)),('Historical accuracy',pct(accuracy(h))),('Features',len(manifest.get('features',[])))])
    st.info('Historical evaluation seasons have been inspected repeatedly. These results are not an independent live accuracy claim.')
    st.subheader('Active model registry'); st.write('Version:',version or 'Unavailable'); st.write('Training cutoff:',manifest.get('cutoff','N/A'))
    table(pd.DataFrame(list(manifest.get('selection',{}).items()),columns=['Task','Selection']))
    st.subheader('Historical outcome performance'); d=stats('metrics.csv');
    if {'Task','Accuracy'}.issubset(d): chart(d[d.Task=='Outcome'].nlargest(10,'Accuracy'),'Model','Accuracy')
    st.caption('Research model configurations may depend on features unavailable in live V4.')

def matches():
    st.title('Match Predictions'); d=live()
    if d.empty:return
    if 'KickoffUTC' in d:
        k=pd.to_datetime(d.KickoffUTC,utc=True,errors='coerce'); upcoming=k>pd.Timestamp.now(tz='UTC'); st.caption(f'{int(upcoming.sum())} upcoming of {len(d)} saved forecasts. Times are UTC.')
        if not st.checkbox('Include started or expired forecasts'): d=d[upcoming]
    if d.empty: st.info('No future fixtures in the saved snapshot. Run the V4 predict command to refresh.'); return
    if not {'HomeTeam','AwayTeam'}.issubset(d):return
    if 'ModelVersion' in d and (d.ModelVersion!=version).any():st.warning('Snapshot contains a different model version from the active registry.')
    opts=[f'{r.HomeTeam} vs {r.AwayTeam}' for r in d.itertuples()]; i=st.selectbox('Fixture',range(len(opts)),format_func=lambda i:opts[i]); r=d.iloc[i]
    st.header(opts[i]); st.caption(f"Generated: {r.get('PredictionGeneratedUTC','N/A')} | Kickoff: {r.get('KickoffUTC','N/A')}")
    metricrow([('Primary',LABEL.get(r.get('PredictedResult'),'N/A')),('Balanced',LABEL.get(r.get('BalancedResult'),'N/A')),('Q-Learning',LABEL.get(r.get('QPrediction'),'N/A'))])
    st.subheader('Outcome probabilities'); rows=[]
    for group in ['Primary','Balanced','Outcome','Q']:
        rows.append({'Strategy':group,**{LABEL[c]:r.get(f'{group}_P_{c}',np.nan) for c in ['H','D','A']}})
    probs=pd.DataFrame(rows).set_index('Strategy'); st.bar_chart(probs);st.caption('Primary and balanced decisions may use draw multipliers. Outcome probabilities are independently selected; the highest probability need not match the decision.')
    metricrow([('Selected goals',number(r.get('PredictedTotalGoals'))),('DC expected goals',number(pd.to_numeric(r.get('ExpectedHomeGoalsDC'),errors='coerce')+pd.to_numeric(r.get('ExpectedAwayGoalsDC'),errors='coerce'))),('Over 2.5',pct(r.get('ProbOver2_5')))])
    st.write('Over/Under decision:',r.get('OverUnder2_5','N/A'),' | Model:',r.get('OverModel','N/A'))
    st.caption('Historical median and prior may be constant across fixtures. Dixon-Coles goal rates are fixture-specific.')
    table(pd.DataFrame([{'Rank':j,'Score':r.get(f'Score{j}'),'Probability':pct(r.get(f'Score{j}Probability'))} for j in range(1,4)]));st.caption('A modal exact score is not the same as the most probable aggregate outcome.')
    table(pd.DataFrame([{'Model':c.removesuffix('_Prediction'),'Prediction':LABEL.get(r[c],r[c])} for c in d if c.endswith('_Prediction')]))
    st.write('Q action:',r.get('QSelectedAction','N/A'),' | Clusters H/A:',r.get('HCluster','N/A'),r.get('ACluster','N/A'));table(d,'v4_fixtures')

def models():
    st.title('Model Comparison'); d=stats('metrics.csv');
    if 'Task' in d:
        z=d[d.Task=='Outcome']; m=st.selectbox('Rank by',['Accuracy','MacroF1','MacroPrecision','MacroRecall','DrawRecall','LogLoss']); z=z.sort_values(m,ascending=m=='LogLoss');chart(z,'Model',m);table(z,'outcome_metrics')
    st.subheader('V4 validation selection');table(read(REG/version/'validation.csv') if version else pd.DataFrame(),'v4_validation')
    st.subheader('V4 historical performance by season');h=historical()
    if 'Season' in h:
        table(pd.DataFrame([{'Season':s,'Matches':len(g),'PrimaryAccuracy':accuracy(g),'BalancedAccuracy':accuracy(g,'BalancedResult'),'QAccuracy':accuracy(g,'QPrediction')} for s,g in h.groupby('Season')]))

def goals():
    st.title('Goal Analysis');d=stats('metrics.csv')
    if 'Task' in d:
        for task,m in [('Goals','MAE'),('OverUnder','Accuracy')]:
            st.subheader(task);z=d[d.Task==task].sort_values(m,ascending=task=='Goals');chart(z,'Model',m);table(z,task+'_metrics')
    st.subheader('Live V4 goal forecasts');d=live(); cols=['HomeTeam','AwayTeam','GoalsMAEModel','PredictedTotalGoals','ExpectedHomeGoalsDC','ExpectedAwayGoalsDC','OverModel','ProbOver2_5','Score1','Score2','Score3'];table(d[[c for c in cols if c in d]],'live_goals')
    st.caption('Selected advanced regression models have not reliably beaten historical baselines.')

def teams():
    st.title('Team Analysis');st.caption('Historical K-Means cluster IDs are arbitrary and may change between fitted models.')
    d=exp('v37','clustering_metrics.csv');
    if 'Selected' in d: d=d[d.Selected.astype(str).str.lower().isin(['true','1'])]
    table(d,'clustering_metrics');
    for name in ['PCA_clusters.png','Silhouette.png','Inertia.png','BootstrapARI.png']:image(OUT/'v37_experiments'/name)
    for name in ['cluster_profiles','cluster_counts','latest_team_profiles','classification']:st.subheader(name.replace('_',' ').title());table(exp('v37',name+'.csv'),name)
    st.subheader('V4 live fixture clusters');d=live();table(d[[c for c in ['HomeTeam','AwayTeam','HCluster','ACluster'] if c in d]])

def strategy():
    st.title('Strategy Maker');st.caption('V3.6 historical Q-Learning experiments; not a claim about V4 live policy accuracy.')
    d=exp('v36','comparison.csv');chart(d,'Strategy','Accuracy');table(d,'q_comparison');table(exp('v36','selection.csv'),'q_selection');table(exp('v36','action_usage.csv'),'q_actions')
    for f in ['QAccuracy_convergence.png','QBalanced_convergence.png','CumulativeReward.png','CumulativeRegretVsFixed.png']:image(OUT/'v36_experiments'/f)
    st.subheader('V4 live policy');d=live();table(d[[c for c in ['HomeTeam','AwayTeam','QSelectedAction','QPrediction'] if c in d]])

def simulation():
    st.title('Season Simulation');st.warning('Legacy V2 simulation only. V4 season simulation has not yet been implemented. These standings may be stale.')
    for f in ['season_simulation_summary.csv','season_probabilities.csv']:
        d=read(ROOT/'data'/'predictions'/f,False)
        if d.empty:d=read(OUT/'metrics'/f,False)
        if d.empty:st.info(f'No saved legacy {f} available.')
        else:table(d,f.removesuffix('.csv'))

def compute():
    st.title('Computational Analysis');d=bench('full_dataset_comparison.csv'); available=[c for c in ['FitWallSeconds','SingleMedianMs','BatchMedianMs','FitPeakRSSMiB','ModelKiB'] if c in d]
    if available: m=st.selectbox('Metric',available);chart(d.sort_values(m),'Model',m)
    table(d,'benchmark_full');table(bench('benchmark_summary.csv'),'benchmark_scaling');image(OUT/'v38_experiments'/'full'/'Scalability.png');table(bench('theoretical_complexity.csv'),'complexity');table(bench('benchmark_errors.csv',False),'benchmark_errors')
    st.caption('Representative configurations; process RSS includes runtime overhead. Feature engineering is excluded.')

def statistics():
    st.title('Statistical Analysis');st.warning('Retrospective evaluation. Confidence intervals are not adjusted for multiple comparisons; Holm-adjusted tests are exploratory.')
    d=stats('confidence_intervals.csv');
    if {'Task','Metric'}.issubset(d):
        task=st.selectbox('Task',d.Task.unique());z=d[d.Task==task]; m=st.selectbox('Metric',z.Metric.unique());table(z[z.Metric==m],'intervals')
    image(OUT/'v39_experiments'/'full'/'accuracy_intervals.png'); d=stats('paired_tests.csv');
    if 'Candidate' in d:
        c=st.selectbox('Candidate',['All']+sorted(d.Candidate.dropna().unique().tolist()));table(d if c=='All' else d[d.Candidate==c],'paired_tests')
    for f in ['Outcome_calibration.png','OverUnder_calibration.png']:image(OUT/'v39_experiments'/'full'/f)
    table(stats('roc_auc.csv'),'roc_auc');table(stats('by_season.csv'),'season_metrics');table(stats('source_audit.csv'),'source_audit')

def archive_rows():
    frames=[]
    for p in sorted((PROD/'prediction_archive').glob('*.csv')):
        d=read(p,False)
        if not d.empty:frames.append(d)
    if not frames:return pd.DataFrame(),0
    d=pd.concat(frames,ignore_index=True);needed={'MatchID','KickoffUTC','PredictionGeneratedUTC'}
    if not needed.issubset(d):return pd.DataFrame(),len(d)
    gen=pd.to_datetime(d.PredictionGeneratedUTC,utc=True,errors='coerce');kick=pd.to_datetime(d.KickoffUTC,utc=True,errors='coerce');good=gen.notna()&kick.notna()&(gen<kick)&d.MatchID.notna();invalid=int((~good).sum());d=d.loc[good].copy();d['_gen']=gen[good];d=d.sort_values('_gen').drop_duplicates('MatchID',keep='first').drop(columns='_gen');return d,invalid

def history_page():
    st.title('Prediction History');d,invalid=archive_rows();metricrow([('Unique pre-match forecasts',len(d)),('Invalid snapshot rows excluded',invalid)])
    st.info('First valid pre-kickoff forecast per MatchID. Automated result settlement and live accuracy are pending; started does not mean settled.')
    if not d.empty:
        d['ArchiveStatus']=np.where(pd.to_datetime(d.KickoffUTC,utc=True)>pd.Timestamp.now(tz='UTC'),'Upcoming','Started / unverified');table(d,'first_forecasts')

def dataset():
    st.title('Dataset Explorer');d=read(ROOT/'data'/'v3_experiments'/'history.csv',False)
    if d.empty:st.info('Historical cache is not present. Showing available raw CSV files instead.'); files=sorted((ROOT/'data'/'raw').glob('*.csv')); 
    else:files=[]
    if files:d=read(st.selectbox('Raw dataset',files,format_func=lambda p:p.name))
    if not d.empty:
        metricrow([('Rows',len(d)),('Columns',len(d.columns))]);
        if 'Season' in d:
            s=st.selectbox('Season',['All']+sorted(d.Season.astype(str).unique().tolist())); d=d if s=='All' else d[d.Season.astype(str)==s]
        table(d,'dataset')
    st.subheader('Active V4 feature list');table(pd.DataFrame({'Feature':manifest.get('features',[])}),'v4_features');table(exp('v32','feature_dictionary.csv',False),'experimental_dictionary')

version,manifest=active()
with st.sidebar:
    st.title('⚽ Football Predictor');st.caption('V4 | Research & production preview');page=st.radio('Navigation',PAGES);st.caption('Read-only dashboard');st.caption('Active: '+(version or 'unavailable'))
    if st.button('Refresh saved data'):st.cache_data.clear();st.rerun()
handlers=dict(zip(PAGES,[overview,matches,models,goals,teams,strategy,simulation,compute,statistics,history_page,dataset]))
handlers[page]()
