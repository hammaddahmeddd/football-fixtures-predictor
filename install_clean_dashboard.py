"""Patch the original dashboard in place, keeping its app.py CSS and navigation."""
from pathlib import Path
import ast, shutil, re, datetime
root=Path(__file__).resolve().parent
D=root/'dashboard'
marker='# V4 CLEAN DASHBOARD ADAPTER'
def rep(s,a,b):
    if a not in s:raise ValueError('Original source anchor missing: '+a[:100])
    return s.replace(a,b,1)
def function(s,name,body):
    n=next(n for n in ast.parse(s).body if isinstance(n,ast.FunctionDef) and n.name==name)
    lines=s.splitlines(keepends=True); start=min([n.lineno]+[x.lineno for x in n.decorator_list])-1
    return ''.join(lines[:start])+f'def {name}():\n'+''.join('    '+x+'\n' for x in body.splitlines())+''.join(lines[n.end_lineno:])
def inject(s):return marker+'\nfrom dashboard import v4_data as v4\n'+s
changes={}
for name in ['overview','match_predictions','model_comparison','goal_analysis','season_table','strategy_maker','prediction_history']:
    p=D/(name+'.py');s=p.read_text(encoding='utf-8-sig')
    if marker in s:continue
    if name=='overview':
        s=re.sub(r'weekly = read_csv\(\s*PREDICTIONS_DIR / "weekly_predictions.csv"\s*\)', 'weekly = v4.weekly()',s)
        s=re.sub(r'season = read_csv\(\s*OUTPUTS_DIR\s*/ "metrics"\s*/ "season_simulation_summary.csv"\s*\)', 'season = v4.season()',s)
        if 'weekly = v4.weekly()' not in s or 'season = v4.season()' not in s:raise ValueError('Overview anchors missing')
        s=s.replace('"Random Forest Regressor"','"Poisson / Ridge / Median"').replace('"python -m src.season_simulation"','"python -m src.refresh_v4"')
        s=rep(s,'st.subheader("Project Overview")','st.subheader("Project Overview")\n    version, manifest = v4.active()\n    st.caption("V4 registry: " + (version or "Unavailable"))')
        s=s.replace('"Historical Matches",\n            f"{len(historical):,}"\n            if historical is not None\n            else "N/A"','"V4 Training Matches",\n            f"{manifest.get(\'training_rows\', \'N/A\')}"')
    elif name=='match_predictions':
        s=function(s,'load_predictions','return v4.weekly()')
        s=s.replace('src.predictions','src.refresh_v4').replace('f"Q-Learning selected the "','f"The primary strategy uses the "').replace('"classifier for this fixture."','"model for this fixture."').replace('"Selected by RL"','"Selected by Q-Learning"').replace('if name == match["SelectedModel"]','if name.replace(" ", "") == str(match.get("QSelectedAction", "")).split("@")[0]').replace('"The Q-Learning policy selected "','"The validation-selected primary model is "').replace('"weekly_predictions.csv"','"weekly_predictions_v4.csv"')
        s=s.replace('"Model Confidence"','"Primary decision probability"').replace('"Expected Goals"','"Selected Goal Estimate"')
        s+='\n    v4.match_extra(match)\n'
    elif name=='model_comparison':
        s=function(s,'load_results','return v4.metrics("Outcome")')
        s=s.replace('"models on historical Premier League data."','"and additional models on the 760-match historical benchmark."').replace('"Evaluation of the four classification "','"Evaluation of classification "').replace('src.train_models','src.accuracy_v39').replace('"Macro_F1",','"Macro_F1",\n            "MacroF1",').replace('st.subheader("Confusion Matrices")','st.subheader("Confusion Matrices (legacy V2)")')
        s+='\n    v4.research()\n'
    elif name=='goal_analysis':
        s=re.sub(r'results = read_optional_csv\(\s*METRICS_DIR\s*/ "regression_results.csv"\s*\)','results = v4.metrics("Goals")',s)
        s=re.sub(r'weekly = read_optional_csv\(\s*PREDICTIONS_DIR\s*/ "weekly_predictions.csv"\s*\)','weekly = v4.weekly()',s)
        start=s.index('    actual_chart =');end=s.index('    st.subheader("Upcoming Fixture Goal Forecasts")',start)
        s=s[:start]+'    v4.goal_charts()\n    st.divider()\n    st.subheader("Over/Under Evaluation")\n    v4.table(v4.metrics("OverUnder"))\n    st.divider()\n'+s[end:]
        s=s.replace('"The saved regression evaluation showed "', '"Historical evaluation of selected regression models showed "').replace('"OverUnder2_5",\n        ]','"OverUnder2_5",\n            "ExpectedHomeGoalsDC", "ExpectedAwayGoalsDC", "Score1",\n        ]')
    elif name=='season_table':
        s=function(s,'load_season','return v4.season()');s=s.replace('"1,000 Monte Carlo simulations."','"V4 Monte Carlo simulations (default 5,000)."').replace('src.season_simulation','src.refresh_v4').replace('file_name="season_simulation_summary.csv"','file_name="season_simulation_v4.csv"')
        start=s.index('    charts = [');end=s.index('    st.divider()',start)
        s=s[:start]+'    for col in ["ExpectedPoints", "Top4Probability", "RelegationProbability"]:\n        st.bar_chart(display.set_index("Team")[[col]])\n\n'+s[end:]
    elif name=='strategy_maker':
        s=re.sub(r'comparison = read_optional_csv\(\s*METRICS_DIR\s*/ "q_learning_comparison.csv"\s*\)','comparison = v4.read(OUTPUTS_DIR / "v36_experiments" / "comparison.csv")',s)
        s=re.sub(r'history = read_optional_csv\(\s*METRICS_DIR\s*/ "q_learning_training_history.csv"\s*\)','history = v4.read(OUTPUTS_DIR / "v36_experiments" / "training_history.csv")',s)
        s=re.sub(r'weekly = read_optional_csv\(\s*PREDICTIONS_DIR\s*/ "weekly_predictions.csv"\s*\)','weekly = v4.weekly()',s)
        s=s.replace('"SelectedModel"','"QSelectedAction"').replace('"PredictedResult"','"QPrediction"').replace('"SelectedConfidence",','').replace('"selects one of four trained classification "','"selects among trained statistical and classification "').replace('"Available Actions"','"Example Available Actions"').replace('"2. It chooses KNN, SVM, Decision Tree "','"2. It can choose KNN, SVM, Decision Tree "').replace('"or Random Forest."','"or other supported models."')
        s=rep(s,'st.subheader("Q-Learning Evaluation")','st.subheader("Q-Learning Evaluation")\n    st.caption("V3.6 research policies; V4 live policy is separately fitted.")')
    else:
        s=function(s,'render','v4.history_page()')
    s=inject(s);ast.parse(s);changes[p]=s
if not (root/'app.py').is_file() or not (D/'v4_data.py').is_file():raise SystemExit('Extract package into the original project root first.')
if changes:
    backup=root.parent/(root.name+'_dashboard_backup_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f'));backup.mkdir()
    for p,s in changes.items():shutil.copy2(p,backup/p.name)
    for p,s in changes.items():p.write_text(s,encoding='utf-8')
    print('Original dashboard modules backed up to:',backup)
print('Done. app.py and its styling/navigation were not changed.')
