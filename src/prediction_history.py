"""Legacy V2 prediction history with explicit nullable text columns."""
import pandas as pd
import numpy as np
from config import PREDICTIONS_DIR, OUTPUTS_DIR
from src.predictions import fetch_match_records

HISTORY_PATH=PREDICTIONS_DIR/'prediction_history.csv'
WEEKLY_PATH=PREDICTIONS_DIR/'weekly_predictions.csv'
METRICS_PATH=OUTPUTS_DIR/'metrics'/'live_accuracy_summary.csv'
PREDICTION_COLUMNS=['MatchID','Matchday','KickoffUTC','HomeTeam','AwayTeam','PredictionGeneratedUTC','PredictedResult','SelectedModel','SelectedConfidence','PredictedTotalGoals','OverUnder2_5','ProbOver2_5']
RESULT_COLUMNS=['Status','ActualHomeGoals','ActualAwayGoals','ActualResult','ActualTotalGoals','ActualOverUnder','ResultCorrect','OverUnderCorrect','AbsoluteGoalError','ResultRegisteredUTC']
ALL_COLUMNS=PREDICTION_COLUMNS+RESULT_COLUMNS
TEXT_COLUMNS=['MatchID','KickoffUTC','HomeTeam','AwayTeam','PredictionGeneratedUTC','PredictedResult','SelectedModel','OverUnder2_5','Status','ActualResult','ActualOverUnder','ResultRegisteredUTC']


def normalize_types(history):
    history=history.copy()
    for col in TEXT_COLUMNS:
        if col in history:
            history[col]=history[col].astype('string')
    for col in set(ALL_COLUMNS)-set(TEXT_COLUMNS):
        if col in history:
            history[col]=pd.to_numeric(history[col],errors='raise').astype('Float64')
    return history


def load_history():
    if not HISTORY_PATH.exists():
        return normalize_types(pd.DataFrame(columns=ALL_COLUMNS))
    history=pd.read_csv(HISTORY_PATH,dtype={c:'string' for c in TEXT_COLUMNS})
    if not set(ALL_COLUMNS).issubset(history.columns):
        raise ValueError('Invalid prediction history schema.')
    if history.MatchID.isna().any() or history.MatchID.duplicated().any():
        raise ValueError('Missing or duplicate predictions in history.')
    return normalize_types(history)


def load_weekly_predictions():
    if not WEEKLY_PATH.exists():
        raise FileNotFoundError('Weekly predictions missing. Run python -m src.predictions first.')
    weekly=pd.read_csv(WEEKLY_PATH,dtype={'MatchID':'string'})
    missing=set(PREDICTION_COLUMNS)-set(weekly.columns)
    if missing: raise ValueError(f'Missing prediction columns: {missing}')
    return weekly


def register_predictions(history,weekly,matches):
    existing_ids=set(history.MatchID.astype(str)); records=[]
    now=pd.Timestamp.now(tz='UTC')
    for _,row in weekly.iterrows():
        match_id=str(row.MatchID)
        if match_id in existing_ids: continue
        match=matches.get(match_id)
        if match is None or match['status'] not in ['SCHEDULED','TIMED']: continue
        kickoff=pd.to_datetime(match['utcDate'],utc=True)
        generated=pd.to_datetime(row.PredictionGeneratedUTC,utc=True)
        if not generated<=now<kickoff: continue
        record={c:row[c] for c in PREDICTION_COLUMNS}
        record['MatchID']=match_id
        record.update({c:None for c in RESULT_COLUMNS}); record['Status']='Pending'
        records.append(record); existing_ids.add(match_id)
    if records:
        history=pd.concat([normalize_types(history),normalize_types(pd.DataFrame(records))],ignore_index=True)
    print('New predictions registered:',len(records))
    return normalize_types(history)


def update_actual_results(history,matches):
    history=normalize_types(history); updated=0
    for index,row in history.iterrows():
        match_id=str(row.MatchID); match=matches.get(match_id)
        if match is None or match['status']!='FINISHED': continue
        kickoff=pd.to_datetime(match['utcDate'],utc=True)
        generated=pd.to_datetime(row.PredictionGeneratedUTC,utc=True)
        if pd.isna(generated) or pd.isna(kickoff) or generated>=kickoff:
            history.at[index,'Status']='Invalid'; continue
        score=(match.get('score') or {}).get('fullTime') or {}
        h,a=score.get('home'),score.get('away')
        if not isinstance(h,int) or not isinstance(a,int) or min(h,a)<0:
            raise ValueError(f'Invalid score for match {match_id}')
        total=h+a; result='H' if h>a else 'A' if a>h else 'D'; ou='Over' if total>=3 else 'Under'
        values={'Status':'Finished','ActualHomeGoals':h,'ActualAwayGoals':a,'ActualResult':result,'ActualTotalGoals':total,'ActualOverUnder':ou,'ResultCorrect':int(row.PredictedResult==result),'OverUnderCorrect':int(row.OverUnder2_5==ou),'AbsoluteGoalError':abs(float(row.PredictedTotalGoals)-total)}
        for c,v in values.items(): history.at[index,c]=v
        if pd.isna(row.ResultRegisteredUTC):
            history.at[index,'ResultRegisteredUTC']=pd.Timestamp.now(tz='UTC').isoformat(); updated+=1
    print('New completed results registered:',updated)
    return history


def calculate_live_accuracy(history):
    finished=history[history.Status=='Finished']; pending=history[history.Status=='Pending']; total=len(finished)
    correct=pd.to_numeric(finished.ResultCorrect).sum(); over=pd.to_numeric(finished.OverUnderCorrect).sum()
    return pd.DataFrame([dict(TotalPredictions=len(history),CompletedMatches=total,PendingMatches=len(pending),CorrectResultPredictions=int(correct),ResultAccuracy=correct/total if total else np.nan,CorrectOverUnderPredictions=int(over),OverUnderAccuracy=over/total if total else np.nan,GoalMAE=pd.to_numeric(finished.AbsoluteGoalError).mean() if total else np.nan)])


def main():
    print('\n===== LIVE PREDICTION TRACKER =====')
    history=load_history(); weekly=load_weekly_predictions()
    matches={str(m['id']):m for m in fetch_match_records()}
    history=register_predictions(history,weekly,matches)
    history=update_actual_results(history,matches)
    summary=calculate_live_accuracy(history)
    HISTORY_PATH.parent.mkdir(parents=True,exist_ok=True); METRICS_PATH.parent.mkdir(parents=True,exist_ok=True)
    history.sort_values(['KickoffUTC','MatchID']).reset_index(drop=True).to_csv(HISTORY_PATH,index=False)
    summary.to_csv(METRICS_PATH,index=False)
    print('\n===== LIVE ACCURACY SUMMARY ====='); print(summary.to_string(index=False))
    print('Prediction history saved:',HISTORY_PATH); print('Live accuracy saved:',METRICS_PATH); print('===== TRACKING COMPLETED =====')

if __name__=='__main__': main()
