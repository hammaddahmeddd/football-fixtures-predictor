import pandas as pd
from src import prediction_history as ph


def test_pending_csv_settles_draw(tmp_path, monkeypatch):
    path=tmp_path/'history.csv'
    row=dict.fromkeys(ph.ALL_COLUMNS)
    row.update(MatchID='1',Matchday=1,KickoffUTC='2026-08-01T12:00:00Z',HomeTeam='A',AwayTeam='B',PredictionGeneratedUTC='2026-08-01T10:00:00Z',PredictedResult='D',SelectedModel='KNN',SelectedConfidence=.4,PredictedTotalGoals=2.5,OverUnder2_5='Under',ProbOver2_5=.4,Status='Pending')
    pd.DataFrame([row]).to_csv(path,index=False)
    monkeypatch.setattr(ph,'HISTORY_PATH',path)
    history=ph.load_history()
    matches={'1':dict(status='FINISHED',utcDate='2026-08-01T12:00:00Z',score={'fullTime':{'home':1,'away':1}})}
    result=ph.update_actual_results(history,matches)
    assert result.iloc[0].ActualResult=='D'
    assert result.iloc[0].ActualOverUnder=='Under'
    assert result.iloc[0].ResultCorrect==1
    assert result.iloc[0].AbsoluteGoalError==.5
    assert result.iloc[0].PredictionGeneratedUTC==row['PredictionGeneratedUTC']
    assert ph.calculate_live_accuracy(result).iloc[0].ResultAccuracy==1
    timestamp=result.iloc[0].ResultRegisteredUTC
    assert ph.update_actual_results(result,matches).iloc[0].ResultRegisteredUTC==timestamp
