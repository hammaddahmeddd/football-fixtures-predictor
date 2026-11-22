"""V4 release tests: deterministic unit tests and saved-output checks."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import pytest
from src.refresh_v4 import parse_schedule, simulate, settle

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs' / 'v4_production'
REG = ROOT / 'models' / 'v4'

@pytest.fixture
def schedule():
    teams = [f'Team{i}' for i in range(20)]
    rows = []
    for h in teams:
        for a in teams:
            if h == a:
                continue
            i = len(rows) + 1
            done = i == 1
            rows.append(dict(id=i, matchday=1, utcDate='2026-08-01T12:00:00Z' if done else '2026-12-01T12:00:00Z', homeTeam={'name':h}, awayTeam={'name':a}, status='FINISHED' if done else 'TIMED', score={'fullTime':{'home':2 if done else None,'away':1 if done else None}}))
    return parse_schedule(rows, lambda x:x, '2026-27')

def test_v4_schedule(schedule):
    assert len(schedule) == 380
    assert schedule.MatchID.is_unique
    assert schedule.Status.eq('FINISHED').sum() == 1
    assert schedule.loc[schedule.Status.eq('FINISHED'),'FTR'].iloc[0] == 'H'

def test_v4_simulation(schedule):
    remaining = schedule[schedule.Status.ne('FINISHED')]
    grid = np.zeros((3,3)); grid[1,1] = 1
    z = simulate(schedule, remaining, [grid]*len(remaining), n=100, seed=42)
    assert len(z) == 20 and z.Team.is_unique
    assert z.Played.sum() == 2
    assert z.CurrentPoints.sum() == 3
    assert np.isclose(z.ExpectedPoints.sum(), 761)
    for c, expected in [('TitleProbability',1),('Top4Probability',4),('RelegationProbability',3)]:
        assert z[c].between(0,1).all()
        assert np.isclose(z[c].sum(),expected)
    assert np.isclose(z.ExpectedRank.sum(),210)

def test_v4_invalid_grid(schedule):
    remaining = schedule[schedule.Status.ne('FINISHED')]
    with pytest.raises(ValueError):
        simulate(schedule, remaining, [np.zeros((2,2))]*len(remaining), n=2)

def test_v4_first_pre_match_prediction(schedule):
    row = schedule.iloc[0][['MatchID','HomeTeam','AwayTeam','Season','KickoffUTC']].to_dict()
    a = pd.DataFrame([{**row,'PredictionGeneratedUTC':stamp,'PredictedResult':pred,'ModelVersion':'test'} for stamp,pred in [('2026-08-01T10:00:00Z','H'),('2026-08-01T11:00:00Z','A'),('2026-08-01T13:00:00Z','D')]])
    z,m = settle(a,schedule,pd.Timestamp('2026-09-01',tz='UTC'))
    assert len(z)==1 and z.iloc[0].PredictedResult=='H'
    assert z.iloc[0].Settlement=='Settled'
    assert m.loc[m.Model.eq('PredictedResult'),'Accuracy'].eq(1).all()
    z,_=settle(a.iloc[2:],schedule,pd.Timestamp('2026-09-01',tz='UTC'))
    assert z.empty

def read(name):
    path=OUT/name
    assert path.is_file(), f'Missing {path}; run python -m src.refresh_v4'
    return pd.read_csv(path)

def test_v4_registry():
    version=(REG/'active.txt').read_text().strip()
    assert version and Path(version).name==version and version not in ('.','..')
    manifest=json.loads((REG/version/'manifest.json').read_text())
    assert manifest['version']==version
    assert len(manifest['features'])==85
    assert (REG/version/'bundle.joblib').stat().st_size>0

def test_v4_saved_schedule_and_standings():
    r=read('remaining_fixtures_v4.csv'); z=read('season_simulation_v4.csv')
    assert len(z)==20 and z.Team.is_unique
    assert r.MatchID.is_unique
    assert int(z.Played.sum())%2==0
    assert int(z.Played.sum())//2+len(r)==380
    assert z.Played.between(0,38).all()
    assert (z.PointsP05<=z.PointsP95).all()
    assert np.isclose(z.ExpectedRank.sum(),210,atol=.001)
    for c,n in [('TitleProbability',1),('Top4Probability',4),('RelegationProbability',3)]:
        assert z[c].between(0,1).all()
        assert np.isclose(z[c].sum(),n,atol=.001)

def test_v4_saved_forecasts():
    z=read('weekly_predictions_v4.csv'); r=read('remaining_fixtures_v4.csv')
    assert z.MatchID.is_unique
    if z.empty: return
    assert set(z.MatchID)<=set(r.MatchID)
    assert (pd.to_datetime(z.PredictionGeneratedUTC,utc=True)<pd.to_datetime(z.KickoffUTC,utc=True)).all()
    for group in ['Primary','Balanced','Outcome','Q','KNN','SVM','DecisionTree','RandomForest']:
        p=z[[f'{group}_P_{c}' for c in 'ADH']].to_numpy(float)
        assert np.isfinite(p).all() and (p>=0).all() and (p<=1).all()
        assert np.allclose(p.sum(1),1)
    for c in ['PredictedResult','BalancedResult','QPrediction']:
        assert z[c].isin(list('ADH')).all()
    assert np.allclose(z.ProbOver2_5+z.ProbUnder2_5,1)
    assert z.ProbOver2_5.between(0,1).all() and z.ProbUnder2_5.between(0,1).all()
    assert (z.PredictedTotalGoals>=0).all()

def test_v4_saved_history():
    z=read('live_history_v4.csv'); m=read('live_metrics_v4.csv')
    assert z.MatchID.is_unique
    if z.empty: return
    assert z.Settlement.isin(['Pending','Settled']).all()
    assert (pd.to_datetime(z.PredictionGeneratedUTC,utc=True)<pd.to_datetime(z.KickoffUTC,utc=True)).all()
    settled=z.Settlement.eq('Settled').sum()
    if settled==0: assert m.empty
    else:
        assert not m.empty and m.Matches.between(1,settled).all()
        if 'Accuracy' in m: assert m.Accuracy.dropna().between(0,1).all()
