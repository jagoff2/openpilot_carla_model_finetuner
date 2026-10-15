import json
import sqlite3
import zlib

from reports.candidate_priority import boundary_risk
from reports.candidate_cache import read_cache, ensure_rankings, PRIORITY


def task(name, family='severe_hairpin', town='Town06', missing=()):
    marks=[{side:dict(type='Solid',width_m=.125) for side in ('left','right')} for _ in range(101)]
    for side,start,stop in missing:
        for i in range(start,stop):marks[i][side]=dict(type='NONE',width_m=0.)
    return dict(group_id=name,family=family,town=town,minimum_ticks=600,nominal_speed_mps=3.5,
        terminal_stop_route_m=80.,maneuver_phase_ranges={'approach':[30.,40.],'recovery':[65.,75.]},
        route=dict(arc_m=list(range(101)),boundary_markings=marks))


def test_unpainted_junctions_are_deprioritized_and_curb_routes_remain_eligible():
    assert boundary_risk(task('paint'))['tier']==0
    assert boundary_risk(task('curb',missing=[('right',15,75)]))['tier']==1
    assert boundary_risk(task('junction',missing=[('left',40,65),('right',40,65)]))['tier']==2
    # Untraversed preview beyond the terminal stop must not penalize a route.
    assert boundary_risk(task('tail',missing=[('left',82,100),('right',82,100)]))['tier']==0
    assert boundary_risk({'route':{}})['tier']==1


def test_ranking_balances_families_and_test_map_before_unmarked_routes(tmp_path):
    path=tmp_path/'screened-pool.sqlite'
    tasks=[task('bad',missing=[('left',30,65),('right',30,65)]),
        task('hairpin-a'),task('hairpin-b'),task('curve','constant_radius'),
        task('straight','straight_recovery'),task('test','straight_recovery','Town05')]
    with sqlite3.connect(path) as db:
        db.execute('CREATE TABLE candidates (id TEXT PRIMARY KEY,priority INTEGER,ticks INTEGER,payload BLOB)')
        for t in tasks:db.execute('INSERT INTO candidates VALUES (?,?,?,?)',
            (t['group_id'],PRIORITY[t['family']],t['minimum_ticks'],zlib.compress(json.dumps(t).encode())))
    db.close()
    before=path.read_bytes()
    actual=list(read_cache(path));names=[t['group_id'] for t in actual]
    assert names==['hairpin-a','curve','straight','test','hairpin-b','bad']
    assert {t['group_id']:t for t in actual}=={t['group_id']:t for t in tasks}
    assert path.read_bytes()==before
    ranking=ensure_rankings(path);modified=ranking.stat().st_mtime_ns
    assert ensure_rankings(path).stat().st_mtime_ns==modified
