from reports.choose_training_candidates import choose_indices


def test_choose_four_per_phase_prioritizes_verified_crop_loss_without_inventing_indices():
    phases={name:[n*100,n*100+100] for n,name in enumerate(('approach','entry','apex','exit','recovery'))}
    candidates=[dict(index=i,support=dict(category='history_crop_loss' if i%100 in (20,30,40) else 'history_supported')) for i in range(0,500,10)]
    reasons={str(v['index']):[] for v in candidates};reasons['20']=['bad evidence']
    chosen=choose_indices(candidates,reasons,list(range(500)),phases)
    assert len(chosen)==20 and 20 not in chosen
    assert all(len([v for v in chosen if lo<=v<hi])==4 for lo,hi in phases.values())
    assert sum(v%100 in (20,30,40) for v in chosen)==14
