from dataclasses import replace
import numpy as np
from lane_dataset.visibility import CameraObservation,probe_lane_evidence
from lane_dataset.depth_sampling import plane_depth_predictor
from lane_dataset.preprocessing import _project_plane_nearest


def observation():
    rgb=np.full((320,640,3),40,np.uint8);rgb[90,100]=240
    row=np.arange(320)[:,None]
    depth=np.broadcast_to(np.divide(260.,row-80,out=np.full((320,1),1000.),where=row>80),(320,640)).astype(np.float32)
    mask=np.zeros((320,640),bool);mask[90,100]=True
    k=np.array([[200.,0,100.],[0,200.,80.],[0,0,1.]])
    return CameraObservation(0,0.,'wide',rgb,depth,np.eye(4),k,np.eye(3,dtype=np.float32),depth_validated=True,
                             marking_mask=mask,surface_normals_world=np.array([[0.,1.,0.]]))


def test_depth_is_compared_at_selected_pixel_not_unrounded_point():
    obs=observation();point=[[0.,1.3,25.]]
    legacy=probe_lane_evidence(replace(obs,surface_normals_world=None),point,['p'],[25.])
    correct=probe_lane_evidence(obs,point,['p'],[25.])
    assert not legacy.crop_visible[0]
    assert correct.crop_visible[0]
    assert correct.crop_depth_residual_m[0]<1e-5
    depth=obs.depth_m.copy();depth[89:92,99:102]=3.
    assert not probe_lane_evidence(replace(obs,depth_m=depth),point,['p'],[25.]).crop_visible[0]


def test_plane_prediction_uses_exact_stock_nearest_source_pixel():
    obs=observation()
    warp=np.array([[1.234,.001,13.23],[.001,1.245,10.31],[0,0,1]],np.float32)
    obs=replace(obs,model_to_source=warp)
    predictor=plane_depth_predictor(obs,np.array([[0.,1.3,25.]]),crop=True)
    expected=_project_plane_nearest(obs.depth_m,warp,(512,256))
    for x,y in ((100,100),(251,150),(12,125)):
        assert abs(predictor(0,x,y)-expected[y,x])<1e-5


def test_carla_half_pixel_center_is_used_after_stock_nearest_lookup():
    obs=replace(observation(),depth_pixel_center_offset=.5)
    predictor=plane_depth_predictor(obs,np.array([[0.,1.3,25.]]))
    assert abs(predictor(0,100,90)-260./10.5)<1e-8
    assert abs(predictor(0,100,90)-260./10.)>.15
