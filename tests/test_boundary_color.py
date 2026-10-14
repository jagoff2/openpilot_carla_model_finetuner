from dataclasses import replace
import numpy as np
import pytest
from lane_dataset.visibility import CameraObservation,probe_lane_evidence
from lane_dataset.preprocessing import image_to_nv12,split_nv12,warp_and_pack_nv12
from lane_dataset.admission import _verify_pixel_claims


def test_yellow_paint_uses_actual_packed_chroma_and_requires_semantics_depth():
    rgb=np.full((320,640,3),200,np.uint8)
    rgb[70:130,98:103]=[220,220,0]
    mask=np.zeros((320,640),bool);mask[70:130,98:103]=True
    obs=CameraObservation(0,0.,'wide',rgb,np.full((320,640),10.,np.float32),
        np.eye(4),np.diag([100.,100.,1.]),np.eye(3,dtype=np.float32),depth_validated=True,marking_mask=mask)
    evidence=probe_lane_evidence(obs,[[10.,10.,10.]],['p'],[10.])
    assert evidence.crop_visible[0] and evidence.crop_boundary_kind[0]==3
    nv12,layout=image_to_nv12(rgb,'RGB');packed=warp_and_pack_nv12(nv12,layout,obs.model_to_source)
    _verify_pixel_claims(evidence,packed,np.zeros(3),np.array([[10.,10.,10.]]))
    broken=packed.copy();broken[4:]=128
    with pytest.raises(ValueError,match='color paint contrast'):
        _verify_pixel_claims(evidence,broken,np.zeros(3),np.array([[10.,10.,10.]]))
    for changed in (replace(obs,marking_mask=np.zeros_like(mask)),replace(obs,depth_m=np.full_like(obs.depth_m,3.))):
        assert not probe_lane_evidence(changed,[[10.,10.,10.]],['p'],[10.]).crop_visible.any()
    y,u,v=split_nv12(nv12,layout)
    reused=probe_lane_evidence(replace(obs,luma=y,chroma_u=u,chroma_v=v),[[10.,10.,10.]],['p'],[10.])
    np.testing.assert_equal(reused.crop_contrast,evidence.crop_contrast)
