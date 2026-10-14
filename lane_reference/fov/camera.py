"""Executable camera assumptions and the first full-frame representation.

Source pinhole constants reproduce the runner, not verified physical optics.
The sensitivity profiles deliberately remain hypotheses until real calibration.
Coordinates: pixels (right,down), rays (right,down,forward), device FRD.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass
import numpy as np

VIEW_FROM_DEVICE = np.array([[0.,1.,0.],[0.,0.,1.],[1.,0.,0.]])
MED_K = np.array([[910.,0.,256.],[0.,910.,47.6],[0.,0.,1.]])
SBIG_K = np.array([[455.,0.,256.],[0.,455.,151.8],[0.,0.,1.]])


@dataclass(frozen=True)
class CameraProfile:
    name: str
    width: int = 1928
    height: int = 1208
    focal_px: float = 567.
    k1: float = 0.
    projection: str = 'brown_pinhole'
    optical_status: str = 'assumed_not_calibrated'

    @property
    def K(self):
        return np.array([[self.focal_px,0.,self.width/2],
                         [0.,self.focal_px,self.height/2],[0.,0.,1.]])

    def project(self, rays):
        rays = np.asarray(rays,np.float64)
        xy = rays[...,:2] / rays[...,2:3]
        factor = 1 + self.k1 * np.sum(xy*xy,axis=-1,keepdims=True)
        pixels = xy * factor * self.focal_px + [self.width/2,self.height/2]
        valid = (rays[...,2] > 0) & np.isfinite(pixels).all(-1)
        valid &= (pixels[...,0]>=0) & (pixels[...,0]<self.width)
        valid &= (pixels[...,1]>=0) & (pixels[...,1]<self.height)
        return pixels,valid

    def unproject(self,pixels):
        distorted = (np.asarray(pixels,np.float64)-[self.width/2,self.height/2])/self.focal_px
        radius_d = np.linalg.norm(distorted,axis=-1,keepdims=True)
        radius = radius_d.copy()
        for _ in range(30):
            derivative = 1 + 3*self.k1*radius*radius
            if np.any(derivative <= 0): raise ValueError('Profile folds within requested pixels')
            radius -= (radius + self.k1*radius**3-radius_d)/derivative
        xy = distorted * np.divide(radius,radius_d,out=np.ones_like(radius),where=radius_d>0)
        rays = np.concatenate([xy,np.ones_like(radius)],axis=-1)
        return rays/np.linalg.norm(rays,axis=-1,keepdims=True)


def sensitivity_profiles():
    # +/-10% focal and modest monotone radial alternatives are an experiment
    # range, not a confidence interval or guaranteed bounds on the real lens.
    return [CameraProfile('source-pinhole'),
            CameraProfile('focal-minus10',focal_px=510.3),
            CameraProfile('focal-plus10',focal_px=623.7),
            CameraProfile('radial-minus',k1=-.015),
            CameraProfile('radial-plus',k1=.04)]


def euler_matrix(rpy):
    # Preserve input precision and the source's operation order. Expanding this
    # product or promoting float32 angles before sin/cos changes warp rounding.
    roll,pitch,yaw = np.asarray(rpy)
    cr,sr=np.cos(roll),np.sin(roll)
    cp,sp=np.cos(pitch),np.sin(pitch)
    cy,sy=np.cos(yaw),np.sin(yaw)
    rx=np.array([[1,0,0],[0,cr,-sr],[0,sr,cr]])
    ry=np.array([[cp,0,sp],[0,1,0],[-sp,0,cp]])
    rz=np.array([[cy,-sy,0],[sy,cy,0],[0,0,1]])
    return rz @ ry @ rx


def original_warp(profile, rpy=(0.,0.,0.), wide=False):
    """Exact source pinhole warp; radial profile distortion is NOT silently applied."""
    model_k = SBIG_K if wide else MED_K
    return (profile.K @ VIEW_FROM_DEVICE @ euler_matrix(rpy)
            @ np.linalg.inv(model_k @ VIEW_FROM_DEVICE)).astype(np.float32)


def full_frame_mapping(width=1928,height=1208,out_width=512,out_height=256):
    """Fit complete sensor rectangle with equal x/y scale and centered padding.

    Destination pixel centers map to source pixel centers; every destination
    cell in the active rectangle covers sensor area. Padding has its own mask.
    No assertion about windshield/body occlusion is implied by sensor validity.
    """
    scale=min(out_width/width,out_height/height)
    extent=np.array([width,height])*scale
    offset=(np.array([out_width,out_height])-extent)/2
    x,y=np.meshgrid(np.arange(out_width,dtype=np.float64)+.5,
                    np.arange(out_height,dtype=np.float64)+.5)
    sensor=np.stack([(x-offset[0])/scale-.5,(y-offset[1])/scale-.5],axis=-1)
    mask=(x>=offset[0]) & (x<offset[0]+extent[0]) & (y>=offset[1]) & (y<offset[1]+extent[1])
    matrix=np.array([[1/scale,0,(.5-offset[0])/scale-.5],
                     [0,1/scale,(.5-offset[1])/scale-.5],[0,0,1]],np.float32)
    return {'source_xy':sensor.astype(np.float32),'valid_mask':mask,'matrix':matrix,
            'scale':float(scale),'offset':offset.tolist(),'active_extent':extent.tolist(),
            'padding_yuv':[16,128,128]}


def specification():
    mapping=full_frame_mapping()
    narrow=CameraProfile('source-narrow',focal_px=2648.)
    return {'schema_version':1,
            'source':'../iqpilot/iqpilot/common/transformations/camera.py and model.py',
            'measured_parameters':[],
            'source_declared_parameters':['1928x1208 sensor dimensions','narrow focal 2648px',
                'wide focal 567px (explicitly approximate)','principal point at image center'],
            'assumptions':['zero mounting rotation until per-recording calibration applied',
                'camera co-location only for simulator probe, physical translation unmeasured',
                'candidate valid mask denotes stored sensor rectangle, not physical obstruction masks',
                'radial/focal profile range is hypothetical, not a confidence bound on real optics'],
            'narrow':asdict(narrow),'wide_profiles':[asdict(p) for p in sensitivity_profiles()],
            'model_narrow_intrinsics':MED_K.tolist(),'model_wide_intrinsics':SBIG_K.tolist(),
            'original_narrow_warp_zero_rotation':original_warp(narrow).tolist(),
            'original_wide_warp_zero_rotation':original_warp(sensitivity_profiles()[0],wide=True).tolist(),
            'candidate':{k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in mapping.items()
                         if k not in ('source_xy','valid_mask')},
            'candidate_narrow':'unchanged MED warp',
            'candidate_wide':'whole stored sensor rectangle fit at equal x/y scale into 512x256 with padding',
            'model_packed_shape':[1,12,128,256],
            'calibration_gate':'pending central and peripheral correspondence validation',
            'cue_visibility_gate':'pending scenario-specific actual-size and compression checks'}


def pack_full_wide_nv12(buffer,layout):
    """Candidate wide representation, retaining the established YUV channel ABI."""
    from .preprocessing import warp_and_pack_nv12
    mapping=full_frame_mapping(layout.width,layout.height)
    packed=warp_and_pack_nv12(buffer,layout,mapping['matrix'])
    mask=mapping['valid_mask']
    planes=[mask[0::2,0::2],mask[1::2,0::2],mask[0::2,1::2],mask[1::2,1::2]]
    # A chroma cell is valid only if its complete 2x2 luma footprint is valid.
    chroma=np.logical_and.reduce(planes)
    packed_mask=np.stack(planes+[chroma,chroma])
    packed=np.where(packed_mask,packed,np.array([16,16,16,16,128,128],np.uint8)[:,None,None])
    return packed.astype(np.uint8),packed_mask
