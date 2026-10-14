"""Replayable RGB contrast at semantic road/sidewalk or road/terrain edges.

Semantics locate candidate boundaries only. Stock RGB contrast and qualified
depth at the adjoining road pixel must independently support every claim.
"""
import numpy as np


def edge_patch(luma, road, other):
    """Return contrast and adjacent pixel pair, or reject an unresolved edge."""
    luma,road,other=np.asarray(luma),np.asarray(road,bool),np.asarray(other,bool)
    if luma.shape!=(9,9) or road.shape!=(9,9) or other.shape!=(9,9) or np.any(road&other):
        raise ValueError('edge proof requires disjoint aligned 9x9 masks')
    if np.count_nonzero(road)<3 or np.count_nonzero(other)<3:
        return None
    pairs=[]
    for y in range(3,6):
        for x in range(3,6):
            if not road[y,x]:continue
            for dy,dx in ((-1,0),(1,0),(0,-1),(0,1)):
                if other[y+dy,x+dx]:
                    pairs.append(((y-4)**2+(x-4)**2,y,x,y+dy,x+dx))
    if not pairs:return None
    _,y,x,oy,ox=min(pairs)
    contrast=float(abs(float(np.median(luma[road]))-float(np.median(luma[other]))))
    return contrast,(y,x),(oy,ox)


def edge_pixel_tests(luma,depth,pixels,expected_depth,road,other,thresholds,depth_predictor=None):
    n=len(pixels)
    visible=np.zeros(n,bool);contrast=np.zeros(n,np.float32)
    residual=np.full(n,np.inf,np.float32);other_residual=residual.copy()
    road_bits=np.zeros((n,11),np.uint8);other_bits=road_bits.copy()
    for i,pixel in enumerate(pixels):
        if not np.isfinite(pixel).all() or expected_depth[i]<=0:continue
        x,y=np.rint(pixel).astype(int)
        if x<4 or y<4 or x>=luma.shape[1]-4 or y>=luma.shape[0]-4:continue
        region=np.s_[y-4:y+5,x-4:x+5]
        r,o=road[region],other[region]
        result=edge_patch(luma[region],r,o)
        if result is None:continue
        contrast[i],(py,px),(oy,ox)=result
        rd,od=float(depth[y-4+py,x-4+px]),float(depth[y-4+oy,x-4+ox])
        re=expected_depth[i] if depth_predictor is None else depth_predictor(i,x-4+px,y-4+py)
        oe=expected_depth[i] if depth_predictor is None else depth_predictor(i,x-4+ox,y-4+oy)
        residual[i]=abs(rd-re);other_residual[i]=abs(od-oe)
        tolerance=thresholds.absolute_depth_tolerance_m+thresholds.relative_depth_tolerance*expected_depth[i]
        # The adjacent exterior must also be at this boundary's depth, excluding
        # foreground objects or a semantic skyline that happens to have contrast.
        visible[i]=(np.isfinite([rd,od,re,oe]).all() and min(rd,od)>0 and
                    max(residual[i],other_residual[i])<=tolerance and
                    contrast[i]>=thresholds.minimum_luma_contrast)
        road_bits[i]=np.packbits(r);other_bits[i]=np.packbits(o)
    return visible,contrast,residual,other_residual,road_bits,other_bits
