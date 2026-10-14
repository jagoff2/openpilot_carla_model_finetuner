"""Paint contrast in the exact NV12 U/V samples available to the student."""
import numpy as np


def paint_color_patch(luma, chroma, marking):
    luma=np.asarray(luma);chroma=np.asarray(chroma,float);marking=np.asarray(marking,bool)
    if luma.shape!=(9,9) or chroma.shape!=(9,9,2) or marking.shape!=(9,9):
        raise ValueError('paint color proof requires aligned 9x9 patches')
    yy,xx=np.mgrid[-4:5,-4:5]
    background=(np.maximum(abs(xx),abs(yy))>=3)&~marking
    if background.sum()<8:return None
    color=np.median(chroma[background],axis=0)
    candidates=[]
    for y in range(3,6):
        for x in range(3,6):
            if marking[y,x] and luma[y,x]>=70:
                candidates.append((float(np.linalg.norm(chroma[y,x]-color)),y,x))
    if not candidates:return None
    contrast,y,x=max(candidates)
    return contrast,(y,x)


def paint_color_tests(luma,chroma,depth,pixels,expected_depth,marking,thresholds,depth_predictor=None):
    n=len(pixels);visible=np.zeros(n,bool);contrast=np.zeros(n,np.float32)
    residual=np.full(n,np.inf,np.float32);bits=np.zeros((n,11),np.uint8)
    for i,pixel in enumerate(pixels):
        if not np.isfinite(pixel).all() or expected_depth[i]<=0:continue
        x,y=np.rint(pixel).astype(int)
        if x<4 or y<4 or x>=luma.shape[1]-4 or y>=luma.shape[0]-4:continue
        region=np.s_[y-4:y+5,x-4:x+5];mask=marking[region]
        result=paint_color_patch(luma[region],chroma[region],mask)
        if result is None:continue
        contrast[i],(py,px)=result
        value=float(depth[y-4+py,x-4+px])
        expected=expected_depth[i] if depth_predictor is None else depth_predictor(i,x-4+px,y-4+py)
        residual[i]=abs(value-expected)
        tolerance=thresholds.absolute_depth_tolerance_m+thresholds.relative_depth_tolerance*expected_depth[i]
        visible[i]=(np.isfinite([value,expected]).all() and value>0 and residual[i]<=tolerance and contrast[i]>=thresholds.minimum_chroma_contrast)
        bits[i]=np.packbits(mask)
    return visible,contrast,residual,bits
