"""Deterministic OpenDRIVE curve fixtures, with split-disjoint shape parameters.

XML uses ASAM inertial axes (Y left, positive curvature left). CARLA's parser
performs its world conversion. Actual lane boundaries and markings must then be
read from CARLA and validated in rendered images before any accepted labels.
Format reference: https://www.asam.net/fileadmin/Standards/OpenDRIVE/ASAM_OpenDRIVE_BS_V1-6-0.html
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import xml.etree.ElementTree as ET

import numpy as np
from scipy.integrate import quad


@dataclass(frozen=True)
class ProceduralParameters:
    family: str
    radius_m: float
    turn: str = "left"
    lane_width_m: float = 3.6
    shoulder_width_m: float = 1.
    approach_m: float = 65.
    exit_m: float = 85.
    grade: float = 0.
    bank_rad: float = 0.
    transition_fraction: float = .6
    split: str = "train"

    def validate(self):
        if self.family not in ("tightening", "s_bend", "hairpin", "constant_radius"):
            raise ValueError("unsupported procedural curve family")
        if self.turn not in ("left","right") or self.split not in ("train","validation","test","development"):
            raise ValueError("invalid turn or split")
        if not (10. <= self.radius_m <= 80. and 3. <= self.lane_width_m <= 4.5):
            raise ValueError("radius/width outside declared feasible generator range")
        if not (.3 <= self.transition_fraction <= .9 and self.approach_m >= 30 and self.exit_m >= 50):
            raise ValueError("insufficient transitions or retained approach/exit")
        if abs(self.grade) > .06 or abs(self.bank_rad) > .10 or self.shoulder_width_m < .5:
            raise ValueError("grade/bank/shoulder outside declared generator range")
        if self.radius_m <= self.lane_width_m+self.shoulder_width_m+2.:
            raise ValueError("inside road edge could collapse")

    @property
    def geometry_group(self):
        # Mirror direction and traversal variants deliberately share a group.
        data=asdict(self);data.pop("turn");data.pop("split")
        return "procedural-"+hashlib.sha256(json.dumps(data,sort_keys=True).encode()).hexdigest()[:20]


def parameters_for_split(split: str, family: str, index: int):
    """Finite parameter grids have no shared radii across splits (>=0.75 m gap).

    Radius is the withheld geometry parameter, not weather or random seed. Each
    group can be rendered in either mirror/traversal direction within its split.
    Requesting an out-of-range group fails instead of recycling geometry.
    """
    offsets={"train":0.,"validation":.75,"test":1.5,"development":2.25}
    if split not in offsets or not 0<=index<10:
        raise ValueError("split requires 0<=index<10 unique procedural groups per family")
    p=ProceduralParameters(family=family,radius_m=11.+3.*index+offsets[split],
        turn="left" if index%2==0 else "right",lane_width_m=(3.3,3.6,3.9)[index%3],
        grade=(0.,.025,-.025)[index%3],bank_rad=(0.,.04,-.04)[(index//3)%3],split=split)
    p.validate();return p


@dataclass(frozen=True)
class CurveSegment:
    length_m: float
    curvature_start: float
    curvature_end: float


def curve_segments(p: ProceduralParameters):
    p.validate();sign=1. if p.turn=="left" else -1.;k=sign/p.radius_m
    transition=p.radius_m*p.transition_fraction
    result=[CurveSegment(p.approach_m,0.,0.)]
    if p.family in ("hairpin","constant_radius"):
        total_angle=math.pi if p.family=="hairpin" else math.pi/2
        arc=p.radius_m*total_angle-transition
        result.extend([CurveSegment(transition,0.,k),CurveSegment(arc,k,k),CurveSegment(transition,k,0.)])
    elif p.family=="tightening":
        result.extend([CurveSegment(transition,0.,k*.25),CurveSegment(1.2*p.radius_m,k*.25,k),
                       CurveSegment(.3*p.radius_m,k,k),CurveSegment(transition,k,0.)])
    else:
        for turn_k in (k,-k):
            result.extend([CurveSegment(transition,0.,turn_k),CurveSegment(.5*p.radius_m,turn_k,turn_k),
                           CurveSegment(transition,turn_k,0.)])
    result.append(CurveSegment(p.exit_m,0.,0.))
    return tuple(result)


def _endpoint(x,y,yaw,segment):
    length=segment.length_m;k0=segment.curvature_start;slope=(segment.curvature_end-k0)/length
    theta=lambda s:yaw+k0*s+.5*slope*s*s
    dx=quad(lambda s:math.cos(theta(s)),0,length,epsabs=1e-10)[0]
    dy=quad(lambda s:math.sin(theta(s)),0,length,epsabs=1e-10)[0]
    return x+dx,y+dy,theta(length)


def sample_reference_line(p: ProceduralParameters, spacing_m=.25):
    """ASAM XYZ samples for generator validation only, not controller admission."""
    if spacing_m<=0: raise ValueError("positive sampling interval required")
    output=[];x=y=yaw=s=0.
    for segment in curve_segments(p):
        count=max(1,math.ceil(segment.length_m/spacing_m))
        for ds in np.linspace(0,segment.length_m,count,endpoint=False):
            partial=CurveSegment(float(max(ds,1e-12)),segment.curvature_start,
                segment.curvature_start+(segment.curvature_end-segment.curvature_start)*ds/segment.length_m)
            xx,yy,_=_endpoint(x,y,yaw,partial) if ds>0 else (x,y,yaw)
            output.append([s+ds,xx,yy,(s+ds)*p.grade])
        x,y,yaw=_endpoint(x,y,yaw,segment);s+=segment.length_m
    output.append([s,x,y,s*p.grade])
    return np.asarray(output)


def build_opendrive(p: ProceduralParameters):
    """Generate a marked two-way road, one driving lane each side and shoulders."""
    p.validate();segments=curve_segments(p);length=sum(seg.length_m for seg in segments)
    root=ET.Element("OpenDRIVE")
    ET.SubElement(root,"header",revMajor="1",revMinor="4",name=p.geometry_group,version="1.00",
                  date="2026-01-01",north="0",south="0",east="0",west="0",vendor="lane-centering-dataset-v1")
    road=ET.SubElement(root,"road",name=p.geometry_group,length=f"{length:.12g}",id="1",junction="-1",rule="RHT")
    ET.SubElement(road,"link")
    kind=ET.SubElement(road,"type",s="0",type="rural");ET.SubElement(kind,"speed",max="10",unit="m/s")
    plan=ET.SubElement(road,"planView");x=y=yaw=s=0.
    for segment in segments:
        element=ET.SubElement(plan,"geometry",s=f"{s:.14g}",x=f"{x:.14g}",y=f"{y:.14g}",
                              hdg=f"{yaw:.14g}",length=f"{segment.length_m:.14g}")
        a,b=segment.curvature_start,segment.curvature_end
        if abs(a)<1e-14 and abs(b)<1e-14: ET.SubElement(element,"line")
        elif abs(a-b)<1e-14: ET.SubElement(element,"arc",curvature=f"{a:.14g}")
        else: ET.SubElement(element,"spiral",curvStart=f"{a:.14g}",curvEnd=f"{b:.14g}")
        x,y,yaw=_endpoint(x,y,yaw,segment);s+=segment.length_m
    elevation=ET.SubElement(road,"elevationProfile")
    ET.SubElement(elevation,"elevation",s="0",a="0",b=str(p.grade),c="0",d="0")
    lateral=ET.SubElement(road,"lateralProfile")
    ET.SubElement(lateral,"superelevation",s="0",a=str(p.bank_rad),b="0",c="0",d="0")
    lanes=ET.SubElement(road,"lanes");section=ET.SubElement(lanes,"laneSection",s="0")
    def marking(lane,color="white"):
        ET.SubElement(lane,"roadMark",sOffset="0",type="solid",weight="standard",color=color,
                      material="standard",width="0.15",laneChange="none",height="0.003")
    for side,sign in (("left",1),("center",0),("right",-1)):
        container=ET.SubElement(section,side)
        if sign==0:
            center=ET.SubElement(container,"lane",id="0",type="none",level="false");marking(center,"yellow")
            continue
        lane=ET.SubElement(container,"lane",id=str(sign),type="driving",level="false")
        ET.SubElement(lane,"link");ET.SubElement(lane,"width",sOffset="0",a=str(p.lane_width_m),b="0",c="0",d="0")
        marking(lane)
        shoulder=ET.SubElement(container,"lane",id=str(2*sign),type="shoulder",level="false")
        ET.SubElement(shoulder,"link");ET.SubElement(shoulder,"width",sOffset="0",a=str(p.shoulder_width_m),b="0",c="0",d="0")
        ET.SubElement(shoulder,"roadMark",sOffset="0",type="none",color="white",width="0")
    ET.SubElement(road,"objects");ET.SubElement(road,"signals")
    ET.indent(root,space="  ")
    return '<?xml version="1.0" encoding="UTF-8"?>\n'+ET.tostring(root,encoding="unicode")+'\n'
