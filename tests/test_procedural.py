import dataclasses
import math
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from lane_dataset.procedural import (ProceduralParameters, build_opendrive, curve_segments,
                                     parameters_for_split, sample_reference_line)


@pytest.mark.parametrize("family",["constant_radius","tightening","s_bend","hairpin"])
def test_continuous_marked_opendrive_geometry(family):
    p=parameters_for_split("train",family,2)
    text=build_opendrive(p);xml=ET.fromstring(text)
    assert len(xml.findall(".//lane[@type='driving']"))==2
    assert len(xml.findall(".//roadMark[@type='solid']"))==3
    segments=curve_segments(p)
    for first,second in zip(segments,segments[1:]):
        assert first.curvature_end==second.curvature_start
    assert all(segment.length_m>0 for segment in segments)
    points=sample_reference_line(p,1.)
    assert np.max(np.abs(np.linalg.norm(np.diff(points[:,1:3],axis=0),axis=1)-np.diff(points[:,0])))<.005
    turn=sum(s.length_m*(s.curvature_start+s.curvature_end)/2 for s in segments)
    if family=="hairpin": assert turn==pytest.approx(math.pi)
    if family=="s_bend": assert turn==pytest.approx(0.)


def test_split_parameters_and_mirror_group_identity():
    parameters={split:[parameters_for_split(split,"hairpin",i) for i in range(10)]
                for split in ("train","validation","test")}
    for first,second in (("train","validation"),("train","test"),("validation","test")):
        assert not {p.radius_m for p in parameters[first]} & {p.radius_m for p in parameters[second]}
        assert not {p.geometry_group for p in parameters[first]} & {p.geometry_group for p in parameters[second]}
    original=parameters["train"][0]
    assert original.geometry_group==dataclasses.replace(original,turn="right").geometry_group
    assert build_opendrive(original)==build_opendrive(original)


def test_unfeasible_radius_rejected():
    with pytest.raises(ValueError): build_opendrive(ProceduralParameters("hairpin",4.))
