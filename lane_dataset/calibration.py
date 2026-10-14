"""Physics-only steering identification maneuvers, separate from corpus episodes."""
from __future__ import annotations

import dataclasses
from pathlib import Path
import time

import numpy as np

from .config import DEFAULT_ROOT, write_json
from .simulator import batch_control, configure_world, connect, spawn_vehicle, state, world_owner


def capture_calibration(root, port=2500, ticks=700):
    root = Path(root)
    out = root / "reports" / f"calibration-{port}-{time.time_ns()}"
    out.mkdir(parents=True)
    client = connect(port)
    rows = []
    ego = None
    with world_owner(port, root):
        world, settings = configure_world(client,"Town04",render=False)
        waypoint = world.get_map().get_waypoint_xodr(38,4,200.)
        if waypoint is None:
            raise RuntimeError("calibration lane missing")
        try:
            ego = spawn_vehicle(world,waypoint)
            physics = ego.get_physics_control()
            wheelbase = abs(physics.wheels[0].position.x-physics.wheels[2].position.x)/100
            metadata = dict(port=port,settings=settings,vehicle=ego.type_id,wheelbase_m=wheelbase,
                            wheel_steer_degrees=[w.max_steer_angle for w in physics.wheels],
                            status="unqualified_calibration_capture",ticks=ticks)
            write_json(out / "manifest.json",metadata)
            pulse = np.array([0.,.02,0.,-.02,0.,.035,0.,-.035,0.,.015,-.015,0.])
            integral = 0.
            for tick in range(ticks):
                speed = np.linalg.norm([ego.get_velocity().x,ego.get_velocity().y])
                target = 5. if tick >= 30 else 0.
                error = target-speed
                integral = float(np.clip(integral+error*.05,-2,2))
                demand = .28*error + .08*integral + (.1 if target else 0)
                steer = float(pulse[min(len(pulse)-1,max(0,(tick-100)//50))]) if tick >= 100 else 0.
                batch_control(client,ego,throttle=max(0,demand),brake=max(0,-demand),steer=steer)
                frame = world.tick()
                snapshot = world.get_snapshot()
                if frame != snapshot.frame:
                    raise RuntimeError("world snapshot mismatch")
                row = state(ego,snapshot)
                row["command_steer"] = steer
                rows.append(row)
            arrays = {key:np.asarray([row[key] for row in rows]) for key in rows[0]}
            np.savez_compressed(out / "samples.npz",**arrays)
            metadata.update(status="captured_not_fitted",sample_count=len(rows),speed_min=float(arrays["speed"][100:].min()),speed_max=float(arrays["speed"][100:].max()))
            write_json(out / "manifest.json",metadata)
        finally:
            if ego is not None:
                ego.destroy()
    return out


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--root",default=str(DEFAULT_ROOT))
    parser.add_argument("--port",type=int,default=2500)
    args = parser.parse_args()
    print(capture_calibration(args.root,args.port))
