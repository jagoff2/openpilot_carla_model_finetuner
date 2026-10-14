"""Machine-local paths. Environment overrides config.local.json, then defaults."""
import json
import os
from pathlib import Path

PROJECT=Path(__file__).resolve().parents[1]


def local_settings():
    path=PROJECT/'config.local.json'
    return json.loads(path.read_text(encoding='utf-8-sig')) if path.exists() else {}


def path_setting(key,environment,default):
    raw=os.environ.get(environment) or local_settings().get(key) or str(default)
    path=Path(os.path.expandvars(raw)).expanduser()
    return (PROJECT/path).resolve() if not path.is_absolute() else path.resolve()
