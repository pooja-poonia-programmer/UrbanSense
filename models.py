from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import urllib.request
import shutil
import config

@dataclass
class ModelStatus:
    name: str
    available: bool
    message: str

@dataclass
class Bundle:
    yolo: object=None
    pothole: object=None
    sign: object=None
    plate: object=None
    ocr: object=None
    statuses: dict=field(default_factory=dict)

def _download_url(url, path):
    path.parent.mkdir(parents=True,exist_ok=True)
    urllib.request.urlretrieve(url,str(path))

def _hf(repo, filename, dest):
    from huggingface_hub import hf_hub_download
    got=hf_hub_download(repo_id=repo,filename=filename,local_dir=str(dest.parent))
    if Path(got).resolve()!=dest.resolve():
        Path(got).replace(dest)

def _load(path, loader):
    try:
        return loader(), None
    except Exception as e:
        return None, str(e)

def _local_or_project(name, preferred):
    """Find a model in the final repo layout or the user's existing project root."""
    candidates = [Path(preferred), Path(config.BASE_DIR) / name]
    for c in candidates:
        if c.exists() and c.stat().st_size > 1_000_000:
            return c
    return Path(preferred)

def load_bundle():
    from ultralytics import YOLO
    b=Bundle()

    # Primary YOLO: use an existing root/model copy, otherwise let Ultralytics download it.
    try:
        yolo_path = _local_or_project("yolo26n.pt", config.YOLO_MODEL)
        if yolo_path.exists():
            b.yolo=YOLO(str(yolo_path))
        else:
            b.yolo=YOLO("yolo26n.pt")
            ckpt=getattr(b.yolo,"ckpt_path",None)
            if ckpt and Path(ckpt).exists():
                try:
                    shutil.copy2(ckpt, config.YOLO_MODEL)
                except Exception:
                    pass
        b.statuses["yolo"]=ModelStatus("YOLO object detector",True,"Ready")
    except Exception as e:
        b.statuses["yolo"]=ModelStatus("YOLO object detector",False,str(e))

    # Optional specialists download automatically when missing.
    try:
        pothole_path = _local_or_project("pothole_model.pt", config.POTHOLE_MODEL)
        if not pothole_path.exists():
            _download_url(config.POTHOLE_URL,config.POTHOLE_MODEL)
            pothole_path = config.POTHOLE_MODEL
        b.pothole=YOLO(str(pothole_path))
        b.statuses["pothole"]=ModelStatus("Pothole model",True,"Ready")
    except Exception as e:
        b.statuses["pothole"]=ModelStatus("Pothole model",False,str(e))

    try:
        sign_path = _local_or_project("traffic_sign_model.pt", config.SIGN_MODEL)
        if not sign_path.exists():
            _hf(config.SIGN_HF_REPO,"best.pt",config.SIGN_MODEL)
            sign_path = config.SIGN_MODEL
        b.sign=YOLO(str(sign_path))
        b.statuses["sign"]=ModelStatus("Traffic sign model",True,"Ready")
    except Exception as e:
        b.statuses["sign"]=ModelStatus("Traffic sign model",False,str(e))

    try:
        plate_path = _local_or_project("license_plate_model.pt", config.PLATE_MODEL)
        if not plate_path.exists():
            _hf(config.PLATE_HF_REPO,"best.pt",config.PLATE_MODEL)
            plate_path = config.PLATE_MODEL
        b.plate=YOLO(str(plate_path))
        b.statuses["plate"]=ModelStatus("License plate detector",True,"Ready")
    except Exception as e:
        b.statuses["plate"]=ModelStatus("License plate detector",False,str(e))

    try:
        import easyocr
        b.ocr=easyocr.Reader(["en"],gpu=False,verbose=False)
        b.statuses["ocr"]=ModelStatus("EasyOCR",True,"Ready")
    except Exception as e:
        b.statuses["ocr"]=ModelStatus("EasyOCR",False,str(e))
    return b
