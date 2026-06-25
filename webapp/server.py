"""
Face Detection Lab · API Server
FastAPI backend serving self-trained YuNet models.
"""
import sys, time, io, base64
from pathlib import Path
import numpy as np
import torch, cv2
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "preprocessing"))
sys.path.insert(0, str(ROOT / "training"))
from process_all import FaceImagePreprocessor
from yunet_torch.model import YuNet

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL = ROOT / "training/outputs/model_full/weights/final_model.pth"

# ─── NMS ───
def _nms(dets, iou=0.45):
    if len(dets) <= 1: return dets
    boxes = np.array([d["bbox"] for d in dets]); scores = np.array([d["conf"] for d in dets])
    x1, y1, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]; x2, y2 = x1 + w, y1 + h
    areas = w * h; order = scores.argsort()[::-1]; keep = []
    while len(order):
        i = order[0]; keep.append(i)
        if len(order) == 1: break
        xx1 = np.maximum(x1[i], x1[order[1:]]); yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]]); yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        order = order[1:][inter / (areas[i] + areas[order[1:]] - inter + 1e-6) <= iou]
    return [dets[k] for k in keep]

# ─── Detector ───
class Detector:
    def __init__(self, path, device="cpu", conf=0.10):
        self.conf = conf; self.device = device
        self.model = YuNet(num_classes=1, use_kps=False)
        ck = torch.load(str(path), map_location=device, weights_only=False)
        self.model.load_state_dict(ck["model_state_dict"])
        self.model.to(device).eval()

    def detect(self, bgr):
        h, w = bgr.shape[:2]; T = 640
        scale = T / max(h, w)
        nh, nw = int(h * scale), int(w * scale)
        resized = cv2.resize(bgr, (nw, nh))
        padded = cv2.copyMakeBorder(resized, 0, T - nh, 0, T - nw, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        t = torch.from_numpy(padded).permute(2, 0, 1).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            cls_s, bbox_s, obj_s, _ = self.model(t)
        results = []
        for lvl, stride in enumerate([8, 16, 32]):
            _, _, fh, fw = cls_s[lvl].shape
            cl = cls_s[lvl][0].permute(1, 2, 0).reshape(-1).sigmoid()
            ob = obj_s[lvl][0].permute(1, 2, 0).reshape(-1).sigmoid()
            bb = bbox_s[lvl][0].permute(1, 2, 0).reshape(-1, 4)
            gy, gx = torch.meshgrid(torch.arange(fh, device=self.device),
                                    torch.arange(fw, device=self.device), indexing="ij")
            cx = gx.float() * stride; cy = gy.float() * stride
            pr = torch.stack([cx, cy, torch.full_like(cx, stride), torch.full_like(cy, stride)], dim=-1).reshape(-1, 4)
            xys = bb[:, :2] * pr[:, 2:] + pr[:, :2]
            whs = bb[:, 2:].exp() * pr[:, 2:]
            x1 = xys[:, 0] - whs[:, 0] / 2; y1 = xys[:, 1] - whs[:, 1] / 2
            x2 = xys[:, 0] + whs[:, 0] / 2; y2 = xys[:, 1] + whs[:, 1] / 2
            sc = cl * ob; ok = sc >= self.conf
            if ok.any():
                bx = torch.stack([x1, y1, x2, y2], dim=-1)[ok]; sx = sc[ok]
                bx[:, [0, 2]] /= scale; bx[:, [1, 3]] /= scale
                bx[:, [0, 2]] = bx[:, [0, 2]].clamp(0, w); bx[:, [1, 3]] = bx[:, [1, 3]].clamp(0, h)
                for b, s in zip(bx.cpu().numpy(), sx.cpu().numpy()):
                    bw, bh = b[2] - b[0], b[3] - b[1]
                    if bw > 2 and bh > 2:
                        results.append({"bbox": [int(b[0]), int(b[1]), int(bw), int(bh)], "conf": round(float(s), 4)})
        return _nms(results)


# ─── Init ───
print(f"Loading model on {DEVICE}...")
preprocessor = FaceImagePreprocessor()
detector = Detector(MODEL, DEVICE)
print("Model ready.")

app = FastAPI(title="Face Detection Lab")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ─── Matching ───
def match_detections(dets_a, dets_b, iou_thr=0.3):
    """Pair detections from two sets by IoU, returning comparison rows."""
    def _iou(a, b):
        x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
        x2 = min(a[0]+a[2], b[0]+b[2]); y2 = min(a[1]+a[3], b[1]+b[3])
        inter = max(0, x2-x1) * max(0, y2-y1)
        area_a = a[2]*a[3]; area_b = b[2]*b[3]
        return inter / (area_a+area_b - inter + 1e-6)

    pairs = []; matched_b = set()
    for da in (dets_a or []):
        best_iou, best_j = 0, -1
        for j, db in enumerate(dets_b or []):
            if j in matched_b: continue
            v = _iou(da["bbox"], db["bbox"])
            if v > best_iou: best_iou, best_j = v, j
        if best_j >= 0 and best_iou >= iou_thr:
            matched_b.add(best_j)
            db = dets_b[best_j]
            diff = round(db["conf"] - da["conf"], 4)
            pairs.append({"bbox": da["bbox"], "orig_conf": da["conf"],
                "proc_conf": db["conf"], "diff": diff,
                "status": "up" if diff > 0.001 else "down" if diff < -0.001 else "same"})
        else:
            pairs.append({"bbox": da["bbox"], "orig_conf": da["conf"],
                "proc_conf": None, "diff": None, "status": "lost"})
    for j, db in enumerate(dets_b or []):
        if j not in matched_b:
            pairs.append({"bbox": db["bbox"], "orig_conf": None,
                "proc_conf": db["conf"], "diff": None, "status": "new"})
    return pairs

# ─── API ───
@app.post("/api/detect")
async def detect(
    image: UploadFile = File(...),
    preprocess: bool = Form(True),
    threshold: float = Form(0.10),
):
    t0 = time.perf_counter()
    contents = await image.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return JSONResponse({"error": "Cannot read image"}, status_code=400)

    detector.conf = threshold

    if preprocess:
        proc = preprocessor.process_image(img.copy())
        t1 = time.perf_counter()
        dets_orig = detector.detect(img)
        ms_orig = (time.perf_counter() - t1) * 1000
        t1 = time.perf_counter()
        dets_proc = detector.detect(proc)
        ms_proc = (time.perf_counter() - t1) * 1000
    else:
        proc = None
        t1 = time.perf_counter()
        dets_orig = detector.detect(img)
        ms_orig = (time.perf_counter() - t1) * 1000
        dets_proc = dets_orig
        ms_proc = 0

    total = (time.perf_counter() - t0) * 1000
    return {
        "preprocess": preprocess,
        "threshold": threshold,
        "detections_orig": dets_orig,
        "detections_proc": dets_proc,
        "conf_comparison": match_detections(dets_orig, dets_proc) if preprocess else None,
        "proc_image": "data:image/jpeg;base64," + base64.b64encode(cv2.imencode(".jpg", proc, [int(cv2.IMWRITE_JPEG_QUALITY), 90])[1]).decode() if preprocess and proc is not None else None,
        "time_orig_ms": round(ms_orig, 1),
        "time_proc_ms": round(ms_proc, 1),
        "time_total_ms": round(total, 1),
    }


# ─── Static files ───
app.mount("/", StaticFiles(directory=str(Path(__file__).parent / "static"), html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=7860)
