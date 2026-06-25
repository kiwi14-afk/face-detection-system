"""Quick evaluation of the new model on WIDER FACE validation set."""
import sys, time, json
from pathlib import Path
import numpy as np, cv2, torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "training"))
from yunet_torch.model import YuNet

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL = ROOT / "training/outputs/model_full/weights/final_model.pth"
VAL_DIR = ROOT / "preprocessing/3226_all/WIDER_val/images"
ANN_FILE = ROOT / "training/data/wider_face_split/wider_face_val_bbx_gt.txt"

# ---- detector ----
class Eval:
    def __init__(self, path, device="cpu", conf=0.05):
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
                        results.append({"bbox": [int(b[0]), int(b[1]), int(bw), int(bh)], "conf": float(s)})
        return self._nms(results)

    def _nms(self, dets, iou=0.45):
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


# ---- parse annotations ----
def parse_ann(path):
    with open(path) as f: lines = f.readlines()
    anns = {}; i = 0
    while i < len(lines):
        img = lines[i].strip()
        if not img: i += 1; continue
        i += 1; n = int(lines[i].strip()); i += 1
        gts = []
        for _ in range(n):
            p = lines[i].strip().split()
            x, y, w, h = float(p[0]), float(p[1]), float(p[2]), float(p[3])
            if w > 0 and h > 0: gts.append([x, y, w, h])
            i += 1
        anns[img] = gts
    return anns


def iou(a, b):
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[0]+a[2], b[0]+b[2]); y2 = min(a[1]+a[3], b[1]+b[3])
    inter = max(0, x2-x1) * max(0, y2-y1)
    area_a = a[2]*a[3]; area_b = b[2]*b[3]
    return inter / (area_a+area_b - inter + 1e-6)


def match(dets, gts, iou_thr=0.5):
    dets_sorted = sorted(enumerate(dets), key=lambda x: x[1]["conf"], reverse=True)
    matched = set(); tp = np.zeros(len(dets)); fp = np.zeros(len(dets))
    for di, det in dets_sorted:
        best, best_gt = 0, -1
        for gi, gt in enumerate(gts):
            if gi in matched: continue
            v = iou(det["bbox"], gt)
            if v > best: best, best_gt = v, gi
        if best >= iou_thr: tp[di] = 1; matched.add(best_gt)
        else: fp[di] = 1
    fn = len(gts) - len(matched)
    return tp, fp, fn


if __name__ == "__main__":
    print("Loading annotations...")
    anns = parse_ann(ANN_FILE)
    total_faces = sum(len(v) for v in anns.values())
    print(f"  {len(anns)} images, {total_faces} faces")

    thresholds = [0.03, 0.04, 0.05, 0.06, 0.08, 0.10]
    print(f"\n{'Threshold':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print("-"*45)

    for thr in thresholds:
        ev = Eval(MODEL, DEVICE, conf=thr)
        total_tp = total_fp = total_fn = 0
        files = sorted(list(VAL_DIR.rglob("*.jpg")))
        for fp in tqdm(files, desc=f"thr={thr:.2f}", leave=False):
            rel = str(fp.relative_to(VAL_DIR)).replace("\\", "/")
            gts = anns.get(rel, [])
            img = cv2.imdecode(np.fromfile(str(fp), dtype=np.uint8), cv2.IMREAD_COLOR)
            if img is None: continue
            dets = ev.detect(img)
            tp, fp, fn = match(dets, gts)
            total_tp += int(tp.sum()); total_fp += int(fp.sum()); total_fn += int(fn)

        p = total_tp/(total_tp+total_fp) if (total_tp+total_fp)>0 else 0
        r = total_tp/(total_tp+total_fn) if (total_tp+total_fn)>0 else 0
        f1 = 2*p*r/(p+r) if (p+r)>0 else 0
        print(f"{thr:10.2f} {p:10.4f} {r:10.4f} {f1:10.4f}")

    print(f"\nTotal faces: {total_faces}")
