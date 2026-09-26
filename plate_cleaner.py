import os
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

# Pre-trained models (both cached locally)
MODEL_V11_PATH = r"C:\Users\Usuario\.cache\huggingface\hub\models--morsetechlab--yolov11-license-plate-detection\snapshots\251a30d7daedca065f56e04b0af04052c907c68f\license-plate-finetune-v1n.pt"
MODEL_V8_PATH = r"C:\Users\Usuario\.cache\huggingface\hub\models--joker5914--yolov8n-license-plate\snapshots\8286762929bd4b111a19186f2a05e0a5940b6088\best.pt"

class PlateCleaner:
    def __init__(self, m1_path=MODEL_V11_PATH, m2_path=MODEL_V8_PATH):
        self.m1 = YOLO(m1_path)
        self.m2 = YOLO(m2_path)

    def detect_plates(self, img_bgr):
        h, w = img_bgr.shape[:2]
        cands = []

        # 1. Dual lightweight YOLO detection
        for model in [self.m1, self.m2]:
            res = model.predict(img_bgr, conf=0.40, verbose=False)[0]
            for b in res.boxes:
                x1, y1, x2, y2 = [int(v) for v in b.xyxy[0].tolist()]
                conf = float(b.conf[0])
                bw, bh = x2 - x1, y2 - y1
                if bw <= 0 or bh <= 0:
                    continue

                # CRITICAL: Exclude top header area (e.g., 'B3 ARQUIVOS STL & MINIATURAS' logo)
                # The user specified never to touch header text in the top of the image
                if y1 < 0.22 * h:
                    continue
                # Discard watermarks at top corners
                if (x1 < 0.18 * w or x2 > 0.82 * w) and y1 < 0.25 * h:
                    continue
                # Discard huge bounding boxes (tailgates, hoods)
                if bw > 0.5 * w or bh > 0.3 * h:
                    continue

                # Discard taillights and red reflectors
                crop_cand = img_bgr[y1:y2, x1:x2]
                if crop_cand.shape[0] > 0 and crop_cand.shape[1] > 0:
                    hsv_c = cv2.cvtColor(crop_cand, cv2.COLOR_BGR2HSV)
                    mean_sat = np.mean(hsv_c[:, :, 1])
                    red_ratio = np.mean((hsv_c[:, :, 0] < 15) | (hsv_c[:, :, 0] > 165))
                    if red_ratio > 0.35 and mean_sat > 40:
                        continue
                    asp = bw / max(1, bh)
                    if asp < 1.30:
                        blue_c = cv2.inRange(hsv_c, (100, 70, 40), (135, 255, 255))
                        if np.sum(blue_c > 0) < 15:
                            continue

                cands.append([x1, y1, x2, y2, conf])

        # 2. Contour + EU blue strip fallback (strictly for EU plates)
        if not cands:
            hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
            blue_mask = cv2.inRange(hsv, (100, 70, 40), (135, 255, 255))
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            cnts, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            for c in cnts:
                rect = cv2.minAreaRect(c)
                (cx, cy), (rw, rh), angle = rect
                area = rw * rh
                long_side = max(rw, rh)
                short_side = min(rw, rh)
                asp = long_side / max(1, short_side)
                # Skip header zone
                if cy < 0.22 * h or (cx < 0.20 * w and cy < 0.25 * h):
                    continue
                if 300 <= area <= 4000 and 2.0 <= asp <= 5.5 and short_side >= 8:
                    x, y, bw, bh = cv2.boundingRect(c)
                    sub_gray = gray[y:y+bh, x:x+bw]
                    if np.mean(sub_gray) > 100:
                        sub_blue = blue_mask[y:y+bh, x:x+bw]
                        # EU blue flag must be strictly on the left edge of the plate
                        if np.sum(sub_blue[:, :int(bw * 0.28)] > 0) >= 15:
                            cands.append([x, y, x + bw, y + bh, 0.70])

        if not cands:
            return []

        # NMS to eliminate overlapping boxes
        boxes = [[c[0], c[1], c[2]-c[0], c[3]-c[1]] for c in cands]
        scores = [c[4] for c in cands]
        idxs = cv2.dnn.NMSBoxes(boxes, scores, 0.25, 0.40)
        best_candidates = [cands[i] for i in idxs]

        # Prioritize bumper plate over grille logo if multiple exist
        if len(best_candidates) > 1:
            best_candidates = [max(best_candidates, key=lambda c: c[4])]

        return [c[:4] for c in best_candidates]

    def clean_plate_crop(self, crop):
        h, w = crop.shape[:2]
        if h < 8 or w < 15:
            return crop, np.zeros((h, w), dtype=np.uint8)

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        # Check internal contrast (plates always have distinct contrast between background and text)
        if int(np.max(gray)) - int(np.min(gray)) < 40:
            return crop, np.zeros((h, w), dtype=np.uint8)

        bg_studio = (crop[:, :, 0] >= 252) & (crop[:, :, 1] >= 252) & (crop[:, :, 2] >= 252)
        non_studio = gray[~bg_studio]
        if len(non_studio) == 0:
            return crop, np.zeros((h, w), dtype=np.uint8)

        p80 = np.percentile(non_studio, 80)
        is_dark = p80 < 135

        if is_dark:
            # 1. Dark plate processing (e.g. Squir, Tesla, black custom plates)
            x_m = max(2, int(w * 0.04))
            y_m = max(2, int(h * 0.08))
            inner = gray[y_m:h-y_m, x_m:w-x_m]
            if inner.shape[0] < 4 or inner.shape[1] < 6:
                return crop, np.zeros((h, w), dtype=np.uint8)
            bg_val = np.median(inner)
            if int(np.max(inner)) - int(bg_val) < 15:
                return crop, np.zeros((h, w), dtype=np.uint8)
            mask_inner = (inner > (bg_val + 6)).astype(np.uint8) * 255
            if np.sum(mask_inner > 0) < 10:
                return crop, np.zeros((h, w), dtype=np.uint8)

            mask = np.zeros((h, w), dtype=np.uint8)
            mask[y_m:h-y_m, x_m:w-x_m] = mask_inner
            mask_dil = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=2)
            clean = cv2.inpaint(crop, mask_dil, 3, cv2.INPAINT_TELEA)
            return clean, mask_dil
        else:
            # 2. Light / EU / Mercosul plate processing (with rotated deskew)
            edges = cv2.Canny(gray, 50, 150)
            cnts, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            best_rect = None
            best_score = -1
            for c in cnts:
                rect = cv2.minAreaRect(c)
                (cx, cy), (rw, rh), angle = rect
                area = rw * rh
                asp = max(rw, rh) / max(1, min(rw, rh))
                if area > 0.15 * (w * h) and 1.8 <= asp <= 6.0:
                    if area > best_score:
                        best_score = area
                        best_rect = rect

            if best_rect is not None:
                (cx, cy), (rw, rh), angle = best_rect
                if rw < rh:
                    rw, rh = rh, rw
                    angle += 90
                tw, th = max(15, int(rw)), max(8, int(rh))
                src_pts = cv2.boxPoints(((cx, cy), (rw, rh), angle)).astype(np.float32)
                s = src_pts.sum(axis=1)
                diff = np.diff(src_pts, axis=1)
                ordered_src = np.array([
                    src_pts[np.argmin(s)],
                    src_pts[np.argmin(diff)],
                    src_pts[np.argmax(s)],
                    src_pts[np.argmax(diff)]
                ], dtype=np.float32)
                dst_pts = np.array([[0, 0], [tw, 0], [tw, th], [0, th]], dtype=np.float32)
                M = cv2.getPerspectiveTransform(ordered_src, dst_pts)
                M_inv = cv2.getPerspectiveTransform(dst_pts, ordered_src)
                warped = cv2.warpPerspective(crop, M, (tw, th))
                gw = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

                bright_rows = [r for r in range(th) if np.max(gw[r, int(tw*0.2):int(tw*0.8)]) > 200]
                if not bright_rows:
                    bright_rows = list(range(int(th * 0.1), int(th * 0.9)))
                y_start = min(bright_rows) + 2
                y_end = max(bright_rows) - 2
                if y_end <= y_start:
                    y_start, y_end = 1, th - 1
                plate_h = y_end - y_start

                hsv_w = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)
                blue_w = cv2.inRange(hsv_w, (100, 70, 40), (135, 255, 255))
                blue_col_counts = np.sum(blue_w[y_start:y_end, :] > 0, axis=0)
                blue_cols = np.where(blue_col_counts >= 0.30 * plate_h)[0]
                if len(blue_cols) > 0 and blue_cols.max() < 0.30 * tw:
                    x_start = blue_cols.max() + 2
                else:
                    bright_cols = [c for c in range(tw) if np.max(gw[y_start:y_end, c]) > 200]
                    x_start = (min(bright_cols) + 2) if bright_cols else int(tw * 0.05)

                bright_cols = [c for c in range(tw) if np.max(gw[y_start:y_end, c]) > 200]
                x_end = (max(bright_cols) - 2) if bright_cols else int(tw * 0.95)
                if x_end <= x_start:
                    x_start, x_end = int(tw * 0.05), int(tw * 0.95)

                mask_w = np.zeros((th, tw), dtype=np.uint8)
                inner_w = gw[y_start:y_end, x_start:x_end]
                mask_w[y_start:y_end, x_start:x_end] = (inner_w < 165).astype(np.uint8) * 255
                if np.sum(mask_w > 0) < 10:
                    return crop, np.zeros((h, w), dtype=np.uint8)

                mask_w_dil = cv2.dilate(mask_w, np.ones((3, 3), np.uint8), iterations=1)
                clean_w = cv2.inpaint(warped, mask_w_dil, 2, cv2.INPAINT_TELEA)

                unwarped_clean = cv2.warpPerspective(clean_w, M_inv, (w, h))
                unwarped_mask = cv2.warpPerspective(mask_w_dil, M_inv, (w, h))
                res = crop.copy()
                res[unwarped_mask > 0] = unwarped_clean[unwarped_mask > 0]
                return res, unwarped_mask
            else:
                # Fallback axis-aligned processing (requires white plate confirmation)
                bg_plate = np.percentile(gray, 85)
                if bg_plate < 170:
                    return crop, np.zeros((h, w), dtype=np.uint8)
                y_start = max(2, int(h * 0.10))
                y_end = h - max(2, int(h * 0.10))
                x_start = max(2, int(w * 0.05))
                x_end = w - max(2, int(w * 0.05))
                mask = np.zeros((h, w), dtype=np.uint8)
                mask[y_start:y_end, x_start:x_end] = (gray[y_start:y_end, x_start:x_end] < (bg_plate - 30)).astype(np.uint8) * 255
                if np.sum(mask > 0) < 10:
                    return crop, np.zeros((h, w), dtype=np.uint8)
                mask_dil = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)
                clean = cv2.inpaint(crop, mask_dil, 2, cv2.INPAINT_TELEA)
                return clean, mask_dil

    def process_image(self, img_path):
        im_pil = Image.open(img_path).convert("RGB")
        img_rgb = np.array(im_pil)
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        boxes = self.detect_plates(img_bgr)
        result_bgr = img_bgr.copy()
        mask_vis = np.zeros_like(img_bgr)

        for box in boxes:
            x1, y1, x2, y2 = box
            crop = result_bgr[y1:y2, x1:x2]
            clean_crop, crop_mask = self.clean_plate_crop(crop)
            result_bgr[y1:y2, x1:x2] = clean_crop

            mask_vis[y1:y2, x1:x2][crop_mask > 0] = [0, 0, 255]
            cv2.rectangle(mask_vis, (x1, y1), (x2, y2), (0, 255, 0), 2)

        return img_bgr, mask_vis, result_bgr, boxes
