from __future__ import annotations

import copy
import os
import traceback
import urllib.request
from contextlib import contextmanager

import cv2
import gradio as gr
import numpy as np
import torch
from PIL import Image

from modules import scripts, shared
from modules.processing import (
    StableDiffusionProcessingImg2Img,
    process_images,
)


AFR_VERSION = "2.0.0"

AFR_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AFR_MODEL_DIR = os.path.join(AFR_DIR, "models")
FACE_LANDMARKER_MODEL_URLS = [
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float32/1/face_landmarker.task",
]
FACE_LANDMARKER_MODEL_PATH = os.path.join(AFR_MODEL_DIR, "face_landmarker.task")

YOLO_FACE_MODEL_URLS = [
    "https://github.com/derronqi/yolov8-face/releases/download/v0.0.0/yolov8n-face.pt",
    "https://huggingface.co/Bingsu/yolov8n-face/resolve/main/yolov8n-face.pt",
]
YOLO_FACE_MODEL_PATH = os.path.join(AFR_MODEL_DIR, "yolov8n-face.pt")

DETECTION_MODES = ["mediapipe", "yolo", "haar"]

LANG_EN = {
    "title": "Advanced Face Refiner",
    "main_accordion": "Advanced Face Refiner (AFR)",
    "enable": "Enable",
    "enable_afr": "Enable AFR",
    "detection_model": "Detection Model",
    "detection_model_desc": "mediapipe = best for real faces, yolo = best for anime/cartoon, haar = fallback",
    "pass1": "Pass 1 - Structure Correction",
    "pass1_denoising": "Pass 1 Denoising Strength",
    "pass1_mask_dilation": "Pass 1 Mask Dilation",
    "pass1_steps": "Pass 1 Steps (0 = use parent)",
    "pass1_cfg": "Pass 1 CFG Scale (0 = use parent)",
    "pass1_sampler": "Pass 1 Sampler (Use parent)",
    "pass1_checkpoint": "Pass 1 Checkpoint (Use parent)",
    "pass2": "Pass 2 - Detail Refinement",
    "pass2_denoising": "Pass 2 Denoising Strength",
    "pass2_mask_dilation": "Pass 2 Mask Dilation",
    "pass2_steps": "Pass 2 Steps (0 = use parent)",
    "pass2_cfg": "Pass 2 CFG Scale (0 = use parent)",
    "pass2_sampler": "Pass 2 Sampler (Use parent)",
    "pass2_checkpoint": "Pass 2 Checkpoint (Use parent)",
    "face_prompts": "Face Prompts",
    "face_prompt": "Face Prompt",
    "face_prompt_placeholder": "detailed face, beautiful eyes, sharp focus, high quality",
    "face_negative_prompt": "Face Negative Prompt",
    "face_negative_prompt_placeholder": "deformed, ugly, bad anatomy, blurry",
    "detection_settings": "Detection Settings",
    "min_detection_confidence": "Min Detection Confidence",
    "max_faces": "Max Faces to Process",
    "inpaint_mask_blur": "Inpaint Mask Blur",
    "inpaint_full_res": "Inpaint Full Resolution",
    "inpaint_full_res_padding": "Inpaint Full Res Padding",
    "no_faces": "No faces detected, skipping.",
    "detected_faces": "Detected {count} face(s), processing...",
    "processing_face": "Processing face {idx}/{total}",
    "pass1_mask_failed": "Pass 1 mask generation failed for face {idx}: {err}",
    "pass1_mask_empty": "Pass 1 mask is empty for face {idx}, skipping.",
    "pass1_complete": "Pass 1 complete for face {idx}",
    "pass1_inpaint_failed": "Pass 1 inpaint failed for face {idx}: {err}",
    "pass2_mask_failed": "Pass 2 mask generation failed for face {idx}: {err}",
    "pass2_mask_empty": "Pass 2 mask is empty for face {idx}, skipping detail pass.",
    "pass2_complete": "Pass 2 complete for face {idx}",
    "pass2_inpaint_failed": "Pass 2 inpaint failed for face {idx}: {err}",
    "all_complete": "All faces processed successfully.",
    "facemesh_failed": "FaceMesh detection failed: {err}",
    "postprocess_error": "Error in postprocess_image: {err}",
    "loaded": "Advanced Face Refiner v{ver} loaded.",
    "use_parent": "Use parent",
    "switching_model": "Switching model to: {model}",
    "model_switch_failed": "Failed to switch model: {err}",
    "yolo_download": "Downloading YOLO face model...",
    "yolo_failed": "YOLO detection failed: {err}",
    "yolo_detected": "YOLO detected {count} face(s)",
}

LANG_ZH = {
    "title": "高级面部修复器",
    "main_accordion": "高级面部修复器 (AFR)",
    "enable": "启用",
    "enable_afr": "启用 AFR",
    "detection_model": "检测模型",
    "detection_model_desc": "mediapipe = 适合真人, yolo = 适合动漫/卡通, haar = 备用",
    "pass1": "第1阶段 - 结构修正",
    "pass1_denoising": "第1阶段重绘强度",
    "pass1_mask_dilation": "第1阶段掩码膨胀",
    "pass1_steps": "第1阶段步数 (0=使用父设置)",
    "pass1_cfg": "第1阶段CFG (0=使用父设置)",
    "pass1_sampler": "第1阶段采样器 (使用父设置)",
    "pass1_checkpoint": "第1阶段大模型 (使用父设置)",
    "pass2": "第2阶段 - 细节精修",
    "pass2_denoising": "第2阶段重绘强度",
    "pass2_mask_dilation": "第2阶段掩码膨胀",
    "pass2_steps": "第2阶段步数 (0=使用父设置)",
    "pass2_cfg": "第2阶段CFG (0=使用父设置)",
    "pass2_sampler": "第2阶段采样器 (使用父设置)",
    "pass2_checkpoint": "第2阶段大模型 (使用父设置)",
    "face_prompts": "面部提示词",
    "face_prompt": "面部正面提示词",
    "face_prompt_placeholder": "精致面部,美丽眼睛,锐利焦点,高质量",
    "face_negative_prompt": "面部负面提示词",
    "face_negative_prompt_placeholder": "畸形,丑陋,结构崩坏,模糊",
    "detection_settings": "检测设置",
    "min_detection_confidence": "最低检测置信度",
    "max_faces": "最多处理人脸数",
    "inpaint_mask_blur": "重绘掩码模糊",
    "inpaint_full_res": "全分辨率重绘",
    "inpaint_full_res_padding": "全分辨率重绘边距",
    "no_faces": "未检测到人脸，跳过。",
    "detected_faces": "检测到 {count} 张人脸，处理中...",
    "processing_face": "正在处理第 {idx}/{total} 张人脸",
    "pass1_mask_failed": "第 {idx} 张人脸第1阶段掩码生成失败: {err}",
    "pass1_mask_empty": "第 {idx} 张人脸第1阶段掩码为空，跳过。",
    "pass1_complete": "第 {idx} 张人脸第1阶段完成",
    "pass1_inpaint_failed": "第 {idx} 张人脸第1阶段重绘失败: {err}",
    "pass2_mask_failed": "第 {idx} 张人脸第2阶段掩码生成失败: {err}",
    "pass2_mask_empty": "第 {idx} 张人脸第2阶段掩码为空，跳过细节阶段。",
    "pass2_complete": "第 {idx} 张人脸第2阶段完成",
    "pass2_inpaint_failed": "第 {idx} 张人脸第2阶段重绘失败: {err}",
    "all_complete": "所有人脸处理完毕。",
    "facemesh_failed": "FaceMesh 检测失败: {err}",
    "postprocess_error": "后处理出错: {err}",
    "loaded": "高级面部修复器 v{ver} 已加载。",
    "use_parent": "使用父设置",
    "switching_model": "切换模型至: {model}",
    "model_switch_failed": "切换模型失败: {err}",
    "yolo_download": "正在下载 YOLO 人脸模型...",
    "yolo_failed": "YOLO 检测失败: {err}",
    "yolo_detected": "YOLO 检测到 {count} 张人脸",
}

LANG_JA = {
    "title": "高度な顔修正器",
    "main_accordion": "高度な顔修正器 (AFR)",
    "enable": "有効化",
    "enable_afr": "AFRを有効化",
    "detection_model": "検出モデル",
    "detection_model_desc": "mediapipe = 実写向け, yolo = アニメ/カートゥーン向け, haar = フォールバック",
    "pass1": "第1段階 - 構造修正",
    "pass1_denoising": "第1段階のノイズ除去強度",
    "pass1_mask_dilation": "第1段階のマスク膨張",
    "pass1_steps": "第1段階のステップ数 (0=親設定)",
    "pass1_cfg": "第1段階のCFGスケール (0=親設定)",
    "pass1_sampler": "第1段階のサンプラー (親設定を使用)",
    "pass1_checkpoint": "第1段階のチェックポイント (親設定を使用)",
    "pass2": "第2段階 - ディテール精錬",
    "pass2_denoising": "第2段階のノイズ除去強度",
    "pass2_mask_dilation": "第2段階のマスク膨張",
    "pass2_steps": "第2段階のステップ数 (0=親設定)",
    "pass2_cfg": "第2段階のCFGスケール (0=親設定)",
    "pass2_sampler": "第2段階のサンプラー (親設定を使用)",
    "pass2_checkpoint": "第2段階のチェックポイント (親設定を使用)",
    "face_prompts": "顔のプロンプト",
    "face_prompt": "顔の正のプロンプト",
    "face_prompt_placeholder": "詳細な顔,美しい目,シャープなフォーカス,高品質",
    "face_negative_prompt": "顔の負のプロンプト",
    "face_negative_prompt_placeholder": "変形,醜い,構造異常,ぼやけ",
    "detection_settings": "検出設定",
    "min_detection_confidence": "最小検出信頼度",
    "max_faces": "処理する最大顔数",
    "inpaint_mask_blur": "インペイントマスクぼかし",
    "inpaint_full_res": "フル解像度でインペイント",
    "inpaint_full_res_padding": "フル解像度インペイントの余白",
    "no_faces": "顔が検出されませんでした。スキップします。",
    "detected_faces": "{count}人の顔を検出しました。処理中...",
    "processing_face": "{idx}/{total}番目の顔を処理中",
    "pass1_mask_failed": "{idx}番目の顔の第1段階マスク生成に失敗: {err}",
    "pass1_mask_empty": "{idx}番目の顔の第1段階マスクが空です。スキップします。",
    "pass1_complete": "{idx}番目の顔の第1段階完了",
    "pass1_inpaint_failed": "{idx}番目の顔の第1段階インペイント失敗: {err}",
    "pass2_mask_failed": "{idx}番目の顔の第2段階マスク生成に失敗: {err}",
    "pass2_mask_empty": "{idx}番目の顔の第2段階マスクが空です。ディテール段階をスキップします。",
    "pass2_complete": "{idx}番目の顔の第2段階完了",
    "pass2_inpaint_failed": "{idx}番目の顔の第2段階インペイント失敗: {err}",
    "all_complete": "すべての顔の処理が完了しました。",
    "facemesh_failed": "FaceMesh検出に失敗: {err}",
    "postprocess_error": "後処理でエラー: {err}",
    "loaded": "高度な顔修正器 v{ver} を読み込みました。",
    "use_parent": "親設定を使用",
    "switching_model": "モデルを切り替え: {model}",
    "model_switch_failed": "モデルの切り替えに失敗: {err}",
    "yolo_download": "YOLO顔モデルをダウンロード中...",
    "yolo_failed": "YOLO検出に失敗: {err}",
    "yolo_detected": "YOLOが{count}人の顔を検出",
}


def get_lang():
    try:
        webui_lang = shared.opts.localization
        if "zh" in webui_lang.lower() or "cn" in webui_lang.lower():
            return LANG_ZH
        elif "ja" in webui_lang.lower() or "jp" in webui_lang.lower():
            return LANG_JA
    except Exception:
        pass
    return LANG_EN


FACE_OVAL_LANDMARKS = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
    397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
    172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109, 10,
]

LEFT_EYE_LANDMARKS = [
    33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246, 33,
]

RIGHT_EYE_LANDMARKS = [
    362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398, 362,
]

LEFT_EYEBROW_LANDMARKS = [
    46, 53, 52, 65, 55, 107, 66, 105, 63, 70, 46,
]

RIGHT_EYEBROW_LANDMARKS = [
    276, 283, 282, 295, 285, 336, 296, 334, 293, 300, 276,
]

LIPS_OUTER_LANDMARKS = [
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185, 61,
]

NOSE_LANDMARKS = [
    168, 6, 197, 195, 5, 4, 1, 19, 94, 2, 164, 0, 11, 12, 13, 14, 15, 16, 17, 18, 200, 199, 175, 152,
]


@contextmanager
def preserve_prompts(p):
    all_pt = copy.copy(p.all_prompts)
    all_ng = copy.copy(p.all_negative_prompts)
    try:
        yield
    finally:
        p.all_prompts = all_pt
        p.all_negative_prompts = all_ng


@contextmanager
def pause_tqdm():
    orig = shared.opts.data.get("multiple_tqdm", True)
    try:
        shared.opts.data["multiple_tqdm"] = False
        yield
    finally:
        shared.opts.data["multiple_tqdm"] = orig


@contextmanager
def change_torch_load():
    orig = torch.load
    try:
        from modules import safe as safe_module
        torch.load = safe_module.unsafe_torch_load
        yield
    except Exception:
        yield
    finally:
        torch.load = orig


def _load_checkpoint(checkpoint_name):
    try:
        from modules import sd_models
        if hasattr(sd_models, "load_model"):
            sd_models.load_model(checkpoint_name)
            return
    except Exception as e:
        print(f"[AFR] sd_models.load_model failed: {e}")
    try:
        shared.opts.sd_model_checkpoint = checkpoint_name
        from modules import sd_models as sm
        if hasattr(sm, "reload_model_weights"):
            sm.reload_model_weights()
            return
    except Exception as e:
        print(f"[AFR] reload_model_weights failed: {e}")
    try:
        shared.opts.sd_model_checkpoint = checkpoint_name
        from modules import sd_models as sm
        model_data = getattr(sm, "model_data", None)
        if model_data is not None and hasattr(model_data, "load"):
            model_data.load(checkpoint_name)
            return
    except Exception as e:
        print(f"[AFR] model_data.load failed: {e}")
    raise RuntimeError(f"[AFR] All model loading methods failed for: {checkpoint_name}")


@contextmanager
def switch_model_context(checkpoint_name):
    if checkpoint_name is None:
        yield
        return
    current_info = getattr(shared.opts, "sd_model_checkpoint", None)
    if current_info == checkpoint_name:
        yield
        return
    try:
        print(f"[AFR] Switching model to: {checkpoint_name}")
        _load_checkpoint(checkpoint_name)
        yield
        print(f"[AFR] Restoring model to: {current_info}")
        _load_checkpoint(current_info)
    except Exception as e:
        print(f"[AFR] Model switch failed: {e}")
        traceback.print_exc()
        try:
            if current_info:
                _load_checkpoint(current_info)
        except Exception:
            pass
        yield


def to_pil_image(image_input) -> Image.Image:
    if isinstance(image_input, Image.Image):
        return image_input.convert("RGB")
    if isinstance(image_input, np.ndarray):
        if image_input.ndim == 2:
            return Image.fromarray(image_input).convert("RGB")
        if image_input.ndim == 3:
            if image_input.shape[2] == 4:
                return Image.fromarray(image_input, mode="RGBA").convert("RGB")
            return Image.fromarray(image_input).convert("RGB")
    raise ValueError(f"[AFR] Cannot convert type {type(image_input)} to PIL Image")


def pil_to_cv2(pil_image: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(pil_image.convert("RGB")), cv2.COLOR_RGB2BGR)


def ensure_face_landmarker_model() -> str:
    if os.path.exists(FACE_LANDMARKER_MODEL_PATH):
        return FACE_LANDMARKER_MODEL_PATH
    os.makedirs(AFR_MODEL_DIR, exist_ok=True)
    last_error = None
    for url in FACE_LANDMARKER_MODEL_URLS:
        print(f"[AFR] Downloading FaceLandmarker model from {url}...")
        try:
            urllib.request.urlretrieve(url, FACE_LANDMARKER_MODEL_PATH)
            print(f"[AFR] Model downloaded to {FACE_LANDMARKER_MODEL_PATH}")
            return FACE_LANDMARKER_MODEL_PATH
        except Exception as e:
            print(f"[AFR] Download failed from {url}: {e}")
            last_error = e
            if os.path.exists(FACE_LANDMARKER_MODEL_PATH):
                os.remove(FACE_LANDMARKER_MODEL_PATH)
    raise RuntimeError(f"[AFR] Failed to download FaceLandmarker model from all URLs. Last error: {last_error}")


def ensure_yolo_face_model() -> str:
    if os.path.exists(YOLO_FACE_MODEL_PATH):
        return YOLO_FACE_MODEL_PATH
    os.makedirs(AFR_MODEL_DIR, exist_ok=True)
    last_error = None
    for url in YOLO_FACE_MODEL_URLS:
        print(f"[AFR] Downloading YOLO face model from {url}...")
        try:
            urllib.request.urlretrieve(url, YOLO_FACE_MODEL_PATH)
            print(f"[AFR] YOLO model downloaded to {YOLO_FACE_MODEL_PATH}")
            return YOLO_FACE_MODEL_PATH
        except Exception as e:
            print(f"[AFR] YOLO download failed from {url}: {e}")
            last_error = e
            if os.path.exists(YOLO_FACE_MODEL_PATH):
                os.remove(YOLO_FACE_MODEL_PATH)
    raise RuntimeError(f"[AFR] Failed to download YOLO face model. Last error: {last_error}")


def _detect_mediapipe_legacy(image_cv2: np.ndarray, min_detection_confidence: float):
    import mediapipe as mp
    if not hasattr(mp, "solutions"):
        return None
    if not hasattr(mp.solutions, "face_mesh"):
        return None
    mp_face_mesh = mp.solutions.face_mesh
    faces = []
    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=10,
        refine_landmarks=True,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=0.5,
    ) as face_mesh:
        rgb_image = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_image)
        if not results.multi_face_landmarks:
            return faces
        h, w = image_cv2.shape[:2]
        for face_landmarks in results.multi_face_landmarks:
            landmarks_px = []
            for lm in face_landmarks.landmark:
                x = int(lm.x * w)
                y = int(lm.y * h)
                landmarks_px.append((x, y))
            faces.append(landmarks_px)
    return faces


def _detect_mediapipe_task_api(image_cv2: np.ndarray, min_detection_confidence: float):
    import mediapipe as mp
    from mediapipe.tasks.python import vision

    model_path = ensure_face_landmarker_model()

    base_options = mp.tasks.BaseOptions(model_asset_path=model_path)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
        num_faces=10,
        min_face_detection_confidence=min_detection_confidence,
    )

    detector = vision.FaceLandmarker.create_from_options(options)

    rgb_image = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
    detection_result = detector.detect(mp_image)

    faces = []
    if detection_result.face_landmarks:
        h, w = image_cv2.shape[:2]
        for face_landmarks in detection_result.face_landmarks:
            landmarks_px = []
            for lm in face_landmarks:
                x = int(lm.x * w)
                y = int(lm.y * h)
                landmarks_px.append((x, y))
            faces.append(landmarks_px)

    detector.close()
    return faces


def _detect_yolo_face(image_cv2: np.ndarray, min_detection_confidence: float):
    try:
        from ultralytics import YOLO
    except ImportError:
        print("[AFR] ultralytics not installed, trying to install...")
        try:
            import subprocess
            import sys
            subprocess.check_call([sys.executable, "-m", "pip", "install", "ultralytics"],
                                  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            from ultralytics import YOLO
        except Exception as e:
            raise ImportError(f"[AFR] Failed to install ultralytics: {e}")

    model_path = ensure_yolo_face_model()
    model = YOLO(model_path)

    results = model(image_cv2, conf=min_detection_confidence, verbose=False)

    faces = []
    h, w = image_cv2.shape[:2]
    for result in results:
        boxes = result.boxes
        if boxes is None:
            continue
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)
            rw = x2 - x1
            rh = y2 - y1
            if rw <= 0 or rh <= 0:
                continue
            kpts = box.keypoints
            if kpts is not None and len(kpts) > 0:
                kpts_data = kpts[0].cpu().numpy()
                landmarks = _yolo_keypoints_to_landmarks(kpts_data, x1, y1, rw, rh, (h, w))
            else:
                landmarks = _generate_approx_landmarks(x1, y1, rw, rh, (h, w))
            faces.append(landmarks)
    return faces


def _yolo_keypoints_to_landmarks(kpts_data, rx, ry, rw, rh, image_shape):
    landmarks = [(0, 0)] * 478
    h, w = image_shape[:2]

    if kpts_data.ndim == 2 and kpts_data.shape[0] >= 5:
        points = kpts_data[:, :2]
        le_x, le_y = int(np.clip(points[0][0], 0, w - 1)), int(np.clip(points[0][1], 0, h - 1))
        re_x, re_y = int(np.clip(points[1][0], 0, w - 1)), int(np.clip(points[1][1], 0, h - 1))
        n_x, n_y = int(np.clip(points[2][0], 0, w - 1)), int(np.clip(points[2][1], 0, h - 1))
        ml_x, ml_y = int(np.clip(points[3][0], 0, w - 1)), int(np.clip(points[3][1], 0, h - 1))
        mr_x, mr_y = int(np.clip(points[4][0], 0, w - 1)), int(np.clip(points[4][1], 0, h - 1))

        cx, cy = rx + rw / 2, ry + rh / 2
        a, b = rw / 2, rh / 2
        for i, idx in enumerate(FACE_OVAL_LANDMARKS):
            if idx < 478:
                angle = 2 * np.pi * i / len(FACE_OVAL_LANDMARKS) - np.pi / 2
                px = int(cx + a * np.cos(angle))
                py = int(cy + b * np.sin(angle))
                landmarks[idx] = (px, py)

        le_w = max(int(rw * 0.13), 3)
        le_h = max(int(rh * 0.04), 2)
        for i, idx in enumerate(LEFT_EYE_LANDMARKS):
            if idx < 478:
                angle = 2 * np.pi * i / len(LEFT_EYE_LANDMARKS)
                landmarks[idx] = (int(le_x + le_w * np.cos(angle)), int(le_y + le_h * np.sin(angle)))

        for i, idx in enumerate(RIGHT_EYE_LANDMARKS):
            if idx < 478:
                angle = 2 * np.pi * i / len(RIGHT_EYE_LANDMARKS)
                landmarks[idx] = (int(re_x + le_w * np.cos(angle)), int(re_y + le_h * np.sin(angle)))

        for i, idx in enumerate(LEFT_EYEBROW_LANDMARKS):
            if idx < 478:
                px = int(le_x - le_w + 2 * le_w * i / len(LEFT_EYEBROW_LANDMARKS))
                py = int(le_y - rh * 0.06 - rh * 0.03 * np.sin(np.pi * i / len(LEFT_EYEBROW_LANDMARKS)))
                landmarks[idx] = (px, py)

        for i, idx in enumerate(RIGHT_EYEBROW_LANDMARKS):
            if idx < 478:
                px = int(re_x - le_w + 2 * le_w * i / len(RIGHT_EYEBROW_LANDMARKS))
                py = int(re_y - rh * 0.06 - rh * 0.03 * np.sin(np.pi * i / len(RIGHT_EYEBROW_LANDMARKS)))
                landmarks[idx] = (px, py)

        m_w = max(int(abs(mr_x - ml_x) * 0.6), 3)
        m_h = max(int(rh * 0.05), 2)
        m_cx = (ml_x + mr_x) / 2
        m_cy = (ml_y + mr_y) / 2
        for i, idx in enumerate(LIPS_OUTER_LANDMARKS):
            if idx < 478:
                angle = 2 * np.pi * i / len(LIPS_OUTER_LANDMARKS)
                landmarks[idx] = (int(m_cx + m_w * np.cos(angle)), int(m_cy + m_h * np.sin(angle)))

        n_w = max(int(rw * 0.08), 2)
        n_h = max(int(rh * 0.06), 2)
        for i, idx in enumerate(NOSE_LANDMARKS):
            if idx < 478:
                angle = 2 * np.pi * i / len(NOSE_LANDMARKS)
                landmarks[idx] = (int(n_x + n_w * np.cos(angle)), int(n_y + n_h * np.sin(angle)))
    else:
        landmarks = _generate_approx_landmarks(rx, ry, rw, rh, image_shape)

    return landmarks


def _detect_opencv_haar(image_cv2: np.ndarray, min_detection_confidence: float):
    cascade_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
    if not os.path.exists(cascade_path):
        cascade_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_alt2.xml")
    if not os.path.exists(cascade_path):
        return None

    face_cascade = cv2.CascadeClassifier(cascade_path)
    gray = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2GRAY)
    face_rects = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.01,
        minNeighbors=3,
        minSize=(32, 32),
    )

    if len(face_rects) == 0:
        return []

    faces = []
    for rx, ry, rw, rh in face_rects:
        landmarks = _generate_approx_landmarks(rx, ry, rw, rh, image_cv2.shape)
        faces.append(landmarks)
    return faces


def _generate_approx_landmarks(rx: int, ry: int, rw: int, rh: int, image_shape: tuple) -> list[tuple[int, int]]:
    landmarks = [(0, 0)] * 478

    cx, cy = rx + rw / 2, ry + rh / 2
    a, b = rw / 2, rh / 2

    for i, idx in enumerate(FACE_OVAL_LANDMARKS):
        if idx < 478:
            angle = 2 * np.pi * i / len(FACE_OVAL_LANDMARKS) - np.pi / 2
            px = int(cx + a * np.cos(angle))
            py = int(cy + b * np.sin(angle))
            landmarks[idx] = (px, py)

    le_cx, le_cy = rx + rw * 0.32, ry + rh * 0.38
    le_w, le_h = rw * 0.13, rh * 0.04
    for i, idx in enumerate(LEFT_EYE_LANDMARKS):
        if idx < 478:
            angle = 2 * np.pi * i / len(LEFT_EYE_LANDMARKS)
            landmarks[idx] = (int(le_cx + le_w * np.cos(angle)), int(le_cy + le_h * np.sin(angle)))

    re_cx, re_cy = rx + rw * 0.68, ry + rh * 0.38
    for i, idx in enumerate(RIGHT_EYE_LANDMARKS):
        if idx < 478:
            angle = 2 * np.pi * i / len(RIGHT_EYE_LANDMARKS)
            landmarks[idx] = (int(re_cx + le_w * np.cos(angle)), int(re_cy + le_h * np.sin(angle)))

    for i, idx in enumerate(LEFT_EYEBROW_LANDMARKS):
        if idx < 478:
            px = int(rx + rw * (0.18 + 0.22 * i / len(LEFT_EYEBROW_LANDMARKS)))
            py = int(ry + rh * 0.28 - rh * 0.03 * np.sin(np.pi * i / len(LEFT_EYEBROW_LANDMARKS)))
            landmarks[idx] = (px, py)

    for i, idx in enumerate(RIGHT_EYEBROW_LANDMARKS):
        if idx < 478:
            px = int(rx + rw * (0.60 + 0.22 * i / len(RIGHT_EYEBROW_LANDMARKS)))
            py = int(ry + rh * 0.28 - rh * 0.03 * np.sin(np.pi * i / len(RIGHT_EYEBROW_LANDMARKS)))
            landmarks[idx] = (px, py)

    m_cx, m_cy = rx + rw * 0.5, ry + rh * 0.75
    m_w, m_h = rw * 0.18, rh * 0.05
    for i, idx in enumerate(LIPS_OUTER_LANDMARKS):
        if idx < 478:
            angle = 2 * np.pi * i / len(LIPS_OUTER_LANDMARKS)
            landmarks[idx] = (int(m_cx + m_w * np.cos(angle)), int(m_cy + m_h * np.sin(angle)))

    n_cx, n_cy = rx + rw * 0.5, ry + rh * 0.58
    n_w, n_h = rw * 0.08, rh * 0.06
    for i, idx in enumerate(NOSE_LANDMARKS):
        if idx < 478:
            angle = 2 * np.pi * i / len(NOSE_LANDMARKS)
            landmarks[idx] = (int(n_cx + n_w * np.cos(angle)), int(n_cy + n_h * np.sin(angle)))

    return landmarks


def detect_faces(image_cv2: np.ndarray, detection_model: str = "mediapipe", min_detection_confidence: float = 0.5):
    print(f"[AFR] Attempting face detection with model: {detection_model}")

    if detection_model == "yolo":
        try:
            result = _detect_yolo_face(image_cv2, min_detection_confidence)
            print(f"[AFR] YOLO detected {len(result)} face(s)")
            return result
        except Exception as e:
            print(f"[AFR] YOLO detection failed: {e}")
            traceback.print_exc()
            print("[AFR] Falling back to mediapipe...")

    if detection_model == "mediapipe" or detection_model == "yolo":
        try:
            result = _detect_mediapipe_legacy(image_cv2, min_detection_confidence)
            if result is not None:
                print(f"[AFR] Using mediapipe legacy API, detected {len(result)} face(s)")
                return result
        except Exception as e:
            print(f"[AFR] Legacy mediapipe API not available: {e}")

        try:
            result = _detect_mediapipe_task_api(image_cv2, min_detection_confidence)
            print(f"[AFR] Using mediapipe Task API, detected {len(result)} face(s)")
            return result
        except Exception as e:
            print(f"[AFR] Mediapipe Task API failed: {e}")
            traceback.print_exc()

        if detection_model == "mediapipe":
            print("[AFR] Falling back to Haar cascade...")

    try:
        result = _detect_opencv_haar(image_cv2, min_detection_confidence)
        if result is not None:
            print(f"[AFR] Using OpenCV Haar cascade, detected {len(result)} face(s)")
            return result
    except Exception as e:
        print(f"[AFR] OpenCV Haar cascade failed: {e}")
        traceback.print_exc()

    raise ImportError(
        "[AFR] All face detection methods failed! "
        "Please ensure mediapipe or ultralytics is installed correctly."
    )


def generate_face_mask(image_cv2: np.ndarray, landmarks_px: list[tuple[int, int]], mask_dilation: int = 10) -> np.ndarray:
    h, w = image_cv2.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    face_oval_pts = []
    for idx in FACE_OVAL_LANDMARKS:
        if idx < len(landmarks_px):
            pt = landmarks_px[idx]
            if pt != (0, 0):
                face_oval_pts.append(pt)
    if len(face_oval_pts) >= 3:
        pts_array = np.array(face_oval_pts, dtype=np.int32)
        hull = cv2.convexHull(pts_array)
        cv2.fillConvexPoly(mask, hull, 255)

    for landmark_group in [LEFT_EYE_LANDMARKS, RIGHT_EYE_LANDMARKS, LIPS_OUTER_LANDMARKS]:
        group_pts = []
        for idx in landmark_group:
            if idx < len(landmarks_px):
                pt = landmarks_px[idx]
                if pt != (0, 0):
                    group_pts.append(pt)
        if len(group_pts) >= 3:
            pts_array = np.array(group_pts, dtype=np.int32)
            hull = cv2.convexHull(pts_array)
            cv2.fillConvexPoly(mask, hull, 255)

    if mask_dilation > 0:
        kernel = np.ones((mask_dilation, mask_dilation), dtype=np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)

    mask = cv2.GaussianBlur(mask, (7, 7), 0)
    return mask


def generate_detail_mask(image_cv2: np.ndarray, landmarks_px: list[tuple[int, int]], mask_dilation: int = 6) -> np.ndarray:
    h, w = image_cv2.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    detail_landmarks = list(
        dict.fromkeys(
            LEFT_EYE_LANDMARKS
            + RIGHT_EYE_LANDMARKS
            + LEFT_EYEBROW_LANDMARKS
            + RIGHT_EYEBROW_LANDMARKS
            + LIPS_OUTER_LANDMARKS
            + NOSE_LANDMARKS
        )
    )

    detail_pts = []
    for idx in detail_landmarks:
        if idx < len(landmarks_px):
            pt = landmarks_px[idx]
            if pt != (0, 0):
                detail_pts.append(pt)

    if len(detail_pts) >= 3:
        pts_array = np.array(detail_pts, dtype=np.int32)
        hull = cv2.convexHull(pts_array)
        cv2.fillConvexPoly(mask, hull, 255)

    face_oval_pts = []
    for idx in FACE_OVAL_LANDMARKS:
        if idx < len(landmarks_px):
            pt = landmarks_px[idx]
            if pt != (0, 0):
                face_oval_pts.append(pt)
    if len(face_oval_pts) >= 3:
        oval_array = np.array(face_oval_pts, dtype=np.int32)
        oval_hull = cv2.convexHull(oval_array)
        oval_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillConvexPoly(oval_mask, oval_hull, 255)
        mask = cv2.bitwise_and(mask, oval_mask)

    if mask_dilation > 0:
        kernel = np.ones((mask_dilation, mask_dilation), dtype=np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)

    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    return mask


def composite_image(original_pil: Image.Image, refined_pil: Image.Image, mask_cv2: np.ndarray) -> Image.Image:
    original_np = np.array(original_pil.convert("RGB"))
    refined_np = np.array(refined_pil.convert("RGB"))
    mask_3ch = cv2.merge([mask_cv2, mask_cv2, mask_cv2])
    mask_float = mask_3ch.astype(np.float32) / 255.0
    result = (original_np.astype(np.float32) * (1.0 - mask_float) + refined_np.astype(np.float32) * mask_float)
    result = np.clip(result, 0, 255).astype(np.uint8)
    return Image.fromarray(result)


def mask_cv2_to_pil(mask_cv2: np.ndarray) -> Image.Image:
    pil_mask = Image.fromarray(mask_cv2)
    if pil_mask.mode != "L":
        pil_mask = pil_mask.convert("L")
    return pil_mask


def get_checkpoint_list():
    checkpoints = []
    try:
        from modules import sd_models
        if hasattr(sd_models, "checkpoint_tiles"):
            checkpoints = list(sd_models.checkpoint_tiles())
            if checkpoints:
                return checkpoints
    except Exception as e:
        print(f"[AFR] checkpoint_tiles failed: {e}")
    try:
        from modules import sd_models
        if hasattr(sd_models, "checkpoints_list") and sd_models.checkpoints_list:
            checkpoints = list(sd_models.checkpoints_list.keys())
            if checkpoints:
                return checkpoints
    except Exception as e:
        print(f"[AFR] checkpoints_list keys failed: {e}")
    try:
        from modules import sd_models
        if hasattr(sd_models, "checkpoint_titles"):
            checkpoints = list(sd_models.checkpoint_titles())
            if checkpoints:
                return checkpoints
    except Exception as e:
        print(f"[AFR] checkpoint_titles failed: {e}")
    try:
        current = getattr(shared.opts, "sd_model_checkpoint", None)
        if current:
            return [current]
    except Exception:
        pass
    return checkpoints


def get_sampler_list():
    try:
        from modules import sd_samplers
        samplers = [s.name for s in sd_samplers.all_samplers]
        return samplers
    except Exception:
        return []


class AdvancedFaceRefinerScript(scripts.Script):
    def __init__(self):
        super().__init__()
        self.lang = get_lang()

    def title(self):
        return self.lang["title"]

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        self.lang = get_lang()
        lang = self.lang

        checkpoints = get_checkpoint_list()
        samplers = get_sampler_list()
        use_parent_label = lang["use_parent"]

        with gr.Accordion(lang["main_accordion"], open=False, elem_id="afr_main_accordion"):
            with gr.Accordion(lang["enable"], open=True, elem_id="afr_enabled"):
                enabled = gr.Checkbox(label=lang["enable_afr"], value=False, elem_id="afr_enable_checkbox")

            with gr.Accordion(lang["detection_settings"], open=True):
                detection_model = gr.Dropdown(
                    choices=DETECTION_MODES,
                    value="mediapipe",
                    label=lang["detection_model"],
                    elem_id="afr_detection_model",
                )
                gr.HTML(f"<small>{lang['detection_model_desc']}</small>")
                min_detection_confidence = gr.Slider(
                    minimum=0.1, maximum=1.0, step=0.05, value=0.5,
                    label=lang["min_detection_confidence"],
                    elem_id="afr_min_confidence",
                )
                max_faces = gr.Slider(
                    minimum=1, maximum=10, step=1, value=4,
                    label=lang["max_faces"],
                    elem_id="afr_max_faces",
                )

            with gr.Accordion(lang["pass1"], open=True):
                pass1_denoising = gr.Slider(
                    minimum=0.05, maximum=1.0, step=0.05, value=0.25,
                    label=lang["pass1_denoising"],
                    elem_id="afr_pass1_denoising",
                )
                pass1_mask_dilation = gr.Slider(
                    minimum=0, maximum=50, step=1, value=12,
                    label=lang["pass1_mask_dilation"],
                    elem_id="afr_pass1_mask_dilation",
                )
                pass1_steps = gr.Slider(
                    minimum=1, maximum=150, step=1, value=0,
                    label=lang["pass1_steps"],
                    elem_id="afr_pass1_steps",
                )
                pass1_cfg = gr.Slider(
                    minimum=1.0, maximum=30.0, step=0.5, value=0,
                    label=lang["pass1_cfg"],
                    elem_id="afr_pass1_cfg",
                )
                pass1_sampler = gr.Dropdown(
                    choices=[use_parent_label] + samplers,
                    value=use_parent_label,
                    label=lang["pass1_sampler"],
                    elem_id="afr_pass1_sampler",
                )
                pass1_checkpoint = gr.Dropdown(
                    choices=[use_parent_label] + checkpoints,
                    value=use_parent_label,
                    label=lang["pass1_checkpoint"],
                    elem_id="afr_pass1_checkpoint",
                )

            with gr.Accordion(lang["pass2"], open=True):
                pass2_denoising = gr.Slider(
                    minimum=0.05, maximum=1.0, step=0.05, value=0.45,
                    label=lang["pass2_denoising"],
                    elem_id="afr_pass2_denoising",
                )
                pass2_mask_dilation = gr.Slider(
                    minimum=0, maximum=50, step=1, value=8,
                    label=lang["pass2_mask_dilation"],
                    elem_id="afr_pass2_mask_dilation",
                )
                pass2_steps = gr.Slider(
                    minimum=1, maximum=150, step=1, value=0,
                    label=lang["pass2_steps"],
                    elem_id="afr_pass2_steps",
                )
                pass2_cfg = gr.Slider(
                    minimum=1.0, maximum=30.0, step=0.5, value=0,
                    label=lang["pass2_cfg"],
                    elem_id="afr_pass2_cfg",
                )
                pass2_sampler = gr.Dropdown(
                    choices=[use_parent_label] + samplers,
                    value=use_parent_label,
                    label=lang["pass2_sampler"],
                    elem_id="afr_pass2_sampler",
                )
                pass2_checkpoint = gr.Dropdown(
                    choices=[use_parent_label] + checkpoints,
                    value=use_parent_label,
                    label=lang["pass2_checkpoint"],
                    elem_id="afr_pass2_checkpoint",
                )

            with gr.Accordion(lang["face_prompts"], open=True):
                face_prompt = gr.Textbox(
                    label=lang["face_prompt"],
                    placeholder=lang["face_prompt_placeholder"],
                    lines=2,
                    elem_id="afr_face_prompt",
                )
                face_negative_prompt = gr.Textbox(
                    label=lang["face_negative_prompt"],
                    placeholder=lang["face_negative_prompt_placeholder"],
                    lines=2,
                    elem_id="afr_face_negative_prompt",
                )

            with gr.Accordion(lang["detection_settings"], open=False):
                mask_blur = gr.Slider(
                    minimum=0, maximum=64, step=1, value=4,
                    label=lang["inpaint_mask_blur"],
                    elem_id="afr_mask_blur",
                )
                inpaint_full_res = gr.Checkbox(
                    label=lang["inpaint_full_res"],
                    value=True,
                    elem_id="afr_inpaint_full_res",
                )
                inpaint_full_res_padding = gr.Slider(
                    minimum=0, maximum=256, step=4, value=32,
                    label=lang["inpaint_full_res_padding"],
                    elem_id="afr_inpaint_padding",
                )

        return [
            enabled,
            detection_model,
            pass1_denoising, pass1_mask_dilation, pass1_steps, pass1_cfg,
            pass1_sampler, pass1_checkpoint,
            pass2_denoising, pass2_mask_dilation, pass2_steps, pass2_cfg,
            pass2_sampler, pass2_checkpoint,
            face_prompt, face_negative_prompt,
            min_detection_confidence, max_faces, mask_blur,
            inpaint_full_res, inpaint_full_res_padding,
        ]

    def _get_prompt(self, ad_prompt: str, all_prompts: list[str], i: int, default: str) -> list[str]:
        prompts = ad_prompt.split("[SEP]")
        prompts = [p.strip() for p in prompts]
        blank_replacement = default
        if all_prompts and i < len(all_prompts):
            blank_replacement = all_prompts[i]
        elif all_prompts:
            blank_replacement = all_prompts[i % len(all_prompts)]
        for n in range(len(prompts)):
            if not prompts[n]:
                prompts[n] = blank_replacement
            elif "[PROMPT]" in prompts[n]:
                prompts[n] = prompts[n].replace("[PROMPT]", f" {blank_replacement} ")
        return prompts

    def _get_seed(self, p, i: int) -> tuple[int, int]:
        if not p.all_seeds:
            seed = p.seed
        elif i < len(p.all_seeds):
            seed = p.all_seeds[i]
        else:
            seed = p.all_seeds[i % len(p.all_seeds)]
        if not p.all_subseeds:
            subseed = p.subseed
        elif i < len(p.all_subseeds):
            subseed = p.all_subseeds[i]
        else:
            subseed = p.all_subseeds[i % len(p.all_subseeds)]
        return seed, subseed

    def _resolve_sampler(self, sampler_name: str, parent_sampler: str) -> str:
        lang = self.lang
        if sampler_name == lang["use_parent"] or not sampler_name:
            return parent_sampler
        return sampler_name

    def _resolve_checkpoint(self, checkpoint_name: str) -> str | None:
        lang = self.lang
        if checkpoint_name == lang["use_parent"] or not checkpoint_name:
            return None
        return checkpoint_name

    def _create_i2i_process(
        self,
        p,
        image: Image.Image,
        mask_pil: Image.Image,
        denoising_strength: float,
        steps: int,
        cfg_scale: float,
        sampler_name: str,
        checkpoint_name: str | None,
        prompt_str: str,
        negative_prompt_str: str,
        mask_blur: int,
        inpaint_full_res: bool,
        inpaint_full_res_padding: int,
    ) -> StableDiffusionProcessingImg2Img:
        i = getattr(p, "_afr_idx", 0)
        seed, subseed = self._get_seed(p, i)

        actual_steps = steps if steps > 0 else p.steps
        actual_cfg = cfg_scale if cfg_scale > 0 else p.cfg_scale
        actual_sampler = sampler_name if sampler_name else p.sampler_name

        override_settings = {}
        if checkpoint_name:
            override_settings["sd_model_checkpoint"] = checkpoint_name

        i2i = StableDiffusionProcessingImg2Img(
            init_images=[image],
            resize_mode=0,
            denoising_strength=denoising_strength,
            mask=mask_pil,
            mask_blur=mask_blur,
            inpainting_fill=1,
            inpaint_full_res=inpaint_full_res,
            inpaint_full_res_padding=inpaint_full_res_padding,
            inpainting_mask_invert=0,
            sd_model=p.sd_model,
            outpath_samples=p.outpath_samples,
            outpath_grids=p.outpath_grids,
            prompt=prompt_str,
            negative_prompt=negative_prompt_str,
            styles=p.styles,
            seed=seed,
            subseed=subseed,
            subseed_strength=p.subseed_strength,
            seed_resize_from_h=p.seed_resize_from_h,
            seed_resize_from_w=p.seed_resize_from_w,
            sampler_name=actual_sampler,
            batch_size=1,
            n_iter=1,
            steps=actual_steps,
            cfg_scale=actual_cfg,
            width=p.width,
            height=p.height,
            restore_faces=False,
            tiling=p.tiling,
            extra_generation_params=p.extra_generation_params,
            do_not_save_samples=True,
            do_not_save_grid=True,
            override_settings=override_settings,
        )

        i2i.cached_c = [None, None]
        i2i.cached_uc = [None, None]
        i2i._disable_adetailer = True
        i2i._afr_processing = True

        try:
            script_runner = copy.copy(p.scripts)
            script_args = copy.deepcopy(p.script_args)
            filtered_alwayson = []
            for script_obj in script_runner.alwayson_scripts:
                filename_lower = script_obj.filename.lower()
                if "afr_main" in filename_lower or "adetailer" in filename_lower:
                    continue
                filtered_alwayson.append(script_obj)
            script_runner.alwayson_scripts = filtered_alwayson
            i2i.scripts = script_runner
            i2i.script_args = script_args
        except Exception as e:
            print(f"[AFR] Script filter failed, using minimal scripts: {e}")
            try:
                i2i.scripts = type(p.scripts)()
                i2i.scripts.alwayson_scripts = []
                i2i.scripts.selectable_scripts = []
                i2i.script_args = []
            except Exception:
                i2i.scripts = None
                i2i.script_args = []

        return i2i

    def _run_inpaint_pass(
        self,
        p,
        image: Image.Image,
        mask_pil: Image.Image,
        denoising_strength: float,
        steps: int,
        cfg_scale: float,
        sampler_name: str,
        checkpoint_name: str | None,
        prompt_str: str,
        negative_prompt_str: str,
        mask_blur: int,
        inpaint_full_res: bool,
        inpaint_full_res_padding: int,
    ) -> Image.Image:
        with switch_model_context(checkpoint_name):
            i2i = self._create_i2i_process(
                p, image, mask_pil, denoising_strength, steps, cfg_scale,
                sampler_name, checkpoint_name,
                prompt_str, negative_prompt_str, mask_blur,
                inpaint_full_res, inpaint_full_res_padding,
            )

            print(f"[AFR] Starting inpaint: denoise={denoising_strength}, steps={i2i.steps}, cfg={i2i.cfg_scale}, sampler={i2i.sampler_name}")
            if checkpoint_name:
                print(f"[AFR] Using checkpoint: {checkpoint_name}")
            print(f"[AFR] Prompt: {prompt_str[:80]}...")
            print(f"[AFR] Image size: {image.size}, Mask mode: {'full_res' if inpaint_full_res else 'whole'}")

            with change_torch_load():
                with pause_tqdm():
                    with preserve_prompts(p):
                        processed = process_images(i2i)

        if processed is None:
            print("[AFR] process_images returned None!")
            return image

        if not processed.images or len(processed.images) == 0:
            print("[AFR] process_images returned no images!")
            return image

        result = processed.images[0]
        if isinstance(result, np.ndarray):
            result = Image.fromarray(result)
        if not isinstance(result, Image.Image):
            print(f"[AFR] Unexpected result type: {type(result)}")
            return image

        print(f"[AFR] Inpaint pass completed, result size: {result.size}")
        return result

    def postprocess_image(self, p, pp, *args):
        self.lang = get_lang()
        try:
            self._postprocess_image_impl(p, pp, args)
        except Exception as e:
            print(f"[AFR] {self.lang['postprocess_error'].format(err=e)}")
            traceback.print_exc()

    def _postprocess_image_impl(self, p, pp, args):
        if getattr(p, "_disable_adetailer", False):
            print("[AFR] Skipping: _disable_adetailer flag detected on parent process.")
            return
        if getattr(p, "_afr_processing", False):
            print("[AFR] Skipping: _afr_processing flag detected (recursive call prevention).")
            return

        print(f"[AFR] postprocess_image called, args count: {len(args)}")

        if len(args) < 20:
            print(f"[AFR] ERROR: Expected 20 args, got {len(args)}. UI may not be connected properly.")
            print(f"[AFR] Args received: {args}")
            return

        (
            enabled,
            detection_model,
            pass1_denoising, pass1_mask_dilation, pass1_steps, pass1_cfg,
            pass1_sampler, pass1_checkpoint,
            pass2_denoising, pass2_mask_dilation, pass2_steps, pass2_cfg,
            pass2_sampler, pass2_checkpoint,
            face_prompt, face_negative_prompt,
            min_detection_confidence, max_faces, mask_blur,
            inpaint_full_res, inpaint_full_res_padding,
        ) = args

        print(f"[AFR] Enabled: {enabled}, Detection model: {detection_model}")

        if not enabled:
            return

        image_pil = pp.image
        if image_pil is None:
            print("[AFR] pp.image is None, skipping.")
            return

        try:
            image_pil = to_pil_image(image_pil)
        except Exception as e:
            print(f"[AFR] Failed to convert pp.image to PIL: {e}, type={type(pp.image)}")
            return

        print(f"[AFR] Image obtained: size={image_pil.size}, mode={image_pil.mode}")

        image_cv2 = pil_to_cv2(image_pil)

        try:
            faces = detect_faces(image_cv2, detection_model, min_detection_confidence)
        except Exception as e:
            print(f"[AFR] {self.lang['facemesh_failed'].format(err=e)}")
            traceback.print_exc()
            return

        if not faces:
            print(f"[AFR] {self.lang['no_faces']}")
            return

        faces = faces[:int(max_faces)]
        print(f"[AFR] {self.lang['detected_faces'].format(count=len(faces))}")

        idx = getattr(pp, "index", 0)
        p._afr_idx = idx
        p._afr_processing = True

        pass1_sampler_resolved = self._resolve_sampler(pass1_sampler, p.sampler_name)
        pass2_sampler_resolved = self._resolve_sampler(pass2_sampler, p.sampler_name)
        pass1_checkpoint_resolved = self._resolve_checkpoint(pass1_checkpoint)
        pass2_checkpoint_resolved = self._resolve_checkpoint(pass2_checkpoint)

        try:
            current_image = image_pil

            for face_idx, landmarks_px in enumerate(faces):
                print(f"[AFR] {self.lang['processing_face'].format(idx=face_idx+1, total=len(faces))}")

                try:
                    face_mask_cv2 = generate_face_mask(image_cv2, landmarks_px, int(pass1_mask_dilation))
                except Exception as e:
                    print(f"[AFR] {self.lang['pass1_mask_failed'].format(idx=face_idx+1, err=e)}")
                    traceback.print_exc()
                    continue

                if np.sum(face_mask_cv2) == 0:
                    print(f"[AFR] {self.lang['pass1_mask_empty'].format(idx=face_idx+1)}")
                    continue

                face_mask_pil = mask_cv2_to_pil(face_mask_cv2)

                prompt_list = self._get_prompt(face_prompt or "", p.all_prompts, idx, p.prompt)
                neg_prompt_list = self._get_prompt(face_negative_prompt or "", p.all_negative_prompts, idx, p.negative_prompt)

                pass1_prompt = prompt_list[0] if prompt_list else p.prompt
                pass1_neg_prompt = neg_prompt_list[0] if neg_prompt_list else p.negative_prompt

                try:
                    pass1_result = self._run_inpaint_pass(
                        p, current_image, face_mask_pil,
                        float(pass1_denoising), int(pass1_steps), float(pass1_cfg),
                        pass1_sampler_resolved, pass1_checkpoint_resolved,
                        pass1_prompt, pass1_neg_prompt,
                        int(mask_blur), bool(inpaint_full_res), int(inpaint_full_res_padding),
                    )
                    current_image = composite_image(current_image, pass1_result, face_mask_cv2)
                    print(f"[AFR] {self.lang['pass1_complete'].format(idx=face_idx+1)}")
                except Exception as e:
                    print(f"[AFR] {self.lang['pass1_inpaint_failed'].format(idx=face_idx+1, err=e)}")
                    traceback.print_exc()
                    continue

                try:
                    detail_mask_cv2 = generate_detail_mask(image_cv2, landmarks_px, int(pass2_mask_dilation))
                except Exception as e:
                    print(f"[AFR] {self.lang['pass2_mask_failed'].format(idx=face_idx+1, err=e)}")
                    traceback.print_exc()
                    continue

                if np.sum(detail_mask_cv2) == 0:
                    print(f"[AFR] {self.lang['pass2_mask_empty'].format(idx=face_idx+1)}")
                    continue

                detail_mask_pil = mask_cv2_to_pil(detail_mask_cv2)

                try:
                    pass2_result = self._run_inpaint_pass(
                        p, current_image, detail_mask_pil,
                        float(pass2_denoising), int(pass2_steps), float(pass2_cfg),
                        pass2_sampler_resolved, pass2_checkpoint_resolved,
                        pass1_prompt, pass1_neg_prompt,
                        int(mask_blur), bool(inpaint_full_res), int(inpaint_full_res_padding),
                    )
                    current_image = composite_image(current_image, pass2_result, detail_mask_cv2)
                    print(f"[AFR] {self.lang['pass2_complete'].format(idx=face_idx+1)}")
                except Exception as e:
                    print(f"[AFR] {self.lang['pass2_inpaint_failed'].format(idx=face_idx+1, err=e)}")
                    traceback.print_exc()
                    continue

            pp.image = current_image
            print(f"[AFR] {self.lang['all_complete']}")

        finally:
            p._afr_processing = False
            if hasattr(p, "_afr_idx"):
                del p._afr_idx


print(f"[AFR] {get_lang()['loaded'].format(ver=AFR_VERSION)}")
