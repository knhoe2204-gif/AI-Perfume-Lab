import json
import os
import tempfile
from pathlib import Path
from typing import Any

import librosa
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from openai import OpenAI

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
MAX_BYTES = 15 * 1024 * 1024
MAX_SECONDS = 45
ALLOWED_EXTENSIONS = {".wav", ".mp3", ".m4a", ".ogg", ".flac", ".webm", ".aac"}
INGREDIENTS = [
    "Bergamot FCF", "Cam ngọt", "Oải hương", "Phong lữ",
    "Hương gỗ tuyết tùng", "Hương gỗ đàn hương", "Hương vani", "Ethanol (dung môi)"
]
MOODS = {
    "serene": "thư giãn, tĩnh tại", "mysterious": "bí ẩn, sâu lắng",
    "romantic": "lãng mạn, mềm mại", "energetic": "năng động, rực rỡ",
    "nostalgic": "hoài niệm, ấm áp"
}

app = FastAPI(title="AI Perfume Lab Sound-to-Scent", version="1.0.0")


@app.get("/")
def home():
    return FileResponse(STATIC_DIR / "index.html")


def safe_float(value: Any, digits: int = 2):
    try:
        value = float(value)
        return round(value, digits) if np.isfinite(value) else None
    except (TypeError, ValueError):
        return None


def analyze_audio(path: str):
    try:
        y, sr = librosa.load(path, sr=22050, mono=True, duration=MAX_SECONDS)
    except Exception as exc:
        raise HTTPException(status_code=422, detail="Không đọc được tệp âm thanh. Hãy thử WAV hoặc MP3 hợp lệ.") from exc
    if y.size == 0 or sr <= 0:
        raise HTTPException(status_code=422, detail="Tệp âm thanh trống hoặc không hợp lệ.")
    duration = len(y) / sr
    if duration > MAX_SECONDS + 0.5:
        raise HTTPException(status_code=422, detail=f"Âm thanh cần dài tối đa {MAX_SECONDS} giây.")

    tempo = safe_float(librosa.feature.tempo(y=y, sr=sr)[0], 1)
    rms = safe_float(np.mean(librosa.feature.rms(y=y)), 5)
    centroid = safe_float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)), 1)
    bandwidth = safe_float(np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)), 1)
    rolloff = safe_float(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)), 1)
    zcr = safe_float(np.mean(librosa.feature.zero_crossing_rate(y)), 5)
    flatness = safe_float(np.mean(librosa.feature.spectral_flatness(y=y)), 5)

    # Interpretations are heuristic descriptions of measured signal features, not emotion detection.
    tempo_meaning = "nhịp chậm" if tempo and tempo < 85 else "nhịp nhanh" if tempo and tempo > 125 else "nhịp trung bình"
    energy_meaning = "năng lượng tín hiệu thấp" if rms is not None and rms < 0.05 else "năng lượng tín hiệu khá cao" if rms is not None and rms > 0.18 else "năng lượng tín hiệu trung bình"
    brightness = "thiên về dải tần cao/sáng hơn" if centroid and centroid > 2500 else "thiên về dải tần thấp/tối hơn" if centroid and centroid < 1200 else "độ sáng phổ ở mức trung gian"
    metrics = [
        {"label": "Thời lượng phân tích", "value": f"{duration:.1f} giây", "meaning": "Đoạn âm thanh được đọc bởi bộ phân tích."},
        {"label": "Nhịp độ ước tính", "value": f"{tempo} BPM" if tempo is not None else "Không xác định", "meaning": tempo_meaning},
        {"label": "RMS trung bình", "value": str(rms), "meaning": energy_meaning},
        {"label": "Tâm phổ trung bình", "value": f"{centroid} Hz" if centroid is not None else "Không xác định", "meaning": brightness},
        {"label": "Băng thông phổ", "value": f"{bandwidth} Hz" if bandwidth is not None else "Không xác định", "meaning": "Mức phân tán năng lượng theo tần số."},
        {"label": "Spectral roll-off", "value": f"{rolloff} Hz" if rolloff is not None else "Không xác định", "meaning": "Tần số nằm dưới ngưỡng chứa phần lớn năng lượng phổ."},
        {"label": "Zero-crossing rate", "value": str(zcr), "meaning": "Tần suất tín hiệu đổi dấu; có thể phản ánh độ nhiễu/đặc trưng âm thanh."},
        {"label": "Spectral flatness", "value": str(flatness), "meaning": "Gợi ý phổ có tính âm sắc hay giống nhiễu hơn; không xác định trực tiếp nhạc cụ."},
    ]
    return {"duration_seconds": round(duration, 2), "sample_rate": sr, "tempo_bpm": tempo,
            "rms_mean": rms, "spectral_centroid_hz": centroid, "spectral_bandwidth_hz": bandwidth,
            "spectral_rolloff_hz": rolloff, "zero_crossing_rate": zcr, "spectral_flatness": flatness}, metrics


def normalize_percentages(values):
    if not isinstance(values, list) or len(values) != len(INGREDIENTS):
        values = [10, 5, 10, 5, 10, 15, 10, 35]
    clean = [max(0, min(100, int(round(float(v))))) for v in values]
    total = sum(clean)
    if total == 0:
        clean = [0, 0, 0, 0, 0, 0, 0, 100]
    else:
        scaled = [v * 100 / total for v in clean]
        clean = [int(v) for v in scaled]
        remainder = 100 - sum(clean)
        order = sorted(range(len(scaled)), key=lambda i: scaled[i] - int(scaled[i]), reverse=True)
        for i in order[:remainder]:
            clean[i] += 1
    return clean


def fallback_formulas(mood: str):
    # Only used if model output is malformed; clearly marked as a deterministic fallback.
    palettes = {
        "serene": [[12, 4, 16, 5, 12, 18, 8, 25], [8, 4, 20, 8, 10, 18, 8, 24], [10, 5, 14, 7, 15, 20, 6, 23]],
        "mysterious": [[5, 2, 7, 4, 20, 24, 10, 28], [4, 2, 8, 5, 18, 25, 12, 26], [6, 3, 6, 4, 22, 22, 9, 26]],
        "romantic": [[8, 4, 12, 16, 10, 15, 12, 29], [6, 4, 10, 18, 12, 16, 14, 26], [9, 5, 11, 15, 12, 17, 10, 27]],
        "energetic": [[20, 12, 8, 6, 10, 10, 6, 36], [17, 12, 10, 8, 10, 12, 8, 31], [22, 10, 7, 7, 12, 12, 5, 33]],
        "nostalgic": [[7, 4, 8, 6, 17, 20, 14, 32], [6, 3, 10, 7, 16, 20, 16, 30], [8, 4, 7, 5, 20, 21, 12, 31]],
    }
    names = ["Echo of Light", "Velvet Resonance", "Scented Silence"]
    profiles = ["Citrus / Woody", "Aromatic / Woody", "Warm / Soft"]
    descs = ["Một hướng hương cân bằng lấy cảm hứng từ đặc trưng tín hiệu.", "Tập trung vào sắc thái mềm mại và độ tương phản.", "Một cách diễn giải ấm áp, có chiều sâu hơn."]
    return [{"letter": chr(65+i), "name": names[i], "values": normalize_percentages(palettes.get(mood, palettes["serene"])[i]),
             "profile": profiles[i], "desc": descs[i]} for i in range(3)]


def ask_model(features, metrics, mood, filename):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=503, detail="Máy chủ chưa cấu hình OPENAI_API_KEY. Hãy thêm API key trong biến môi trường của dịch vụ triển khai.")
    client = OpenAI(api_key=api_key)
    model = os.getenv("OPENAI_MODEL", "gpt-5-mini")
    prompt = f"""Bạn là trợ lý học tập về cảm hứng mùi hương. Tạo ba concept nước hoa lấy cảm hứng từ đặc trưng tín hiệu âm thanh được đo và cảm xúc người dùng chọn.
Tên tệp (chỉ là metadata, KHÔNG suy ra âm thanh chỉ từ tên): {filename}
Cảm xúc người dùng chọn: {MOODS.get(mood, MOODS['serene'])}
Đặc trưng đo được (JSON): {json.dumps(features, ensure_ascii=False)}
Các chỉ số hiển thị (JSON): {json.dumps(metrics, ensure_ascii=False)}
Danh sách nguyên liệu DUY NHẤT, đúng thứ tự: {json.dumps(INGREDIENTS, ensure_ascii=False)}
Trả về JSON thuần, không Markdown, theo cấu trúc:
{{"interpretation":"2-3 câu phân biệt chỉ số đo và diễn giải; không khẳng định chắc chắn cảm xúc/nhạc cụ/loại âm thanh nếu không được đo","mapping_explanation":"2-3 câu giải thích liên tưởng từ đặc trưng âm thanh và cảm xúc sang nhóm hương; nói rõ đây là sáng tạo chủ quan","formulas":[{{"letter":"A","name":"tên ngắn","profile":"nhóm hương","desc":"mô tả ngắn","values":[8 số nguyên theo thứ tự nguyên liệu] }}, ... B và C]}}
Mỗi công thức phải có 8 số không âm, tổng đúng 100. Ethanol là dung môi trong ý tưởng minh họa, không được xem các tỷ lệ là hướng dẫn pha chế an toàn. Không tuyên bố công thức đã kiểm nghiệm hoặc an toàn khi dùng trên da. Giữ câu trả lời bằng tiếng Việt, ngắn gọn, phù hợp dự án học tập."""
    try:
        response = client.responses.create(model=model, input=prompt, max_output_tokens=1400)
        raw = response.output_text.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        data = json.loads(raw)
        formulas = []
        for i, item in enumerate(data.get("formulas", [])[:3]):
            vals = normalize_percentages(item.get("values", []))
            formulas.append({"letter": chr(65+i), "name": str(item.get("name", f"Concept {chr(65+i)}"))[:80],
                             "profile": str(item.get("profile", "Concept"))[:80], "desc": str(item.get("desc", "Ý tưởng mùi hương minh họa."))[:400], "values": vals})
        if len(formulas) != 3:
            raise ValueError("Model did not return 3 formulas")
        return {"interpretation": str(data.get("interpretation", "Các chỉ số mô tả tín hiệu âm thanh; phần cảm xúc là diễn giải."))[:1000],
                "mapping_explanation": str(data.get("mapping_explanation", "Liên tưởng mùi hương mang tính sáng tạo."))[:1000], "formulas": formulas, "model": model}
    
    except HTTPException:
        raise
    except Exception as exc:
        import logging
        from openai import APIStatusError, APIConnectionError

        logger = logging.getLogger("uvicorn.error")

        if isinstance(exc, APIStatusError):
            logger.error(
                "OpenAI API error: status=%s, type=%s, message=%s, request_id=%s",
                exc.status_code,
                type(exc).__name__,
                str(exc),
                getattr(exc, "request_id", None),
            )
        elif isinstance(exc, APIConnectionError):
            logger.error(
                "OpenAI connection/timeout error: type=%s, message=%s",
                type(exc).__name__,
                str(exc),
            )
        else:
            logger.exception(
                "Sound-to-Scent AI response parsing/processing failed"
            )

        raise HTTPException(
            status_code=502,
            detail="Không thể tạo phản hồi AI hợp lệ. "
                   "Hãy kiểm tra log máy chủ để biết nguyên nhân."
        ) from exc



@app.post("/api/sound-to-scent")
async def sound_to_scent(file: UploadFile = File(...), mood: str = Form("serene")):
    suffix = Path(file.filename or "audio.wav").suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=415, detail="Định dạng chưa được hỗ trợ. Hãy dùng WAV, MP3, M4A, OGG, FLAC hoặc AAC.")
    content = await file.read(MAX_BYTES + 1)
    if len(content) > MAX_BYTES:
        raise HTTPException(status_code=413, detail="Tệp quá lớn. Giới hạn là 15 MB.")
    if not content:
        raise HTTPException(status_code=400, detail="Tệp âm thanh trống.")
    if mood not in MOODS:
        mood = "serene"
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(content)
            temp_path = tmp.name
        features, metrics = analyze_audio(temp_path)
        ai = ask_model(features, metrics, mood, Path(file.filename or "audio").name[:120])
        return {"filename": Path(file.filename or "audio").name, "mood_label": MOODS[mood], "audio_analysis": metrics,
                "interpretation": ai["interpretation"], "mapping_explanation": ai["mapping_explanation"],
                "formulas": ai["formulas"], "model": ai["model"], "note": "Liên tưởng mùi hương là ý tưởng sáng tạo, không phải phép đo khoa học về mùi."}
    finally:
        if temp_path:
            try:
                os.unlink(temp_path)
            except OSError:
                pass
