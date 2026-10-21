import json
import os
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from openai import OpenAI

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = FastAPI(title="AI Perfume Lab API", version="1.0.0")
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")

INGREDIENTS = [
    "Bergamot FCF", "Cam ngọt", "Oải hương", "Phong lữ",
    "Hương gỗ tuyết tùng", "Hương gỗ đàn hương", "Hương vani",
    "Ethanol (dung môi)"
]

class PromptRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=5000)
    context: str = ""
    emotion: str = ""
    baseFormula: dict[str, Any] | None = None
    baseName: str = "Công thức mới"

def get_client() -> OpenAI:
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise HTTPException(status_code=503, detail="Máy chủ chưa cấu hình OPENAI_API_KEY.")
    return OpenAI(api_key=key)

def ask_json(instructions: str, payload: dict[str, Any]) -> dict[str, Any]:
    client = get_client()
    model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
    try:
        response = client.responses.create(
            model=model,
            instructions=instructions + "\nTrả về JSON hợp lệ, không markdown. Nội dung bằng tiếng Việt. Không khẳng định công thức đã an toàn để dùng trên da.",
            input=json.dumps(payload, ensure_ascii=False),
        )
        text = response.output_text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        data = json.loads(text)
        return data
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"AI chưa thể xử lý yêu cầu: {str(exc)[:300]}") from exc

def normalize_formula(formula: dict[str, Any]) -> dict[str, Any]:
    values = formula.get("ingredients", {})
    clean = {}
    for name in INGREDIENTS:
        try:
            value = max(0.0, float(values.get(name, 0)))
        except (TypeError, ValueError):
            value = 0.0
        clean[name] = value
    total = sum(clean.values())
    if total <= 0:
        clean = {name: 0.0 for name in INGREDIENTS}
        clean["Ethanol (dung môi)"] = 100.0
    else:
        # Normalize to 100.0 while retaining one decimal place.
        clean = {name: round(value * 100 / total, 1) for name, value in clean.items()}
        diff = round(100.0 - sum(clean.values()), 1)
        clean["Ethanol (dung môi)"] = round(clean["Ethanol (dung môi)"] + diff, 1)
    formula["ingredients"] = clean
    return formula

FORMULA_RULES = """Bạn là trợ lý giáo dục về ý tưởng nước hoa. Tạo đúng 3 công thức ý tưởng khác nhau, có thể so sánh.
Mỗi công thức phải có name, description (ngắn), profile, style, notes (mảng chuỗi), ingredients (object chứa chính xác các tên nguyên liệu sau với giá trị số):
Bergamot FCF, Cam ngọt, Oải hương, Phong lữ, Hương gỗ tuyết tùng, Hương gỗ đàn hương, Hương vani, Ethanol (dung môi).
Mỗi công thức phải có đủ cả 8 nguyên liệu, giá trị không âm. Tổng tỷ lệ mỗi công thức là 100. Tỷ lệ chỉ mang tính minh họa học tập; không đưa ra hướng dẫn sử dụng trên da hoặc tuyên bố an toàn. Dùng ethanol làm dung môi trong mô hình ý tưởng nhưng tránh khẳng định đây là công thức pha chế thực tế đã kiểm nghiệm.
Định dạng JSON: {"formulas":[{...},{...},{...}]}"""

@app.get("/")
def home():
    return FileResponse(os.path.join(BASE_DIR, "static", "index.html"))

@app.get("/health")
def health():
    return {"status": "ok", "service": "AI Perfume Lab"}

@app.post("/api/formula")
def formula(req: PromptRequest):
    data = ask_json(FORMULA_RULES, {"description": req.prompt})
    formulas = data.get("formulas", [])
    if len(formulas) != 3:
        raise HTTPException(status_code=502, detail="AI không trả về đủ ba công thức. Vui lòng thử lại.")
    return {"tool": "formula", "formulas": [normalize_formula(f) for f in formulas]}

@app.post("/api/memory")
def memory(req: PromptRequest):
    instructions = """Phân tích ký ức thành ý tưởng nước hoa. Trước hết tạo memory_analysis gồm emotions, context, details, translation; sau đó tạo đúng 3 công thức khác nhau.
Mỗi công thức có name, description, profile, style, notes (mảng), ingredients object với chính xác các khóa: Bergamot FCF, Cam ngọt, Oải hương, Phong lữ, Hương gỗ tuyết tùng, Hương gỗ đàn hương, Hương vani, Ethanol (dung môi). Giá trị số không âm, tổng 100. Đây là tỷ lệ ý tưởng, không phải hướng dẫn pha chế đã xác nhận an toàn.
Trả JSON: {"memory_analysis":{"emotions":"","context":"","details":"","translation":""},"formulas":[{...},{...},{...}]}"""
    data = ask_json(instructions, {"experience": req.prompt, "context": req.context, "emotion": req.emotion})
    formulas = data.get("formulas", [])
    if len(formulas) != 3:
        raise HTTPException(status_code=502, detail="AI không trả về đủ ba công thức. Vui lòng thử lại.")
    data["formulas"] = [normalize_formula(f) for f in formulas]
    data["tool"] = "memory"
    return data

@app.post("/api/qa")
def qa(req: PromptRequest):
    instructions = """Trả lời câu hỏi về khoa học nước hoa bằng tiếng Việt, ngắn gọn nhưng đủ dễ hiểu cho học sinh phổ thông. Phân biệt nguyên lý chung với điều chưa thể kết luận khi thiếu dữ liệu. Không bịa dữ kiện, không đưa ra tuyên bố an toàn da. JSON: {"title":"", "answer":"", "checks":["...","..."]}. Giữ answer khoảng 2-4 đoạn ngắn, checks tối đa 3 mục."""
    data = ask_json(instructions, {"question": req.prompt})
    data["tool"] = "qa"
    return data

@app.post("/api/whatif")
def whatif(req: PromptRequest):
    instructions = """Mô phỏng giả thuyết về mùi hương một cách định tính. Trả về JSON:
{"before":{"profile":"mô tả hồ sơ mùi trước","formula":{"name":"Công thức trước","ingredients":{"Bergamot FCF":0,"Cam ngọt":0,"Oải hương":0,"Phong lữ":0,"Hương gỗ tuyết tùng":0,"Hương gỗ đàn hương":0,"Hương vani":0,"Ethanol (dung môi)":100}}},"after":{"profile":"mô tả hồ sơ mùi sau","formula":{"name":"Công thức sau","ingredients":{...}}},"analysis":"giải thích thay đổi dự kiến","limitations":"nêu giới hạn của dự đoán"}.
Cả hai công thức phải có đúng 8 nguyên liệu với giá trị số không âm, tổng tỷ lệ 100. Nếu có công thức cơ sở, hãy dùng làm before và giữ gần với tỷ lệ cơ sở; nếu không, hãy tạo một công thức cơ sở phù hợp với giả thuyết. after cần thể hiện giả thuyết. Đây chỉ là mô hình ý tưởng, không khẳng định kết quả cảm quan hoặc an toàn thực tế."""
    data = ask_json(instructions, {"hypothesis": req.prompt, "base_formula": req.baseFormula, "base_name": req.baseName})
    for key in ("before", "after"):
        if isinstance(data.get(key), dict) and isinstance(data[key].get("formula"), dict):
            data[key]["formula"] = normalize_formula(data[key]["formula"])
    data["tool"] = "whatif"
    return data
