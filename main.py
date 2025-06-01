from rembg import remove, new_session
from pathlib import Path
from PIL import Image
import io

# 모델 경로 설정
model_path = "models/u2net_pill_136000.onnx"
session = new_session(model_name='u2net_custom', model_path=model_path)

# 입력 파일 경로
input_path = Path("input_images/입력이미지.png")  # 원본 이미지 경로
output_path = input_path.with_name(f"{input_path.stem}_nobg.png")  # 저장 경로

# 이미지 읽기
with input_path.open("rb") as f:
    input_bytes = f.read()

# 배경 제거
output_bytes = remove(input_bytes, session=session)

# 저장
with open(output_path, "wb") as f:
    f.write(output_bytes)

print(f"✅ 배경 제거 이미지 저장 완료: {output_path}")
