# U²-Net Pill Background Removal Model (for `rembg`)

이 프로젝트는 [MEDISEG 데이터셋](https://figshare.com/articles/dataset/MEDISEG/28574786?file=52926545)을 기반으로 학습된  
**알약 전용 배경 제거 모델**을 [`rembg`](https://github.com/danielgatis/rembg) 모듈에 적용하기 위해 파인튜닝한  
U²-Net → ONNX 모델입니다.

---

## 사용한 데이터셋

- **출처**: [MEDISEG (Figshare)](https://figshare.com/articles/dataset/MEDISEG/28574786?file=52926545)  
- **형태**: COCO-format annotations (`annotations.json`) 포함  
- **변환 방식**: 본 프로젝트에 포함된 `coco2binary.py` 스크립트를 사용하여  
  COCO-style polygon segmentation을 binary 마스크(.png)로 자동 변환  
- **라이선스**: CC BY 4.0 (데이터셋에 한함)

---

## 학습 방법

- 전체 데이터를 무작위로 7:3으로 분할하여 (`dataset_splitter.py`)  
  - 70% 학습용, 30% 테스트용으로 사용  
- 학습 모델: U²-Net (PyTorch 기반)  
- 학습이 완료된 `.pth` 모델을 ONNX 포맷으로 변환하여 `rembg`에서 사용 가능하도록 구성

---

## 실행 방법

- [모델 파일](https://drive.google.com/file/d/1inuIfO1hHVhKxUwxeA0lSy1etcYV_osY/view?usp=sharing) 을 다운로드하여 `models` 폴더에 비치
- main.py 실행

---

## 배경 제거 예시

![배경 제거 예시](README_images/rembg-example.png)

---
## 성능 평가

- 평가 지표: Intersection over Union (IoU), Dice coefficient
- 본 프로젝트의 `evaluate_u2net.py` 로 성능 평가 진행
- Iteration별 성능 지표 정리

|  Iteration | IoU       | Dice      |
|------------|-----------|-----------|
| 2000       | 0.5286    | 0.6058    |
| 4000       | 0.5855    | 0.6565    |
| 6000       | 0.6788    | 0.7435    |
| 8000       | 0.642     | 0.7099    |
| 10000      | 0.7229    | 0.7849    |
| 30000      | 0.7784    | 0.8377    |
| 74000      | 0.7950    | 0.8501    |
| 80000      | 0.7967    | 0.8533    |
| 90000      | 0.8019    | 0.8579    |
| 122000     | 0.8132    | 0.8679    |
| **136000**     | **0.8148**    | **0.8668**    |
| 148000     | 0.8149    | 0.8678    |

- 정량적 평가 이외에도 직접 촬영한 고해상도 이미지로 테스트하였을 때, Iteration 136000의 모델이 가장 적절하다고 판단되었고 이후로는 유의미한 성능 개선이 없고, 픽셀 깨짐 현상이 발견되어 학습을 중단하였음.

---

## 학습 환경

| 항목       | 사양                                               |
|------------|----------------------------------------------------|
| CPU        | AMD Ryzen Threadripper PRO 5975WX (32-Cores, 3.60GHz) |
| RAM        | 256GB                                              |
| GPU        | NVIDIA RTX A6000 × 2                               |
| PyTorch    | 2.4.1 (CUDA 11.8)                                  |
| 장소       | 동국대학교 인간-컴퓨터 상호작용 연구실 장비 사용 |

---

## 라이선스

- **본 프로젝트 코드 및 생성된 모델**: MIT License  
- **사용된 데이터셋 (MEDISEG)**: CC BY 4.0

데이터셋은 CC BY 4.0 라이선스를 따르므로, 해당 데이터로 생성된 결과물을 사용할 경우 반드시 출처를 명시해야 합니다.  
본 프로젝트의 코드와 학습된 모델(ONNX)은 MIT 라이선스로 자유롭게 사용 및 수정할 수 있습니다.


---

## 프로젝트 소속

이 프로젝트는 **동국대학교 2025-1 종합설계1 02분반 5팀**의  
**"알약 식별 프로그램 개발"** 과제의 일부로 수행되었습니다.

> 통합된 전체 프로젝트는 [여기에서 확인할 수 있습니다](https://github.com/illujun/Capstone-Design).
