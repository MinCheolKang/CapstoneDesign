# Smart Traffic Light System for Vulnerable Pedestrians

ESP32-CAM과 엣지 서버 간 통신을 통해 횡단보도 영역 내 보행 약자를 실시간 감지·분류하고, 보행 신호 시간을 동적으로 연장 제어하는 하드웨어-소프트웨어 통합 시스템입니다.

---

## 📌 프로젝트 개요

* **목적**: 
  * 횡단보도 진입 영역 내 교통약자(어린이, 노인, 휠체어, 유모차 등) 감지
  * 객체 및 연령 판별 결과에 따른 보행 신호 시간 동적 연장 제어
  * 시간대별/일별/월별 보행 데이터 자동 집계 및 통계 구축

---

## 🛠 시스템 아키텍처

1. **영상 수신 (ESP32-CAM)**: HTTP 프로토콜을 통해 실시간 스트리밍 제어 및 스냅샷 캡처
2. **이원화 AI 분석 파이프라인 (Edge Server)**:
   * **객체 탐지 (YOLOv5)**: 휠체어, 유모차, 지팡이 등 교통약자 보조기구 및 객체 인식
   * **연령 분류 (OpenCV + TensorFlow/Keras)**: Haar Cascade 안면 검출 및 4단계 연령대(`kid`, `youth`, `midlife`, `old`) 분류
3. **하드웨어 제어 (Arduino)**: PySerial 기반 시리얼 통신으로 산출된 보행 시간 데이터를 전송받아 신호등 LED 점등 및 점멸 제어
4. **데이터 파이프라인 (Pandas)**: 2시간 단위, 일별, 월별(윤년 대응) 누적 통계 CSV 생성

---

## ⚙️ 주요 기능

### 1. 이원화 딥러닝 모델 파이프라인
* **교통약자 보조기구 탐지**: YOLOv5 기반 커스텀 가중치(`best.pt`) 적용, 바운딩 박스 및 신뢰도 점수 산출
* **안면 기반 연령대 분류**: Haar Cascade 검출 영역을 128×128 크기로 리사이징 후 Keras 분류 모델 추론

### 2. 동적 신호 시간 제어
* 보행 약자 객체 또는 취약 연령대(Kid/Old) 감지 시 기본 신호 시간(10초)에 추가 연장 시간 가산
* 시리얼 통신을 통해 아두이노로 가산된 초 단위 데이터를 패킷으로 전달

### 3. 신호등 펌웨어 (Arduino C/C++)
* 수신된 `seconds` 값에 맞춰 초록불 유지
* 신호 종료 10초 전부터 500ms 주기의 점멸 경고 로직 동작 후 빨간불 전환

### 4. 통계 로깅 및 배치 처리
* 매일 자정(`00:00:00`) 전일 2시간 단위 통계 집계 (`daily/prediction_YYYY-MM-DD.csv`)
* 매월 1일 자정 전달의 일별 통계를 취합하여 월간 리포트 파일 생성 (`monthly/prediction_YYYY-MM.csv`)

---

## 💻 실행 환경 및 구성

### 하드웨어
* Arduino Uno / Nano
* ESP32-CAM
* 신호등 제어용 LED 회로 (Green/Red)

### 소프트웨어 요구사항
```bash
pip install numpy pandas tensorflow opencv-python pyserial requests
```

### 실행 방법

1. Arduino에 `traffic_light_controller.ino` 펌웨어 업로드
2. ESP32-CAM 웹서버 구동 및 IP 주소 설정 (`url = 'http://<ESP32_IP>/'`)
3. 파이썬 관제 스크립트 실행:

```bash
python main.py
```