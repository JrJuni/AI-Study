import cv2
import time
from ultralytics import YOLO

# 1. YOLOv8 모델 로드 (사전 학습된 모델 사용)
model = YOLO("yolov8n.pt")  # 'n'은 nano 모델, 다른 크기 사용 가능 (예: yolov8s.pt, yolov8m.pt 등)

# 노트북 내장 카메라 열기 (일반적으로 0번 카메라)
cap = cv2.VideoCapture(0)

# 카메라 설정 확인
if not cap.isOpened():
    print("카메라를 열 수 없습니다. 카메라가 연결되어 있는지 확인해주세요.")
    exit()
    
# FPS 측정을 위한 초기화
frame_count = 0
start_time = time.time()

while True:
    # 프레임 읽기
    ret, frame = cap.read()
    
    # 프레임 읽기 실패 시 종료
    if not ret:
        print("프레임을 읽을 수 없습니다.")
        break
    
    # YOLOv8로 객체 탐지 수행
    # YOLOv8로 객체 탐지 수행 (시작 시간 기록)
    inference_start = time.time()
    results = model(frame, conf=0.3)  # 결과는 Results 객체 또는 리스트로 반환됨
    inference_end = time.time()
    
    # Results 객체에서 시각화된 프레임 가져오기
    for result in results:
        annotated_frame = result.plot()  # 각 Results 객체의 plot() 메서드 호출
        
    # FPS 계산 (현재 프레임 처리 시간 포함)
    frame_count += 1
    elapsed_time = time.time() - start_time
    fps = frame_count / elapsed_time
    
    short_elapsed = inference_end - inference_start
    short_fps = 1 / short_elapsed
    
        # FPS 표시
    cv2.putText(
        annotated_frame,
        f"FPS: {fps:.2f}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    
            # Short FPS 표시
    cv2.putText(
        annotated_frame,
        f"FPS: {short_fps:.2f}",
        (10, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 0),
        2,
        cv2.LINE_AA,
    )
    
    # 결과 프레임 표시
    cv2.imshow('YOLOv8 Object Detection', annotated_frame)
    
    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 리소스 해제
cap.release()
cv2.destroyAllWindows()

'''
# 2. 객체 탐지 수행
results = model("path/to/your/image.jpg")  # 입력 이미지 경로

# 3. 결과 시각화
results.show()  # 탐지 결과를 이미지에 표시하고 출력

for box in results.boxes:
    print(f"Class: {box.cls}, Confidence: {box.conf}, Coordinates: {box.xyxy}")

# 4. 결과 저장 (선택 사항)
results.save("path/to/save/results/")  # 결과를 파일로 저장
'''