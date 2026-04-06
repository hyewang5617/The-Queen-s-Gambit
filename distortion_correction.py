import cv2 as cv
import numpy as np

# calibration 데이터 불러오기
data = np.load("calibration_data.npz")
mtx = data["mtx"]
dist = data["dist"]

# 영상 또는 이미지
cap = cv.VideoCapture("chessboard_d.mp4")  # 또는 0

if not cap.isOpened():
    print("영상을 열 수 없습니다.")
    exit()

print("왜곡 보정 실행 중... (ESC 종료)")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]

    # 최적 카메라 매트릭스
    newcameramtx, roi = cv.getOptimalNewCameraMatrix(
        mtx, dist, (w, h), 1, (w, h)
    )

    # 왜곡 보정
    dst = cv.undistort(frame, mtx, dist, None, newcameramtx)

    # 비교 (왼쪽: 원본 / 오른쪽: 보정)
    dst = cv.resize(dst, (frame.shape[1], frame.shape[0]))
    combined = np.hstack((frame, dst))

    scale = 0.5  # 크기 줄이기 (0.3~0.7 사이 추천)
    combined = cv.resize(combined, None, fx=scale, fy=scale)

    cv.imshow("Original vs Undistorted", combined)

    if cv.waitKey(30) & 0xFF == 27:
        break

cap.release()
cv.destroyAllWindows()