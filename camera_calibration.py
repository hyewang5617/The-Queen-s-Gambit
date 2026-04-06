import cv2 as cv
import numpy as np

# 체스보드 내부 코너 개수
CHECKERBOARD = (7, 5)

# 3D 좌표 (z=0 평면)
objp = np.zeros((CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

objpoints = []
imgpoints = []

cap = cv.VideoCapture("chessboard.mp4")  # 또는 0

if not cap.isOpened():
    print("동영상을 열 수 없습니다.")
    exit()

print("캘리브레이션 진행 중... (ESC 종료)")

gray = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    found, corners = cv.findChessboardCorners(
        gray,
        CHECKERBOARD,
        cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_NORMALIZE_IMAGE
    )

    if found:
        corners2 = cv.cornerSubPix(
            gray,
            corners,
            (11, 11),
            (-1, -1),
            (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        )

        objpoints.append(objp)
        imgpoints.append(corners2)

        cv.drawChessboardCorners(frame, CHECKERBOARD, corners2, found)

    cv.imshow("Calibration", frame)

    if cv.waitKey(30) & 0xFF == 27:
        break

cap.release()
cv.destroyAllWindows()

if len(objpoints) < 10:
    print("데이터 부족 (최소 10장 필요)")
    exit()

# 캘리브레이션
ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(
    objpoints, imgpoints, gray.shape[::-1], None, None
)

# ===== 결과 출력 =====
fx = mtx[0][0]
fy = mtx[1][1]
cx = mtx[0][2]
cy = mtx[1][2]

print("\n=== Camera Matrix ===")
print(mtx)

print("\n=== Intrinsic Parameters ===")
print(f"fx = {fx}")
print(f"fy = {fy}")
print(f"cx = {cx}")
print(f"cy = {cy}")

print("\n=== Distortion Coefficients ===")
print(dist)

print("\n=== RMSE ===")
print(ret)

# 저장
np.savez("calibration_data.npz", mtx=mtx, dist=dist)

print("\ncalibration_data.npz 저장 완료")