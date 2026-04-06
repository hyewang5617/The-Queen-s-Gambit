import cv2 as cv
import numpy as np

# 체스보드 내부 코너 개수 (너 기준)
CHECKERBOARD = (7, 5)

# 3D 좌표 생성 (z=0 평면)
objp = np.zeros((CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

objpoints = []  # 3D points
imgpoints = []  # 2D points

cap = cv.VideoCapture("chessboard.mp4")

if not cap.isOpened():
    print("동영상을 열 수 없습니다.")
    exit()

print("캘리브레이션 진행 중... (ESC 종료)")

image_size = None

while True:
    ret, frame = cap.read()

    # 영상 끝 처리 (중요)
    if not ret or frame is None:
        print("영상 끝")
        break

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # 이미지 크기 저장 (한 번만)
    if image_size is None:
        image_size = gray.shape[::-1]

    # 체스보드 코너 찾기
    found, corners = cv.findChessboardCorners(
        gray,
        CHECKERBOARD,
        cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_NORMALIZE_IMAGE
    )

    if found and len(objpoints) < 30:
        corners2 = cv.cornerSubPix(
            gray,
            corners,
            (11, 11),
            (-1, -1),
            (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    )

        objpoints.append(objp.copy())
        imgpoints.append(corners2)

        # 화면에 표시
        cv.drawChessboardCorners(frame, CHECKERBOARD, corners2, found)

    cv.imshow("Calibration", frame)

    # ESC 종료
    if cv.waitKey(30) & 0xFF == 27:
        print("ESC 종료")
        break

cap.release()
cv.destroyAllWindows()

print("\n=== 수집 결과 ===")
print("objpoints 개수:", len(objpoints))
print("imgpoints 개수:", len(imgpoints))

# 데이터 부족 체크
if len(objpoints) < 10:
    print("데이터 부족 (최소 10장 필요)")
    exit()

if image_size is None:
    print("image_size 없음")
    exit()

print("\n캘리브레이션 시작...")

# 캘리브레이션 수행
ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(
    objpoints,
    imgpoints,
    image_size,
    None,
    None
)

print("캘리브레이션 완료!")

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

# 결과 저장
np.savez("calibration_data.npz", mtx=mtx, dist=dist)

print("\ncalibration_data.npz 저장 완료")