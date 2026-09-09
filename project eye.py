import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import time
from collections import deque

# ---------- SETTINGS ----------
pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0

CAL_POINTS = [
    (0.1, 0.1), (0.5, 0.1), (0.9, 0.1),
    (0.1, 0.5), (0.5, 0.5), (0.9, 0.5),
    (0.1, 0.9), (0.5, 0.9), (0.9, 0.9)
]

CAL_FRAMES = 25
SMOOTHING_ALPHA = 0.15

# Blink
BLINK_THRESHOLD = 0.20
BLINK_FRAMES = 2

# Dwell (optional)
DWELL_TIME = 1.0
DWELL_RADIUS = 20

# ---------- INIT ----------
screen_w, screen_h = pyautogui.size()

mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Eye landmarks (for EAR)
LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]

# ---------- FUNCTIONS ----------
def landmarks_to_np(landmarks, w, h):
    return np.array([(int(p.x * w), int(p.y * h)) for p in landmarks])

def eye_roi_from_landmarks(img, lm_coords):
    x = min(lm_coords[:,0]); y = min(lm_coords[:,1])
    X = max(lm_coords[:,0]); Y = max(lm_coords[:,1])
    pad = 5
    x, y, X, Y = max(x-pad,0), max(y-pad,0), min(X+pad,img.shape[1]-1), min(Y+pad,img.shape[0]-1)
    return img[y:Y, x:X], (x, y, X, Y)

def pupil_center_from_eye(eye_img):
    gray = cv2.cvtColor(eye_img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    _, th = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    c = max(contours, key=cv2.contourArea)
    M = cv2.moments(c)
    if M['m00'] == 0:
        return None

    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])
    return (cx, cy)

def eye_aspect_ratio(eye):
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    return (A + B) / (2.0 * C)

def fit_affine(X, Y):
    ones = np.ones((X.shape[0], 1))
    A = np.hstack([X, ones])
    params, _, _, _ = np.linalg.lstsq(A, Y, rcond=None)
    return params

def apply_affine(params, x):
    x = np.append(x, 1.0)
    return x.dot(params)

# ---------- CALIBRATION ----------
def calibrate(cap):
    print("Look at points for calibration...")
    feats, scr = [], []

    for (fx, fy) in CAL_POINTS:
        px, py = int(fx * screen_w), int(fy * screen_h)
        frames = []

        while len(frames) < CAL_FRAMES:
            ret, frame = cap.read()
            if not ret:
                continue

            frame = cv2.resize(frame, (640, 480))
            h,w,_ = frame.shape

            disp = frame.copy()
            cv2.circle(disp, (int(fx*w), int(fy*h)), 10, (0,0,255), -1)
            cv2.imshow("Calibration", disp)

            res = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if res.multi_face_landmarks:
                lm = landmarks_to_np(res.multi_face_landmarks[0].landmark, w, h)
                eye_img, _ = eye_roi_from_landmarks(frame, lm[LEFT_EYE_IDX])
                pc = pupil_center_from_eye(eye_img)

                if pc:
                    cx, cy = pc
                    nx = cx / eye_img.shape[1]
                    ny = cy / eye_img.shape[0]
                    frames.append((nx, ny))

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        if frames:
            feats.append(np.mean(frames, axis=0))
            scr.append([px, py])

    if len(feats) < 3:
        return None

    return fit_affine(np.array(feats), np.array(scr))

# ---------- MAIN ----------
def main():
    cap = cv2.VideoCapture(0)
    params = calibrate(cap)

    if params is None:
        print("Calibration failed")
        return

    last_pos = np.array([screen_w//2, screen_h//2], dtype=float)
    pos_history = deque(maxlen=5)

    blink_counter = 0
    dwell_start = None

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.resize(frame, (640, 480))
        h,w,_ = frame.shape

        res = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        vis = frame.copy()

        if res.multi_face_landmarks:
            lm = landmarks_to_np(res.multi_face_landmarks[0].landmark, w, h)

            # -------- BLINK DETECTION --------
            leftEAR = eye_aspect_ratio(lm[LEFT_EYE_IDX])
            rightEAR = eye_aspect_ratio(lm[RIGHT_EYE_IDX])
            ear = (leftEAR + rightEAR) / 2

            if ear < BLINK_THRESHOLD:
                blink_counter += 1
            else:
                if blink_counter >= BLINK_FRAMES:
                    pyautogui.click()
                blink_counter = 0

            # -------- EYE TRACKING --------
            eye_img, (x,y,X,Y) = eye_roi_from_landmarks(frame, lm[LEFT_EYE_IDX])
            pc = pupil_center_from_eye(eye_img)

            if pc:
                cx, cy = pc
                norm = np.array([cx/eye_img.shape[1], cy/eye_img.shape[0]])

                scr = apply_affine(params, norm)
                scr[0] = np.clip(scr[0], 0, screen_w)
                scr[1] = np.clip(scr[1], 0, screen_h)

                last_pos = (1-SMOOTHING_ALPHA)*last_pos + SMOOTHING_ALPHA*scr
                pos_history.append(last_pos.copy())

                pyautogui.moveTo(int(last_pos[0]), int(last_pos[1]))

                # -------- DWELL CLICK --------
                if len(pos_history) == pos_history.maxlen:
                    avg = np.mean(pos_history, axis=0)
                    dist = np.linalg.norm(last_pos - avg)

                    if dist < DWELL_RADIUS:
                        if dwell_start is None:
                            dwell_start = time.time()
                        elif time.time() - dwell_start > DWELL_TIME:
                            pyautogui.click()
                            dwell_start = None
                    else:
                        dwell_start = None

                # visuals
                cv2.circle(vis, (x+cx, y+cy), 3, (0,255,0), -1)
                cv2.rectangle(vis, (x,y), (X,Y), (255,0,0), 1)

        cv2.imshow("Eye Mouse", vis)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# ---------- RUN ----------
if __name__ == "__main__":
    main()