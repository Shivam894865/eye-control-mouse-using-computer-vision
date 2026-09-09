import time
from collections import deque

import av
import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
from streamlit_webrtc import VideoProcessorBase, webrtc_streamer


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Eye-Controlled Mouse",
    page_icon="👁️",
    layout="wide"
)


# ============================================================
# SETTINGS
# ============================================================

CAL_POINTS = [
    (0.1, 0.1),
    (0.5, 0.1),
    (0.9, 0.1),
    (0.1, 0.5),
    (0.5, 0.5),
    (0.9, 0.5),
    (0.1, 0.9),
    (0.5, 0.9),
    (0.9, 0.9),
]

CAL_FRAMES = 25

SMOOTHING_ALPHA = 0.15

BLINK_THRESHOLD = 0.20
BLINK_FRAMES = 2

DWELL_TIME = 1.0
DWELL_RADIUS = 0.04


# ============================================================
# MEDIAPIPE
# ============================================================

mp_face = mp.solutions.face_mesh

LEFT_EYE_IDX = [
    33, 160, 158, 133, 153, 144
]

RIGHT_EYE_IDX = [
    362, 385, 387, 263, 373, 380
]


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def landmarks_to_np(landmarks, w, h):
    return np.array(
        [
            (int(p.x * w), int(p.y * h))
            for p in landmarks
        ],
        dtype=np.int32
    )


def eye_roi_from_landmarks(img, lm_coords):

    x = np.min(lm_coords[:, 0])
    y = np.min(lm_coords[:, 1])

    X = np.max(lm_coords[:, 0])
    Y = np.max(lm_coords[:, 1])

    pad = 5

    x = max(x - pad, 0)
    y = max(y - pad, 0)

    X = min(
        X + pad,
        img.shape[1] - 1
    )

    Y = min(
        Y + pad,
        img.shape[0] - 1
    )

    roi = img[y:Y, x:X]

    return roi, (x, y, X, Y)


def pupil_center_from_eye(eye_img):

    if eye_img is None:
        return None

    if eye_img.size == 0:
        return None

    gray = cv2.cvtColor(
        eye_img,
        cv2.COLOR_BGR2GRAY
    )

    gray = cv2.equalizeHist(gray)

    _, th = cv2.threshold(
        gray,
        50,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (3, 3)
    )

    th = cv2.morphologyEx(
        th,
        cv2.MORPH_OPEN,
        kernel
    )

    contours, _ = cv2.findContours(
        th,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return None

    # Ignore extremely small contours
    contours = [
        c for c in contours
        if cv2.contourArea(c) >= 2
    ]

    if not contours:
        return None

    c = max(
        contours,
        key=cv2.contourArea
    )

    M = cv2.moments(c)

    if M["m00"] == 0:
        return None

    cx = int(
        M["m10"] / M["m00"]
    )

    cy = int(
        M["m01"] / M["m00"]
    )

    return cx, cy


def eye_aspect_ratio(eye):

    A = np.linalg.norm(
        eye[1] - eye[5]
    )

    B = np.linalg.norm(
        eye[2] - eye[4]
    )

    C = np.linalg.norm(
        eye[0] - eye[3]
    )

    if C == 0:
        return 0

    return (A + B) / (2.0 * C)


def fit_affine(X, Y):

    ones = np.ones(
        (X.shape[0], 1)
    )

    A = np.hstack(
        [X, ones]
    )

    params, _, _, _ = np.linalg.lstsq(
        A,
        Y,
        rcond=None
    )

    return params


def apply_affine(params, x):

    x = np.append(
        x,
        1.0
    )

    return x.dot(params)


# ============================================================
# STREAMLIT VIDEO PROCESSOR
# ============================================================

class EyeTracker(VideoProcessorBase):

    def __init__(self):

        self.face_mesh = mp_face.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        self.params = None

        self.calibration_features = []
        self.calibration_screen = []

        self.calibration_index = 0
        self.calibration_frames = []

        self.calibrated = False

        self.last_pos = np.array(
            [0.5, 0.5],
            dtype=float
        )

        self.pos_history = deque(
            maxlen=5
        )

        self.blink_counter = 0

        self.dwell_start = None

        self.click_event = False

        self.gaze_x = 0.5
        self.gaze_y = 0.5

        self.status = "Waiting..."

    # --------------------------------------------------------
    # Reset
    # --------------------------------------------------------

    def reset_calibration(self):

        self.params = None

        self.calibration_features = []
        self.calibration_screen = []

        self.calibration_index = 0
        self.calibration_frames = []

        self.calibrated = False

        self.last_pos = np.array(
            [0.5, 0.5],
            dtype=float
        )

        self.pos_history.clear()

        self.status = "Calibration reset"

    # --------------------------------------------------------
    # Process frame
    # --------------------------------------------------------

    def recv(self, frame):

        img = frame.to_ndarray(
            format="bgr24"
        )

        img = cv2.resize(
            img,
            (640, 480)
        )

        h, w, _ = img.shape

        rgb = cv2.cvtColor(
            img,
            cv2.COLOR_BGR2RGB
        )

        result = self.face_mesh.process(rgb)

        output = img.copy()

        self.click_event = False

        # ----------------------------------------------------
        # FACE DETECTED
        # ----------------------------------------------------

        if result.multi_face_landmarks:

            landmarks = landmarks_to_np(
                result.multi_face_landmarks[0].landmark,
                w,
                h
            )

            # =================================================
            # CALIBRATION
            # =================================================

            if not self.calibrated:

                fx, fy = CAL_POINTS[
                    self.calibration_index
                ]

                target_x = int(
                    fx * w
                )

                target_y = int(
                    fy * h
                )

                cv2.circle(
                    output,
                    (target_x, target_y),
                    15,
                    (0, 0, 255),
                    -1
                )

                cv2.putText(
                    output,
                    f"Look at point "
                    f"{self.calibration_index + 1}/"
                    f"{len(CAL_POINTS)}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2
                )

                eye_img, _ = eye_roi_from_landmarks(
                    img,
                    landmarks[LEFT_EYE_IDX]
                )

                pupil = pupil_center_from_eye(
                    eye_img
                )

                if pupil is not None:

                    cx, cy = pupil

                    if eye_img.shape[1] > 0 and eye_img.shape[0] > 0:

                        nx = (
                            cx /
                            eye_img.shape[1]
                        )

                        ny = (
                            cy /
                            eye_img.shape[0]
                        )

                        self.calibration_frames.append(
                            (nx, ny)
                        )

                # Collect enough frames
                if len(self.calibration_frames) >= CAL_FRAMES:

                    if self.calibration_frames:

                        mean_feature = np.mean(
                            self.calibration_frames,
                            axis=0
                        )

                        self.calibration_features.append(
                            mean_feature
                        )

                        self.calibration_screen.append(
                            [fx, fy]
                        )

                    self.calibration_frames = []

                    self.calibration_index += 1

                    if self.calibration_index >= len(CAL_POINTS):

                        if len(
                            self.calibration_features
                        ) >= 3:

                            self.params = fit_affine(
                                np.array(
                                    self.calibration_features
                                ),
                                np.array(
                                    self.calibration_screen
                                )
                            )

                            self.calibrated = True

                            self.status = (
                                "Calibration complete"
                            )

                        else:

                            self.status = (
                                "Calibration failed"
                            )

            # =================================================
            # TRACKING
            # =================================================

            else:

                # ---------------------------------------------
                # BLINK DETECTION
                # ---------------------------------------------

                left_ear = eye_aspect_ratio(
                    landmarks[LEFT_EYE_IDX]
                )

                right_ear = eye_aspect_ratio(
                    landmarks[RIGHT_EYE_IDX]
                )

                ear = (
                    left_ear +
                    right_ear
                ) / 2.0

                if ear < BLINK_THRESHOLD:

                    self.blink_counter += 1

                else:

                    if (
                        self.blink_counter
                        >= BLINK_FRAMES
                    ):

                        self.click_event = True

                    self.blink_counter = 0

                # ---------------------------------------------
                # PUPIL TRACKING
                # ---------------------------------------------

                eye_img, bbox = eye_roi_from_landmarks(
                    img,
                    landmarks[LEFT_EYE_IDX]
                )

                pupil = pupil_center_from_eye(
                    eye_img
                )

                if pupil is not None:

                    cx, cy = pupil

                    if (
                        eye_img.shape[1] > 0
                        and eye_img.shape[0] > 0
                    ):

                        normalized = np.array(
                            [
                                cx /
                                eye_img.shape[1],

                                cy /
                                eye_img.shape[0]
                            ]
                        )

                        gaze = apply_affine(
                            self.params,
                            normalized
                        )

                        gaze[0] = np.clip(
                            gaze[0],
                            0,
                            1
                        )

                        gaze[1] = np.clip(
                            gaze[1],
                            0,
                            1
                        )

                        self.last_pos = (
                            (1 - SMOOTHING_ALPHA)
                            * self.last_pos
                            +
                            SMOOTHING_ALPHA
                            * gaze
                        )

                        self.pos_history.append(
                            self.last_pos.copy()
                        )

                        self.gaze_x = float(
                            self.last_pos[0]
                        )

                        self.gaze_y = float(
                            self.last_pos[1]
                        )

                        # -------------------------------------
                        # DWELL CLICK
                        # -------------------------------------

                        if len(
                            self.pos_history
                        ) == self.pos_history.maxlen:

                            avg = np.mean(
                                self.pos_history,
                                axis=0
                            )

                            distance = np.linalg.norm(
                                self.last_pos -
                                avg
                            )

                            if distance < DWELL_RADIUS:

                                if self.dwell_start is None:

                                    self.dwell_start = (
                                        time.time()
                                    )

                                elif (
                                    time.time()
                                    -
                                    self.dwell_start
                                    >
                                    DWELL_TIME
                                ):

                                    self.click_event = True

                                    self.dwell_start = None

                            else:

                                self.dwell_start = None

                        # -------------------------------------
                        # DRAW EYE ROI
                        # -------------------------------------

                        x, y, X, Y = bbox

                        cv2.rectangle(
                            output,
                            (x, y),
                            (X, Y),
                            (255, 0, 0),
                            1
                        )

                        cv2.circle(
                            output,
                            (x + cx, y + cy),
                            4,
                            (0, 255, 0),
                            -1
                        )

                # ---------------------------------------------
                # VIRTUAL CURSOR
                # ---------------------------------------------

                cursor_x = int(
                    self.gaze_x * w
                )

                cursor_y = int(
                    self.gaze_y * h
                )

                cv2.circle(
                    output,
                    (cursor_x, cursor_y),
                    12,
                    (0, 255, 255),
                    2
                )

                cv2.circle(
                    output,
                    (cursor_x, cursor_y),
                    4,
                    (0, 255, 255),
                    -1
                )

                cv2.putText(
                    output,
                    "Eye Cursor",
                    (
                        max(cursor_x - 40, 5),
                        max(cursor_y - 18, 20)
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 255),
                    1
                )

                # ---------------------------------------------
                # CLICK INDICATOR
                # ---------------------------------------------

                if self.click_event:

                    cv2.putText(
                        output,
                        "CLICK",
                        (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (0, 255, 0),
                        3
                    )

        else:

            cv2.putText(
                output,
                "Face not detected",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )

        return av.VideoFrame.from_ndarray(
            output,
            format="bgr24"
        )


# ============================================================
# SESSION STATE
# ============================================================

if "running" not in st.session_state:
    st.session_state.running = False


# ============================================================
# UI
# ============================================================

st.title("👁️ Eye-Controlled Mouse")

st.markdown(
    """
    ### Computer Vision Based Eye-Controlled Interface

    Control the **virtual cursor** using your eye movement.
    Blinking and dwelling can be used as click actions.
    """
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ Settings")

    st.write(
        "Calibration uses 9 points."
    )

    st.write(
        f"Frames per point: {CAL_FRAMES}"
    )

    st.write(
        f"Smoothing: {SMOOTHING_ALPHA}"
    )

    st.write(
        f"Blink threshold: {BLINK_THRESHOLD}"
    )

    st.write(
        f"Dwell time: {DWELL_TIME}s"
    )

    st.info(
        "Allow camera permission when "
        "your browser asks."
    )


# ============================================================
# INSTRUCTIONS
# ============================================================

st.subheader("📋 How to use")

st.markdown(
    """
    1. Click **START**.
    2. Allow webcam access.
    3. Look at each red calibration point.
    4. After calibration, move your eyes.
    5. The yellow circle represents the virtual cursor.
    6. Blink or dwell on a position to generate a click.
    """
)


# ============================================================
# CAMERA
# ============================================================

ctx = webrtc_streamer(
    key="eye-control",
    video_processor_factory=EyeTracker,
    media_stream_constraints={
        "video": True,
        "audio": False,
    },
    async_processing=True,
)


# ============================================================
# STATUS
# ============================================================

if ctx.video_processor:

    processor = ctx.video_processor

    st.divider()

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Gaze X",
            f"{processor.gaze_x:.2f}"
        )

    with col2:

        st.metric(
            "Gaze Y",
            f"{processor.gaze_y:.2f}"
        )

    with col3:

        if processor.calibrated:

            st.success(
                "Calibrated"
            )

        else:

            st.warning(
                "Calibrating..."
            )


# ============================================================
# VIRTUAL SCREEN
# ============================================================

st.divider()

st.subheader("🖱️ Virtual Eye Cursor")

st.markdown(
    """
    The area below represents the browser screen.
    The cursor position is controlled by your eye movement.
    """
)

if ctx.video_processor:

    processor = ctx.video_processor

    x_percent = int(
        processor.gaze_x * 100
    )

    y_percent = int(
        processor.gaze_y * 100
    )

    st.markdown(
        f"""
        <div style="
            position: relative;
            width: 100%;
            height: 400px;
            border: 3px solid #777;
            border-radius: 15px;
            background: #f5f5f5;
            overflow: hidden;
        ">

            <div style="
                position: absolute;
                left: {x_percent}%;
                top: {y_percent}%;
                transform: translate(-50%, -50%);
                width: 25px;
                height: 25px;
                border-radius: 50%;
                border: 4px solid red;
                background: yellow;
                box-shadow: 0 0 15px rgba(255,0,0,0.5);
            "></div>

        </div>

        <p style="text-align:center;">
            Cursor position:
            <b>({x_percent}%, {y_percent}%)</b>
        </p>
        """,
        unsafe_allow_html=True
    )

else:

    st.info(
        "Start the camera to activate "
        "the virtual cursor."
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Eye-Controlled Mouse using "
    "Computer Vision | "
    "OpenCV + MediaPipe + Streamlit"
)
