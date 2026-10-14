import cv2
import numpy as np
import mediapipe as mp

LIP_LANDMARKS = [
    61,146,91,181,84,17,314,405,321,375,291,
    185,40,39,37,0,267,269,270,409,
    78,95,88,178,87,14,317,402,318,324,308
]

def crop_lip_region(rgb_image, detection_result, padding=12, size=(96,96)):
    if not detection_result.face_landmarks:
        return None

    face_landmarks = detection_result.face_landmarks[0]
    h, w, _ = rgb_image.shape

    xs = []
    ys = []

    for idx in LIP_LANDMARKS:
        lm = face_landmarks[idx]
        xs.append(int(lm.x * w))
        ys.append(int(lm.y * h))

    x_min = max(min(xs) - padding, 0)
    x_max = min(max(xs) + padding, w)

    y_min = max(min(ys) - padding, 0)
    y_max = min(max(ys) + padding, h)

    lip_crop = rgb_image[y_min:y_max, x_min:x_max]

    if lip_crop.size == 0:
        return None

    lip_crop = cv2.resize(lip_crop, size)

    return lip_crop


def check_and_correct_orientation(frame, detector):
    """
    Checks frame orientation using mediapipe.
    If the face is detected sideways (e.g. eyes are vertically aligned instead of horizontally), 
    rotates the frame. Returns the corrected frame.
    """
    # Quick low-res check for speed
    small_frame = cv2.resize(frame, (320, int(320 * frame.shape[0] / frame.shape[1])))
    frame_rgb = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    
    detection_result = detector.detect(mp_image)
    
    if not detection_result.face_landmarks:
        # If no face is found normally, let's try rotating it 90 deg clockwise to see if a face appears
        rot_frame90 = cv2.rotate(small_frame, cv2.ROTATE_90_CLOCKWISE)
        mp_image90 = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(rot_frame90, cv2.COLOR_BGR2RGB))
        res90 = detector.detect(mp_image90)
        
        if res90.face_landmarks:
            return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE), True
            
        # Try counter-clockwise just in case
        rot_frame270 = cv2.rotate(small_frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        mp_image270 = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(rot_frame270, cv2.COLOR_BGR2RGB))
        res270 = detector.detect(mp_image270)
        
        if res270.face_landmarks:
            return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE), True
            
    return frame, False


def extract_lip_frames(video_path, detector, correct_orientation=False):
    """
    Extracts frames from a video file and crops the lip region dynamically
    using the provided MediaPipe Tasks API detector.
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    prev_crop = None
    
    needs_rotation = False
    rotation_type = None
    first_frame_checked = False

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        if correct_orientation and not first_frame_checked:
            # Check orientation only on the first frame to save processing time
            corrected_frame, did_rotate = check_and_correct_orientation(frame, detector)
            if did_rotate:
                needs_rotation = True
                # Determine which rotation was applied by checking shapes or just applying the successful one
                # For simplicity here, we assume cameras usually need 90_CLOCKWISE or 90_COUNTERCLOCKWISE.
                # check_and_correct_orientation logic rotated finding a face
                # We'll just run it again frame-by-frame optimally below
            first_frame_checked = True

        if correct_orientation and needs_rotation:
             # It's an rotated iPhone video (face wasn't detectable unless rotated)
             # Let's rotate it back based on basic heuristics or just blindly 90 clockwise assuming portrait
             # Since check_and_correct_orientation returned true, let's just do clockwise for now
             frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Convert to mp.Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        
        detection_result = detector.detect(mp_image)
        
        crop = crop_lip_region(frame_rgb, detection_result, padding=12, size=(96,96))
        
        if crop is not None:
            frames.append(crop)
            prev_crop = crop
        elif needs_rotation and crop is None and correct_orientation:
            # Try counter clockwise if clockwise failed
             frame_ccw = cv2.rotate(frame, cv2.ROTATE_180) # 90CW + 180 = 90CCW
             frame_rgb_ccw = cv2.cvtColor(frame_ccw, cv2.COLOR_BGR2RGB)
             mp_image_ccw = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb_ccw)
             res = detector.detect(mp_image_ccw)
             crop = crop_lip_region(frame_rgb_ccw, res, padding=12, size=(96,96))
             if crop is not None:
                 frames.append(crop)
                 prev_crop = crop
             else:
                 if prev_crop is not None:
                    frames.append(prev_crop)
        else:
            # Fallback to previous crop if tracked face drops for a frame
            if prev_crop is not None:
                frames.append(prev_crop)

    cap.release()
    
    return frames
