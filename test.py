import cv2
import numpy as np

def extract_name_from_file(file_path):
    last_slash_pos = file_path.rfind('/')
    second_last_slash_pos = file_path.rfind('/', 0, last_slash_pos - 1)
    directory_name = file_path[second_last_slash_pos + 1:last_slash_pos]
    return directory_name

def calc_iou(bbox1, bbox2):
    x1_box1, y1_box1, w_box1, h_box1 = bbox1
    x1_box2, y1_box2, w_box2, h_box2 = bbox2
    
    x2_box1, y2_box1 = x1_box1 + w_box1, y1_box1 + h_box1
    x2_box2, y2_box2 = x1_box2 + w_box2, y1_box2 + h_box2
    
    intersection_width = max(0, min(x2_box1, x2_box2) - max(x1_box1, x1_box2))
    intersection_height = max(0, min(y2_box1, y2_box2) - max(y1_box1, y1_box2))
    
    intersection_area = intersection_width * intersection_height
    area_box1 = w_box1 * h_box1
    area_box2 = w_box2 * h_box2
    
    union_area = area_box1 + area_box2 - intersection_area
    
    iou = intersection_area / union_area if intersection_area > 0 and union_area > 0 else 0.0
    return iou

def draw(frame, bbox, person_name):
    x, y, w, h = bbox
    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)
    cv2.putText(frame, person_name, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

def detect(srcfd, current_frame):
    filtered_detected_boxes = []
    detected_boxes = srcfd.detect(current_frame)
    for detected_box in detected_boxes:
        x, y, w, h = detected_box.box.rect()
        if w >= 30 and h >= 30:
            filtered_detected_boxes.append(detected_box)
    return filtered_detected_boxes

def recognition_one_person(frame, recognition, face_contents, detected_box):
    x, y, w, h = detected_box
    cropped_image = frame[y:y+h, x:x+w]
    face_to_compare = recognition.detect(cropped_image)
    person_name = "Unknown"
    if face_to_compare.flag:
        for file_path, known_face in face_contents.items():
            if known_face.flag:
                sim = np.dot(face_to_compare.embedding, known_face.embedding)
                if sim > 0.6:
                    person_name = extract_name_from_file(file_path)
                    break
    return person_name

def recognitor(frame, recognition, face_contents, detected_boxes):
    for detected_box in detected_boxes:
        x, y, w, h = detected_box.box.rect()
        cropped_image = frame[y:y+h, x:x+w]
        face_to_compare = recognition.detect(cropped_image)
        person_name = "Unknown"
        if face_to_compare.flag:
            for file_path, known_face in face_contents.items():
                if known_face.flag:
                    sim = np.dot(face_to_compare.embedding, known_face.embedding)
                    if sim > 0.6:
                        person_name = extract_name_from_file(file_path)
                        break
        draw(frame, (x, y, w, h), person_name)

# Load Model
scrfd_path = "../model/scrfd_500m_kps.onnx"
faceRecog_path = "../model/faceRecogByMobileFaceNet.onnx"
scrfd = lite.onnxruntime.cv.face.detect.SCRFD(scrfd_path)
recognition = lite.onnxruntime.cv.faceid.MobileSEFocalFace(faceRecog_path)

# Load Known Face Data
faceContents = {}
data_path = "../data/*.jpg"
fn = cv2.glob(data_path)
for file_path in fn:
    img = cv2.imread(file_path)
    faceContent = recognition.detect(img)
    faceContents[file_path] = faceContent

# Load video
video_path = "../input/testVideo2.mp4"
videoCapture = cv2.VideoCapture(video_path)
if not videoCapture.isOpened():
    print("Error: Could not open the video file.")
    exit(1)

frameCount = 0
detectedBoxesPrev = []

# Face Detection and Recognition from Video
startPr = time.time()

while True:
    ret, currentFrame = videoCapture.read()
    if not ret:
        break
    
    start = time.time()

    if frameCount == 0:
        detectedBoxesPrev = detect(scrfd, currentFrame)
        recognitor(currentFrame, recognition, faceContents, detectedBoxesPrev)
    else:
        detectedBoxes = detect(scrfd, currentFrame)
        for i in range(len(detectedBoxes)):
            iou = calc_iou(detectedBoxes[i].box.rect(), detectedBoxesPrev[i].box.rect())

            if iou > 0.6:
                # person_name = recognition_one_person(current_frame, recognition, faceContents, detected_boxes_prev[i].box.rect())
                # draw(current_frame, detected_boxes[i].box.rect(), person_name)
                pass
            else:
                detectedBoxesPrev = detectedBoxes
                person_name = recognition_one_person(current_frame, recognition, faceContents, detected_boxes_prev[i].box.rect())
                draw(current_frame, detected_boxes[i].box.rect(), person_name)

        detectedBoxesPrev = detectedBoxes
    
    frameCount += 1

    stop = time.time()
    duration = (stop - start) * 1000  # Convert to milliseconds
    print("\nTime taken by frame", frameCount, ":", duration, "ms")

    cv2.imshow("Frame", currentFrame)
    
    # Press 'ESC' key to exit
    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break

videoCapture.release()
cv2.destroyAllWindows()

stopDetect = time.time()
programDuration = stopDetect - startPr
print("\nTime taken:", programDuration, "s",
      "FPS:", frameCount / programDuration)
