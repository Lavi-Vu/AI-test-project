#include "lite/lite.h"
#include "iostream"
#include <chrono>

using namespace std::chrono;

std::string extractNameFromFile(const std::string &filePath) {
    size_t lastSlashPos = filePath.find_last_of('/');
    size_t secondLastSlashPos = filePath.find_last_of('/', lastSlashPos - 1);
    std::string directoryName = filePath.substr(secondLastSlashPos + 1, lastSlashPos - secondLastSlashPos - 1);
    return directoryName;
}

void face_detect(cv::Mat &img_bgr,
    lite::onnxruntime::cv::face::detect::SCRFD *yolov5face,
    lite::onnxruntime::cv::faceid::MobileSEFocalFace *recognition,
    const std::unordered_map<std::string, lite::types::FaceContent> &faceContents) {

    std::vector<lite::types::BoxfWithLandmarks> detected_boxes;
    yolov5face->detect(img_bgr, detected_boxes);

    for (const auto detected_box : detected_boxes) {
        const lite::types::BoxfWithLandmarks &box_kps = detected_box;
        cv::Rect Rect(box_kps.box.rect().x, box_kps.box.rect().y, box_kps.box.rect().width, box_kps.box.rect().height);
        cv::Mat croppedImage = img_bgr(Rect);

        lite::types::FaceContent face_to_compare;
        recognition->detect(croppedImage, face_to_compare);

        std::string personName = "Unknown";
        if (face_to_compare.flag) {
            for (const auto &faceContent : faceContents) {
                const lite::types::FaceContent &knownFace = faceContent.second;
                if (knownFace.flag) {
                    float sim = lite::utils::math::cosine_similarity<float>(
                        face_to_compare.embedding, knownFace.embedding);
                    if (sim > 0.6) {
                        personName = extractNameFromFile(faceContent.first);
                        break;
                    }
                
                }
            }
        }
      
        cv::rectangle(img_bgr, box_kps.box.rect(), cv::Scalar(255, 255, 0), 2);
        cv::putText(img_bgr, personName, cv::Point(box_kps.box.rect().x, box_kps.box.rect().y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.6f, cv::Scalar(0, 255, 0), 2); //-5 for the text go up
        // std::cout << personName << std::endl;
        
        
    }
}
void detect(lite::onnxruntime::cv::face::detect::SCRFD* srcfd,
            cv::Mat &currentFrame){
    std::vector<lite::types::BoxfWithLandmarks> currentDetectedBoxes;
    srcfd->detect(currentFrame, currentDetectedBoxes);
    lite::utils::draw_boxes_with_landmarks_inplace(currentFrame, currentDetectedBoxes);


}

int main() {
    // Load Models
    std::string yolo5face_path = "../model/scrfd_500m_kps.onnx";
    std::string faceRecog_path = "../model/faceRecogByMobileFaceNet.onnx";
    lite::onnxruntime::cv::face::detect::SCRFD yolov5face(yolo5face_path);
    lite::onnxruntime::cv::faceid::MobileSEFocalFace recognition(faceRecog_path);

    // Load Known Face Data
    std::unordered_map<std::string, lite::types::FaceContent> faceContents;
    cv::String data_path = "../data/*.jpg";
    std::vector<cv::String> fn;
    cv::glob(data_path, fn, true);
    for (const auto &filePath : fn) {
        cv::Mat img = cv::imread(filePath);
        lite::types::FaceContent faceContent;
        recognition.detect(img, faceContent);
        faceContents[filePath] = faceContent;
    }

    // Open video capture
    std::string video_path = "../input/testVideo2.mp4";
    cv::VideoCapture videoCapture(video_path);
    if (!videoCapture.isOpened()) {
        std::cout << "Error: Could not open the video file." << std::endl;
        return -1;
    }

    int k = 0;
    cv::Mat frame;

    // Face Detection and Recognition from Video
    while (videoCapture.read(frame)) {
        auto start = high_resolution_clock::now();
        face_detect(frame, &yolov5face, &recognition, faceContents);
        auto stop = high_resolution_clock::now();
        auto duration = duration_cast<milliseconds>(stop - start);
        std::cout << "\nTime taken by frame " << k << " : " << duration.count() << " ms" << std::endl;
            cv::imshow( "Frame", frame );
 
    // Press  ESC on keyboard to exit
    char c=(char)cv::waitKey(25);
    if(c==27)
      break;
  }
        // std::string save_img_path = "../output/frame" + std::to_string(k) + ".jpg";
        // cv::imwrite(save_img_path, frame);
        
    

    videoCapture.release();

    return 0;
}
