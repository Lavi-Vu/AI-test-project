#include "lite/lite.h"
#include "iostream"
#include <chrono>

using namespace std::chrono;


std::string extractNameFromFile(const std::string& filePath){
  size_t lastSlashPos = filePath.find_last_of('/');
  size_t secondLastSlashPos = filePath.find_last_of('/', lastSlashPos - 1);
  std::string directoryName = filePath.substr(secondLastSlashPos + 1, lastSlashPos - secondLastSlashPos - 1);
  return directoryName;
}

void face_detect(cv::Mat& img_bgr, 
                lite::onnxruntime::cv::face::detect::YOLO5Face* yolov5face, 
                lite::onnxruntime::cv::faceid::MobileSEFocalFace* recognition, 
                const std::unordered_map<std::string, lite::types::FaceContent>& faceContents) 
{
    
    std::vector<lite::types::BoxfWithLandmarks> detected_boxes;
    yolov5face->detect(img_bgr, detected_boxes);

    for (size_t i = 0; i < detected_boxes.size(); ++i) {
        const lite::types::BoxfWithLandmarks& box_kps = detected_boxes[i];
        cv::Rect Rect(box_kps.box.rect().x, box_kps.box.rect().y, box_kps.box.rect().width, box_kps.box.rect().height);
        cv::Mat croppedImage = img_bgr(Rect);

        lite::types::FaceContent face_to_compare;
        recognition->detect(croppedImage, face_to_compare);

        std::string personName = "Unknown";
        if (face_to_compare.flag) {
            for (const auto& faceContent : faceContents) {
                const lite::types::FaceContent& knownFace = faceContent.second;
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
        std::cout << detected_boxes[i].box.rect();

        cv::rectangle(img_bgr, box_kps.box.rect(), cv::Scalar(255, 255, 0), 2);
        cv::putText(img_bgr, personName, cv::Point(box_kps.box.rect().x, box_kps.box.rect().y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.6f, cv::Scalar(0, 255, 0), 2);//-5 for the text go up
        // std::cout << personName << std::endl;
    }
}



int main() {

    // Load Models
    std::string yolo5face_path = "../model/yolov5n-face.onnx";
    std::string faceRecog_path = "../model/faceRecogByMobileFaceNet.onnx";
    lite::onnxruntime::cv::face::detect::YOLO5Face yolov5face(yolo5face_path);
    lite::onnxruntime::cv::faceid::MobileSEFocalFace recognition(faceRecog_path);

    // Load Known Face Data
    std::unordered_map<std::string, lite::types::FaceContent> faceContents;
    cv::String data_path = "../data/*.jpg";
    std::vector<cv::String> fn;
    cv::glob(data_path, fn, true);
    for (const auto& filePath : fn) {
        cv::Mat img = cv::imread(filePath);
        lite::types::FaceContent faceContent;
        recognition.detect(img, faceContent);
        faceContents[filePath] = faceContent;
    }

    // Load Test Image
    std::string test_img_path = "../input/*.jpg";
    std::vector<cv::String> input_data;
    cv::glob(test_img_path, input_data, true);
    int k = 0;

    auto start = high_resolution_clock::now();
    // Face Detection and Recognition
    for (const auto& filePath : input_data){
      
      cv::Mat img_bgr = cv::imread(filePath);
      face_detect(img_bgr, &yolov5face, &recognition, faceContents);
      std::string save_img_path = "../output/test_lite_yolov5face" + std::to_string(k) +".jpg";
      cv::imwrite(save_img_path, img_bgr);
      ++k;
    }
    // Save Result Image

    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(stop - start);
    std::cout << "\nTime taken by function : " << duration.count() << " milliseconds" << std::endl;

    return 0;
}
