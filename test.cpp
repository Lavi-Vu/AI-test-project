#include "lite/lite.h"
#include <chrono>
using namespace std::chrono;
static void test_default()
{
  std::string onnx_path = "/home/lavi/Desktop/LiteAITest/model/scrfd_500m_kps.onnx";
  std::string test_img_path = "/home/lavi/Desktop/LiteAITest/input/im2.jpg";
  std::string save_img_path = "/home/lavi/Desktop/LiteAITest/output/test_lite_yolov5facehmm.jpg";
  
  auto *scrfd = new lite::cv::face::detect::SCRFD(onnx_path);
  
  std::vector<lite::types::BoxfWithLandmarks> detected_boxes;
  cv::Mat img_bgr = cv::imread(test_img_path);
    auto start = high_resolution_clock::now();
  scrfd->detect(img_bgr, detected_boxes);
  auto stop = high_resolution_clock::now();
  lite::utils::draw_boxes_with_landmarks_inplace(img_bgr, detected_boxes);
  auto duration = duration_cast<milliseconds>(stop - start);
  std::cout << "\nTime taken :"<< duration.count() << "ms" << std::endl;
  cv::imwrite(save_img_path, img_bgr);
  
  delete scrfd;
}
int main(){
    test_default();
}