#include "lite/lite.h"
#include "iostream"
#include <chrono>
#include <string>
#include <algorithm>
using namespace std::chrono;

std::string extractNameFromFile(const std::string &filePath)
{
    size_t lastSlashPos = filePath.find_last_of('/');
    size_t secondLastSlashPos = filePath.find_last_of('/', lastSlashPos - 1);
    std::string directoryName = filePath.substr(secondLastSlashPos + 1, lastSlashPos - secondLastSlashPos - 1);
    return directoryName;
}

float calc_iou(cv::Rect bbox1, cv::Rect bbox2)
{
    // bbox x, y is top-left point

    // cacl x2 y2 ( bottom right point )
    float x2_box1 = bbox1.x + bbox1.width;
    float y2_box1 = bbox1.y + bbox1.height;

    // cacl x2 y2 ( bottom right point )
    float x2_box2 = bbox2.x + bbox2.width;
    float y2_box2 = bbox2.y + bbox2.height;

    // cacl intersection  w h
    float intersection_width = std::min(x2_box1, x2_box2) - std::max(bbox1.x, bbox2.x);
    float intersection_height = std::min(y2_box1, y2_box2) - std::max(bbox1.y, bbox2.y);

    float iou;

    if (intersection_height > 0 && intersection_width > 0)
    {
        float intersection_area = intersection_width * intersection_height;

        float area_box1 = bbox1.width * bbox1.height;
        float area_box2 = bbox2.width * bbox2.height;

        float union_area = area_box1 + area_box2 - intersection_area;

        iou = intersection_area / union_area;

        return iou;
    }
    else
        return 0.0f;
}

void draw(cv::Mat &frame, cv::Rect bbox, std::string personName)
{
    cv::rectangle(frame, bbox, cv::Scalar(255, 255, 0), 2);
    cv::putText(frame, personName, cv::Point(bbox.x, bbox.y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.6f, cv::Scalar(0, 255, 0), 2); //-5 for the text go up
}

std::vector<lite::types::BoxfWithLandmarks> detect(lite::onnxruntime::cv::face::detect::SCRFD *srcfd,
                                                   cv::Mat &currentFrame)
{
    std::vector<lite::types::BoxfWithLandmarks> filteredDetectedBoxes;
    std::vector<lite::types::BoxfWithLandmarks> detectedBoxes;
    srcfd->detect(currentFrame, detectedBoxes);
    for (const auto detectedBox : detectedBoxes)
    {
        if (detectedBox.box.rect().width >= 30 && detectedBox.box.rect().height >= 30)
        {
            filteredDetectedBoxes.push_back(detectedBox);
        }
    }
    return filteredDetectedBoxes;
}
std::string recognitionOnePerson(cv::Mat &frame,
                                 lite::onnxruntime::cv::faceid::MobileSEFocalFace *recognition,
                                 const std::unordered_map<std::string, lite::types::FaceContent> &faceContents,
                                 cv::Rect detectedBox)
{
    std::cout << "cai gi" << std::endl;
    cv::Rect Rect(detectedBox.x, detectedBox.y, detectedBox.width, detectedBox.height);
    cv::Mat croppedImage = frame(Rect);
    lite::types::FaceContent face_to_compare;
    recognition->detect(croppedImage, face_to_compare);
    std::string personName = "Unknown";
    if (face_to_compare.flag)
    {
        for (const auto &faceContent : faceContents)
        {
            const lite::types::FaceContent &knownFace = faceContent.second;
            if (knownFace.flag)
            {
                float sim = lite::utils::math::cosine_similarity<float>(
                    face_to_compare.embedding, knownFace.embedding);
                if (sim > 0.6)
                {
                    personName = extractNameFromFile(faceContent.first);
                    break;
                }
            }
        }
    }
        return personName;
}
void recognitor(cv::Mat &frame,
                lite::onnxruntime::cv::faceid::MobileSEFocalFace *recognition,
                const std::unordered_map<std::string, lite::types::FaceContent> &faceContents,
                std::vector<lite::types::BoxfWithLandmarks> detectedBoxes)
{
    std::cout << "cai" << std::endl;
    for (const auto detectedBox : detectedBoxes)
    {
        cv::Rect Rect(detectedBox.box.rect().x, detectedBox.box.rect().y, detectedBox.box.rect().width, detectedBox.box.rect().height);
        cv::Mat croppedImage = frame(Rect);
        lite::types::FaceContent face_to_compare;
        recognition->detect(croppedImage, face_to_compare);
        std::string personName = "Unknown";
        if (face_to_compare.flag)
        {
            for (const auto &faceContent : faceContents)
            {
                const lite::types::FaceContent &knownFace = faceContent.second;
                if (knownFace.flag)
                {
                    float sim = lite::utils::math::cosine_similarity<float>(
                        face_to_compare.embedding, knownFace.embedding);
                    if (sim > 0.6)
                    {
                        personName = extractNameFromFile(faceContent.first);
                        break;
                    }
                }
            }
        }
        draw(frame, detectedBox.box.rect(), personName);
    }
}

int main()
{
    // Load Model
    std::string scrfd_path = "../model/scrfd_500m_kps.onnx";
    std::string faceRecog_path = "../model/faceRecogByMobileFaceNet.onnx";
    lite::onnxruntime::cv::face::detect::SCRFD scrfd(scrfd_path);
    lite::onnxruntime::cv::faceid::MobileSEFocalFace recognition(faceRecog_path);

    // Load Known Face Data
    std::unordered_map<std::string, lite::types::FaceContent> faceContents;
    cv::String data_path = "../data/*.jpg";
    std::vector<cv::String> fn;
    cv::glob(data_path, fn, true);
    for (const auto &filePath : fn)
    {
        cv::Mat img = cv::imread(filePath);
        lite::types::FaceContent faceContent;
        recognition.detect(img, faceContent);
        faceContents[filePath] = faceContent;
    }

    // Load video
    std::string video_path = "../input/testVideo.mp4";
    cv::VideoCapture videoCapture(video_path);
    if (!videoCapture.isOpened())
    {
        std::cout << "Error: Could not open the video file." << std::endl;
        return -1;
    }

    int frameCount = 0;
    cv::Mat currentFrame, prevFrame;
    std::vector<lite::types::BoxfWithLandmarks> detectedBoxes, detectedBoxesPrev;
    // Face Detection and Recognition from Video
    auto startPr = high_resolution_clock::now();

    while (videoCapture.read(currentFrame))
    {
        auto start = high_resolution_clock::now();
        if (frameCount == 0)
        {
            detectedBoxesPrev = detect(&scrfd, currentFrame);
            recognitor(currentFrame, &recognition, faceContents, detectedBoxesPrev);
        }
        else
        {
            detectedBoxes = detect(&scrfd, currentFrame);
            for (size_t i = 0; i < detectedBoxes.size(); ++i)
            {
                float iou = calc_iou(detectedBoxes[i].box.rect(), detectedBoxesPrev[i].box.rect());

                if (iou > 0.1)
                {   
                    // std::string personName = recognitionOnePerson(currentFrame, &recognition, faceContents, detectedBoxesPrev[i].box.rect());
                    draw(currentFrame, detectedBoxes[i].box.rect(), std::to_string(i));
                }
                else
                {
                    detectedBoxesPrev = detectedBoxes;
                    std::string personName = recognitionOnePerson(currentFrame, &recognition, faceContents, detectedBoxesPrev[i].box.rect());
                    draw(currentFrame, detectedBoxes[i].box.rect(), personName);
                }
            }
        detectedBoxesPrev = detectedBoxes;
        }
     
        frameCount++;

        auto stop = high_resolution_clock::now();
        auto duration = duration_cast<milliseconds>(stop - start);

        std::cout << "\nTime taken by frame " << frameCount << " : " << duration.count() << " ms" << std::endl;

        cv::imshow("Frame", currentFrame);

        // Press  ESC on keyboard to exit
        char c = (char)cv::waitKey(1);
        if (c == 27)
            break;
    }
    auto stopDetect = high_resolution_clock::now();
    auto programDuration = duration_cast<seconds>(stopDetect - startPr);
    std::cout << "\nTime taken :" << programDuration.count() << "s "
              << "FPS : " << frameCount / (programDuration.count()) << std::endl;
    videoCapture.release();

    return 0;
}