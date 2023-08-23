#include "lite/lite.h"
#include "iostream"
#include <chrono>
#include <string>
#include <algorithm>

#include <set>
#include <iterator>
#include <curl/curl.h>
using namespace std::chrono;
typedef struct TrackingBox
{
    cv::Rect box;
    std::string personName;
    int frame;
} TrackingBox;
std::string extractNameFromFile(const std::string &filePath)
{
    size_t lastSlashPos = filePath.find_last_of('/');
    size_t secondLastSlashPos = filePath.find_last_of('/', lastSlashPos - 1);
    std::string directoryName = filePath.substr(secondLastSlashPos + 1, lastSlashPos - secondLastSlashPos - 1);
    return directoryName;
}

float calculateIoU(cv::Rect bbox1, cv::Rect bbox2)
{
    // Calculate bottom-right coordinates
    float x2_bbox1 = bbox1.x + bbox1.width;
    float y2_bbox1 = bbox1.y + bbox1.height;
    float x2_bbox2 = bbox2.x + bbox2.width;
    float y2_bbox2 = bbox2.y + bbox2.height;

    // Calculate intersection dimensions
    float intersection_width = std::max(0.0f, std::min(x2_bbox1, x2_bbox2) - std::max(bbox1.x, bbox2.x));
    float intersection_height = std::max(0.0f, std::min(y2_bbox1, y2_bbox2) - std::max(bbox1.y, bbox2.y));

    // Calculate intersection area
    float intersection_area = intersection_width * intersection_height;

    // Calculate individual bounding box areas
    float area_bbox1 = bbox1.width * bbox1.height;
    float area_bbox2 = bbox2.width * bbox2.height;

    // Calculate union area
    float union_area = area_bbox1 + area_bbox2 - intersection_area;

    // Calculate IoU
    if (union_area > 0)
    {
        float iou = intersection_area / union_area;
        return iou;
    }
    else
    {
        return 0.0f;
    }
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
bool hasPersonName(const std::vector<TrackingBox> &detData, const std::string &name)
{
    for (const TrackingBox &box : detData)
    {
        if (box.personName == name)
        {
            return true;
        }
    }
    return false;
}
void addToStruct(std::vector<TrackingBox> &detData, const TrackingBox &newBox)
{
    if (!hasPersonName(detData, newBox.personName))
    {
        std::cout << "Add new data to struct .........................................................................";
        detData.push_back(newBox);
    }
}

int main()
{
    // Load Model
    std::string scrfd_path = "../model/scrfd_500m_kps.onnx";
    std::string faceRecog_path = "../model/faceRecogByMobileFaceNet.onnx";
    lite::onnxruntime::cv::face::detect::SCRFD scrfd(scrfd_path, 2);
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
    std::vector<TrackingBox> detData;
    std::vector<std::vector<double>> iouMatrix;
    // Face Detection and Recognition from Video
    auto startPr = high_resolution_clock::now();

    while (videoCapture.read(currentFrame))
    {
        auto start = high_resolution_clock::now();
        if (detData.size() == 0)
        {
            detectedBoxesPrev = detect(&scrfd, currentFrame);
            for (size_t i = 0; i < detectedBoxesPrev.size(); ++i)
            {
                std::string personName = recognitionOnePerson(currentFrame, &recognition, faceContents, detectedBoxesPrev[i].box.rect());
                draw(currentFrame, detectedBoxesPrev[i].box.rect(), personName);
                TrackingBox newTrackingBox;
                newTrackingBox.box = detectedBoxesPrev[i].box.rect();
                newTrackingBox.personName = personName;
                newTrackingBox.frame = frameCount;
                addToStruct(detData, newTrackingBox);
            }
        }
        else
        {
            detectedBoxes = detect(&scrfd, currentFrame);
            iouMatrix.clear();
            iouMatrix.resize(detectedBoxes.size(), std::vector<double>(detectedBoxesPrev.size(), 0));
            int trkNum = detectedBoxes.size();
            int detNum = detectedBoxesPrev.size();

            if (detNum == trkNum)
            {
                for (size_t i = 0; i < detectedBoxes.size(); i++)
                {
                    for (size_t j = 0; j < detectedBoxesPrev.size(); j++)
                    {
                        iouMatrix[i][j] = calculateIoU(detectedBoxes[i].box.rect(), detectedBoxesPrev[j].box.rect());

                        if (iouMatrix[i][j] > 0.1)
                        {
                            std::cout << "IOUuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuu" << std::endl;
                            // update new box for detData
                            detData[j].box = detectedBoxes[i].box.rect();
                            draw(currentFrame, detData[j].box, detData[j].personName);
                        }
                    }
                }
            }
            else if (detNum < trkNum){

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