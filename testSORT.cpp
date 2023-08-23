#include "lite/lite.h"
#include "iostream"
#include <chrono>
#include <string>
#include <algorithm>
#include <set>
#include <vector>
#include <iterator>
#include <curl/curl.h>
#include <nlohmann/json.hpp>
using namespace std::chrono;

typedef struct TrackingBox
{
	cv::Rect box;
	std::string personName;
	int id;
	int frame;
} TrackingBox;
class TrackingBoxManager
{
private:
	int lastId;

public:
	TrackingBoxManager() : lastId(0) {}

	bool hasPersonName(const std::string &name, std::vector<TrackingBox> &detData)
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

	void addTrackingBox(const TrackingBox &newBox, std::vector<TrackingBox> &detData)
	{
		lastId++;
		TrackingBox boxWithId = newBox;
		boxWithId.id = lastId;
		detData.push_back(boxWithId);
	}
	void updateTrackingBox(std::vector<TrackingBox> &detData, const TrackingBox &updatedBox)
	{
		for (TrackingBox &box : detData)
		{
			if (box.personName == updatedBox.personName)
			{
				box.box = updatedBox.box;
				box.frame = updatedBox.frame;
				return; // Exit the loop once updated
			}
		}
	}
};
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
		if (detectedBox.box.rect().width >= 30 && detectedBox.box.rect().height >= 30 && detectedBox.box.rect().width <= 640 && detectedBox.box.rect().height <= 640)
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
				if (sim > 0.5)
				{
					personName = extractNameFromFile(faceContent.first);
					break;
				}
			}
		}
	}
	return personName;
}
void cURL()
{
}

int main()
{
	// init server and cURL
	CURL *curl = curl_easy_init();
	if (!curl)
	{
		std::cerr << "cURL initialization failed." << std::endl;
		return 1;
	}
	std::string url = "http://localhost:3000/";

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
	cv::Mat currentFrame;
	std::vector<lite::types::BoxfWithLandmarks> detectedBoxes;
	std::vector<TrackingBox> detData;
	std::vector<std::vector<double>> iouMatrix;
	int frameCount = 0;
	unsigned int trkNum = 0;
	unsigned int detNum = 0;

	// Face Detection and Recognition from Video
	auto startPr = high_resolution_clock::now();
	TrackingBoxManager manager;
	while (videoCapture.read(currentFrame))
	{
		auto start = high_resolution_clock::now();
		// track face using scrfd
		if (detData.size() == 0)
		{
			std::cout << "detData == 0" << std::endl;
			detectedBoxes = detect(&scrfd, currentFrame);
			for (size_t i = 0; i < detectedBoxes.size(); ++i)
			{
				std::string personName = recognitionOnePerson(currentFrame, &recognition, faceContents, detectedBoxes[i].box.rect());
				if (personName != "Unknown")
				{
					TrackingBox newTrackingBox;
					newTrackingBox.box = detectedBoxes[i].box.rect();
					newTrackingBox.personName = personName;
					newTrackingBox.frame = frameCount;
					manager.addTrackingBox(newTrackingBox, detData);
					draw(currentFrame, detectedBoxes[i].box.rect(), std::to_string(detData[i].id));
				}
			}
			// OutPut of first Det
			// for (const auto &detBox : detData)
			// {
			// 	std::cout << "Person Name: " << detBox.personName << std::endl;
			// 	std::cout << "Detected Box: ";
			// 	// In ra thông tin detectedBox ở đây, ví dụ:
			// 	std::cout << "x: " << detBox.box.x << ", "
			// 			  << "y: " << detBox.box.y << ", "
			// 			  << "width: " << detBox.box.width << ", "
			// 			  << "height: " << detBox.box.height << std::endl;
			// 	std::cout << std::endl;
			// }
			continue;
		}

		std::vector<lite::types::BoxfWithLandmarks> trackBoxes = detect(&scrfd, currentFrame);
		trkNum = trackBoxes.size();
		detNum = detectedBoxes.size();
		iouMatrix.clear();
		iouMatrix.resize(trkNum, std::vector<double>(detNum, 0));

		if (detNum == trkNum)
		{
			for (unsigned int i = 0; i < trkNum; i++)
			{
				for (unsigned int j = 0; j < detNum; j++)
				{
					iouMatrix[i][j] = calculateIoU(trackBoxes[i].box.rect(), detData[j].box);
					if (iouMatrix[i][j] > 0.4)
					{
						std::cout << "IOUuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuu" << std::endl;
						detData[j].box = trackBoxes[i].box.rect();
						draw(currentFrame, detData[j].box, detData[j].personName);
					}
				}
			}
		}
		else
		{
			if (trackBoxes.size() == 0){
				std::cout << "0 Detected .........................." << std::endl;
			}
			for (size_t i = 0; i < trackBoxes.size(); ++i)
			{
				std::string personName = recognitionOnePerson(currentFrame, &recognition, faceContents, trackBoxes[i].box.rect());
				if (personName != "Unknown")
					for (unsigned int j = 0; j < detectedBoxes.size(); j++)
					{
						iouMatrix[i][j] = calculateIoU(trackBoxes[i].box.rect(), detData[j].box);
						if (iouMatrix[i][j] > 0.4)
						{
							std::cout << "IOUuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuu222222222222222" << std::endl;
							detData[j].box = trackBoxes[i].box.rect();
							draw(currentFrame, detData[j].box, detData[j].personName);
						}
						else
						{
							TrackingBox newTrackingBox;
							newTrackingBox.box = trackBoxes[i].box.rect();
							newTrackingBox.personName = personName;
							newTrackingBox.frame = frameCount;
							draw(currentFrame, trackBoxes[i].box.rect(), personName);
							if (!manager.hasPersonName(newTrackingBox.personName, detData))
							{
								std::cout << "Add new data to struct ..............................." << std::endl;
								manager.addTrackingBox(newTrackingBox, detData);
								draw(currentFrame, trackBoxes[i].box.rect(), personName);
							}
							else
							{
								std::cout << "Update data to struct ..............................." << std::endl;
								manager.updateTrackingBox(detData, newTrackingBox);
								draw(currentFrame, trackBoxes[i].box.rect(), personName);
							}
						}
					}
			}
		}

		detectedBoxes = trackBoxes;

		// for (const auto &detBox : detData)
		// {
		// 	nlohmann::json jsonData;

		// 	if (detBox.personName != "Unknown")
		// 	{
		// 		jsonData["personName"] = detBox.personName;
		// 		std::string jsonString = jsonData.dump();
		// 		curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
		// 		curl_easy_setopt(curl, CURLOPT_POST, 1L);
		// 		curl_easy_setopt(curl, CURLOPT_POSTFIELDS, jsonString.c_str()); // Set POST data directly

		// 		CURLcode res = curl_easy_perform(curl);
		// 		sentPersonNames.insert(detBox.personName);
		// 	}
		// 	// std::cout << "Person Name: " << detBox.personName << std::endl;
		// 	// std::cout << "Detected Box: ";
		// 	// std::cout << "x: " << detBox.box.x << ", "
		// 	// 		  << "y: " << detBox.box.y << ", "
		// 	// 		  << "width: " << detBox.box.width << ", "
		// 	// 		  << "height: " << detBox.box.height << std::endl;
		// 	// std::cout << std::endl;
		// }

		auto stop = high_resolution_clock::now();
		auto duration = duration_cast<milliseconds>(stop - start);
		frameCount++;

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
	for (const auto &detBox : detData)
	{
		std::cout << "Person Name: " << detBox.personName << std::endl;
		std::cout << "Detected Box: ";
		std::cout << "x: " << detBox.box.x << ", "
				  << "y: " << detBox.box.y << ", "
				  << "width: " << detBox.box.width << ", "
				  << "height: " << detBox.box.height << std::endl;
		std::cout << "Frame: " << detBox.frame << std::endl;
		std::cout << "ID: " << detBox.id << std::endl;
		std::cout << std::endl;
	}
	videoCapture.release();
	// curl_easy_cleanup(curl);
	return 0;
}