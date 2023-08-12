#include <iostream>
#include <curl/curl.h>

size_t WriteCallback(void* contents, size_t size, size_t nmemb, void* user_data) {
    size_t total_size = size * nmemb;
    std::string* response = static_cast<std::string*>(user_data);
    response->append(static_cast<char*>(contents), total_size);
    return total_size;
}

int main() {
    CURL* curl = curl_easy_init();
    if (!curl) {
        std::cerr << "cURL initialization failed." << std::endl;
        return 1;
    }

    std::string url = "http://localhost:3000";
    std::string response;
    std::string post_data = "This is the data sent in the POST request.";

    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_POST, 1L);
    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, post_data.c_str());
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);

    CURLcode res = curl_easy_perform(curl);
    if (res != CURLE_OK) {
        std::cerr << "cURL request failed: " << curl_easy_strerror(res) << std::endl;
    } else {
        std::cout << "Response:\n" << response << std::endl;
    }

    curl_easy_cleanup(curl);

    return 0;
}
