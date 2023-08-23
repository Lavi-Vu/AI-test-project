#include <iostream>
#include <curl/curl.h>
#include <chrono>
#include <thread>

int main() {
    CURL* curl = curl_easy_init();
    if (!curl) {
        std::cerr << "cURL initialization failed." << std::endl;
        return 1;
    }

    std::string url = "http://localhost:3000/";
    std::string data = "Hello from C++";  // Data to send

    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_POST, 1L);
    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, data.c_str()); // Set POST data directly

    while (true) {
        CURLcode res = curl_easy_perform(curl);
        if (res != CURLE_OK) {
            std::cerr << "cURL request failed: " << curl_easy_strerror(res) << std::endl;
        } else {
            std::cout << "Data sent successfully." << std::endl;
        }

        // Wait for a while before sending the next request
        // std::this_thread::sleep_for(std::chrono::seconds(10));
    }

    curl_easy_cleanup(curl);

    return 0;
}
