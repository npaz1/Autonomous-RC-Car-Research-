/*
 * SPDX-FileCopyrightText: 2024-2025 Espressif Systems (Shanghai) CO LTD
 *
 * SPDX-License-Identifier: ESPRESSIF MIT
 */

#include <string.h>
#include <math.h>
#include "esp_wifi.h"
#include "esp_netif.h"
#include "esp_event.h"
#include "driver/ledc.h"
#include "driver/uart.h"
#include <fcntl.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/param.h>
#include <sys/errno.h>
#include "sdkconfig.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/semphr.h"
#include "cJSON.h"
#include "esp_err.h"
#include "esp_check.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "nvs_flash.h"
#include "esp_check.h"
#include "esp_http_server.h"
#include "protocol_examples_common.h"
#include "mdns.h"
#include "lwip/inet.h"
#include "lwip/sockets.h"
#include "lwip/netdb.h"
#include "lwip/apps/netbiosns.h"
#include "example_video_common.h"

#define EXAMPLE_CAMERA_VIDEO_BUFFER_NUMBER  CONFIG_EXAMPLE_CAMERA_VIDEO_BUFFER_NUMBER

#define RC_CAMERA_JPEG_QUALITY               45
#define V4_STREAM_MIN_FRAME_INTERVAL_US       50000  /* 20 FPS max */
#define EXAMPLE_JPEG_ENC_QUALITY            RC_CAMERA_JPEG_QUALITY

/* Stream diagnostics: log only unusually slow stages to avoid serial-log spam. */
#define RC_STREAM_SLOW_DQBUF_MS              120
#define RC_STREAM_SLOW_ENCODE_MS             80
#define RC_STREAM_SLOW_SEND_MS               120
#define RC_STREAM_SLOW_TOTAL_MS              180

#define EXAMPLE_MDNS_INSTANCE               CONFIG_EXAMPLE_MDNS_INSTANCE
#define EXAMPLE_MDNS_HOST_NAME              CONFIG_EXAMPLE_MDNS_HOST_NAME

#define EXAMPLE_PART_BOUNDARY               CONFIG_EXAMPLE_HTTP_PART_BOUNDARY


/* ========================= RC CAR CONTROL ========================= */
#define RC_STEERING_GPIO                    4
#define RC_ESC_GPIO                         5

#define RC_PWM_FREQUENCY_HZ                 50
#define RC_PWM_RESOLUTION                   LEDC_TIMER_16_BIT
#define RC_PWM_TIMER                        LEDC_TIMER_0
#define RC_PWM_MODE                         LEDC_LOW_SPEED_MODE

#define RC_STEERING_CHANNEL                 LEDC_CHANNEL_0
#define RC_ESC_CHANNEL                      LEDC_CHANNEL_1

#define RC_PWM_PERIOD_US                    20000U
#define RC_PWM_MAX_DUTY                     65535U

#define RC_STEERING_MIN_US                  1000U
#define RC_STEERING_CENTER_US               1500U
#define RC_STEERING_MAX_US                  2000U

/* Conservative keyboard steering values for the first live test. */
#define RC_STEERING_LEFT_US                 1300U
#define RC_STEERING_RIGHT_US                1700U

#define RC_ESC_MIN_US                       1450U
#define RC_ESC_NEUTRAL_US                   1500U
#define RC_ESC_MAX_US                       1550U

#define RC_UDP_PORT                         4210
#define RC_FAILSAFE_TIMEOUT_MS              500
#define RC_UDP_RX_BUFFER_SIZE               128
/* ================================================================= */


/* ============================ LD19 LIDAR ============================ */
#define LIDAR_UART_PORT                     UART_NUM_1
#define LIDAR_UART_RX_GPIO                  6
#define LIDAR_UART_BAUD_RATE                230400
#define LIDAR_UART_RX_BUFFER_SIZE           4096
#define LIDAR_TASK_STACK_SIZE               4096
#define LD19_PACKET_HEADER                  0x54
#define LD19_VER_LEN                        0x2C
#define LD19_PACKET_SIZE                    47
#define LIDAR_SCAN_BINS                      360
#define LIDAR_MIN_CONFIDENCE                 5
#define LIDAR_MAX_DISTANCE_MM                12000
/* =================================================================== */


static const char *STREAM_CONTENT_TYPE = "multipart/x-mixed-replace;boundary=" EXAMPLE_PART_BOUNDARY;
static const char *STREAM_BOUNDARY = "\r\n--" EXAMPLE_PART_BOUNDARY "\r\n";
static const char *STREAM_PART = "Content-Type: image/jpeg\r\nContent-Length: %u\r\nX-Timestamp: %d.%06d\r\n\r\n";

extern const uint8_t index_html_gz_start[] asm("_binary_index_html_gz_start");
extern const uint8_t index_html_gz_end[] asm("_binary_index_html_gz_end");
extern const uint8_t loading_jpg_gz_start[] asm("_binary_loading_jpg_gz_start");
extern const uint8_t loading_jpg_gz_end[] asm("_binary_loading_jpg_gz_end");
extern const uint8_t favicon_ico_gz_start[] asm("_binary_favicon_ico_gz_start");
extern const uint8_t favicon_ico_gz_end[] asm("_binary_favicon_ico_gz_end");
extern const uint8_t assets_index_js_gz_start[] asm("_binary_index_js_gz_start");
extern const uint8_t assets_index_js_gz_end[] asm("_binary_index_js_gz_end");
extern const uint8_t assets_index_css_gz_start[] asm("_binary_index_css_gz_start");
extern const uint8_t assets_index_css_gz_end[] asm("_binary_index_css_gz_end");

/**
 * @brief Web cam control structure
 */
typedef struct web_cam_video {
    int fd;
    uint8_t index;

    example_encoder_handle_t encoder_handle;
    uint8_t *jpeg_out_buf;
    uint32_t jpeg_out_size;

    uint8_t *buffer[EXAMPLE_CAMERA_VIDEO_BUFFER_NUMBER];
    uint32_t buffer_size;

    uint32_t width;
    uint32_t height;
    uint32_t pixel_format;
    uint8_t jpeg_quality;

    uint32_t frame_rate;

    SemaphoreHandle_t sem;

    uint32_t support_control_jpeg_quality   : 1;
} web_cam_video_t;

typedef struct web_cam {
    uint8_t video_count;
    web_cam_video_t video[0];
} web_cam_t;

typedef struct web_cam_video_config {
    const char *dev_name;
    uint32_t buffer_count;
} web_cam_video_config_t;

typedef struct request_desc {
    int index;
} request_desc_t;

static const char *TAG = "example";


typedef struct {
    volatile bool receiving;
    volatile uint32_t bytes_per_sec;
    volatile uint32_t valid_packets_per_sec;
    volatile uint32_t crc_errors_per_sec;
    volatile uint32_t framing_errors_per_sec;
    volatile uint32_t total_valid_packets;
    volatile uint32_t total_crc_errors;
    volatile uint32_t total_framing_errors;
    volatile uint16_t last_speed_deg_s;
    volatile float last_start_angle_deg;
    volatile float last_end_angle_deg;
    volatile uint16_t last_first_distance_mm;
    volatile uint8_t last_first_confidence;
    volatile int64_t last_packet_time_us;
} lidar_status_t;

static lidar_status_t g_lidar_status = {0};


typedef struct {
    uint16_t distance_mm;
    uint8_t confidence;
    uint32_t update_seq;
} lidar_scan_bin_t;

static lidar_scan_bin_t g_lidar_scan[LIDAR_SCAN_BINS] = {0};
static SemaphoreHandle_t g_lidar_scan_mutex = NULL;
static volatile uint32_t g_lidar_scan_sequence = 0;

static bool is_valid_web_cam(web_cam_video_t *video)
{
    return video->fd != -1;
}

static esp_err_t decode_request(web_cam_t *web_cam, httpd_req_t *req, request_desc_t *desc)
{
    esp_err_t ret;
    int index = -1;
    char buffer[32];

    if ((ret = httpd_req_get_url_query_str(req, buffer, sizeof(buffer))) != ESP_OK) {
        return ret;
    }
    ESP_LOGD(TAG, "source: %s", buffer);

    for (int i = 0; i < web_cam->video_count; i++) {
        char source_str[16];

        if (snprintf(source_str, sizeof(source_str), "source=%d", i) <= 0) {
            return ESP_FAIL;
        }

        if (strcmp(buffer, source_str) == 0) {
            index = i;
            break;
        }
    }
    if (index == -1) {
        return ESP_ERR_INVALID_ARG;
    }

    desc->index = index;
    return ESP_OK;
}

static esp_err_t capture_video_image(httpd_req_t *req, web_cam_video_t *video, bool is_jpeg)
{
    esp_err_t ret;
    struct v4l2_buffer buf;
    const char *type_str = is_jpeg ? "JPEG" : "binary";
    uint32_t jpeg_encoded_size;

    memset(&buf, 0, sizeof(buf));
    buf.type   = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buf.memory = V4L2_MEMORY_MMAP;
    ESP_RETURN_ON_ERROR(ioctl(video->fd, VIDIOC_DQBUF, &buf), TAG, "failed to receive video frame");
    if (!(buf.flags & V4L2_BUF_FLAG_DONE)) {
        return ESP_ERR_INVALID_RESPONSE;
    }

    if (!is_jpeg || video->pixel_format == V4L2_PIX_FMT_JPEG) {
        /* Directly send the buffer of raw data */
        ESP_GOTO_ON_ERROR(httpd_resp_send(req, (char *)video->buffer[buf.index], buf.bytesused), fail0, TAG, "failed to send %s", type_str);
        jpeg_encoded_size = buf.bytesused;
    } else {
        ESP_GOTO_ON_FALSE(xSemaphoreTake(video->sem, portMAX_DELAY) == pdPASS, ESP_FAIL, fail0, TAG, "failed to take semaphore");
        ret = example_encoder_process(video->encoder_handle, video->buffer[buf.index], video->buffer_size,
                                      video->jpeg_out_buf, video->jpeg_out_size, &jpeg_encoded_size);
        xSemaphoreGive(video->sem);
        ESP_GOTO_ON_ERROR(ret, fail0, TAG, "failed to encode video frame");
        ESP_GOTO_ON_ERROR(httpd_resp_send(req, (char *)video->jpeg_out_buf, jpeg_encoded_size), fail0, TAG, "failed to send %s", type_str);
    }

    ESP_RETURN_ON_ERROR(ioctl(video->fd, VIDIOC_QBUF, &buf), TAG, "failed to queue video frame");

    ESP_GOTO_ON_ERROR(httpd_resp_sendstr_chunk(req, NULL), fail0, TAG, "failed to send null");

    ESP_LOGD(TAG, "send %s image%d size: %" PRIu32, type_str, video->index, jpeg_encoded_size);

    return ESP_OK;

fail0:
    ioctl(video->fd, VIDIOC_QBUF, &buf);
    return ret;
}

static char *get_cameras_json(web_cam_t *web_cam)
{
    cJSON *root = cJSON_CreateObject();
    cJSON *cameras = cJSON_CreateArray();
    cJSON_AddItemToObject(root, "cameras", cameras);

    for (int i = 0; i < web_cam->video_count; i++) {
        char src_str[32];

        if (!is_valid_web_cam(&web_cam->video[i])) {
            continue;
        }

        cJSON *camera = cJSON_CreateObject();
        cJSON_AddNumberToObject(camera, "index", i);
        assert(snprintf(src_str, sizeof(src_str), ":%d/stream", i + 81) > 0);
        cJSON_AddStringToObject(camera, "src", src_str);
        cJSON_AddNumberToObject(camera, "currentFrameRate", web_cam->video[i].frame_rate);
        cJSON_AddNumberToObject(camera, "currentImageFormat", 0);
        assert(snprintf(src_str, sizeof(src_str), "JPEG %" PRIu32 "x%" PRIu32, web_cam->video[i].width, web_cam->video[i].height) > 0);
        cJSON_AddStringToObject(camera, "currentImageFormatDescription", src_str);

        if (web_cam->video[i].support_control_jpeg_quality) {
            cJSON_AddNumberToObject(camera, "currentQuality", web_cam->video[i].jpeg_quality);
        }

        cJSON *current_resolution = cJSON_CreateObject();
        cJSON_AddNumberToObject(current_resolution, "width", web_cam->video[i].width);
        cJSON_AddNumberToObject(current_resolution, "height", web_cam->video[i].height);
        cJSON_AddItemToObject(camera, "currentResolution", current_resolution);

        cJSON *image_formats = cJSON_CreateArray();
        cJSON *image_format = cJSON_CreateObject();
        cJSON_AddNumberToObject(image_format, "id", 0);
        assert(snprintf(src_str, sizeof(src_str), "JPEG %" PRIu32 "x%" PRIu32, web_cam->video[i].width, web_cam->video[i].height) > 0);
        cJSON_AddStringToObject(image_format, "description", src_str);

        if (web_cam->video[i].support_control_jpeg_quality) {
            cJSON *image_format_quality = cJSON_CreateObject();

            int min_quality = 1;
            int max_quality = 100;
            int step_quality = 1;
            int default_quality = EXAMPLE_JPEG_ENC_QUALITY;
            if (web_cam->video[i].pixel_format == V4L2_PIX_FMT_JPEG) {
                struct v4l2_query_ext_ctrl qctrl = {0};

                qctrl.id = V4L2_CID_JPEG_COMPRESSION_QUALITY;
                if (ioctl(web_cam->video[i].fd, VIDIOC_QUERY_EXT_CTRL, &qctrl) == 0) {
                    min_quality = qctrl.minimum;
                    max_quality = qctrl.maximum;
                    step_quality = qctrl.step;
                    default_quality = qctrl.default_value;
                }
            }

            cJSON_AddNumberToObject(image_format_quality, "min", min_quality);
            cJSON_AddNumberToObject(image_format_quality, "max", max_quality);
            cJSON_AddNumberToObject(image_format_quality, "step", step_quality);
            cJSON_AddNumberToObject(image_format_quality, "default", default_quality);
            cJSON_AddItemToObject(image_format, "quality", image_format_quality);
        }
        cJSON_AddItemToArray(image_formats, image_format);

        cJSON_AddItemToObject(camera, "imageFormats", image_formats);
        cJSON_AddItemToArray(cameras, camera);
    }

    char *output = cJSON_Print(root);
    cJSON_Delete(root);
    return output;
}

static esp_err_t set_camera_jpeg_quality(web_cam_video_t *video, int quality)
{
    esp_err_t ret = ESP_OK;
    int quality_reset = quality;

    if (video->pixel_format == V4L2_PIX_FMT_JPEG) {
        struct v4l2_ext_controls controls = {0};
        struct v4l2_ext_control control[1];
        struct v4l2_query_ext_ctrl qctrl = {0};

        qctrl.id = V4L2_CID_JPEG_COMPRESSION_QUALITY;
        if (ioctl(video->fd, VIDIOC_QUERY_EXT_CTRL, &qctrl) == 0) {
            if ((quality > qctrl.maximum) || (quality < qctrl.minimum) ||
                    (((quality - qctrl.minimum) % qctrl.step) != 0)) {

                if (quality > qctrl.maximum) {
                    quality_reset = qctrl.maximum;
                } else if (quality < qctrl.minimum) {
                    quality_reset = qctrl.minimum;
                } else {
                    quality_reset = qctrl.minimum + ((quality - qctrl.minimum) / qctrl.step) * qctrl.step;
                }

                ESP_LOGW(TAG, "video%d: JPEG compression quality=%d is out of sensor's range, reset to %d", video->index, quality, quality_reset);
            }

            controls.ctrl_class = V4L2_CID_JPEG_CLASS;
            controls.count = 1;
            controls.controls = control;
            control[0].id = V4L2_CID_JPEG_COMPRESSION_QUALITY;
            control[0].value = quality_reset;
            ESP_RETURN_ON_ERROR(ioctl(video->fd, VIDIOC_S_EXT_CTRLS, &controls), TAG, "failed to set jpeg compression quality");

            video->jpeg_quality = quality_reset;
            video->support_control_jpeg_quality = 1;
        } else {
            video->support_control_jpeg_quality = 0;
            ESP_LOGW(TAG, "video%d: JPEG compression quality control is not supported", video->index);
        }
    } else {
        ESP_RETURN_ON_ERROR(example_encoder_set_jpeg_quality(video->encoder_handle, quality_reset), TAG, "failed to set jpeg quality");
        video->jpeg_quality = quality_reset;
    }

    if (video->support_control_jpeg_quality) {
        ESP_LOGI(TAG, "video%d: set jpeg quality %d success", video->index, quality_reset);
    }

    return ret;
}

static esp_err_t camera_info_handler(httpd_req_t *req)
{
    esp_err_t ret;
    web_cam_t *web_cam = (web_cam_t *)req->user_ctx;
    char *output = get_cameras_json(web_cam);

    httpd_resp_set_type(req, "application/json");
    ret = httpd_resp_sendstr(req, output);
    free(output);

    return ret;
}

static esp_err_t camera_settings_handler(httpd_req_t *req)
{
    esp_err_t ret;
    char *content;
    web_cam_t *web_cam = (web_cam_t *)req->user_ctx;

    content = (char *)calloc(1, req->content_len + 1);
    ESP_RETURN_ON_FALSE(content, ESP_ERR_NO_MEM, TAG, "failed to allocate memory");

    ESP_GOTO_ON_FALSE(httpd_req_recv(req, content, req->content_len) > 0, ESP_FAIL, fail0, TAG, "failed to recv content");
    ESP_LOGD(TAG, "content: %s", content);

    cJSON *json_root = cJSON_Parse(content);
    free(content);
    content = NULL;
    ESP_GOTO_ON_FALSE(json_root, ESP_FAIL, fail0, TAG, "failed to parse JSON");

    cJSON *json_index = cJSON_GetObjectItem(json_root, "index");
    ESP_GOTO_ON_FALSE(json_index && cJSON_IsNumber(json_index), ESP_ERR_INVALID_ARG, fail1, TAG, "missing or invalid index field");
    int index = json_index->valueint;
    ESP_GOTO_ON_FALSE(index >= 0 && index < web_cam->video_count && is_valid_web_cam(&web_cam->video[index]), ESP_ERR_INVALID_ARG, fail1, TAG, "invalid index");

    cJSON *json_image_format = cJSON_GetObjectItem(json_root, "image_format");
    ESP_GOTO_ON_FALSE(json_image_format && cJSON_IsNumber(json_image_format), ESP_ERR_INVALID_ARG, fail1, TAG, "missing or invalid image_format field");
    int image_format = json_image_format->valueint;

    cJSON *json_jpeg_quality = cJSON_GetObjectItem(json_root, "jpeg_quality");
    ESP_GOTO_ON_FALSE(json_jpeg_quality && cJSON_IsNumber(json_jpeg_quality), ESP_ERR_INVALID_ARG, fail1, TAG, "missing or invalid jpeg_quality field");
    int jpeg_quality = json_jpeg_quality->valueint;

    ESP_LOGI(TAG, "JSON parse success - index:%d, image_format:%d, jpeg_quality:%d", index, image_format, jpeg_quality);
    cJSON_Delete(json_root);
    json_root = NULL;

    ESP_GOTO_ON_ERROR(set_camera_jpeg_quality(&web_cam->video[index], jpeg_quality), fail1, TAG, "failed to set camera jpeg quality");

    httpd_resp_sendstr(req, "OK");
    return ESP_OK;

fail1:
    if (json_root) {
        cJSON_Delete(json_root);
    }
fail0:
    if (ret == HTTPD_SOCK_ERR_TIMEOUT) {
        httpd_resp_send_408(req);
    } else {
        httpd_resp_send_err(req, HTTPD_400_BAD_REQUEST, "Invalid JSON format");
    }
    if (content) {
        free(content);
    }
    return ret;
}



static float ld19_normalize_angle(float angle_deg)
{
    while (angle_deg < 0.0f) angle_deg += 360.0f;
    while (angle_deg >= 360.0f) angle_deg -= 360.0f;
    return angle_deg;
}

static void ld19_store_packet_points(const uint8_t *packet)
{
    uint16_t start_raw = (uint16_t)packet[4] | ((uint16_t)packet[5] << 8);
    uint16_t end_raw   = (uint16_t)packet[42] | ((uint16_t)packet[43] << 8);

    float start_deg = start_raw / 100.0f;
    float end_deg = end_raw / 100.0f;
    float delta = end_deg - start_deg;
    if (delta < 0.0f) delta += 360.0f;

    uint32_t seq = ++g_lidar_scan_sequence;

    if (g_lidar_scan_mutex != NULL &&
        xSemaphoreTake(g_lidar_scan_mutex, pdMS_TO_TICKS(5)) == pdTRUE) {

        for (int i = 0; i < 12; ++i) {
            int base = 6 + i * 3;
            uint16_t distance_mm =
                (uint16_t)packet[base] |
                ((uint16_t)packet[base + 1] << 8);
            uint8_t confidence = packet[base + 2];

            float angle_deg = start_deg + delta * ((float)i / 11.0f);
            angle_deg = ld19_normalize_angle(angle_deg);

            int bin = (int)lroundf(angle_deg);
            if (bin >= 360) bin = 0;

            if (distance_mm > 0 &&
                distance_mm <= LIDAR_MAX_DISTANCE_MM &&
                confidence >= LIDAR_MIN_CONFIDENCE) {
                g_lidar_scan[bin].distance_mm = distance_mm;
                g_lidar_scan[bin].confidence = confidence;
                g_lidar_scan[bin].update_seq = seq;
            }
        }

        xSemaphoreGive(g_lidar_scan_mutex);
    }
}

static esp_err_t lidar_scan_handler(httpd_req_t *req)
{
    /*
     * Compact dashboard-oriented LiDAR payload.
     *
     * Instead of returning up to 360 JSON point triplets every request,
     * reduce the rolling 1-degree scan to 72 x 5-degree bins on the P4.
     * The nearest valid return wins in each bin.
     *
     * This substantially reduces Wi-Fi/HTTP traffic while preserving the
     * same angular resolution the existing DWA dashboard already uses.
     *
     * Response:
     * {
     *   "sequence": 12345,
     *   "bin_deg": 5,
     *   "scan_rate_hz": 9.93,
     *   "distances_mm": [72 values]
     * }
     */
    uint16_t distances[72] = {0};

    if (g_lidar_scan_mutex != NULL &&
        xSemaphoreTake(g_lidar_scan_mutex, pdMS_TO_TICKS(10)) == pdTRUE) {

        for (int angle = 0; angle < LIDAR_SCAN_BINS; ++angle) {
            uint16_t distance = g_lidar_scan[angle].distance_mm;

            if (distance == 0) {
                continue;
            }

            int bin = angle / 5;

            if (bin < 0 || bin >= 72) {
                continue;
            }

            if (distances[bin] == 0 || distance < distances[bin]) {
                distances[bin] = distance;
            }
        }

        xSemaphoreGive(g_lidar_scan_mutex);
    }

    const size_t buf_size = 1024;
    char *json = malloc(buf_size);

    if (json == NULL) {
        return ESP_ERR_NO_MEM;
    }

    float scan_rate_hz = g_lidar_status.last_speed_deg_s / 360.0f;

    size_t used = 0;

    int n = snprintf(
        json + used,
        buf_size - used,
        "{\"sequence\":%" PRIu32
        ",\"bin_deg\":5"
        ",\"scan_rate_hz\":%.2f"
        ",\"distances_mm\":[",
        g_lidar_scan_sequence,
        scan_rate_hz
    );

    if (n < 0 || (size_t)n >= buf_size - used) {
        free(json);
        return ESP_FAIL;
    }

    used += (size_t)n;

    for (int i = 0; i < 72; ++i) {
        n = snprintf(
            json + used,
            buf_size - used,
            "%s%u",
            (i == 0) ? "" : ",",
            distances[i]
        );

        if (n < 0 || (size_t)n >= buf_size - used) {
            free(json);
            return ESP_FAIL;
        }

        used += (size_t)n;
    }

    n = snprintf(json + used, buf_size - used, "]}");

    if (n < 0 || (size_t)n >= buf_size - used) {
        free(json);
        return ESP_FAIL;
    }

    used += (size_t)n;

    httpd_resp_set_type(req, "application/json");
    httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
    httpd_resp_set_hdr(req, "Cache-Control", "no-store");

    esp_err_t ret = httpd_resp_send(req, json, used);

    free(json);
    return ret;
}

static esp_err_t lidar_status_handler(httpd_req_t *req)
{
    int64_t now_us = esp_timer_get_time();
    int64_t age_ms = -1;

    if (g_lidar_status.last_packet_time_us > 0) {
        age_ms = (now_us - g_lidar_status.last_packet_time_us) / 1000;
    }

    bool healthy = g_lidar_status.receiving &&
                   g_lidar_status.valid_packets_per_sec > 0 &&
                   age_ms >= 0 &&
                   age_ms < 1000;

    char json[768];

    int len = snprintf(
        json,
        sizeof(json),
        "{"
        "\"healthy\":%s,"
        "\"receiving\":%s,"
        "\"packet_age_ms\":%" PRId64 ","
        "\"bytes_per_sec\":%" PRIu32 ","
        "\"valid_packets_per_sec\":%" PRIu32 ","
        "\"crc_errors_per_sec\":%" PRIu32 ","
        "\"framing_errors_per_sec\":%" PRIu32 ","
        "\"total_valid_packets\":%" PRIu32 ","
        "\"total_crc_errors\":%" PRIu32 ","
        "\"total_framing_errors\":%" PRIu32 ","
        "\"speed_deg_s\":%u,"
        "\"start_angle_deg\":%.2f,"
        "\"end_angle_deg\":%.2f,"
        "\"first_distance_mm\":%u,"
        "\"first_confidence\":%u"
        "}",
        healthy ? "true" : "false",
        g_lidar_status.receiving ? "true" : "false",
        age_ms,
        g_lidar_status.bytes_per_sec,
        g_lidar_status.valid_packets_per_sec,
        g_lidar_status.crc_errors_per_sec,
        g_lidar_status.framing_errors_per_sec,
        g_lidar_status.total_valid_packets,
        g_lidar_status.total_crc_errors,
        g_lidar_status.total_framing_errors,
        g_lidar_status.last_speed_deg_s,
        g_lidar_status.last_start_angle_deg,
        g_lidar_status.last_end_angle_deg,
        g_lidar_status.last_first_distance_mm,
        g_lidar_status.last_first_confidence
    );

    if (len < 0 || len >= (int)sizeof(json)) {
        return ESP_FAIL;
    }

    httpd_resp_set_type(req, "application/json");
    httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
    return httpd_resp_send(req, json, len);
}

static esp_err_t static_file_handler(httpd_req_t *req)
{
    const char *uri = req->uri;

    /* Route to appropriate static file based on URI */
    if (strcmp(uri, "/") == 0) {
        httpd_resp_set_type(req, "text/html");
        httpd_resp_set_hdr(req, "Content-Encoding", "gzip");
        return httpd_resp_send(req, (const char *)index_html_gz_start, index_html_gz_end - index_html_gz_start);
    } else if (strcmp(uri, "/loading.jpg") == 0) {
        httpd_resp_set_type(req, "image/jpeg");
        httpd_resp_set_hdr(req, "Content-Encoding", "gzip");
        return httpd_resp_send(req, (const char *)loading_jpg_gz_start, loading_jpg_gz_end - loading_jpg_gz_start);
    } else if (strcmp(uri, "/favicon.ico") == 0) {
        httpd_resp_set_type(req, "image/x-icon");
        httpd_resp_set_hdr(req, "Content-Encoding", "gzip");
        return httpd_resp_send(req, (const char *)favicon_ico_gz_start, favicon_ico_gz_end - favicon_ico_gz_start);
    } else if (strcmp(uri, "/assets/index.js") == 0) {
        httpd_resp_set_type(req, "application/javascript");
        httpd_resp_set_hdr(req, "Content-Encoding", "gzip");
        return httpd_resp_send(req, (const char *)assets_index_js_gz_start, assets_index_js_gz_end - assets_index_js_gz_start);
    } else if (strcmp(uri, "/assets/index.css") == 0) {
        httpd_resp_set_type(req, "text/css");
        httpd_resp_set_hdr(req, "Content-Encoding", "gzip");
        return httpd_resp_send(req, (const char *)assets_index_css_gz_start, assets_index_css_gz_end - assets_index_css_gz_start);
    }

    /* If no static file matches, return 404 */
    ESP_LOGW(TAG, "File not found: %s", uri);
    httpd_resp_send_404(req);
    return ESP_FAIL;
}

static esp_err_t image_stream_handler(httpd_req_t *req)
{
    esp_err_t ret = ESP_OK;
    struct v4l2_buffer buf;
    char http_string[128];
    web_cam_video_t *video = (web_cam_video_t *)req->user_ctx;

    uint8_t *tx_buf = NULL;
    uint32_t tx_buf_size = 0;

    bool encoder_locked = false;
    bool camera_buffer_dequeued = false;
    int64_t last_stream_frame_us = 0;

    ESP_RETURN_ON_FALSE(
        snprintf(
            http_string,
            sizeof(http_string),
            "%" PRIu32,
            video->frame_rate
        ) > 0,
        ESP_FAIL,
        TAG,
        "failed to format framerate buffer"
    );

    ESP_RETURN_ON_ERROR(
        httpd_resp_set_type(req, STREAM_CONTENT_TYPE),
        TAG,
        "failed to set content type"
    );

    ESP_RETURN_ON_ERROR(
        httpd_resp_set_hdr(
            req,
            "Access-Control-Allow-Origin",
            "*"
        ),
        TAG,
        "failed to set access control allow origin"
    );

    ESP_RETURN_ON_ERROR(
        httpd_resp_set_hdr(
            req,
            "X-Framerate",
            http_string
        ),
        TAG,
        "failed to set x framerate"
    );

    while (1) {
        int hlen;
        struct timespec ts;

        uint32_t jpeg_encoded_size = 0;


        /*
         * Cap MJPEG transmission to 20 FPS. The previous stream could generate
         * frames faster than the SoftAP/TCP link could drain them, producing
         * multi-second httpd_resp_send_chunk() stalls.
         */
        if (last_stream_frame_us != 0) {
            int64_t elapsed_us = esp_timer_get_time() - last_stream_frame_us;

            if (elapsed_us < V4_STREAM_MIN_FRAME_INTERVAL_US) {
                int64_t remaining_us =
                    V4_STREAM_MIN_FRAME_INTERVAL_US - elapsed_us;

                vTaskDelay(
                    pdMS_TO_TICKS(
                        (remaining_us + 999) / 1000
                    )
                );
            }
        }

        last_stream_frame_us = esp_timer_get_time();

        int64_t frame_start_us = esp_timer_get_time();
        int64_t dq_done_us = 0;
        int64_t encode_done_us = 0;
        int64_t qbuf_done_us = 0;
        int64_t send_done_us = 0;

        encoder_locked = false;
        camera_buffer_dequeued = false;

        memset(
            &buf,
            0,
            sizeof(buf)
        );

        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_MMAP;

        ret = ioctl(
            video->fd,
            VIDIOC_DQBUF,
            &buf
        );

        if (ret != 0) {
            ESP_LOGE(
                TAG,
                "failed to receive video frame"
            );
            return ESP_FAIL;
        }

        camera_buffer_dequeued = true;
        dq_done_us = esp_timer_get_time();

        if (!(buf.flags & V4L2_BUF_FLAG_DONE)) {
            ret = ioctl(
                video->fd,
                VIDIOC_QBUF,
                &buf
            );

            camera_buffer_dequeued = false;

            if (ret != 0) {
                ESP_LOGE(
                    TAG,
                    "failed to queue incomplete video frame"
                );
                return ESP_FAIL;
            }

            continue;
        }

        /*
         * IMPORTANT:
         * Do not hold a V4L2 capture buffer while sending JPEG data over Wi-Fi.
         *
         * For a camera that already outputs JPEG, copy the JPEG into a private
         * transmit buffer first.
         *
         * For non-JPEG camera output, encode into the encoder output buffer,
         * then copy that JPEG into the private transmit buffer.
         *
         * Once the private copy is complete, immediately QBUF the camera frame
         * back to the driver BEFORE any HTTP transmission.
         */

        if (video->pixel_format == V4L2_PIX_FMT_JPEG) {
            jpeg_encoded_size = buf.bytesused;

            if (jpeg_encoded_size == 0) {
                ret = ESP_ERR_INVALID_SIZE;
                goto stream_fail;
            }

            if (tx_buf_size < jpeg_encoded_size) {
                uint8_t *new_buf = realloc(
                    tx_buf,
                    jpeg_encoded_size
                );

                if (new_buf == NULL) {
                    ret = ESP_ERR_NO_MEM;
                    goto stream_fail;
                }

                tx_buf = new_buf;
                tx_buf_size = jpeg_encoded_size;
            }

            memcpy(
                tx_buf,
                video->buffer[buf.index],
                jpeg_encoded_size
            );

            encode_done_us = esp_timer_get_time();
        } else {
            if (
                xSemaphoreTake(
                    video->sem,
                    portMAX_DELAY
                ) != pdPASS
            ) {
                ret = ESP_FAIL;
                goto stream_fail;
            }

            encoder_locked = true;

            ret = example_encoder_process(
                video->encoder_handle,
                video->buffer[buf.index],
                video->buffer_size,
                video->jpeg_out_buf,
                video->jpeg_out_size,
                &jpeg_encoded_size
            );

            if (ret != ESP_OK) {
                ESP_LOGE(
                    TAG,
                    "failed to encode video frame"
                );
                goto stream_fail;
            }

            if (jpeg_encoded_size == 0) {
                ret = ESP_ERR_INVALID_SIZE;
                goto stream_fail;
            }

            if (tx_buf_size < jpeg_encoded_size) {
                uint8_t *new_buf = realloc(
                    tx_buf,
                    jpeg_encoded_size
                );

                if (new_buf == NULL) {
                    ret = ESP_ERR_NO_MEM;
                    goto stream_fail;
                }

                tx_buf = new_buf;
                tx_buf_size = jpeg_encoded_size;
            }

            memcpy(
                tx_buf,
                video->jpeg_out_buf,
                jpeg_encoded_size
            );

            xSemaphoreGive(
                video->sem
            );

            encoder_locked = false;
            encode_done_us = esp_timer_get_time();
        }

        /*
         * Return the capture buffer immediately, before doing any TCP send.
         */
        ret = ioctl(
            video->fd,
            VIDIOC_QBUF,
            &buf
        );

        if (ret != 0) {
            ret = ESP_FAIL;
            ESP_LOGE(
                TAG,
                "failed to re-queue video frame"
            );
            camera_buffer_dequeued = false;
            goto stream_fail;
        }

        camera_buffer_dequeued = false;
        qbuf_done_us = esp_timer_get_time();

        /*
         * Only now begin the potentially slow network transmission.
         */
        ret = httpd_resp_send_chunk(
            req,
            STREAM_BOUNDARY,
            strlen(STREAM_BOUNDARY)
        );

        if (ret != ESP_OK) {
            goto stream_fail;
        }

        if (
            clock_gettime(
                CLOCK_MONOTONIC,
                &ts
            ) != 0
        ) {
            ret = ESP_FAIL;
            goto stream_fail;
        }

        hlen = snprintf(
            http_string,
            sizeof(http_string),
            STREAM_PART,
            jpeg_encoded_size,
            (int)ts.tv_sec,
            (int)(ts.tv_nsec / 1000)
        );

        if (hlen <= 0) {
            ret = ESP_FAIL;
            goto stream_fail;
        }

        ret = httpd_resp_send_chunk(
            req,
            http_string,
            hlen
        );

        if (ret != ESP_OK) {
            goto stream_fail;
        }

        ret = httpd_resp_send_chunk(
            req,
            (char *)tx_buf,
            jpeg_encoded_size
        );

        if (ret != ESP_OK) {
            goto stream_fail;
        }

        send_done_us = esp_timer_get_time();

        /*
         * Timing diagnostics. These only print when a stage is unusually slow.
         */
        int64_t wait_ms = (
            dq_done_us
            - frame_start_us
        ) / 1000;

        int64_t camera_owned_ms = (
            qbuf_done_us
            - dq_done_us
        ) / 1000;

        int64_t send_ms = (
            send_done_us
            - qbuf_done_us
        ) / 1000;

        int64_t total_ms = (
            send_done_us
            - frame_start_us
        ) / 1000;

        if (wait_ms >= 200) {
            ESP_LOGW(
                TAG,
                "slow camera frame wait: %" PRId64 " ms",
                wait_ms
            );
        }

        if (camera_owned_ms >= 150) {
            ESP_LOGW(
                TAG,
                "slow camera-owned stage: %" PRId64
                " ms, JPEG=%" PRIu32 " bytes",
                camera_owned_ms,
                jpeg_encoded_size
            );
        }

        if (send_ms >= 200) {
            ESP_LOGW(
                TAG,
                "slow stream send: %" PRId64
                " ms, JPEG=%" PRIu32 " bytes",
                send_ms,
                jpeg_encoded_size
            );
        }

        if (total_ms >= 300) {
            ESP_LOGW(
                TAG,
                "slow stream frame: total=%" PRId64
                " ms, wait=%" PRId64
                " ms, camera_owned=%" PRId64
                " ms, send=%" PRId64 " ms",
                total_ms,
                wait_ms,
                camera_owned_ms,
                send_ms
            );
        }
    }

stream_fail:
    if (encoder_locked) {
        xSemaphoreGive(
            video->sem
        );
        encoder_locked = false;
    }

    if (camera_buffer_dequeued) {
        ioctl(
            video->fd,
            VIDIOC_QBUF,
            &buf
        );
        camera_buffer_dequeued = false;
    }

    free(
        tx_buf
    );

    tx_buf = NULL;
    tx_buf_size = 0;

    return ret;
}

static esp_err_t capture_image_handler(httpd_req_t *req)
{
    web_cam_t *web_cam = (web_cam_t *)req->user_ctx;

    request_desc_t desc;
    ESP_RETURN_ON_ERROR(decode_request(web_cam, req, &desc), TAG, "failed to decode request");

    char type_ptr[32];
    ESP_RETURN_ON_FALSE(snprintf(type_ptr, sizeof(type_ptr), "image/jpeg;name=image%d.jpg", desc.index) > 0, ESP_FAIL, TAG, "failed to format buffer");
    ESP_RETURN_ON_ERROR(httpd_resp_set_type(req, type_ptr), TAG, "failed to set content type");

    return capture_video_image(req, &web_cam->video[desc.index], true);
}

static esp_err_t capture_binary_handler(httpd_req_t *req)
{
    web_cam_t *web_cam = (web_cam_t *)req->user_ctx;

    request_desc_t desc;
    ESP_RETURN_ON_ERROR(decode_request(web_cam, req, &desc), TAG, "failed to decode request");

    char type_ptr[56];
    ESP_RETURN_ON_FALSE(snprintf(type_ptr, sizeof(type_ptr), "application/octet-stream;name=image_binary%d.bin", desc.index) > 0, ESP_FAIL, TAG, "failed to format buffer");
    ESP_RETURN_ON_ERROR(httpd_resp_set_type(req, type_ptr), TAG, "failed to set content type");

    return capture_video_image(req, &web_cam->video[desc.index], false);
}

static esp_err_t init_web_cam_video(web_cam_video_t *video, const web_cam_video_config_t *config, int index)
{
    int fd;
    int ret;
    struct v4l2_format format;
    struct v4l2_streamparm sparm;
    struct v4l2_requestbuffers req;
    struct v4l2_captureparm *cparam = &sparm.parm.capture;
    struct v4l2_fract *timeperframe = &cparam->timeperframe;

    fd = open(config->dev_name, O_RDWR);
    ESP_RETURN_ON_FALSE(fd >= 0, ESP_ERR_NOT_FOUND, TAG, "Open video device %s failed", config->dev_name);

    memset(&format, 0, sizeof(struct v4l2_format));
    format.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    ESP_GOTO_ON_ERROR(ioctl(fd, VIDIOC_G_FMT, &format), fail0, TAG, "Failed get fmt from %s", config->dev_name);

#if CONFIG_EXAMPLE_SELECT_JPEG_HW_DRIVER
    if (format.fmt.pix.pixelformat == V4L2_PIX_FMT_RGB565X) {
#if CONFIG_ESP_VIDEO_ENABLE_SWAP_BYTE
        ESP_LOGW(TAG, "The hardware JPEG encoder does not support RGB565 big endian. Instead, use RGB565 little endian by enabling the byte swap function.");

        format.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        format.fmt.pix.pixelformat = V4L2_PIX_FMT_RGB565;
        ESP_GOTO_ON_ERROR(ioctl(fd, VIDIOC_S_FMT, &format), fail0, TAG, "failed to set fmt to %s", config->dev_name);
#else
        ESP_GOTO_ON_ERROR(ESP_FAIL, fail0, TAG, "The hardware JPEG encoder does not support RGB565 big endian. Please enable the byte swap function ESP_VIDEO_ENABLE_SWAP_BYTE in menuconfig.");
#endif
    }
#endif

    memset(&sparm, 0, sizeof(sparm));
    sparm.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    ESP_GOTO_ON_ERROR(ioctl(fd, VIDIOC_G_PARM, &sparm), fail0, TAG, "failed to get frame rate from %s", config->dev_name);
    video->frame_rate = timeperframe->denominator / timeperframe->numerator;

#if CONFIG_EXAMPLE_ENABLE_MIPI_CSI_CROP
    /**
     * Command VIDIOC_S_SELECTION should be called before VIDIOC_REQBUFS.
     */

    struct v4l2_selection selection;

    memset(&selection, 0, sizeof(selection));
    selection.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    selection.target = V4L2_SEL_TGT_CROP;
    selection.r.left = CONFIG_EXAMPLE_MIPI_CSI_CROP_LEFT;
    selection.r.width = CONFIG_EXAMPLE_MIPI_CSI_CROP_WIDTH;
    selection.r.top = CONFIG_EXAMPLE_MIPI_CSI_CROP_TOP;
    selection.r.height = CONFIG_EXAMPLE_MIPI_CSI_CROP_HEIGHT;
    if (ioctl(fd, VIDIOC_S_SELECTION, &selection) != 0) {
        ESP_LOGE(TAG, "failed to set selection");
    }
#endif

    memset(&req, 0, sizeof(req));
    req.count  = EXAMPLE_CAMERA_VIDEO_BUFFER_NUMBER;
    req.type   = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    req.memory = V4L2_MEMORY_MMAP;
    ESP_GOTO_ON_ERROR(ioctl(fd, VIDIOC_REQBUFS, &req), fail0, TAG, "failed to req buffers from %s", config->dev_name);

    for (int i = 0; i < EXAMPLE_CAMERA_VIDEO_BUFFER_NUMBER; i++) {
        struct v4l2_buffer buf;

        memset(&buf, 0, sizeof(buf));
        buf.type        = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory      = V4L2_MEMORY_MMAP;
        buf.index       = i;
        ESP_GOTO_ON_ERROR(ioctl(fd, VIDIOC_QUERYBUF, &buf), fail0, TAG, "failed to query vbuf from %s", config->dev_name);

        video->buffer[i] = mmap(NULL, buf.length, PROT_READ | PROT_WRITE, MAP_SHARED, fd, buf.m.offset);
        ESP_GOTO_ON_FALSE(video->buffer[i] != MAP_FAILED, ESP_ERR_NO_MEM, fail0, TAG, "failed to mmap buffer");
        video->buffer_size = buf.length;

        ESP_GOTO_ON_ERROR(ioctl(fd, VIDIOC_QBUF, &buf), fail0, TAG, "failed to queue frame vbuf from %s", config->dev_name);
    }

    video->fd = fd;
    video->width = format.fmt.pix.width;
    video->height = format.fmt.pix.height;
    video->pixel_format = format.fmt.pix.pixelformat;
    video->jpeg_quality = EXAMPLE_JPEG_ENC_QUALITY;

    if (video->pixel_format == V4L2_PIX_FMT_JPEG) {
        ESP_GOTO_ON_ERROR(set_camera_jpeg_quality(video, EXAMPLE_JPEG_ENC_QUALITY), fail0, TAG, "failed to set jpeg quality");
    } else {
        example_encoder_config_t encoder_config = {0};

        encoder_config.width = video->width;
        encoder_config.height = video->height;
        encoder_config.pixel_format = video->pixel_format;
        encoder_config.quality = EXAMPLE_JPEG_ENC_QUALITY;
        ESP_GOTO_ON_ERROR(example_encoder_init(&encoder_config, &video->encoder_handle), fail0, TAG, "failed to init encoder");

        ESP_GOTO_ON_ERROR(example_encoder_alloc_output_buffer(video->encoder_handle, &video->jpeg_out_buf, &video->jpeg_out_size),
                          fail1, TAG, "failed to alloc jpeg output buf");

        video->support_control_jpeg_quality = 1;
    }

    video->sem = xSemaphoreCreateBinary();
    ESP_GOTO_ON_FALSE(video->sem, ESP_ERR_NO_MEM, fail2, TAG, "failed to create semaphore");
    xSemaphoreGive(video->sem);

    return ESP_OK;

fail2:
    if (video->pixel_format != V4L2_PIX_FMT_JPEG) {
        example_encoder_free_output_buffer(video->encoder_handle, video->jpeg_out_buf);
        video->jpeg_out_buf = NULL;
    }
fail1:
    if (video->pixel_format != V4L2_PIX_FMT_JPEG) {
        example_encoder_deinit(video->encoder_handle);
        video->encoder_handle = NULL;
    }
fail0:
    close(fd);
    video->fd = -1;
    return ret;
}

static esp_err_t deinit_web_cam_video(web_cam_video_t *video)
{
    if (video->sem) {
        vSemaphoreDelete(video->sem);
        video->sem = NULL;
    }

    if (video->pixel_format != V4L2_PIX_FMT_JPEG) {
        example_encoder_free_output_buffer(video->encoder_handle, video->jpeg_out_buf);
        example_encoder_deinit(video->encoder_handle);
    }

    close(video->fd);
    return ESP_OK;
}

static esp_err_t new_web_cam(const web_cam_video_config_t *config, int config_count, web_cam_t **ret_wc)
{
    int i;
    web_cam_t *wc;
    esp_err_t ret = ESP_FAIL;
    int type = V4L2_BUF_TYPE_VIDEO_CAPTURE;

    wc = calloc(1, sizeof(web_cam_t) + config_count * sizeof(web_cam_video_t));
    ESP_RETURN_ON_FALSE(wc, ESP_ERR_NO_MEM, TAG, "failed to alloc web cam");
    wc->video_count = config_count;

    for (i = 0; i < config_count; i++) {
        wc->video[i].index = i;
        wc->video[i].fd = -1;

        ret = init_web_cam_video(&wc->video[i], &config[i], i);
        if (ret == ESP_ERR_NOT_FOUND) {
            ESP_LOGW(TAG, "failed to find web_cam %d", i);
            continue;
        } else if (ret != ESP_OK) {
            ESP_LOGE(TAG, "failed to initialize web_cam %d", i);
            goto fail0;
        }

        ESP_LOGI(TAG, "video%d: width=%" PRIu32 " height=%" PRIu32 " format=" V4L2_FMT_STR, i, wc->video[i].width,
                 wc->video[i].height, V4L2_FMT_STR_ARG(wc->video[i].pixel_format));
    }

    for (i = 0; i < config_count; i++) {
        if (is_valid_web_cam(&wc->video[i])) {
            ESP_GOTO_ON_ERROR(ioctl(wc->video[i].fd, VIDIOC_STREAMON, &type), fail1, TAG, "failed to start stream");
        }
    }

    *ret_wc = wc;

    return ESP_OK;

fail1:
    for (int j = i - 1; j >= 0; j--) {
        if (is_valid_web_cam(&wc->video[j])) {
            ioctl(wc->video[j].fd, VIDIOC_STREAMOFF, &type);
        }
    }
    i = config_count; // deinit all web_cam
fail0:
    for (int j = i - 1; j >= 0; j--) {
        if (is_valid_web_cam(&wc->video[j])) {
            deinit_web_cam_video(&wc->video[j]);
        }
    }
    free(wc);
    return ret;
}

static void free_web_cam(web_cam_t *web_cam)
{
    for (int i = 0; i < web_cam->video_count; i++) {
        if (is_valid_web_cam(&web_cam->video[i])) {
            deinit_web_cam_video(&web_cam->video[i]);
        }
    }
    free(web_cam);
}

static esp_err_t http_server_init(web_cam_t *web_cam)
{
    httpd_handle_t stream_httpd;
    httpd_config_t config = HTTPD_DEFAULT_CONFIG();
    /* Prevent a congested MJPEG client from blocking a send for multiple seconds. */
    config.send_wait_timeout = 1;
    config.uri_match_fn = httpd_uri_match_wildcard;

    /* Unified static file handler for all static resources */
    httpd_uri_t static_file_uri = {
        .uri = "/*",
        .method = HTTP_GET,
        .handler = static_file_handler,
        .user_ctx = (void *)web_cam
    };

    /* API handlers */
    httpd_uri_t capture_image_uri = {
        .uri = "/api/capture_image",
        .method = HTTP_GET,
        .handler = capture_image_handler,
        .user_ctx = (void *)web_cam
    };

    httpd_uri_t capture_binary_uri = {
        .uri = "/api/capture_binary",
        .method = HTTP_GET,
        .handler = capture_binary_handler,
        .user_ctx = (void *)web_cam
    };

    httpd_uri_t camera_info_uri = {
        .uri = "/api/get_camera_info",
        .method = HTTP_GET,
        .handler = camera_info_handler,
        .user_ctx = (void *)web_cam
    };

    httpd_uri_t camera_settings_uri = {
        .uri = "/api/set_camera_config",
        .method = HTTP_POST,
        .handler = camera_settings_handler,
        .user_ctx = (void *)web_cam
    };


    httpd_uri_t lidar_status_uri = {
        .uri = "/api/lidar_status",
        .method = HTTP_GET,
        .handler = lidar_status_handler,
        .user_ctx = NULL
    };


    httpd_uri_t lidar_scan_uri = {
        .uri = "/api/lidar_scan",
        .method = HTTP_GET,
        .handler = lidar_scan_handler,
        .user_ctx = NULL
    };

    config.stack_size = 1024 * 6;
    ESP_LOGI(TAG, "Starting stream server on port: '%d'", config.server_port);
    if (httpd_start(&stream_httpd, &config) == ESP_OK) {
        /* Register API handlers (more specific URIs) */
        httpd_register_uri_handler(stream_httpd, &capture_image_uri);
        httpd_register_uri_handler(stream_httpd, &capture_binary_uri);
        httpd_register_uri_handler(stream_httpd, &camera_info_uri);
        httpd_register_uri_handler(stream_httpd, &camera_settings_uri);
        httpd_register_uri_handler(stream_httpd, &lidar_status_uri);
        httpd_register_uri_handler(stream_httpd, &lidar_scan_uri);

        /* Register wildcard static file handler to catch all other requests */
        httpd_register_uri_handler(stream_httpd, &static_file_uri);
    }

    for (int i = 0; i < web_cam->video_count; i++) {
        if (!is_valid_web_cam(&web_cam->video[i])) {
            continue;
        }

        httpd_uri_t stream_0_uri = {
            .uri = "/stream",
            .method = HTTP_GET,
            .handler = image_stream_handler,
            .user_ctx = (void *) &web_cam->video[i]
        };

        config.stack_size = 1024 * 6;
        config.server_port += 1;
        config.ctrl_port += 1;
        if (httpd_start(&stream_httpd, &config) == ESP_OK) {
            httpd_register_uri_handler(stream_httpd, &stream_0_uri);
        }
    }

    return ESP_OK;
}

static esp_err_t start_cam_web_server(const web_cam_video_config_t *config, int config_count)
{
    esp_err_t ret;
    web_cam_t *web_cam;

    ESP_RETURN_ON_ERROR(new_web_cam(config, config_count, &web_cam), TAG, "Failed to new web cam");
    ESP_GOTO_ON_ERROR(http_server_init(web_cam), fail0, TAG, "Failed to init http server");

    return ESP_OK;

fail0:
    free_web_cam(web_cam);
    return ret;
}


static uint32_t rc_pulse_us_to_duty(uint32_t pulse_us)
{
    if (pulse_us > RC_PWM_PERIOD_US) {
        pulse_us = RC_PWM_PERIOD_US;
    }

    return (uint32_t)(((uint64_t)pulse_us * RC_PWM_MAX_DUTY) / RC_PWM_PERIOD_US);
}

static esp_err_t rc_set_channel_us(ledc_channel_t channel, uint32_t pulse_us)
{
    uint32_t duty = rc_pulse_us_to_duty(pulse_us);

    ESP_RETURN_ON_ERROR(
        ledc_set_duty(RC_PWM_MODE, channel, duty),
        TAG,
        "Failed to set PWM duty"
    );

    ESP_RETURN_ON_ERROR(
        ledc_update_duty(RC_PWM_MODE, channel),
        TAG,
        "Failed to update PWM duty"
    );

    return ESP_OK;
}

static esp_err_t rc_set_steering_us(uint32_t pulse_us)
{
    if (pulse_us < RC_STEERING_MIN_US) {
        pulse_us = RC_STEERING_MIN_US;
    } else if (pulse_us > RC_STEERING_MAX_US) {
        pulse_us = RC_STEERING_MAX_US;
    }

    return rc_set_channel_us(RC_STEERING_CHANNEL, pulse_us);
}

static esp_err_t rc_set_esc_us(uint32_t pulse_us)
{
    if (pulse_us < RC_ESC_MIN_US) {
        pulse_us = RC_ESC_MIN_US;
    } else if (pulse_us > RC_ESC_MAX_US) {
        pulse_us = RC_ESC_MAX_US;
    }

    return rc_set_channel_us(RC_ESC_CHANNEL, pulse_us);
}

static void rc_go_safe(void)
{
    rc_set_steering_us(RC_STEERING_CENTER_US);
    rc_set_esc_us(RC_ESC_NEUTRAL_US);
}

static void rc_pwm_init(void)
{
    ledc_timer_config_t timer_cfg = {
        .speed_mode = RC_PWM_MODE,
        .duty_resolution = RC_PWM_RESOLUTION,
        .timer_num = RC_PWM_TIMER,
        .freq_hz = RC_PWM_FREQUENCY_HZ,
        .clk_cfg = LEDC_AUTO_CLK,
        .deconfigure = false,
    };

    ESP_ERROR_CHECK(ledc_timer_config(&timer_cfg));

    ledc_channel_config_t steering_cfg = {
        .gpio_num = RC_STEERING_GPIO,
        .speed_mode = RC_PWM_MODE,
        .channel = RC_STEERING_CHANNEL,
        .intr_type = LEDC_INTR_DISABLE,
        .timer_sel = RC_PWM_TIMER,
        .duty = rc_pulse_us_to_duty(RC_STEERING_CENTER_US),
        .hpoint = 0,
        .sleep_mode = LEDC_SLEEP_MODE_NO_ALIVE_NO_PD,
        .flags.output_invert = 0,
    };

    ledc_channel_config_t esc_cfg = {
        .gpio_num = RC_ESC_GPIO,
        .speed_mode = RC_PWM_MODE,
        .channel = RC_ESC_CHANNEL,
        .intr_type = LEDC_INTR_DISABLE,
        .timer_sel = RC_PWM_TIMER,
        .duty = rc_pulse_us_to_duty(RC_ESC_NEUTRAL_US),
        .hpoint = 0,
        .sleep_mode = LEDC_SLEEP_MODE_NO_ALIVE_NO_PD,
        .flags.output_invert = 0,
    };

    ESP_ERROR_CHECK(ledc_channel_config(&steering_cfg));
    ESP_ERROR_CHECK(ledc_channel_config(&esc_cfg));

    ESP_ERROR_CHECK(rc_set_steering_us(RC_STEERING_CENTER_US));
    ESP_ERROR_CHECK(rc_set_esc_us(RC_ESC_NEUTRAL_US));

    ESP_LOGI(TAG, "RC PWM initialized");
    ESP_LOGI(TAG, "Steering GPIO %d -> %u us center", RC_STEERING_GPIO, RC_STEERING_CENTER_US);
    ESP_LOGI(TAG, "ESC GPIO %d -> %u us neutral", RC_ESC_GPIO, RC_ESC_NEUTRAL_US);
}

static bool rc_process_udp_command(char *cmd)
{
    while (*cmd == ' ' || *cmd == '\t' || *cmd == '\r' || *cmd == '\n') {
        cmd++;
    }

    size_t len = strlen(cmd);
    while (len > 0 &&
           (cmd[len - 1] == ' ' || cmd[len - 1] == '\t' ||
            cmd[len - 1] == '\r' || cmd[len - 1] == '\n')) {
        cmd[--len] = '\0';
    }

    if (len == 0) {
        return false;
    }

    /*
     * Single-character commands:
     * A = steer left
     * D = steer right
     * C = steering center
     * W = forward test throttle
     * S = reverse test throttle
     * X = throttle neutral
     * N = neutral EVERYTHING immediately
     *
     * The ESC may remain physically disconnected during steering-only testing.
     */
    if (len == 1) {
        switch (cmd[0]) {
            case 'A':
            case 'a':
                ESP_ERROR_CHECK(rc_set_steering_us(RC_STEERING_LEFT_US));
                return true;

            case 'D':
            case 'd':
                ESP_ERROR_CHECK(rc_set_steering_us(RC_STEERING_RIGHT_US));
                return true;

            case 'C':
            case 'c':
                ESP_ERROR_CHECK(rc_set_steering_us(RC_STEERING_CENTER_US));
                return true;

            case 'W':
            case 'w':
                ESP_ERROR_CHECK(rc_set_esc_us(RC_ESC_MAX_US));
                return true;

            case 'S':
            case 's':
                ESP_ERROR_CHECK(rc_set_esc_us(RC_ESC_MIN_US));
                return true;

            case 'X':
            case 'x':
                ESP_ERROR_CHECK(rc_set_esc_us(RC_ESC_NEUTRAL_US));
                return true;

            case 'N':
            case 'n':
                rc_go_safe();
                return true;

            default:
                return false;
        }
    }

    /*
     * Numeric commands for the later Python dashboard/DWA integration:
     *
     *   STEER:1500
     *   ESC:1500
     *   CTRL:1500,1500
     */
    unsigned int steer_us = 0;
    unsigned int esc_us = 0;

    if (sscanf(cmd, "STEER:%u", &steer_us) == 1) {
        ESP_ERROR_CHECK(rc_set_steering_us(steer_us));
        return true;
    }

    if (sscanf(cmd, "ESC:%u", &esc_us) == 1) {
        ESP_ERROR_CHECK(rc_set_esc_us(esc_us));
        return true;
    }

    if (sscanf(cmd, "CTRL:%u,%u", &steer_us, &esc_us) == 2) {
        ESP_ERROR_CHECK(rc_set_steering_us(steer_us));
        ESP_ERROR_CHECK(rc_set_esc_us(esc_us));
        return true;
    }

    return false;
}

static void rc_udp_control_task(void *arg)
{
    (void)arg;

    char rx_buffer[RC_UDP_RX_BUFFER_SIZE];
    struct sockaddr_in listen_addr = {
        .sin_family = AF_INET,
        .sin_port = htons(RC_UDP_PORT),
        .sin_addr.s_addr = htonl(INADDR_ANY),
    };

    int sock = socket(AF_INET, SOCK_DGRAM, IPPROTO_IP);
    if (sock < 0) {
        ESP_LOGE(TAG, "Unable to create UDP socket: errno %d", errno);
        rc_go_safe();
        vTaskDelete(NULL);
        return;
    }

    struct timeval timeout = {
        .tv_sec = 0,
        .tv_usec = 50000,  /* 50 ms receive timeout */
    };

    if (setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout)) < 0) {
        ESP_LOGW(TAG, "Could not set UDP receive timeout: errno %d", errno);
    }

    if (bind(sock, (struct sockaddr *)&listen_addr, sizeof(listen_addr)) < 0) {
        ESP_LOGE(TAG, "Unable to bind UDP socket to port %d: errno %d", RC_UDP_PORT, errno);
        close(sock);
        rc_go_safe();
        vTaskDelete(NULL);
        return;
    }

    ESP_LOGI(TAG, "RC UDP control listening on 192.168.4.1:%d", RC_UDP_PORT);
    ESP_LOGI(TAG, "Commands: A/D/C steering, W/S/X throttle, N full neutral");
    ESP_LOGI(TAG, "Failsafe: %d ms without valid command -> center + neutral", RC_FAILSAFE_TIMEOUT_MS);

    int64_t last_valid_command_us = esp_timer_get_time();
    bool failsafe_active = false;

    while (1) {
        struct sockaddr_storage source_addr;
        socklen_t source_addr_len = sizeof(source_addr);

        int len = recvfrom(
            sock,
            rx_buffer,
            sizeof(rx_buffer) - 1,
            0,
            (struct sockaddr *)&source_addr,
            &source_addr_len
        );

        if (len > 0) {
            rx_buffer[len] = '\0';

            if (rc_process_udp_command(rx_buffer)) {
                last_valid_command_us = esp_timer_get_time();

                if (failsafe_active) {
                    ESP_LOGI(TAG, "RC control link restored");
                    failsafe_active = false;
                }
            } else {
                ESP_LOGW(TAG, "Unknown RC command: '%s'", rx_buffer);
            }
        } else if (len < 0 && errno != EAGAIN && errno != EWOULDBLOCK) {
            ESP_LOGW(TAG, "UDP recvfrom error: errno %d", errno);
        }

        int64_t now_us = esp_timer_get_time();
        int64_t elapsed_ms = (now_us - last_valid_command_us) / 1000;

        if (elapsed_ms >= RC_FAILSAFE_TIMEOUT_MS && !failsafe_active) {
            rc_go_safe();
            failsafe_active = true;
            ESP_LOGW(TAG, "RC FAILSAFE -> steering center, throttle neutral");
        }
    }
}


static uint8_t ld19_crc8(const uint8_t *data, size_t len)
{
    static const uint8_t table[256] = {
        0x00,0x4d,0x9a,0xd7,0x79,0x34,0xe3,0xae,0xf2,0xbf,0x68,0x25,0x8b,0xc6,0x11,0x5c,
        0xa9,0xe4,0x33,0x7e,0xd0,0x9d,0x4a,0x07,0x5b,0x16,0xc1,0x8c,0x22,0x6f,0xb8,0xf5,
        0x1f,0x52,0x85,0xc8,0x66,0x2b,0xfc,0xb1,0xed,0xa0,0x77,0x3a,0x94,0xd9,0x0e,0x43,
        0xb6,0xfb,0x2c,0x61,0xcf,0x82,0x55,0x18,0x44,0x09,0xde,0x93,0x3d,0x70,0xa7,0xea,
        0x3e,0x73,0xa4,0xe9,0x47,0x0a,0xdd,0x90,0xcc,0x81,0x56,0x1b,0xb5,0xf8,0x2f,0x62,
        0x97,0xda,0x0d,0x40,0xee,0xa3,0x74,0x39,0x65,0x28,0xff,0xb2,0x1c,0x51,0x86,0xcb,
        0x21,0x6c,0xbb,0xf6,0x58,0x15,0xc2,0x8f,0xd3,0x9e,0x49,0x04,0xaa,0xe7,0x30,0x7d,
        0x88,0xc5,0x12,0x5f,0xf1,0xbc,0x6b,0x26,0x7a,0x37,0xe0,0xad,0x03,0x4e,0x99,0xd4,
        0x7c,0x31,0xe6,0xab,0x05,0x48,0x9f,0xd2,0x8e,0xc3,0x14,0x59,0xf7,0xba,0x6d,0x20,
        0xd5,0x98,0x4f,0x02,0xac,0xe1,0x36,0x7b,0x27,0x6a,0xbd,0xf0,0x5e,0x13,0xc4,0x89,
        0x63,0x2e,0xf9,0xb4,0x1a,0x57,0x80,0xcd,0x91,0xdc,0x0b,0x46,0xe8,0xa5,0x72,0x3f,
        0xca,0x87,0x50,0x1d,0xb3,0xfe,0x29,0x64,0x38,0x75,0xa2,0xef,0x41,0x0c,0xdb,0x96,
        0x42,0x0f,0xd8,0x95,0x3b,0x76,0xa1,0xec,0xb0,0xfd,0x2a,0x67,0xc9,0x84,0x53,0x1e,
        0xeb,0xa6,0x71,0x3c,0x92,0xdf,0x08,0x45,0x19,0x54,0x83,0xce,0x60,0x2d,0xfa,0xb7,
        0x5d,0x10,0xc7,0x8a,0x24,0x69,0xbe,0xf3,0xaf,0xe2,0x35,0x78,0xd6,0x9b,0x4c,0x01,
        0xf4,0xb9,0x6e,0x23,0x8d,0xc0,0x17,0x5a,0x06,0x4b,0x9c,0xd1,0x7f,0x32,0xe5,0xa8
    };
    uint8_t crc = 0;
    for (size_t i = 0; i < len; ++i) crc = table[crc ^ data[i]];
    return crc;
}

static void lidar_uart_init(void)
{
    uart_config_t cfg = {
        .baud_rate = LIDAR_UART_BAUD_RATE,
        .data_bits = UART_DATA_8_BITS,
        .parity = UART_PARITY_DISABLE,
        .stop_bits = UART_STOP_BITS_1,
        .flow_ctrl = UART_HW_FLOWCTRL_DISABLE,
        .source_clk = UART_SCLK_DEFAULT,
    };

    ESP_ERROR_CHECK(uart_driver_install(LIDAR_UART_PORT, LIDAR_UART_RX_BUFFER_SIZE, 0, 0, NULL, 0));
    ESP_ERROR_CHECK(uart_param_config(LIDAR_UART_PORT, &cfg));
    ESP_ERROR_CHECK(uart_set_pin(LIDAR_UART_PORT, UART_PIN_NO_CHANGE, LIDAR_UART_RX_GPIO,
                                 UART_PIN_NO_CHANGE, UART_PIN_NO_CHANGE));

    ESP_LOGI(TAG, "LD19 UART ready: RX GPIO %d, %d baud", LIDAR_UART_RX_GPIO, LIDAR_UART_BAUD_RATE);
}

static void lidar_rx_task(void *arg)
{
    (void)arg;
    uint8_t rx[256];
    uint8_t packet[LD19_PACKET_SIZE];
    size_t pos = 0;
    uint32_t bytes = 0, valid = 0, crc_bad = 0, framing_bad = 0;
    int64_t last = esp_timer_get_time();

    ESP_LOGI(TAG, "LD19 receive task started; waiting for packets");

    while (1) {
        int n = uart_read_bytes(LIDAR_UART_PORT, rx, sizeof(rx), pdMS_TO_TICKS(100));
        if (n > 0) {
            bytes += (uint32_t)n;
            for (int i = 0; i < n; ++i) {
                uint8_t b = rx[i];

                if (pos == 0) {
                    if (b == LD19_PACKET_HEADER) packet[pos++] = b;
                    continue;
                }
                if (pos == 1) {
                    if (b == LD19_VER_LEN) {
                        packet[pos++] = b;
                    } else {
                        framing_bad++;
                        g_lidar_status.total_framing_errors++;
                        if (b == LD19_PACKET_HEADER) {
                            packet[0] = b;
                            pos = 1;
                        } else {
                            pos = 0;
                        }
                    }
                    continue;
                }

                packet[pos++] = b;
                if (pos == LD19_PACKET_SIZE) {
                    if (ld19_crc8(packet, LD19_PACKET_SIZE - 1) == packet[LD19_PACKET_SIZE - 1]) {
                        valid++;

                        uint16_t speed = packet[2] | ((uint16_t)packet[3] << 8);
                        uint16_t a0 = packet[4] | ((uint16_t)packet[5] << 8);
                        uint16_t a1 = packet[42] | ((uint16_t)packet[43] << 8);
                        uint16_t d0 = packet[6] | ((uint16_t)packet[7] << 8);
                        uint8_t conf0 = packet[8];

                        g_lidar_status.receiving = true;
                        g_lidar_status.total_valid_packets++;
                        g_lidar_status.last_speed_deg_s = speed;
                        g_lidar_status.last_start_angle_deg = a0 / 100.0f;
                        g_lidar_status.last_end_angle_deg = a1 / 100.0f;
                        g_lidar_status.last_first_distance_mm = d0;
                        g_lidar_status.last_first_confidence = conf0;
                        g_lidar_status.last_packet_time_us = esp_timer_get_time();
                        ld19_store_packet_points(packet);
                    } else {
                        crc_bad++;
                        g_lidar_status.total_crc_errors++;
                    }
                    pos = 0;
                }
            }
        }

        int64_t now = esp_timer_get_time();
        if (now - last >= 1000000LL) {
            g_lidar_status.bytes_per_sec = bytes;
            g_lidar_status.valid_packets_per_sec = valid;
            g_lidar_status.crc_errors_per_sec = crc_bad;
            g_lidar_status.framing_errors_per_sec = framing_bad;

            if (valid == 0 && g_lidar_status.last_packet_time_us > 0 &&
                (now - g_lidar_status.last_packet_time_us) > 1000000LL) {
                g_lidar_status.receiving = false;
            }

            bytes = valid = crc_bad = framing_bad = 0;
            last = now;
        }
    }
}


static void initialise_mdns(void)
{
    ESP_ERROR_CHECK(mdns_init());
    ESP_ERROR_CHECK(mdns_hostname_set(EXAMPLE_MDNS_HOST_NAME));
    ESP_ERROR_CHECK(mdns_instance_name_set(EXAMPLE_MDNS_INSTANCE));

    mdns_txt_item_t serviceTxtData[] = {
        {"board", CONFIG_IDF_TARGET},
        {"path", "/"}
    };

    ESP_ERROR_CHECK(mdns_service_add("ESP32-WebServer", "_http", "_tcp", 80, serviceTxtData,
                                     sizeof(serviceTxtData) / sizeof(serviceTxtData[0])));
}


static void start_rc_car_softap(void)
{
    esp_netif_t *ap_netif = esp_netif_create_default_wifi_ap();
    assert(ap_netif != NULL);

    wifi_init_config_t wifi_init_cfg = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&wifi_init_cfg));

    wifi_config_t wifi_config = {
        .ap = {
            .ssid = "RC_CAR_WIFI",
            .ssid_len = 0,
            .channel = 6,
            .password = "rc_car_2026",
            .max_connection = 4,
            .authmode = WIFI_AUTH_WPA2_PSK,
            .pmf_cfg = {
                .required = false,
            },
        },
    };

    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_AP));
    ESP_ERROR_CHECK(esp_wifi_set_config(WIFI_IF_AP, &wifi_config));
    ESP_ERROR_CHECK(esp_wifi_start());
    /* Reduce SoftAP latency/jitter for MJPEG + UDP + LiDAR traffic. */
    ESP_ERROR_CHECK(esp_wifi_set_ps(WIFI_PS_NONE));

    esp_netif_ip_info_t ip_info;
    if (esp_netif_get_ip_info(ap_netif, &ip_info) == ESP_OK) {
        ESP_LOGI(TAG, "RC car SoftAP started");
        ESP_LOGI(TAG, "SSID: RC_CAR_WIFI");
        ESP_LOGI(TAG, "Password: rc_car_2026");
        ESP_LOGI(TAG, "AP IP: " IPSTR, IP2STR(&ip_info.ip));
        ESP_LOGI(TAG, "Camera server: http://" IPSTR, IP2STR(&ip_info.ip));
    } else {
        ESP_LOGI(TAG, "RC car SoftAP started");
        ESP_LOGI(TAG, "SSID: RC_CAR_WIFI");
        ESP_LOGI(TAG, "Password: rc_car_2026");
        ESP_LOGI(TAG, "Camera server: http://192.168.4.1");
    }
}

void app_main(void)
{
    ESP_LOGI(TAG, "Camera stream V5: bandwidth optimized, JPEG quality 45, target 20 FPS");

    esp_err_t ret = nvs_flash_init();
    if (ret == ESP_ERR_NVS_NO_FREE_PAGES || ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        // NVS partition was truncated and needs to be erased
        // Retry nvs_flash_init
        ESP_ERROR_CHECK(nvs_flash_erase());
        ESP_ERROR_CHECK(nvs_flash_init());
    }

    /*For camera devices that require the host to provide XCLK, the video_init() must be called immediately after the device is restarted,
    otherwise the camera device may not be able to start due to the lack of the main clock.*/
    ESP_ERROR_CHECK(example_video_init());

    ESP_ERROR_CHECK(esp_netif_init());
    ESP_ERROR_CHECK(esp_event_loop_create_default());

    rc_pwm_init();

    lidar_uart_init();

    g_lidar_scan_mutex = xSemaphoreCreateMutex();
    if (g_lidar_scan_mutex == NULL) {
        ESP_LOGE(TAG, "Failed to create LiDAR scan mutex");
    }

    start_rc_car_softap();

    initialise_mdns();
    netbiosns_init();
    netbiosns_set_name(EXAMPLE_MDNS_HOST_NAME);

    web_cam_video_config_t config[] = {
#if EXAMPLE_ENABLE_MIPI_CSI_CAM_SENSOR
        {
            .dev_name = ESP_VIDEO_MIPI_CSI_DEVICE_NAME,
        },
#endif /* EXAMPLE_ENABLE_MIPI_CSI_CAM_SENSOR */
#if EXAMPLE_ENABLE_DVP_CAM_SENSOR
        {
            .dev_name = ESP_VIDEO_DVP_DEVICE_NAME,
        },
#endif /* EXAMPLE_ENABLE_DVP_CAM_SENSOR */
#if EXAMPLE_ENABLE_SPI_CAM_0_SENSOR
        {
            .dev_name = ESP_VIDEO_SPI_DEVICE_NAME,
        },
#endif /* EXAMPLE_ENABLE_SPI_CAM_0_SENSOR */
#if EXAMPLE_ENABLE_SPI_CAM_1_SENSOR
        {
            .dev_name = ESP_VIDEO_SPI_DEVICE_1_NAME,
        },
#endif /* EXAMPLE_ENABLE_SPI_CAM_1_SENSOR */
#if EXAMPLE_ENABLE_USB_UVC_CAM_SENSOR
        {
            .dev_name = ESP_VIDEO_USB_UVC_DEVICE_NAME(0),
        },
#endif /* EXAMPLE_ENABLE_USB_UVC_CAM_SENSOR */
    };

    int config_count = sizeof(config) / sizeof(config[0]);

    assert(config_count > 0);
    ESP_ERROR_CHECK(start_cam_web_server(config, config_count));

    ESP_LOGI(TAG, "Camera web server starts");
    ESP_LOGI(TAG, "Camera JPEG quality: %d", EXAMPLE_JPEG_ENC_QUALITY);

    BaseType_t task_ok = xTaskCreate(
        rc_udp_control_task,
        "rc_udp_control",
        4096,
        NULL,
        6,
        NULL
    );

    if (task_ok != pdPASS) {
        ESP_LOGE(TAG, "Failed to create RC UDP control task");
        rc_go_safe();
    }

    BaseType_t lidar_task_ok = xTaskCreate(
        lidar_rx_task,
        "lidar_rx",
        LIDAR_TASK_STACK_SIZE,
        NULL,
        5,
        NULL
    );

    if (lidar_task_ok != pdPASS) {
        ESP_LOGE(TAG, "Failed to create LD19 receive task");
    }
}