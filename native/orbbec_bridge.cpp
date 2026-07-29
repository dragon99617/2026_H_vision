#include <libobsensor/ObSensor.hpp>

#include <algorithm>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>

extern "C" {

struct ObBridgeCalibration {
    int color_width;
    int color_height;
    float color_fx;
    float color_fy;
    float color_cx;
    float color_cy;
    float color_distortion[8];  // OpenCV order: k1,k2,p1,p2,k3,k4,k5,k6
    int depth_width;
    int depth_height;
    float depth_fx;
    float depth_fy;
    float depth_cx;
    float depth_cy;
    float depth_distortion[8];
    float depth_to_color_rotation[9];
    float depth_to_color_translation_m[3];
};

struct ObBridgeFrameInfo {
    uint64_t color_index;
    uint64_t depth_index;
    uint64_t color_timestamp_us;
    uint64_t depth_timestamp_us;
    uint32_t color_size;
    uint32_t depth_size;
    float depth_scale_m;
};

struct ObBridgeDeviceInfo {
    char name[128];
    char serial[128];
    char firmware[64];
    char connection[32];
    int vid;
    int pid;
};

struct ObBridgeColorControls {
    int auto_exposure_supported;
    int exposure_supported;
    int gain_supported;
    int auto_exposure;
    int exposure_current;
    int exposure_min;
    int exposure_max;
    int exposure_step;
    int exposure_default;
    int gain_current;
    int gain_min;
    int gain_max;
    int gain_step;
    int gain_default;
};

}

namespace {

thread_local std::string last_error;

void copy_distortion(const OBCameraDistortion &source, float *target) {
    target[0] = source.k1;
    target[1] = source.k2;
    target[2] = source.p1;
    target[3] = source.p2;
    target[4] = source.k3;
    target[5] = source.k4;
    target[6] = source.k5;
    target[7] = source.k6;
}

class Bridge {
public:
    Bridge(int color_width, int color_height, int color_fps, int depth_width, int depth_height, int depth_fps) {
        pipe_ = std::make_shared<ob::Pipeline>();
        const auto color_profiles = pipe_->getStreamProfileList(OB_SENSOR_COLOR);
        color_profile_ =
            color_profiles->getVideoStreamProfile(color_width, color_height, OB_FORMAT_MJPG, color_fps);

        auto config = std::make_shared<ob::Config>();
        config->enableStream(color_profile_);
        if(depth_width > 0 && depth_height > 0 && depth_fps > 0) {
            const auto depth_profiles = pipe_->getStreamProfileList(OB_SENSOR_DEPTH);
            depth_profile_ =
                depth_profiles->getVideoStreamProfile(depth_width, depth_height, OB_FORMAT_Y16, depth_fps);
            config->enableStream(depth_profile_);
        }
        // Produce every 60 Hz color frame. When depth is enabled, a depth frame is
        // present on approximately every second frameset at 30 Hz.
        config->setFrameAggregateOutputMode(OB_FRAME_AGGREGATE_OUTPUT_COLOR_FRAME_REQUIRE);
        config->setAlignMode(ALIGN_DISABLE);
        pipe_->start(config);
        device_ = pipe_->getDevice();
    }

    ~Bridge() {
        try {
            if(pipe_) {
                pipe_->stop();
            }
        }
        catch(...) {
        }
    }

    void calibration(ObBridgeCalibration *output) const {
        if(!output) {
            throw std::runtime_error("calibration output is null");
        }
        std::memset(output, 0, sizeof(*output));
        const auto color_intrinsic = color_profile_->getIntrinsic();
        const auto color_distortion = color_profile_->getDistortion();
        output->color_width = color_intrinsic.width;
        output->color_height = color_intrinsic.height;
        output->color_fx = color_intrinsic.fx;
        output->color_fy = color_intrinsic.fy;
        output->color_cx = color_intrinsic.cx;
        output->color_cy = color_intrinsic.cy;
        copy_distortion(color_distortion, output->color_distortion);
        if(depth_profile_) {
            const auto depth_intrinsic = depth_profile_->getIntrinsic();
            const auto depth_distortion = depth_profile_->getDistortion();
            const auto extrinsic = depth_profile_->getExtrinsicTo(color_profile_);
            output->depth_width = depth_intrinsic.width;
            output->depth_height = depth_intrinsic.height;
            output->depth_fx = depth_intrinsic.fx;
            output->depth_fy = depth_intrinsic.fy;
            output->depth_cx = depth_intrinsic.cx;
            output->depth_cy = depth_intrinsic.cy;
            copy_distortion(depth_distortion, output->depth_distortion);
            for(int index = 0; index < 9; ++index) {
                output->depth_to_color_rotation[index] = extrinsic.rot[index];
            }
            for(int index = 0; index < 3; ++index) {
                output->depth_to_color_translation_m[index] = extrinsic.trans[index] * 0.001f;
            }
        }
    }

    void device_info(ObBridgeDeviceInfo *output) const {
        if(!output) {
            throw std::runtime_error("device info output is null");
        }
        std::memset(output, 0, sizeof(*output));
        const auto info = device_->getDeviceInfo();
        std::snprintf(output->name, sizeof(output->name), "%s", info->getName());
        std::snprintf(output->serial, sizeof(output->serial), "%s", info->getSerialNumber());
        std::snprintf(output->firmware, sizeof(output->firmware), "%s", info->getFirmwareVersion());
        std::snprintf(output->connection, sizeof(output->connection), "%s", info->getConnectionType());
        output->vid = info->getVid();
        output->pid = info->getPid();
    }

    void color_controls(ObBridgeColorControls *output) const {
        if(!output) {
            throw std::runtime_error("color controls output is null");
        }
        std::memset(output, 0, sizeof(*output));
        output->auto_exposure_supported =
            device_->isPropertySupported(OB_PROP_COLOR_AUTO_EXPOSURE_BOOL, OB_PERMISSION_READ_WRITE) ? 1 : 0;
        output->exposure_supported =
            device_->isPropertySupported(OB_PROP_COLOR_EXPOSURE_INT, OB_PERMISSION_READ_WRITE) ? 1 : 0;
        output->gain_supported =
            device_->isPropertySupported(OB_PROP_COLOR_GAIN_INT, OB_PERMISSION_READ_WRITE) ? 1 : 0;
        if(output->auto_exposure_supported) {
            output->auto_exposure = device_->getBoolProperty(OB_PROP_COLOR_AUTO_EXPOSURE_BOOL) ? 1 : 0;
        }
        if(output->exposure_supported) {
            const auto range = device_->getIntPropertyRange(OB_PROP_COLOR_EXPOSURE_INT);
            output->exposure_current = range.cur;
            output->exposure_min = range.min;
            output->exposure_max = range.max;
            output->exposure_step = range.step;
            output->exposure_default = range.def;
        }
        if(output->gain_supported) {
            const auto range = device_->getIntPropertyRange(OB_PROP_COLOR_GAIN_INT);
            output->gain_current = range.cur;
            output->gain_min = range.min;
            output->gain_max = range.max;
            output->gain_step = range.step;
            output->gain_default = range.def;
        }
    }

    void set_color_controls(int auto_exposure, int exposure, int gain) {
        if(auto_exposure >= 0) {
            if(!device_->isPropertySupported(
                   OB_PROP_COLOR_AUTO_EXPOSURE_BOOL, OB_PERMISSION_READ_WRITE)) {
                throw std::runtime_error("color auto exposure control is unsupported");
            }
            device_->setBoolProperty(OB_PROP_COLOR_AUTO_EXPOSURE_BOOL, auto_exposure != 0);
        }
        if(exposure >= 0) {
            if(!device_->isPropertySupported(OB_PROP_COLOR_EXPOSURE_INT, OB_PERMISSION_READ_WRITE)) {
                throw std::runtime_error("color exposure control is unsupported");
            }
            device_->setIntProperty(OB_PROP_COLOR_EXPOSURE_INT, exposure);
        }
        if(gain >= 0) {
            if(!device_->isPropertySupported(OB_PROP_COLOR_GAIN_INT, OB_PERMISSION_READ_WRITE)) {
                throw std::runtime_error("color gain control is unsupported");
            }
            device_->setIntProperty(OB_PROP_COLOR_GAIN_INT, gain);
        }
    }

    int wait(uint8_t *color_data, uint32_t color_capacity, uint16_t *depth_data, uint32_t depth_capacity_bytes,
             ObBridgeFrameInfo *info, uint32_t timeout_ms) {
        if(!color_data || !depth_data || !info) {
            throw std::runtime_error("wait output buffer is null");
        }
        std::memset(info, 0, sizeof(*info));
        auto frameset = pipe_->waitForFrameset(timeout_ms);
        if(!frameset) {
            return 0;
        }
        auto color = frameset->getColorFrame();
        if(!color) {
            return 0;
        }
        if(color->getFormat() != OB_FORMAT_MJPG) {
            throw std::runtime_error("Orbbec color frame is not MJPEG");
        }
        const uint32_t color_size = color->getDataSize();
        if(color_size > color_capacity) {
            throw std::runtime_error("MJPEG frame exceeds bridge buffer capacity");
        }
        std::memcpy(color_data, color->getData(), color_size);
        info->color_index = color->getIndex();
        info->color_timestamp_us = color->getTimeStampUs();
        info->color_size = color_size;

        auto depth = frameset->getDepthFrame();
        if(depth && depth->getIndex() != last_depth_index_) {
            if(depth->getFormat() != OB_FORMAT_Y16) {
                throw std::runtime_error("Orbbec depth frame is not Y16");
            }
            const uint32_t depth_size = depth->getDataSize();
            if(depth_size > depth_capacity_bytes) {
                throw std::runtime_error("depth frame exceeds bridge buffer capacity");
            }
            std::memcpy(depth_data, depth->getData(), depth_size);
            last_depth_index_ = depth->getIndex();
            info->depth_index = depth->getIndex();
            info->depth_timestamp_us = depth->getTimeStampUs();
            info->depth_size = depth_size;
            info->depth_scale_m = depth->getValueScale() * 0.001f;
        }
        return 1;
    }

private:
    std::shared_ptr<ob::Pipeline> pipe_;
    std::shared_ptr<ob::Device> device_;
    std::shared_ptr<ob::VideoStreamProfile> color_profile_;
    std::shared_ptr<ob::VideoStreamProfile> depth_profile_;
    uint64_t last_depth_index_ = 0;
};

template <typename F>
int protect(F &&function) {
    try {
        last_error.clear();
        function();
        return 1;
    }
    catch(const ob::Error &error) {
        last_error = std::string(error.what()) + " [" + error.getFunction() + "]";
    }
    catch(const std::exception &error) {
        last_error = error.what();
    }
    catch(...) {
        last_error = "unknown Orbbec bridge error";
    }
    return 0;
}

}  // namespace

extern "C" {

void *ob_bridge_create(int color_width, int color_height, int color_fps, int depth_width, int depth_height,
                       int depth_fps) {
    Bridge *bridge = nullptr;
    if(!protect([&]() { bridge = new Bridge(color_width, color_height, color_fps, depth_width, depth_height, depth_fps); })) {
        return nullptr;
    }
    return bridge;
}

void ob_bridge_destroy(void *handle) {
    delete static_cast<Bridge *>(handle);
}

int ob_bridge_get_calibration(void *handle, ObBridgeCalibration *output) {
    if(!handle) {
        last_error = "Orbbec bridge handle is null";
        return 0;
    }
    return protect([&]() { static_cast<Bridge *>(handle)->calibration(output); });
}

int ob_bridge_get_device_info(void *handle, ObBridgeDeviceInfo *output) {
    if(!handle) {
        last_error = "Orbbec bridge handle is null";
        return 0;
    }
    return protect([&]() { static_cast<Bridge *>(handle)->device_info(output); });
}

int ob_bridge_get_color_controls(void *handle, ObBridgeColorControls *output) {
    if(!handle) {
        last_error = "Orbbec bridge handle is null";
        return 0;
    }
    return protect([&]() { static_cast<Bridge *>(handle)->color_controls(output); });
}

int ob_bridge_set_color_controls(void *handle, int auto_exposure, int exposure, int gain) {
    if(!handle) {
        last_error = "Orbbec bridge handle is null";
        return 0;
    }
    return protect([&]() { static_cast<Bridge *>(handle)->set_color_controls(auto_exposure, exposure, gain); });
}

int ob_bridge_wait(void *handle, uint8_t *color_data, uint32_t color_capacity, uint16_t *depth_data,
                   uint32_t depth_capacity_bytes, ObBridgeFrameInfo *info, uint32_t timeout_ms) {
    if(!handle) {
        last_error = "Orbbec bridge handle is null";
        return -1;
    }
    int result = -1;
    const int ok = protect([&]() {
        result = static_cast<Bridge *>(handle)->wait(color_data, color_capacity, depth_data, depth_capacity_bytes,
                                                     info, timeout_ms);
    });
    return ok ? result : -1;
}

const char *ob_bridge_last_error() {
    return last_error.c_str();
}

}
