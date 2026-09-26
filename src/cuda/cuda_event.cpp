#include <soar/cuda/cuda_event.hpp>

namespace soar::cuda {

void CudaEvent::ensure_created(int device_index) {
    if (!is_created_) {
        device_index_ = device_index;
        SOAR_CUDA_CHECK(cudaEventCreateWithFlags(&event_, flags_));
        is_created_ = true;
    }
}

void CudaEvent::destroy() noexcept {
    if (is_created_ && event_) {
        cudaEventDestroy(event_);
        event_ = nullptr;
        is_created_ = false;
        was_recorded_ = false;
    }
}

void CudaEvent::move_from(CudaEvent&& other) noexcept {
    event_ = other.event_;
    flags_ = other.flags_;
    device_index_ = other.device_index_;
    is_created_ = other.is_created_;
    was_recorded_ = other.was_recorded_;

    other.event_ = nullptr;
    other.is_created_ = false;
    other.was_recorded_ = false;
}

void CudaEvent::record(const CudaStream& stream) {
    ensure_created(stream.device_index());
    SOAR_CUDA_CHECK(cudaEventRecord(event_, stream.stream()));
    was_recorded_ = true;
}

void CudaEvent::record_once(const CudaStream& stream) {
    if (!was_recorded_) {
        record(stream);
    }
}

void CudaEvent::block(const CudaStream& stream) const {
    if (is_created_) {
        SOAR_CUDA_CHECK(cudaStreamWaitEvent(stream.stream(), event_, 0));
    }
}

void CudaEvent::synchronize() const {
    if (is_created_) {
        SOAR_CUDA_CHECK(cudaEventSynchronize(event_));
    }
}

bool CudaEvent::query() const noexcept {
    if (!is_created_) {
        return true;
    }
    return cudaEventQuery(event_) == cudaSuccess;
}

float CudaEvent::elapsed_time(const CudaEvent& end) const {
    if (!is_created_ || !end.is_created_) {
        throw CudaException(cudaErrorInvalidValue, "Event not created for elapsed_time", __FILE__, __LINE__);
    }
    float ms = 0.0f;
    SOAR_CUDA_CHECK(cudaEventElapsedTime(&ms, event_, end.event_));
    return ms;
}

} // namespace soar::cuda
