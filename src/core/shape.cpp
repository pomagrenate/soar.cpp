#include <soar/core/shape.hpp>
#include <sstream>

namespace soar::core {

std::string Shape::to_string() const {
    std::ostringstream oss;
    oss << "[";
    for (size_t i = 0; i < dims_.size(); ++i) {
        oss << dims_[i] << (i + 1 < dims_.size() ? ", " : "");
    }
    oss << "]";
    return oss.str();
}

} // namespace soar::core
