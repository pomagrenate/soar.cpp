#include <soar/memory/palloc_allocator.hpp>
#include <palloc-new-delete.h>

// Global instantiation of palloc C++ overrides
namespace soar::memory {

bool is_palloc_active() noexcept {
    return ::pa_version() > 0;
}

} // namespace soar::memory
