#include "esm_cpp/status.h"

namespace esm {

namespace {
const char* CodeName(StatusCode code) {
  switch (code) {
    case StatusCode::kOk: return "Ok";
    case StatusCode::kInvalidArgument: return "InvalidArgument";
    case StatusCode::kOutOfRange: return "OutOfRange";
    case StatusCode::kNotFound: return "NotFound";
    case StatusCode::kIoError: return "IoError";
    case StatusCode::kShapeMismatch: return "ShapeMismatch";
    case StatusCode::kInternal: return "Internal";
  }
  return "Unknown";
}
}  // namespace

std::string Status::ToString() const {
  std::string out(CodeName(code_));
  if (!message_.empty()) {
    out.append(": ");
    out.append(message_);
  }
  return out;
}

}  // namespace esm
