#pragma once

#include <string>
#include <string_view>
#include <utility>

namespace esm {

enum class StatusCode {
  kOk = 0,
  kInvalidArgument,
  kOutOfRange,
  kNotFound,
  kIoError,
  kShapeMismatch,
  kInternal,
};

class Status {
 public:
  Status() : code_(StatusCode::kOk) {}
  Status(StatusCode code, std::string message)
      : code_(code), message_(std::move(message)) {}

  static Status Ok() { return Status{}; }

  bool ok() const { return code_ == StatusCode::kOk; }
  StatusCode code() const { return code_; }
  std::string_view message() const { return message_; }

  std::string ToString() const;

 private:
  StatusCode code_;
  std::string message_;
};

}  // namespace esm
