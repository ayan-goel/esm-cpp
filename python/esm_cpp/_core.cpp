#include <pybind11/pybind11.h>

#include "esm_cpp/version.h"

namespace py = pybind11;

PYBIND11_MODULE(_core, m) {
  m.doc() = "esm.cpp internal bindings";
  m.attr("__version__") = esm::kVersionString;
}
