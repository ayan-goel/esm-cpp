#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "esm_cpp/tokenizer.h"
#include "esm_cpp/version.h"

namespace py = pybind11;

PYBIND11_MODULE(_core, m) {
  m.doc() = "esm.cpp internal bindings";
  m.attr("__version__") = esm::kVersionString;

  py::class_<esm::Tokenizer>(m, "Tokenizer", "ESM-2 tokenizer (33 tokens, UR50 frequency order).")
      .def(py::init<>())
      .def(
          "encode",
          [](const esm::Tokenizer& self, std::string_view text,
             bool add_special, bool truncate) {
            return self.Encode(text, add_special, truncate);
          },
          py::arg("text"), py::arg("add_special") = true,
          py::arg("truncate") = true,
          "Encode a sequence to token IDs.")
      .def(
          "decode",
          [](const esm::Tokenizer& self, const std::vector<int32_t>& ids,
             bool skip_special_tokens) {
            return self.Decode(std::span<const int32_t>(ids.data(), ids.size()),
                               skip_special_tokens);
          },
          py::arg("ids"), py::arg("skip_special_tokens") = false,
          "Decode token IDs back to a sequence string.")
      .def("token_to_id", &esm::Tokenizer::TokenToId, py::arg("token"))
      .def("id_to_token",
           [](const esm::Tokenizer& self, int32_t id) {
             return std::string(self.IdToToken(id));
           },
           py::arg("id"))
      .def_property_readonly_static(
          "vocab_size",
          [](const py::object&) { return esm::Tokenizer::kVocabSize; })
      .def_property_readonly_static(
          "cls_id", [](const py::object&) { return esm::Tokenizer::kClsId; })
      .def_property_readonly_static(
          "pad_id", [](const py::object&) { return esm::Tokenizer::kPadId; })
      .def_property_readonly_static(
          "eos_id", [](const py::object&) { return esm::Tokenizer::kEosId; })
      .def_property_readonly_static(
          "unk_id", [](const py::object&) { return esm::Tokenizer::kUnkId; })
      .def_property_readonly_static(
          "mask_id", [](const py::object&) { return esm::Tokenizer::kMaskId; })
      .def_property_readonly_static(
          "model_max_length", [](const py::object&) {
            return esm::Tokenizer::kModelMaxLength;
          });
}
