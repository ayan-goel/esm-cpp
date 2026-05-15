#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "esm_cpp/model.h"
#include "esm_cpp/tokenizer.h"
#include "esm_cpp/version.h"

namespace py = pybind11;

namespace {

py::array_t<float> ForwardToNumpy(const esm::Model& self,
                                  py::array_t<std::int32_t, py::array::c_style |
                                                            py::array::forcecast>
                                      input_ids,
                                  py::object attention_mask,
                                  bool return_hidden_states,
                                  py::list* hidden_states_out) {
  if (input_ids.ndim() != 1) {
    throw std::invalid_argument("input_ids must be a 1-D int32 array");
  }
  const auto* ids_ptr = input_ids.data();
  std::span<const std::int32_t> ids(ids_ptr, static_cast<std::size_t>(input_ids.shape(0)));

  std::vector<std::int32_t> mask_storage;
  std::span<const std::int32_t> mask_span;
  if (!attention_mask.is_none()) {
    auto m = py::cast<py::array_t<std::int32_t, py::array::c_style |
                                                   py::array::forcecast>>(
        attention_mask);
    if (m.ndim() != 1 || m.shape(0) != input_ids.shape(0)) {
      throw std::invalid_argument(
          "attention_mask must be a 1-D int32 array of the same length as input_ids");
    }
    mask_storage.assign(m.data(), m.data() + m.shape(0));
    mask_span = mask_storage;
  }

  std::vector<std::vector<float>> hs_storage;
  std::vector<float> logits;
  {
    py::gil_scoped_release release;
    logits = self.ForwardWithHiddenStates(
        ids, mask_span, return_hidden_states ? &hs_storage : nullptr);
  }

  const auto& cfg = self.config();
  const py::ssize_t L = input_ids.shape(0);
  const py::ssize_t V = cfg.vocab_size;
  py::array_t<float> arr({L, V});
  std::memcpy(arr.mutable_data(), logits.data(),
              logits.size() * sizeof(float));

  if (return_hidden_states && hidden_states_out) {
    hidden_states_out->attr("clear")();
    for (auto& h : hs_storage) {
      py::array_t<float> a({L, static_cast<py::ssize_t>(cfg.hidden_size)});
      std::memcpy(a.mutable_data(), h.data(), h.size() * sizeof(float));
      hidden_states_out->append(std::move(a));
    }
  }
  return arr;
}

}  // namespace

PYBIND11_MODULE(_core, m) {
  m.doc() = "esm.cpp internal bindings";
  m.attr("__version__") = esm::kVersionString;

  py::class_<esm::Tokenizer>(m, "Tokenizer",
                              "ESM-2 tokenizer (33 tokens, UR50 frequency order).")
      .def(py::init<>())
      .def("encode",
           [](const esm::Tokenizer& self, std::string_view text,
              bool add_special, bool truncate) {
             return self.Encode(text, add_special, truncate);
           },
           py::arg("text"), py::arg("add_special") = true,
           py::arg("truncate") = true)
      .def("decode",
           [](const esm::Tokenizer& self, const std::vector<int32_t>& ids,
              bool skip_special_tokens) {
             return self.Decode(
                 std::span<const int32_t>(ids.data(), ids.size()),
                 skip_special_tokens);
           },
           py::arg("ids"), py::arg("skip_special_tokens") = false)
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
      .def_property_readonly_static("model_max_length", [](const py::object&) {
        return esm::Tokenizer::kModelMaxLength;
      });

  py::class_<esm::Config>(m, "Config")
      .def_readonly("num_hidden_layers", &esm::Config::num_hidden_layers)
      .def_readonly("hidden_size", &esm::Config::hidden_size)
      .def_readonly("num_attention_heads", &esm::Config::num_attention_heads)
      .def_readonly("head_dim", &esm::Config::head_dim)
      .def_readonly("intermediate_size", &esm::Config::intermediate_size)
      .def_readonly("vocab_size", &esm::Config::vocab_size)
      .def_readonly("layer_norm_eps", &esm::Config::layer_norm_eps)
      .def_readonly("token_dropout", &esm::Config::token_dropout)
      .def_readonly("mask_token_id", &esm::Config::mask_token_id);

  py::class_<esm::Model>(m, "Model")
      .def_static("load_from_safetensors", &esm::Model::LoadFromSafetensors,
                  py::arg("path"))
      .def_property_readonly("config", &esm::Model::config)
      .def(
          "forward",
          [](const esm::Model& self,
             py::array_t<std::int32_t, py::array::c_style | py::array::forcecast>
                 input_ids,
             py::object attention_mask) {
            return ForwardToNumpy(self, input_ids, attention_mask, false, nullptr);
          },
          py::arg("input_ids"), py::arg("attention_mask") = py::none())
      .def(
          "forward_with_hidden_states",
          [](const esm::Model& self,
             py::array_t<std::int32_t, py::array::c_style | py::array::forcecast>
                 input_ids,
             py::object attention_mask) {
            py::list hs;
            auto logits =
                ForwardToNumpy(self, input_ids, attention_mask, true, &hs);
            return py::make_tuple(logits, hs);
          },
          py::arg("input_ids"), py::arg("attention_mask") = py::none(),
          "Returns (logits, [hidden_state_0, ..., hidden_state_N]) where N = num_hidden_layers. "
          "hidden_state_0 is post-embed; hidden_state_N is post-final-LN (matches HF semantics).");
}
