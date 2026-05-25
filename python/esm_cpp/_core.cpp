#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "esm_cpp/batch.h"
#include "esm_cpp/cpu_features.h"
#include "esm_cpp/model.h"
#include "esm_cpp/observer.h"
#include "esm_cpp/scheduler.h"
#include "esm_cpp/smoothquant.h"
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

  m.def("current_isa", []() {
    return std::string(esm::IsaToString(esm::CurrentIsa()));
  }, "Returns the selected ISA name (ref|neon|avx2|avx512|avx512vnni|amx) "
     "honoring ESM_FORCE_ISA.");
  m.def("host_isa", []() {
    return std::string(esm::IsaToString(esm::HostIsa()));
  }, "Returns the host's best-available ISA (ignoring ESM_FORCE_ISA).");

  py::class_<esm::ActivationObserver>(m, "ActivationObserver",
                                       "Per-tensor activation observer for "
                                       "SmoothQuant calibration.")
      .def(py::init<>())
      .def("percentile", &esm::ActivationObserver::Percentile, py::arg("pctile"))
      .def("clear", &esm::ActivationObserver::Clear);

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
      .def_readonly("mask_token_id", &esm::Config::mask_token_id)
      .def_readonly("weights_quantized", &esm::Config::weights_quantized)
      .def_readonly("first_block_fc1_fp16",
                     &esm::Config::first_block_fc1_fp16);

  py::class_<esm::Model>(m, "Model")
      .def_static("load_from_safetensors", &esm::Model::LoadFromSafetensors,
                  py::arg("path"))
      .def_static("load_from_gguf", &esm::Model::LoadFromGguf,
                  py::arg("path"),
                  "Load an ESM-2 model from a GGUF v3 file (produced by "
                  "Model.save_to_gguf or `python -m esm_cpp.convert`).")
      .def_static("load", &esm::Model::Load, py::arg("path"),
                  "Auto-detect by magic byte: GGUF if the file starts "
                  "with the GGUF magic, safetensors otherwise.")
      .def("save_to_gguf", &esm::Model::SaveToGguf, py::arg("path"),
           "Serialize the current model state to GGUF v3. FP32 weights "
           "are stored as-is; Slice 5 adds Q8_ESM for quantized state.")
      .def("load_amx_artifacts", &esm::Model::LoadAmxArtifacts, py::arg("dir"),
           py::call_guard<py::gil_scoped_release>(),
           "Phase 11: load per-Linear fp16 BNNSGraph artifacts produced by "
           "tools/build_amx_artifacts.py. Missing artifacts are silently "
           "skipped; per-Linear fallback to the default INT8/FP32 path. "
           "Apple-only at runtime (returns 0 elsewhere). As of Phase 14 "
           "this engages by default in the forward when artifacts are loaded; "
           "set ESM_APPLE_AMX=off to disable. Returns the count of loaded contexts.")
      .def_property_readonly(
          "amx_artifacts_path",
          [](const esm::Model& m) { return m.amx_artifacts_path(); },
          "Phase 14: directory the auto-discovery loaded AMX artifacts from "
          "during the most recent load_from_safetensors / load_from_gguf, or "
          "empty if none. Useful for confirming the auto-engage path fired.")
      .def("load_ane_artifacts", &esm::Model::LoadAneArtifacts, py::arg("dir"),
           py::call_guard<py::gil_scoped_release>(),
           "Phase 12: load per-Linear-per-bucket ANE CoreML artifacts built by "
           "`build_amx_artifacts.py --compute-units CPU_AND_NE --buckets …`. "
           "`dir` contains M-<m>/<linear>.mlmodelc subdirs. Apple-only; "
           "returns 0 elsewhere. Engaged via ESM_APPLE_ANE=on; the dispatcher "
           "pads runtime M to the nearest bucket (or chunks at the max bucket "
           "when M exceeds it). ANE wins on shapes that fit; missing buckets "
           "/ over-large M / non-Apple all fall back to AMX (if loaded) or SDOT.")
      .def("load_whole_graph_artifact",
           [](esm::Model& self, const std::string& dir, int B, int L,
              const std::string& compute_units) {
             auto cu = esm::WholeGraphComputeUnits::kCpuAndNeuralEngine;
             if (compute_units == "cpu_only") {
               cu = esm::WholeGraphComputeUnits::kCpuOnly;
             } else if (compute_units == "cpu_and_ne") {
               cu = esm::WholeGraphComputeUnits::kCpuAndNeuralEngine;
             } else if (compute_units == "all") {
               cu = esm::WholeGraphComputeUnits::kAll;
             } else {
               throw std::invalid_argument(
                   "compute_units must be one of {cpu_only, cpu_and_ne, all}");
             }
             return self.LoadWholeGraphArtifact(dir, B, L, cu);
           },
           py::arg("dir"), py::arg("batch"), py::arg("seq_len"),
           py::arg("compute_units") = std::string("cpu_and_ne"),
           py::call_guard<py::gil_scoped_release>(),
           "Phase 13: register a whole-graph CoreML artifact (one MLModel for "
           "the entire ESM forward) for fixed shape (batch, seq_len). `dir` "
           "is a `.mlmodelc` bundle produced by "
           "`tools/build_whole_graph_artifacts.py`. Apple-only; returns False "
           "elsewhere. Engaged via ESM_APPLE_ANE_GRAPH=on in ForwardScheduled "
           "when all sequences share a length L AND there's a registered "
           "(B, L). Falls through to the default scheduled path otherwise.")
      .def_property_readonly("config", &esm::Model::config)
      .def("set_first_block_fc1_fp16", &esm::Model::SetFirstBlockFc1Fp16,
           py::arg("enabled"),
           "Slice 5 sensitivity escape: round the activation feeding "
           "layer 0's fc1 to FP16 precision (FP32 -> half -> FP32). "
           "Used when SmoothQuant + INT8 alone doesn't hit the PPPL gate "
           "(research-report: layer 0 outliers are the usual culprit).")
      .def("quantize_weights", &esm::Model::QuantizeWeights,
           "Quantize all per-layer Linear weights in place to per-channel "
           "symmetric INT8. lm_head stays FP32 (Slice 5 escape list). After "
           "this call Forward/ForwardBatch route the per-layer projections "
           "through LinearInt8.")
      .def("apply_smoothquant", &esm::Model::ApplySmoothQuant,
           py::arg("act_stats"), py::arg("alpha"),
           "Migrate activation outliers into per-channel weight scales "
           "(SmoothQuant). Identity-preserving for the FP32 forward to "
           "round-off; quantization-friendly because activation channels "
           "now have smoother dynamic range.")
      .def(
          "forward_with_observer",
          [](const esm::Model& self,
             py::array_t<std::int32_t, py::array::c_style | py::array::forcecast>
                 input_ids,
             py::object attention_mask, esm::ActivationObserver& observer) {
            if (input_ids.ndim() != 1) {
              throw std::invalid_argument("input_ids must be a 1-D int32 array");
            }
            std::span<const std::int32_t> ids(input_ids.data(),
                                              static_cast<std::size_t>(
                                                  input_ids.shape(0)));
            std::vector<std::int32_t> mask_storage;
            std::span<const std::int32_t> mask_span;
            if (!attention_mask.is_none()) {
              auto m_arr = py::cast<py::array_t<
                  std::int32_t,
                  py::array::c_style | py::array::forcecast>>(attention_mask);
              if (m_arr.ndim() != 1 || m_arr.shape(0) != input_ids.shape(0)) {
                throw std::invalid_argument(
                    "attention_mask must be a 1-D int32 array of the same "
                    "length as input_ids");
              }
              mask_storage.assign(m_arr.data(), m_arr.data() + m_arr.shape(0));
              mask_span = mask_storage;
            }
            std::vector<float> logits;
            {
              py::gil_scoped_release release;
              logits = self.ForwardWithObserver(ids, mask_span, &observer);
            }
            const auto& cfg = self.config();
            py::array_t<float> arr({static_cast<py::ssize_t>(input_ids.shape(0)),
                                     static_cast<py::ssize_t>(cfg.vocab_size)});
            std::memcpy(arr.mutable_data(), logits.data(),
                        logits.size() * sizeof(float));
            return arr;
          },
          py::arg("input_ids"), py::arg("attention_mask") = py::none(),
          py::arg("observer"),
          "Run a forward pass and feed every Linear-input activation into "
          "the observer at well-known site keys for SmoothQuant calibration.")
      .def_property_readonly("workspace_capacity_bytes",
                              &esm::Model::workspace_capacity_bytes,
                              "Bytes the per-forward scratch arena holds. "
                              "After the first forward at a given seq_len, this "
                              "should stay constant on subsequent calls at the "
                              "same length (zero-alloc inner loop).")
      .def_property_readonly_static(
          "num_threads",
          [](const py::object&) { return esm::Model::num_threads(); },
          "Size of the process-global thread pool, sized from "
          "ESM_NUM_THREADS at first model load (default physical-core count).")
      .def(
          "forward_batch",
          [](const esm::Model& self, const std::vector<py::array_t<
                                          std::int32_t,
                                          py::array::c_style |
                                              py::array::forcecast>>& batch_ids,
             py::object batch_masks) {
            std::vector<std::vector<std::int32_t>> ids_vec;
            ids_vec.reserve(batch_ids.size());
            for (const auto& arr : batch_ids) {
              if (arr.ndim() != 1) {
                throw std::invalid_argument(
                    "each input_ids entry must be a 1-D int32 array");
              }
              ids_vec.emplace_back(arr.data(), arr.data() + arr.shape(0));
            }
            std::vector<std::vector<std::int32_t>> masks_vec;
            if (!batch_masks.is_none()) {
              auto py_masks = py::cast<std::vector<py::array_t<
                  std::int32_t,
                  py::array::c_style | py::array::forcecast>>>(batch_masks);
              if (py_masks.size() != batch_ids.size()) {
                throw std::invalid_argument(
                    "attention_masks length must match input_ids length");
              }
              masks_vec.reserve(py_masks.size());
              for (std::size_t i = 0; i < py_masks.size(); ++i) {
                if (py_masks[i].ndim() != 1 ||
                    py_masks[i].shape(0) != batch_ids[i].shape(0)) {
                  throw std::invalid_argument(
                      "attention_mask[i] must be a 1-D int32 array of the "
                      "same length as input_ids[i]");
                }
                masks_vec.emplace_back(py_masks[i].data(),
                                       py_masks[i].data() + py_masks[i].shape(0));
              }
            }
            std::vector<std::vector<float>> outputs;
            {
              py::gil_scoped_release release;
              outputs = self.ForwardBatch(ids_vec, masks_vec);
            }
            const auto V = static_cast<py::ssize_t>(self.config().vocab_size);
            py::list result;
            for (std::size_t b = 0; b < outputs.size(); ++b) {
              const auto L = static_cast<py::ssize_t>(ids_vec[b].size());
              py::array_t<float> arr({L, V});
              std::memcpy(arr.mutable_data(), outputs[b].data(),
                          outputs[b].size() * sizeof(float));
              result.append(std::move(arr));
            }
            return result;
          },
          py::arg("input_ids"), py::arg("attention_masks") = py::none(),
          "Run a batch of sequences in parallel across the process-global "
          "thread pool. Returns a list of [seq_len, vocab_size] logit "
          "arrays. Each sequence may have a different length.\n\n"
          "Soft-deprecated in v0.1.0 in favor of forward_scheduled, which "
          "packs sequences into single cu_seqlens forwards with optional "
          "length-bucketing. forward_batch is kept as a perf-comparison "
          "baseline; switch to forward_scheduled for new code.")
      .def(
          "forward_packed",
          [](const esm::Model& self,
             py::array_t<std::int32_t, py::array::c_style |
                                            py::array::forcecast>
                 packed_ids,
             py::array_t<std::int32_t, py::array::c_style |
                                            py::array::forcecast>
                 cu_seqlens,
             py::object packed_masks) {
            if (packed_ids.ndim() != 1) {
              throw std::invalid_argument(
                  "packed_ids must be a 1-D int32 array");
            }
            if (cu_seqlens.ndim() != 1 || cu_seqlens.shape(0) < 2) {
              throw std::invalid_argument(
                  "cu_seqlens must be a 1-D int32 array of size batch + 1");
            }
            std::span<const std::int32_t> ids(
                packed_ids.data(),
                static_cast<std::size_t>(packed_ids.shape(0)));
            std::span<const std::int32_t> seqlens(
                cu_seqlens.data(),
                static_cast<std::size_t>(cu_seqlens.shape(0)));
            std::vector<std::int32_t> masks_storage;
            std::span<const std::int32_t> masks_span;
            if (!packed_masks.is_none()) {
              auto m = py::cast<
                  py::array_t<std::int32_t, py::array::c_style |
                                                 py::array::forcecast>>(
                  packed_masks);
              if (m.ndim() != 1 || m.shape(0) != packed_ids.shape(0)) {
                throw std::invalid_argument(
                    "packed_masks must be a 1-D int32 array of the same "
                    "length as packed_ids");
              }
              masks_storage.assign(m.data(), m.data() + m.shape(0));
              masks_span = masks_storage;
            }
            const int batch_size =
                static_cast<int>(cu_seqlens.shape(0)) - 1;
            esm::BatchView view(ids, masks_span, seqlens, batch_size);
            std::vector<std::vector<float>> outputs;
            {
              py::gil_scoped_release release;
              outputs = self.ForwardPacked(view);
            }
            const auto V =
                static_cast<py::ssize_t>(self.config().vocab_size);
            py::list result;
            for (int b = 0; b < batch_size; ++b) {
              const py::ssize_t L_b = static_cast<py::ssize_t>(
                  seqlens.data()[b + 1] - seqlens.data()[b]);
              py::array_t<float> arr({L_b, V});
              std::memcpy(arr.mutable_data(),
                          outputs[static_cast<std::size_t>(b)].data(),
                          outputs[static_cast<std::size_t>(b)].size() *
                              sizeof(float));
              result.append(std::move(arr));
            }
            return result;
          },
          py::arg("packed_ids"), py::arg("cu_seqlens"),
          py::arg("packed_masks") = py::none(),
          "Run a packed batch as a single concatenated forward through the "
          "encoder. packed_ids is a 1-D int32 array of all sequences "
          "concatenated; cu_seqlens has size batch+1 with cu_seqlens[0] = 0 "
          "and cu_seqlens[-1] = len(packed_ids). Returns a list of "
          "[L_b, vocab_size] logit arrays.")
      .def(
          "forward_scheduled",
          [](const esm::Model& self,
             const std::vector<py::array_t<
                 std::int32_t,
                 py::array::c_style | py::array::forcecast>>& batch_ids,
             py::object batch_masks, float imbalance_threshold,
             int max_batch_size) {
            std::vector<std::vector<std::int32_t>> ids_vec;
            ids_vec.reserve(batch_ids.size());
            for (const auto& arr : batch_ids) {
              if (arr.ndim() != 1) {
                throw std::invalid_argument(
                    "each input_ids entry must be a 1-D int32 array");
              }
              ids_vec.emplace_back(arr.data(), arr.data() + arr.shape(0));
            }
            std::vector<std::vector<std::int32_t>> masks_vec;
            if (!batch_masks.is_none()) {
              auto py_masks = py::cast<std::vector<py::array_t<
                  std::int32_t,
                  py::array::c_style | py::array::forcecast>>>(batch_masks);
              if (py_masks.size() != batch_ids.size()) {
                throw std::invalid_argument(
                    "attention_masks length must match input_ids length");
              }
              masks_vec.reserve(py_masks.size());
              for (std::size_t i = 0; i < py_masks.size(); ++i) {
                if (py_masks[i].ndim() != 1 ||
                    py_masks[i].shape(0) != batch_ids[i].shape(0)) {
                  throw std::invalid_argument(
                      "attention_mask[i] must be a 1-D int32 array of the "
                      "same length as input_ids[i]");
                }
                masks_vec.emplace_back(
                    py_masks[i].data(),
                    py_masks[i].data() + py_masks[i].shape(0));
              }
            }
            esm::SchedulerConfig cfg;
            cfg.imbalance_threshold = imbalance_threshold;
            cfg.max_batch_size = max_batch_size;
            std::vector<std::vector<float>> outputs;
            {
              py::gil_scoped_release release;
              outputs = self.ForwardScheduled(ids_vec, masks_vec, cfg);
            }
            const auto V =
                static_cast<py::ssize_t>(self.config().vocab_size);
            py::list result;
            for (std::size_t b = 0; b < outputs.size(); ++b) {
              const auto L =
                  static_cast<py::ssize_t>(ids_vec[b].size());
              py::array_t<float> arr({L, V});
              std::memcpy(arr.mutable_data(), outputs[b].data(),
                          outputs[b].size() * sizeof(float));
              result.append(std::move(arr));
            }
            return result;
          },
          py::arg("input_ids"), py::arg("attention_masks") = py::none(),
          py::arg("imbalance_threshold") = 1.2f,
          py::arg("max_batch_size") = 256,
          "Run a list of sequences through one or more packed forwards "
          "with length-aware bucketing. Returns logits in input order. "
          "imbalance_threshold gates the bucket-split (default 1.2 = "
          "split when max(L)/mean(L) > 1.2). max_batch_size caps the "
          "number of sequences in a single packed dispatch.")
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
