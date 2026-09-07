/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2024 The LCZero Authors

  Leela Chess is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  Leela Chess is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with Leela Chess.  If not, see <http://www.gnu.org/licenses/>.

  Additional permission under GNU GPL version 3 section 7

  If you modify this Program, or any covered work, by linking or
  combining it with NVIDIA Corporation's libraries from the NVIDIA CUDA
  Toolkit and the NVIDIA CUDA Deep Neural Network library (or a
  modified version of those libraries), containing parts covered by the
  terms of the respective license agreement, the licensors of this
  Program grant you additional permission to convey the resulting work.
*/

#include <algorithm>
#include <cassert>
#include <sstream>

#include "neural/backends/xla/xla_runner.h"
#include "neural/factory.h"
#include "neural/network.h"
#include "neural/onnx/converter.h"
#include "neural/xla/onnx2hlo.h"
#include "utils/bf16_utils.h"
#include "utils/bititer.h"
#include "utils/fp16_utils.h"

namespace lczero {
namespace {

class XlaNetwork;
class XlaComputation : public NetworkComputation {
 public:
  XlaComputation(const XlaNetwork* network);
  void AddInput(InputPlanes&& input) override;
  int GetBatchSize() const override;
  void ComputeBlocking() override;
  float GetQVal(int sample) const override;
  float GetDVal(int sample) const override;
  float GetPVal(int sample, int move_id) const override;
  float GetMVal(int sample) const override;

 private:
  const XlaNetwork* network_;
  XlaMutableTensor input_tensor_;
  const std::vector<const XlaMutableTensor*>* outputs_ = nullptr;
};

// Indices of various heads in the HLO output.
struct XlaNetworkOptions {
  struct IOInfo {
    size_t idx;
    pblczero::XlaShapeProto::Type type;
  };
  std::optional<IOInfo> input;
  std::optional<IOInfo> output_value;
  std::optional<IOInfo> output_wdl;
  std::optional<IOInfo> output_policy;
  std::optional<IOInfo> output_mlh;
};

class XlaNetwork : public Network {
 public:
  XlaNetwork(std::unique_ptr<XlaRunner> runner,
             const XlaNetworkOptions& options,
             const pblczero::NetworkFormat& format,
             int mini_batch_size = -1);

  const NetworkCapabilities& GetCapabilities() const override {
    return capabilities_;
  }
  std::unique_ptr<NetworkComputation> NewComputation() override {
    return std::make_unique<XlaComputation>(this);
  }
  int GetMiniBatchSize() const override {
    return mini_batch_size_ > 0 ? mini_batch_size_ : runner_->GetMaxBatchSize();
  }
  int GetPreferredBatchStep() const override {
    return runner_->GetPreferredBatchStep();
  }

 private:
  std::unique_ptr<XlaRunner> runner_;
  XlaNetworkOptions options_;
  NetworkCapabilities capabilities_;
  int mini_batch_size_;

  friend class XlaComputation;
};

static inline float ReadOutputScalar(const XlaMutableTensor* tensor, size_t idx) {
  if (tensor->type() == pblczero::XlaShapeProto::BF16) {
    return BF16toFP32(static_cast<const uint16_t*>(tensor->data())[idx]);
  } else if (tensor->type() == pblczero::XlaShapeProto::F16) {
    return FP16toFP32(static_cast<const uint16_t*>(tensor->data())[idx]);
  } else {
    return static_cast<const float*>(tensor->data())[idx];
  }
}

XlaComputation::XlaComputation(const XlaNetwork* network)
    : network_(network),
      input_tensor_(
          network->options_.input->type,
          std::vector<int64_t>{0, kInputPlanes, 8, 8},
          XlaMutableTensor::GetBufferSize(
              network->options_.input->type,
              std::vector<int64_t>{
                  static_cast<int64_t>(network->runner_->GetMaxBatchSize()),
                  kInputPlanes, 8, 8})) {}

void XlaComputation::AddInput(InputPlanes&& input) {
  auto new_shape = input_tensor_.shape();
  const size_t sample_idx = new_shape[0];
  ++new_shape[0];
  input_tensor_.Reshape(new_shape);
  const auto input_type = input_tensor_.type();
  if (input_type == pblczero::XlaShapeProto::BF16) {
    uint16_t* ptr = static_cast<uint16_t*>(input_tensor_.mutable_data()) +
                    sample_idx * 8 * 8 * kInputPlanes;
    memset(ptr, 0, 8 * 8 * kInputPlanes * sizeof(uint16_t));
    for (const auto& plane : input) {
      const uint16_t val = (plane.value == 1.0f) ? 0x3F80 : FP32toBF16(plane.value);
      for (auto bit : IterateBits(plane.mask)) ptr[bit] = val;
      ptr += 8 * 8;
    }
  } else if (input_type == pblczero::XlaShapeProto::F16) {
    uint16_t* ptr = static_cast<uint16_t*>(input_tensor_.mutable_data()) +
                    sample_idx * 8 * 8 * kInputPlanes;
    memset(ptr, 0, 8 * 8 * kInputPlanes * sizeof(uint16_t));
    for (const auto& plane : input) {
      const uint16_t val = (plane.value == 1.0f) ? 0x3C00 : FP32toFP16(plane.value);
      for (auto bit : IterateBits(plane.mask)) ptr[bit] = val;
      ptr += 8 * 8;
    }
  } else {
    float* ptr = static_cast<float*>(input_tensor_.mutable_data()) +
                 sample_idx * 8 * 8 * kInputPlanes;
    memset(ptr, 0, 8 * 8 * kInputPlanes * sizeof(float));
    for (const auto& plane : input) {
      for (auto bit : IterateBits(plane.mask)) ptr[bit] = plane.value;
      ptr += 8 * 8;
    }
  }
}

float XlaComputation::GetQVal(int sample) const {
  if (network_->options_.output_wdl) {
    const auto* tensor =
        (*outputs_)[network_->options_.output_wdl->idx];
    return ReadOutputScalar(tensor, sample * 3 + 0) -
           ReadOutputScalar(tensor, sample * 3 + 2);
  } else {
    const auto* tensor =
        (*outputs_)[network_->options_.output_value->idx];
    return ReadOutputScalar(tensor, sample);
  }
}

float XlaComputation::GetDVal(int sample) const {
  if (network_->options_.output_wdl) {
    const auto* tensor =
        (*outputs_)[network_->options_.output_wdl->idx];
    return ReadOutputScalar(tensor, sample * 3 + 1);
  }
  return 0.0f;
}

float XlaComputation::GetPVal(int sample, int move_id) const {
  const auto* tensor =
      (*outputs_)[network_->options_.output_policy->idx];
  return ReadOutputScalar(tensor, sample * 1858 + move_id);
}

float XlaComputation::GetMVal(int sample) const {
  if (network_->options_.output_mlh) {
    const auto* tensor =
        (*outputs_)[network_->options_.output_mlh->idx];
    return ReadOutputScalar(tensor, sample);
  }
  return 0.0f;
}

int XlaComputation::GetBatchSize() const { return input_tensor_.shape()[0]; }

void XlaComputation::ComputeBlocking() {
  if (input_tensor_.type() != network_->options_.input->type) {
    input_tensor_.Cast(network_->options_.input->type);
  }
  outputs_ = &network_->runner_->ExecuteBlocking({&input_tensor_});
}

XlaNetwork::XlaNetwork(std::unique_ptr<XlaRunner> runner,
                       const XlaNetworkOptions& options,
                       const pblczero::NetworkFormat& format,
                       int mini_batch_size)
    : runner_(std::move(runner)),
      options_(options),
      capabilities_{format.input(), format.output(), format.moves_left()},
      mini_batch_size_(mini_batch_size) {}

// Converts ONNX model to HLO (for various batch sizes) and adds them to the
// XlaRunner.
XlaNetworkOptions FillXlaRunnerFromOnnx(
    const pblczero::OnnxModel& onnx_model, XlaRunner* runner,
    const std::vector<size_t>& batch_sizes,
    std::optional<pblczero::XlaShapeProto::Type> io_type) {
  pblczero::ModelProto onnx;
  onnx.ParseFromString(onnx_model.model());

  using IOInfo = XlaNetworkOptions::IOInfo;
  std::unordered_map<std::string, IOInfo> constant_to_parameter_idx;
  std::unordered_map<std::string, IOInfo> input_to_parameter_idx;
  std::unordered_map<std::string, IOInfo> output_to_parameter_idx;

  auto add_tensors = [](const std::vector<Onnx2HloResult::NamedTensor>& tensors,
                        std::unordered_map<std::string, IOInfo>& map) {
    for (const auto& tensor : tensors) {
      auto iter = map.find(tensor.name);
      if (iter == map.end()) {
        map.emplace(tensor.name,
                    IOInfo{tensor.param_idx, tensor.shape.element_type()});
      } else if (iter->second.idx != tensor.param_idx) {
        throw Exception("Inconsistent index for " + tensor.name);
      } else if (iter->second.type != tensor.shape.element_type()) {
        throw Exception("Inconsistent type for " + tensor.name);
      }
    }
  };

  Onnx2HloOptions onnx2hlo_options{};
  if (onnx_model.has_output_value()) {
    onnx2hlo_options.outputs_override.emplace_back(onnx_model.output_value());
  }
  if (onnx_model.has_output_wdl()) {
    onnx2hlo_options.outputs_override.emplace_back(onnx_model.output_wdl());
  }
  if (onnx_model.has_output_policy()) {
    onnx2hlo_options.outputs_override.emplace_back(onnx_model.output_policy());
  }
  if (onnx_model.has_output_mlh()) {
    onnx2hlo_options.outputs_override.emplace_back(onnx_model.output_mlh());
  }
  onnx2hlo_options.io_type = io_type;

  for (size_t batch_size : batch_sizes) {
    CERR << "Building HLO for batch size " << batch_size << "...";
    auto conversion = ConvertOnnxToHlo(onnx, batch_size, onnx2hlo_options);
    add_tensors(conversion.constants, constant_to_parameter_idx);
    add_tensors(conversion.inputs, input_to_parameter_idx);
    add_tensors(conversion.outputs, output_to_parameter_idx);
    runner->AddModule(batch_size, conversion.hlo_module);
  }

  std::vector<std::unique_ptr<XlaTensor>> constants;
  constants.resize(constant_to_parameter_idx.size() +
                   input_to_parameter_idx.size());
  for (const auto& initializer : onnx.graph().initializer()) {
    auto iter = constant_to_parameter_idx.find(std::string(initializer.name()));
    if (iter == constant_to_parameter_idx.end()) continue;
    auto io_info = iter->second;
    assert(io_info.idx < constants.size());
    constants[io_info.idx] = OnnxTensorToXlaTensor(initializer);
  }

  CERR << "Transferring constants...";
  runner->SetFrozenInputs(std::move(constants));
  CERR << "Done.";

  XlaNetworkOptions options;
  if (input_to_parameter_idx.size() != 1 ||
      input_to_parameter_idx.begin()->first != onnx_model.input_planes()) {
    throw Exception("Expected a single input named " +
                    std::string(onnx_model.input_planes()));
  }
  options.input = input_to_parameter_idx.begin()->second;
  if (onnx_model.has_output_value()) {
    options.output_value =
        output_to_parameter_idx.at(std::string(onnx_model.output_value()));
  }
  if (onnx_model.has_output_wdl()) {
    options.output_wdl =
        output_to_parameter_idx.at(std::string(onnx_model.output_wdl()));
  }
  if (onnx_model.has_output_policy()) {
    options.output_policy =
        output_to_parameter_idx.at(std::string(onnx_model.output_policy()));
  }
  if (onnx_model.has_output_mlh()) {
    options.output_mlh =
        output_to_parameter_idx.at(std::string(onnx_model.output_mlh()));
  }
  return options;
}

// Makes an XLA network. First converts the weights to ONNX, and then calls
// FillXlaRunnerFromOnnx to convert them further to HLO and them compile them.
std::unique_ptr<Network> MakeXlaNetwork(const std::optional<WeightsFile>& w,
                                        const OptionsDict& opts) {
  if (!w) throw Exception("The XLA backend requires a network file.");
  int device = opts.GetOrDefault<int>("device", 0);
  bool is_tpu = opts.GetOrDefault<bool>("tpu", false);

  std::string default_plugin =
      is_tpu ? "libtpu.so" : "./pjrt_c_api_gpu_plugin.so";
  std::string plugin_path =
      opts.GetOrDefault<std::string>("plugin_path", default_plugin);

#if !defined(_WIN32)
  if (is_tpu) {
    const char* cur_args = getenv("LIBTPU_INIT_ARGS");
    if (!cur_args || strlen(cur_args) == 0) {
      setenv("LIBTPU_INIT_ARGS", "--xla_tpu_use_aggressive_scheduling=true", 1);
    }
  }
  if (opts.Exists<std::string>("libtpu_args")) {
    setenv("LIBTPU_INIT_ARGS", opts.Get<std::string>("libtpu_args").c_str(), 1);
  }
#endif

  auto runner = std::make_unique<XlaRunner>(plugin_path.c_str(), device);

  std::vector<size_t> batch_sizes;
  if (opts.Exists<std::string>("batch_buckets")) {
    std::string buckets_str = opts.Get<std::string>("batch_buckets");
    std::replace(buckets_str.begin(), buckets_str.end(), ':', ',');
    std::stringstream ss(buckets_str);
    std::string item;
    while (std::getline(ss, item, ',')) {
      if (!item.empty()) {
        batch_sizes.push_back(std::stoul(item));
      }
    }
  } else if (is_tpu) {
    // TPU MXUs work with 128x128 systolic matrix multipliers.
    // Compiling key batch buckets (16, 24, 64, 128, 256) avoids wasteful
    // padding for chess tree search while minimizing startup compilation latency.
    int max_batch_size = opts.GetOrDefault<int>("max_batch", 256);
    for (size_t b : {16UL, 24UL, 64UL, 128UL, 256UL}) {
      if (b <= static_cast<size_t>(max_batch_size)) {
        batch_sizes.push_back(b);
      }
    }
    if (batch_sizes.empty() ||
        batch_sizes.back() < static_cast<size_t>(max_batch_size)) {
      batch_sizes.push_back(max_batch_size);
    }
  } else {
    int max_batch_size = opts.GetOrDefault<int>("max_batch", 512);
    int steps = opts.GetOrDefault<int>("steps", 16);
    for (size_t i = 0; i < static_cast<size_t>(steps); ++i) {
      batch_sizes.push_back(max_batch_size * (i + 1) / steps);
    }
  }

  std::sort(batch_sizes.begin(), batch_sizes.end());
  batch_sizes.erase(std::unique(batch_sizes.begin(), batch_sizes.end()),
                    batch_sizes.end());
  if (batch_sizes.empty()) {
    batch_sizes.push_back(16);
  }

  std::string default_datatype = is_tpu ? "bf16" : "f32";
  std::string datatype_str =
      opts.GetOrDefault<std::string>("datatype", default_datatype);

  XlaNetworkOptions options;
  std::optional<pblczero::XlaShapeProto::Type> io_type;
  if (opts.Exists<std::string>("io_datatype")) {
    io_type = StringToXlaType(opts.Get<std::string>("io_datatype"));
  } else if (is_tpu) {
    io_type = pblczero::XlaShapeProto::BF16;
  }

  if (w->has_onnx_model()) {
    options = FillXlaRunnerFromOnnx(w->onnx_model(), runner.get(),
                                    batch_sizes, io_type);
  } else {
    CERR << "Converting weights to ONNX first (" << datatype_str << ").";
    WeightsToOnnxConverterOptions onnx_converter_options;
    onnx_converter_options.data_type =
        WeightsToOnnxConverterOptions::StringToDataType(datatype_str);
    onnx_converter_options.opset = 22;  // For full onnx bfloat16 support.
    onnx_converter_options.alt_mish =
        opts.GetOrDefault<bool>("alt_mish", false);
    auto converted = ConvertWeightsToOnnx(*w, onnx_converter_options);
    options = FillXlaRunnerFromOnnx(converted.onnx_model(), runner.get(),
                                    batch_sizes, io_type);
  }

  int mini_batch = opts.GetOrDefault<int>("batch", -1);
  return std::make_unique<XlaNetwork>(std::move(runner), options,
                                      w->format().network_format(),
                                      mini_batch);
}

REGISTER_NETWORK("xla", MakeXlaNetwork, 34)

}  // namespace
}  // namespace lczero
