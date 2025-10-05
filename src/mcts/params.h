/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2018-2019 The LCZero Authors

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

#pragma once

#include <cmath>
#include <vector>
#include "neural/encoder.h"
#include "utils/optionsdict.h"
#include "utils/optionsparser.h"
#include "utils/random.h"

namespace lczero {

enum class ContemptMode { PLAY, WHITE, BLACK, NONE };

enum class EntropyUpdateMode { PERIODIC, FREQUENT };

// START: ADDED FOR DYNAMIC HYBRID RATIO
enum class HybridRatioMode {
  STATIC,
  MANUAL_SCHEDULE,
  LINEAR,
  LOGARITHMIC,
  POWER,
  ROOT,
  SIGMOID,
  EXPONENTIAL,
  STEP_DECAY,
  INVERSE_SIGMOID,
  STEPS,
  PLATEAU,
  GAUSSIAN_PEAK,
  DOUBLE_PEAK,
  SAWTOOTH_WAVE,
  OSCILLATING,
  HEARTBEAT,
  MULTI_TIMESCALE,
  THERMAL_ANNEALING,
  ASYMPTOTIC_APPROACH,
  CHAOTIC
  // Fibonacci and Golden Ratio are complex and stateful, omitted for simplicity
  // unless a stateful evaluation mechanism is added.
};
// END: ADDED FOR DYNAMIC HYBRID RATIO

class SearchParams {
 public:
  SearchParams(const OptionsDict& options);
  SearchParams(const SearchParams&) = delete;

  // Use struct for WDLRescaleParams calculation to make them const.
  struct WDLRescaleParams {
    WDLRescaleParams(float r, float d) {
      ratio = r;
      diff = d;
    }
    float ratio;
    float diff;
  };

  // Populates UciOptions with search parameters.
  static void Populate(OptionsParser* options);

  // START: ADDED FOR DYNAMIC HYBRID RATIO
  // The main function to calculate the ratio based on the selected mode.
  float GetDynamicHybridRatio(int node_count) const;
  // END: ADDED FOR DYNAMIC HYBRID RATIO

  // Parameter getters.
  int GetMiniBatchSize() const { return kMiniBatchSize; }
  int GetMaxPrefetchBatch() const {
    return options_.Get<int>(kMaxPrefetchBatchId);
  }
  float GetCpuct(bool at_root) const { return at_root ? kCpuctAtRoot : kCpuct; }
  float GetCpuctBase(bool at_root) const {
    return at_root ? kCpuctBaseAtRoot : kCpuctBase;
  }
  float GetCpuctFactor(bool at_root) const {
    return at_root ? kCpuctFactorAtRoot : kCpuctFactor;
  }
  bool GetTwoFoldDraws() const { return kTwoFoldDraws; }
  float GetTemperature() const { return options_.Get<float>(kTemperatureId); }
  int GetScLimit() const { return options_.Get<int>(kScLimitId); }
  float GetHybridSamplingRatio() const { return options_.Get<float>(kHybridSamplingRatioId); }
  HybridRatioMode GetHybridRatioMode() const { return kHybridRatioMode; }
  const std::vector<std::pair<int, float>>& GetHybridRatioSchedule() const { return kHybridRatioSchedule; }
  float GetHybridMinRatio() const { return options_.Get<float>(kHybridMinRatioId); }
  float GetHybridMaxRatio() const { return options_.Get<float>(kHybridMaxRatioId); }
  int GetHybridScalingFactor() const { return options_.Get<int>(kHybridScalingFactorId); }
  float GetHybridShapeParam1() const { return options_.Get<float>(kHybridShapeParam1Id); }
  float GetHybridShapeParam2() const { return options_.Get<float>(kHybridShapeParam2Id); }

  float GetTemperatureVisitOffset() const {
    return options_.Get<float>(kTemperatureVisitOffsetId);
  }
  int GetTempDecayMoves() const { return options_.Get<int>(kTempDecayMovesId); }
  int GetTempDecayDelayMoves() const {
    return options_.Get<int>(kTempDecayDelayMovesId);
  }
  int GetTemperatureCutoffMove() const {
    return options_.Get<int>(kTemperatureCutoffMoveId);
  }
  float GetTemperatureEndgame() const {
    return options_.Get<float>(kTemperatureEndgameId);
  }
  float GetTemperatureWinpctCutoff() const {
    return options_.Get<float>(kTemperatureWinpctCutoffId);
  }
  float GetNoiseEpsilon() const { return kNoiseEpsilon; }
  float GetNoiseAlpha() const { return kNoiseAlpha; }
  bool GetVerboseStats() const { return options_.Get<bool>(kVerboseStatsId); }
  bool GetLogLiveStats() const { return options_.Get<bool>(kLogLiveStatsId); }
  bool GetFpuAbsolute(bool at_root) const {
    return at_root ? kFpuAbsoluteAtRoot : kFpuAbsolute;
  }
  float GetFpuValue(bool at_root) const {
    return at_root ? kFpuValueAtRoot : kFpuValue;
  }
  int GetCacheHistoryLength() const { return kCacheHistoryLength; }
  float GetPolicySoftmaxTemp() const { return kPolicySoftmaxTemp; }
  int GetMaxCollisionEvents() const { return kMaxCollisionEvents; }
  int GetMaxCollisionVisits() const { return kMaxCollisionVisits; }
  bool GetOutOfOrderEval() const { return kOutOfOrderEval; }
  bool GetStickyEndgames() const { return kStickyEndgames; }
  bool GetSyzygyFastPlay() const { return kSyzygyFastPlay; }
  int GetMultiPv() const { return options_.Get<int>(kMultiPvId); }
  bool GetPerPvCounters() const { return options_.Get<bool>(kPerPvCountersId); }
  std::string GetScoreType() const {
    return options_.Get<std::string>(kScoreTypeId);
  }
  FillEmptyHistory GetHistoryFill() const { return kHistoryFill; }
  float GetMovesLeftMaxEffect() const { return kMovesLeftMaxEffect; }
  float GetMovesLeftThreshold() const { return kMovesLeftThreshold; }
  float GetMovesLeftSlope() const { return kMovesLeftSlope; }
  float GetMovesLeftConstantFactor() const { return kMovesLeftConstantFactor; }
  float GetMovesLeftScaledFactor() const { return kMovesLeftScaledFactor; }
  float GetMovesLeftQuadraticFactor() const {
    return kMovesLeftQuadraticFactor;
  }
  bool GetDisplayCacheUsage() const { return kDisplayCacheUsage; }
  int GetMaxConcurrentSearchers() const { return kMaxConcurrentSearchers; }
  float GetDrawScore() const { return kDrawScore; }
  ContemptMode GetContemptMode() const {
    std::string mode = options_.Get<std::string>(kContemptModeId);
    if (mode == "play") return ContemptMode::PLAY;
    if (mode == "white_side_analysis") return ContemptMode::WHITE;
    if (mode == "black_side_analysis") return ContemptMode::BLACK;
    assert(mode == "disable");
    return ContemptMode::NONE;
  }
  int GetContemptModeTB() const { return kContemptModeTB; }
  float GetWDLRescaleRatio() const { return kWDLRescaleParams.ratio; }
  float GetWDLRescaleDiff() const { return kWDLRescaleParams.diff; }
  float GetWDLMaxS() const { return kWDLMaxS; }
  float GetWDLEvalObjectivity() const { return kWDLEvalObjectivity; }
  bool GetSwapColors() const { return kSwapColors; }
  float GetMaxOutOfOrderEvalsFactor() const {
    return kMaxOutOfOrderEvalsFactor;
  }
  float GetNpsLimit() const { return kNpsLimit; }
  int GetSolidTreeThreshold() const { return kSolidTreeThreshold; }

  int GetTaskWorkersPerSearchWorker() const {
    return kTaskWorkersPerSearchWorker;
  }
  int GetMinimumWorkSizeForProcessing() const {
    return kMinimumWorkSizeForProcessing;
  }
  int GetMinimumWorkSizeForPicking() const {
    return kMinimumWorkSizeForPicking;
  }
  int GetMinimumRemainingWorkSizeForPicking() const {
    return kMinimumRemainingWorkSizeForPicking;
  }
  int GetMinimumWorkPerTaskForProcessing() const {
    return kMinimumWorkPerTaskForProcessing;
  }
  int GetIdlingMinimumWork() const { return kIdlingMinimumWork; }
  int GetThreadIdlingThreshold() const { return kThreadIdlingThreshold; }
  int GetMaxCollisionVisitsScalingStart() const {
    return kMaxCollisionVisitsScalingStart;
  }
  int GetMaxCollisionVisitsScalingEnd() const {
    return kMaxCollisionVisitsScalingEnd;
  }
  float GetMaxCollisionVisitsScalingPower() const {
    return kMaxCollisionVisitsScalingPower;
  }
  bool GetSearchSpinBackoff() const { return kSearchSpinBackoff; }

  // Entropy parameters
  float GetEntropyBeta() const { return options_.Get<float>(kEntropyBetaId); }
  float GetEntropyDelta0() const { return options_.Get<float>(kEntropyDelta0Id); }
  EntropyUpdateMode GetEntropyUpdateMode() const { return kEntropyUpdateMode; }


  // Search parameter IDs.
  static const OptionId kMiniBatchSizeId;
  static const OptionId kMaxPrefetchBatchId;
  static const OptionId kCpuctId;
  static const OptionId kCpuctAtRootId;
  static const OptionId kCpuctBaseId;
  static const OptionId kCpuctBaseAtRootId;
  static const OptionId kCpuctFactorId;
  static const OptionId kCpuctFactorAtRootId;
  static const OptionId kRootHasOwnCpuctParamsId;
  static const OptionId kTwoFoldDrawsId;
  static const OptionId kTemperatureId;
  static const OptionId kScLimitId;
  static const OptionId kHybridSamplingRatioId;
  // START: ADDED FOR DYNAMIC HYBRID RATIO
  static const OptionId kHybridRatioModeId;
  static const OptionId kHybridRatioScheduleId;
  static const OptionId kHybridMinRatioId;
  static const OptionId kHybridMaxRatioId;
  static const OptionId kHybridScalingFactorId;
  static const OptionId kHybridShapeParam1Id;
  static const OptionId kHybridShapeParam2Id;
  // END: ADDED FOR DYNAMIC HYBRID RATIO
  static const OptionId kTempDecayMovesId;
  static const OptionId kTempDecayDelayMovesId;
  static const OptionId kTemperatureCutoffMoveId;
  static const OptionId kTemperatureEndgameId;
  static const OptionId kTemperatureWinpctCutoffId;
  static const OptionId kTemperatureVisitOffsetId;
  static const OptionId kNoiseEpsilonId;
  static const OptionId kNoiseAlphaId;
  static const OptionId kVerboseStatsId;
  static const OptionId kLogLiveStatsId;
  static const OptionId kFpuStrategyId;
  static const OptionId kFpuValueId;
  static const OptionId kFpuStrategyAtRootId;
  static const OptionId kFpuValueAtRootId;
  static const OptionId kCacheHistoryLengthId;
  static const OptionId kPolicySoftmaxTempId;
  static const OptionId kMaxCollisionEventsId;
  static const OptionId kMaxCollisionVisitsId;
  static const OptionId kOutOfOrderEvalId;
  static const OptionId kStickyEndgamesId;
  static const OptionId kSyzygyFastPlayId;
  static const OptionId kMultiPvId;
  static const OptionId kPerPvCountersId;
  static const OptionId kScoreTypeId;
  static const OptionId kHistoryFillId;
  static const OptionId kMovesLeftMaxEffectId;
  static const OptionId kMovesLeftThresholdId;
  static const OptionId kMovesLeftConstantFactorId;
  static const OptionId kMovesLeftScaledFactorId;
  static const OptionId kMovesLeftQuadraticFactorId;
  static const OptionId kMovesLeftSlopeId;
  static const OptionId kDisplayCacheUsageId;
  static const OptionId kMaxConcurrentSearchersId;
  static const OptionId kDrawScoreId;
  static const OptionId kContemptModeId;
  static const OptionId kContemptId;
  static const OptionId kContemptMaxValueId;
  static const OptionId kContemptModeTBId;
  static const OptionId kWDLCalibrationEloId;
  static const OptionId kWDLContemptAttenuationId;
  static const OptionId kWDLMaxSId;
  static const OptionId kWDLEvalObjectivityId;
  static const OptionId kWDLDrawRateTargetId;
  static const OptionId kWDLDrawRateReferenceId;
  static const OptionId kWDLBookExitBiasId;
  static const OptionId kSwapColorsId;
  static const OptionId kMaxOutOfOrderEvalsFactorId;
  static const OptionId kNpsLimitId;
  static const OptionId kSolidTreeThresholdId;
  static const OptionId kTaskWorkersPerSearchWorkerId;
  static const OptionId kMinimumWorkSizeForProcessingId;
  static const OptionId kMinimumWorkSizeForPickingId;
  static const OptionId kMinimumRemainingWorkSizeForPickingId;
  static const OptionId kMinimumWorkPerTaskForProcessingId;
  static const OptionId kIdlingMinimumWorkId;
  static const OptionId kThreadIdlingThresholdId;
  static const OptionId kMaxCollisionVisitsScalingStartId;
  static const OptionId kMaxCollisionVisitsScalingEndId;
  static const OptionId kMaxCollisionVisitsScalingPowerId;
  static const OptionId kUCIOpponentId;
  static const OptionId kUCIRatingAdvId;
  static const OptionId kSearchSpinBackoffId;

  // Entropy parameter IDs
  static const OptionId kEntropyBetaId;
  static const OptionId kEntropyDelta0Id;
  static const OptionId kEntropyUpdateModeId;

 private:
  const OptionsDict& options_;
  // Cached parameter values. Values have to be cached if either:
  // 1. Parameter is accessed often and has to be cached for performance
  // reasons.
  // 2. Parameter has to stay the same during the search.
  // TODO(crem) Some of those parameters can be converted to be dynamic after
  //            trivial search optimizations.
  const float kCpuct;
  const float kCpuctAtRoot;
  const float kCpuctBase;
  const float kCpuctBaseAtRoot;
  const float kCpuctFactor;
  const float kCpuctFactorAtRoot;
  const bool kTwoFoldDraws;
  const float kNoiseEpsilon;
  const float kNoiseAlpha;
  const bool kFpuAbsolute;
  const float kFpuValue;
  const bool kFpuAbsoluteAtRoot;
  const float kFpuValueAtRoot;
  const int kCacheHistoryLength;
  const float kPolicySoftmaxTemp;
  const int kMaxCollisionEvents;
  const int kMaxCollisionVisits;
  const bool kOutOfOrderEval;
  const bool kStickyEndgames;
  const bool kSyzygyFastPlay;
  const FillEmptyHistory kHistoryFill;
  const int kMiniBatchSize;
  const float kMovesLeftMaxEffect;
  const float kMovesLeftThreshold;
  const float kMovesLeftSlope;
  const float kMovesLeftConstantFactor;
  const float kMovesLeftScaledFactor;
  const float kMovesLeftQuadraticFactor;
  const bool kDisplayCacheUsage;
  const int kMaxConcurrentSearchers;
  const float kDrawScore;
  const float kContempt;
  const int kContemptModeTB;
  const WDLRescaleParams kWDLRescaleParams;
  const float kWDLMaxS;
  const float kWDLEvalObjectivity;
  const bool kSwapColors;
  const float kMaxOutOfOrderEvalsFactor;
  const float kNpsLimit;
  const int kSolidTreeThreshold;
  const int kTaskWorkersPerSearchWorker;
  const int kMinimumWorkSizeForProcessing;
  const int kMinimumWorkSizeForPicking;
  const int kMinimumRemainingWorkSizeForPicking;
  const int kMinimumWorkPerTaskForProcessing;
  const int kIdlingMinimumWork;
  const int kThreadIdlingThreshold;
  const int kMaxCollisionVisitsScalingStart;
  const int kMaxCollisionVisitsScalingEnd;
  const float kMaxCollisionVisitsScalingPower;
  const bool kSearchSpinBackoff;
  const EntropyUpdateMode kEntropyUpdateMode;

  // START: ADDED FOR DYNAMIC HYBRID RATIO
  const HybridRatioMode kHybridRatioMode;
  const std::vector<std::pair<int, float>> kHybridRatioSchedule;
  mutable float chaotic_state_{0.5f}; // Mutable for chaotic function state
  // END: ADDED FOR DYNAMIC HYBRID RATIO
};

}  // namespace lczero
