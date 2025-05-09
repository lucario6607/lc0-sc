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
#include <cstdint>
#include <cstring>

namespace lczero {

inline float FastLog2(const float a) { return std::log2(a); }

inline float FastExp2(const float a) { return std::exp2(a); }

inline float FastLog(const float a) { return std::log(a); }

inline float FastExp(const float a) { return std::exp(a); }

// Safeguarded logistic function.
inline float FastLogistic(const float a) {
  if (a > 20.0f) {return 1.0f;}
  if (a < -20.0f) {return 0.0f;}
  return 1.0f / (1.0f + std::exp(-a));
}

inline float FastSign(const float a) {
  // Microsoft compiler does not have a builtin for copysign and emits a
  // library call which is too expensive for hot paths.
#if defined(_MSC_VER)
  // This doesn't treat signed 0's the same way that copysign does, but it
  // should be good enough, for our use case.
  return a < 0 ? -1.0f : 1.0f;
#else
  return std::copysign(1.0f, a);
#endif
}

}  // namespace lczero
