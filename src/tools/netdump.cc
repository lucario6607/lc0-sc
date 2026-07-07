/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2026 The LCZero Authors

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

#include "tools/netdump.h"

#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>

#include "chess/board.h"
#include "neural/encoder.h"
#include "neural/register.h"
#include "neural/shared_params.h"
#include "search/classic/node.h"
#include "utils/optionsparser.h"

namespace lczero {
namespace {

const OptionId kFenId{"fen", "", "FEN of the position to dump (default startpos)."};
const OptionId kMovesId{"moves", "",
                        "Space-separated UCI moves to play from the FEN "
                        "before dumping."};

std::vector<std::string> SplitMoves(const std::string& moves) {
  std::vector<std::string> result;
  std::istringstream iss(moves);
  std::string move;
  while (iss >> move) result.push_back(move);
  return result;
}

}  // namespace

void NetDumpCmd() {
  OptionsParser options;
  SharedBackendParams::Populate(&options);
  options.Add<StringOption>(kFenId) = ChessBoard::kStartposFen;
  options.Add<StringOption>(kMovesId) = "";

  if (!options.ProcessAllFlags()) return;

  try {
    const OptionsDict& dict = options.GetOptionsDict();
    auto backend = BackendManager::Get()->CreateFromParams(dict);

    classic::NodeTree tree;
    tree.ResetToPosition(dict.Get<std::string>(kFenId),
                         SplitMoves(dict.Get<std::string>(kMovesId)));
    const PositionHistory& history = tree.GetPositionHistory();
    const Position& current = history.Last();

    // --- Section 1: raw Ceres TPG byte encoding. ---
    std::vector<uint8_t> tpg = EncodePositionForCeresTPG(
        history, 8, FillEmptyHistory::FEN_ONLY);
    std::cout << "BEGIN_TPG_BYTES " << tpg.size() << "\n";
    for (size_t i = 0; i < tpg.size(); i++) {
      std::cout << std::setw(2) << std::setfill('0') << std::hex
                << static_cast<int>(tpg[i]);
      std::cout << ((i % 137 == 136) ? '\n' : ' ');
    }
    std::cout << std::dec << "END_TPG_BYTES\n";

    // --- Section 2: legal moves in absolute (UCI) coordinates, in the same
    // order as board.GenerateLegalMoves(), so policy entries below can be
    // labelled. ---
    const ChessBoard& board = current.GetBoard();
    MoveList legal = board.GenerateLegalMoves();
    const bool is_black = current.IsBlackToMove();
    std::vector<std::string> uci_moves;
    uci_moves.reserve(legal.size());
    for (Move m : legal) {
      if (is_black) m.Flip();
      uci_moves.push_back(m.ToString(/*is_chess960=*/false));
    }

    // --- Section 3: raw NN output (policy/wdl/mlh) via the real backend
    // (goes through the same Ceres TPG encode+ONNX path as normal play). ---
    EvalPosition eval_pos{history.GetPositions(), legal};
    EvalResult result;
    result.p.resize(legal.size());
    // Request per-move action-head WDL (filled only for nets that have the
    // head, e.g. Ceres C3; stays NaN otherwise).
    result.action.assign(legal.size() * 3,
                         std::numeric_limits<float>::quiet_NaN());
    auto computation = backend->CreateComputation();
    computation->AddInput(eval_pos, result.AsPtr());
    computation->ComputeBlocking();

    std::cout << "BEGIN_NN_OUTPUT\n";
    std::cout << "Q " << std::setprecision(8) << result.q << "\n";
    std::cout << "D " << std::setprecision(8) << result.d << "\n";
    std::cout << "M " << std::setprecision(8) << result.m << "\n";
    std::cout << "BEGIN_POLICY " << result.p.size() << "\n";
    for (size_t i = 0; i < result.p.size(); i++) {
      std::cout << uci_moves[i] << " " << std::setprecision(8) << result.p[i]
                << "\n";
    }
    std::cout << "END_POLICY\n";
    // Action head (Ceres C3): per-move (W,D,L). Omitted when the net has no
    // action head (values stay NaN).
    if (!result.action.empty() && !std::isnan(result.action[0])) {
      std::cout << "BEGIN_ACTION " << result.p.size() << "\n";
      for (size_t i = 0; i < result.p.size(); i++) {
        std::cout << uci_moves[i] << " " << std::setprecision(8)
                  << result.action[i * 3 + 0] << " " << result.action[i * 3 + 1]
                  << " " << result.action[i * 3 + 2] << "\n";
      }
      std::cout << "END_ACTION\n";
    }
    std::cout << "END_NN_OUTPUT\n";
  } catch (Exception& ex) {
    std::cerr << ex.what() << std::endl;
  }
}

}  // namespace lczero
