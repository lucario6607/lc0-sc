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

#include "chess/board.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <sstream>

#include "utils/exception.h"

#if not defined(NO_PEXT)
// Include header for pext instruction.
#include <immintrin.h>
#endif

namespace lczero {

const char* ChessBoard::kStartposFen =
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

const ChessBoard ChessBoard::kStartposBoard(ChessBoard::kStartposFen);

const BitBoard ChessBoard::kPawnMask = 0x00FFFFFFFFFFFF00ULL;

void ChessBoard::Clear() {
  *this = ChessBoard();
}

void ChessBoard::Mirror() {
  our_pieces_.Mirror();
  their_pieces_.Mirror();
  std::swap(our_pieces_, their_pieces_);
  rooks_.Mirror();
  bishops_.Mirror();
  pawns_.Mirror();
  our_king_.Mirror();
  their_king_.Mirror();
  std::swap(our_king_, their_king_);
  castlings_.Mirror();
  flipped_ = !flipped_;
}

namespace {
static const std::pair<int, int> kKingMoves[] = {
    {-1, -1}, {-1, 0}, {-1, 1}, {0, -1}, {0, 1}, {1, -1}, {1, 0}, {1, 1}};

static const std::pair<int, int> kRookDirections[] = {
    {1, 0}, {-1, 0}, {0, 1}, {0, -1}};

static const std::pair<int, int> kBishopDirections[] = {
    {1, 1}, {-1, 1}, {1, -1}, {-1, -1}};

// Which squares can rook attack from every of squares.
static const BitBoard kRookAttacks[] = {
    0x01010101010101FEULL, 0x02020202020202FDULL, 0x04040404040404FBULL,
    0x08080808080808F7ULL, 0x10101010101010EFULL, 0x20202020202020DFULL,
    0x40404040404040BFULL, 0x808080808080807FULL, 0x010101010101FE01ULL,
    0x020202020202FD02ULL, 0x040404040404FB04ULL, 0x080808080808F708ULL,
    0x101010101010EF10ULL, 0x202020202020DF20ULL, 0x404040404040BF40ULL,
    0x8080808080807F80ULL, 0x0101010101FE0101ULL, 0x0202020202FD0202ULL,
    0x0404040404FB0404ULL, 0x0808080808F70808ULL, 0x1010101010EF1010ULL,
    0x2020202020DF2020ULL, 0x4040404040BF4040ULL, 0x80808080807F8080ULL,
    0x01010101FE010101ULL, 0x02020202FD020202ULL, 0x04040404FB040404ULL,
    0x08080808F7080808ULL, 0x10101010EF101010ULL, 0x20202020DF202020ULL,
    0x40404040BF404040ULL, 0x808080807F808080ULL, 0x010101FE01010101ULL,
    0x020202FD02020202ULL, 0x040404FB04040404ULL, 0x080808F708080808ULL,
    0x101010EF10101010ULL, 0x202020DF20202020ULL, 0x404040BF40404040ULL,
    0x8080807F80808080ULL, 0x0101FE0101010101ULL, 0x0202FD0202020202ULL,
    0x0404FB0404040404ULL, 0x0808F70808080808ULL, 0x1010EF1010101010ULL,
    0x2020DF2020202020ULL, 0x4040BF4040404040ULL, 0x80807F8080808080ULL,
    0x01FE010101010101ULL, 0x02FD020202020202ULL, 0x04FB040404040404ULL,
    0x08F7080808080808ULL, 0x10EF101010101010ULL, 0x20DF202020202020ULL,
    0x40BF404040404040ULL, 0x807F808080808080ULL, 0xFE01010101010101ULL,
    0xFD02020202020202ULL, 0xFB04040404040404ULL, 0xF708080808080808ULL,
    0xEF10101010101010ULL, 0xDF20202020202020ULL, 0xBF40404040404040ULL,
    0x7F80808080808080ULL};
// Which squares can bishop attack.
static const BitBoard kBishopAttacks[] = {
    0x8040201008040200ULL, 0x0080402010080500ULL, 0x0000804020110A00ULL,
    0x0000008041221400ULL, 0x0000000182442800ULL, 0x0000010204885000ULL,
    0x000102040810A000ULL, 0x0102040810204000ULL, 0x4020100804020002ULL,
    0x8040201008050005ULL, 0x00804020110A000AULL, 0x0000804122140014ULL,
    0x0000018244280028ULL, 0x0001020488500050ULL, 0x0102040810A000A0ULL,
    0x0204081020400040ULL, 0x2010080402000204ULL, 0x4020100805000508ULL,
    0x804020110A000A11ULL, 0x0080412214001422ULL, 0x0001824428002844ULL,
    0x0102048850005088ULL, 0x02040810A000A010ULL, 0x0408102040004020ULL,
    0x1008040200020408ULL, 0x2010080500050810ULL, 0x4020110A000A1120ULL,
    0x8041221400142241ULL, 0x0182442800284482ULL, 0x0204885000508804ULL,
    0x040810A000A01008ULL, 0x0810204000402010ULL, 0x0804020002040810ULL,
    0x1008050005081020ULL, 0x20110A000A112040ULL, 0x4122140014224180ULL,
    0x8244280028448201ULL, 0x0488500050880402ULL, 0x0810A000A0100804ULL,
    0x1020400040201008ULL, 0x0402000204081020ULL, 0x0805000508102040ULL,
    0x110A000A11204080ULL, 0x2214001422418000ULL, 0x4428002844820100ULL,
    0x8850005088040201ULL, 0x10A000A010080402ULL, 0x2040004020100804ULL,
    0x0200020408102040ULL, 0x0500050810204080ULL, 0x0A000A1120408000ULL,
    0x1400142241800000ULL, 0x2800284482010000ULL, 0x5000508804020100ULL,
    0xA000A01008040201ULL, 0x4000402010080402ULL, 0x0002040810204080ULL,
    0x0005081020408000ULL, 0x000A112040800000ULL, 0x0014224180000000ULL,
    0x0028448201000000ULL, 0x0050880402010000ULL, 0x00A0100804020100ULL,
    0x0040201008040201ULL};
// Which squares can knight attack.
static const BitBoard kKnightAttacks[] = {
    0x0000000000020400ULL, 0x0000000000050800ULL, 0x00000000000A1100ULL,
    0x0000000000142200ULL, 0x0000000000284400ULL, 0x0000000000508800ULL,
    0x0000000000A01000ULL, 0x0000000000402000ULL, 0x0000000002040004ULL,
    0x0000000005080008ULL, 0x000000000A110011ULL, 0x0000000014220022ULL,
    0x0000000028440044ULL, 0x0000000050880088ULL, 0x00000000A0100010ULL,
    0x0000000040200020ULL, 0x0000000204000402ULL, 0x0000000508000805ULL,
    0x0000000A1100110AULL, 0x0000001422002214ULL, 0x0000002844004428ULL,
    0x0000005088008850ULL, 0x000000A0100010A0ULL, 0x0000004020002040ULL,
    0x0000020400040200ULL, 0x0000050800080500ULL, 0x00000A1100110A00ULL,
    0x0000142200221400ULL, 0x0000284400442800ULL, 0x0000508800885000ULL,
    0x0000A0100010A000ULL, 0x0000402000204000ULL, 0x0002040004020000ULL,
    0x0005080008050000ULL, 0x000A1100110A0000ULL, 0x0014220022140000ULL,
    0x0028440044280000ULL, 0x0050880088500000ULL, 0x00A0100010A00000ULL,
    0x0040200020400000ULL, 0x0204000402000000ULL, 0x0508000805000000ULL,
    0x0A1100110A000000ULL, 0x1422002214000000ULL, 0x2844004428000000ULL,
    0x5088008850000000ULL, 0xA0100010A0000000ULL, 0x4020002040000000ULL,
    0x0400040200000000ULL, 0x0800080500000000ULL, 0x1100110A00000000ULL,
    0x2200221400000000ULL, 0x4400442800000000ULL, 0x8800885000000000ULL,
    0x100010A000000000ULL, 0x2000204000000000ULL, 0x0004020000000000ULL,
    0x0008050000000000ULL, 0x00110A0000000000ULL, 0x0022140000000000ULL,
    0x0044280000000000ULL, 0x0088500000000000ULL, 0x0010A00000000000ULL,
    0x0020400000000000ULL};
// Opponent pawn attacks
static const BitBoard kPawnAttacks[] = {
    0x0000000000000200ULL, 0x0000000000000500ULL, 0x0000000000000A00ULL,
    0x0000000000001400ULL, 0x0000000000002800ULL, 0x0000000000005000ULL,
    0x000000000000A000ULL, 0x0000000000004000ULL, 0x0000000000020000ULL,
    0x0000000000050000ULL, 0x00000000000A0000ULL, 0x0000000000140000ULL,
    0x0000000000280000ULL, 0x0000000000500000ULL, 0x0000000000A00000ULL,
    0x0000000000400000ULL, 0x0000000002000000ULL, 0x0000000005000000ULL,
    0x000000000A000000ULL, 0x0000000014000000ULL, 0x0000000028000000ULL,
    0x0000000050000000ULL, 0x00000000A0000000ULL, 0x0000000040000000ULL,
    0x0000000200000000ULL, 0x0000000500000000ULL, 0x0000000A00000000ULL,
    0x0000001400000000ULL, 0x0000002800000000ULL, 0x0000005000000000ULL,
    0x000000A000000000ULL, 0x0000004000000000ULL, 0x0000020000000000ULL,
    0x0000050000000000ULL, 0x00000A0000000000ULL, 0x0000140000000000ULL,
    0x0000280000000000ULL, 0x0000500000000000ULL, 0x0000A00000000000ULL,
    0x0000400000000000ULL, 0x0002000000000000ULL, 0x0005000000000000ULL,
    0x000A000000000000ULL, 0x0014000000000000ULL, 0x0028000000000000ULL,
    0x0050000000000000ULL, 0x00A0000000000000ULL, 0x0040000000000000ULL,
    0x0000000000000000ULL, 0x0000000000000000ULL, 0x0000000000000000ULL,
    0x0000000000000000ULL, 0x0000000000000000ULL, 0x0000000000000000ULL,
    0x0000000000000000ULL, 0x0000000000000000ULL, 0x0000000000000000ULL,
    0x0000000000000000ULL, 0x0000000000000000ULL, 0x0000000000000000ULL,
    0x0000000000000000ULL, 0x0000000000000000ULL, 0x0000000000000000ULL,
    0x0000000000000000ULL};

static const Move::Promotion kPromotions[] = {
    Move::Promotion::Queen,
    Move::Promotion::Rook,
    Move::Promotion::Bishop,
    Move::Promotion::Knight,
};

// Magic bitboard routines and structures.
// We use so-called "fancy" magic bitboards.

// Structure holding all relevant magic parameters per square.
struct MagicParams {
  // Relevant occupancy mask.
  uint64_t mask_;
  // Pointer to lookup table.
  BitBoard* attacks_table_;
#if defined(NO_PEXT)
  // Magic number.
  uint64_t magic_number_;
  // Number of bits to shift.
  uint8_t shift_bits_;
#endif
};

#if defined(NO_PEXT)
// Magic numbers determined via trial and error with random number generator
// such that the number of relevant occupancy bits suffice to index the attacks
// tables with only constructive collisions.
static const BitBoard kRookMagicNumbers[] = {
    0x088000102088C001ULL, 0x10C0200040001000ULL, 0x83001041000B2000ULL,
    0x0680280080041000ULL, 0x488004000A080080ULL, 0x0100180400010002ULL,
    0x040001C401021008ULL, 0x02000C04A980C302ULL, 0x0000800040082084ULL,
    0x5020C00820025000ULL, 0x0001002001044012ULL, 0x0402001020400A00ULL,
    0x00C0800800040080ULL, 0x4028800200040080ULL, 0x00A0804200802500ULL,
    0x8004800040802100ULL, 0x0080004000200040ULL, 0x1082810020400100ULL,
    0x0020004010080040ULL, 0x2004818010042800ULL, 0x0601010008005004ULL,
    0x4600808002001400ULL, 0x0010040009180210ULL, 0x020412000406C091ULL,
    0x040084228000C000ULL, 0x8000810100204000ULL, 0x0084110100402000ULL,
    0x0046001A00204210ULL, 0x2001040080080081ULL, 0x0144020080800400ULL,
    0x0840108400080229ULL, 0x0480308A0000410CULL, 0x0460324002800081ULL,
    0x620080A001804000ULL, 0x2800802000801006ULL, 0x0002809000800800ULL,
    0x4C09040080802800ULL, 0x4808800C00800200ULL, 0x0200311004001802ULL,
    0x0400008402002141ULL, 0x0410800140008020ULL, 0x000080C001050020ULL,
    0x004080204A020010ULL, 0x0224201001010038ULL, 0x0109001108010004ULL,
    0x0282004844020010ULL, 0x8228180110040082ULL, 0x0001000080C10002ULL,
    0x024000C120801080ULL, 0x0001406481060200ULL, 0x0101243200418600ULL,
    0x0108800800100080ULL, 0x4022080100100D00ULL, 0x0000843040600801ULL,
    0x8301000200CC0500ULL, 0x1000004500840200ULL, 0x1100104100800069ULL,
    0x2001008440001021ULL, 0x2002008830204082ULL, 0x0010145000082101ULL,
    0x01A2001004200842ULL, 0x1007000608040041ULL, 0x000A08100203028CULL,
    0x02D4048040290402ULL};
static const BitBoard kBishopMagicNumbers[] = {
    0x0008201802242020ULL, 0x0021040424806220ULL, 0x4006360602013080ULL,
    0x0004410020408002ULL, 0x2102021009001140ULL, 0x08C2021004000001ULL,
    0x6001031120200820ULL, 0x1018310402201410ULL, 0x401CE00210820484ULL,
    0x001029D001004100ULL, 0x2C00101080810032ULL, 0x0000082581000010ULL,
    0x10000A0210110020ULL, 0x200002016C202000ULL, 0x0201018821901000ULL,
    0x006A0300420A2100ULL, 0x0010014005450400ULL, 0x1008C12008028280ULL,
    0x00010010004A0040ULL, 0x3000820802044020ULL, 0x0000800405A02820ULL,
    0x8042004300420240ULL, 0x10060801210D2000ULL, 0x0210840500511061ULL,
    0x0008142118509020ULL, 0x0021109460040104ULL, 0x00A1480090019030ULL,
    0x0102008808008020ULL, 0x884084000880E001ULL, 0x040041020A030100ULL,
    0x3000810104110805ULL, 0x04040A2006808440ULL, 0x0044040404C01100ULL,
    0x4122B80800245004ULL, 0x0044020502380046ULL, 0x0100400888020200ULL,
    0x01C0002060020080ULL, 0x4008811100021001ULL, 0x8208450441040609ULL,
    0x0408004900008088ULL, 0x0294212051220882ULL, 0x000041080810E062ULL,
    0x10480A018E005000ULL, 0x80400A0204201600ULL, 0x2800200204100682ULL,
    0x0020200400204441ULL, 0x0A500600A5002400ULL, 0x801602004A010100ULL,
    0x0801841008040880ULL, 0x10010880C4200028ULL, 0x0400004424040000ULL,
    0x0401000142022100ULL, 0x00A00010020A0002ULL, 0x1010400204010810ULL,
    0x0829910400840000ULL, 0x0004235204010080ULL, 0x1002008143082000ULL,
    0x11840044440C2080ULL, 0x2802A02104030440ULL, 0x6100000900840401ULL,
    0x1C20A15A90420200ULL, 0x0088414004480280ULL, 0x0000204242881100ULL,
    0x0240080802809010ULL};
#endif

// Magic parameters for rooks/bishops.
static MagicParams rook_magic_params[64];
static MagicParams bishop_magic_params[64];

// Precomputed attacks bitboard tables.
static BitBoard rook_attacks_table[102400];
static BitBoard bishop_attacks_table[5248];

// Builds rook or bishop attacks table.
static void BuildAttacksTable(MagicParams* magic_params,
                              BitBoard* attacks_table,
                              const std::pair<int, int>* directions) {
  // Offset into lookup table.
  uint32_t table_offset = 0;

  // Initialize for all board squares.
  for (unsigned square = 0; square < 64; square++) {
    const BoardSquare b_sq(square);

    // Calculate relevant occupancy masks.
    BitBoard mask = {0};

    for (int j = 0; j < 4; j++) {
      auto direction = directions[j];
      auto dst_row = b_sq.row();
      auto dst_col = b_sq.col();
      while (true) {
        dst_row += direction.first;
        dst_col += direction.second;
        // If the next square in this direction is invalid, the current square
        // is at the board's edge and should not be added.
        if (!BoardSquare::IsValid(dst_row + direction.first,
                                  dst_col + direction.second))
          break;
        const BoardSquare destination(dst_row, dst_col);
        mask.set(destination);
      }
    }

    // Set mask.
    magic_params[square].mask_ = mask.as_int();

    // Cache relevant occupancy board squares.
    std::vector<BoardSquare> occupancy_squares;

    for (auto occ_sq : BitBoard(magic_params[square].mask_)) {
      occupancy_squares.emplace_back(occ_sq);
    }

#if defined(NO_PEXT)
    // Set number of shifted bits. The magic numbers have been chosen such that
    // the number of relevant occupancy bits suffice to index the attacks table.
    magic_params[square].shift_bits_ = 64 - occupancy_squares.size();
#endif

    // Set pointer to lookup table.
    magic_params[square].attacks_table_ = &attacks_table[table_offset];

    // Clear attacks table (used for sanity check later on).
    for (int i = 0; i < (1 << occupancy_squares.size()); i++) {
      attacks_table[table_offset + i] = 0;
    }

    // Build square attacks table for every possible relevant occupancy
    // bitboard.
    for (int i = 0; i < (1 << occupancy_squares.size()); i++) {
      BitBoard occupancy(0);

      for (size_t bit = 0; bit < occupancy_squares.size(); bit++) {
        occupancy.set_if(occupancy_squares[bit], (1 << bit) & i);
      }

      // Calculate attacks bitboard corresponding to this occupancy bitboard.
      BitBoard attacks(0);

      for (int j = 0; j < 4; j++) {
        auto direction = directions[j];
        auto dst_row = b_sq.row();
        auto dst_col = b_sq.col();
        while (true) {
          dst_row += direction.first;
          dst_col += direction.second;
          if (!BoardSquare::IsValid(dst_row, dst_col)) break;
          const BoardSquare destination(dst_row, dst_col);
          attacks.set(destination);
          if (occupancy.get(destination)) break;
        }
      }

#if defined(NO_PEXT)
      // Calculate magic index.
      uint64_t index = occupancy.as_int();
      index *= magic_params[square].magic_number_;
      index >>= magic_params[square].shift_bits_;

      // Sanity check. The magic numbers have been chosen such that
      // the number of relevant occupancy bits suffice to index the attacks
      // table. If the table already contains an attacks bitboard, possible
      // collisions should be constructive.
      if (attacks_table[table_offset + index] != 0 &&
          attacks_table[table_offset + index] != attacks) {
        throw Exception("Invalid magic number!");
      }
#else
      uint64_t index =
          _pext_u64(occupancy.as_int(), magic_params[square].mask_);
#endif

      // Update table.
      attacks_table[table_offset + index] = attacks;
    }

    // Update table offset.
    table_offset += (1 << occupancy_squares.size());
  }
}

// Returns the rook attacks bitboard for the given rook board square and the
// given occupied piece bitboard.
static inline BitBoard GetRookAttacks(const BoardSquare rook_square,
                                      const BitBoard pieces) {
  // Calculate magic index.
  const uint8_t square = rook_square.as_int();

#if defined(NO_PEXT)
  uint64_t index = pieces.as_int() & rook_magic_params[square].mask_;
  index *= rook_magic_params[square].magic_number_;
  index >>= rook_magic_params[square].shift_bits_;
#else
  uint64_t index = _pext_u64(pieces.as_int(), rook_magic_params[square].mask_);
#endif

  // Return attacks bitboard.
  return rook_magic_params[square].attacks_table_[index];
}

// Returns the bishop attacks bitboard for the given bishop board square and
// the given occupied piece bitboard.
static inline BitBoard GetBishopAttacks(const BoardSquare bishop_square,
                                        const BitBoard pieces) {
  // Calculate magic index.
  const uint8_t square = bishop_square.as_int();

#if defined(NO_PEXT)
  uint64_t index = pieces.as_int() & bishop_magic_params[square].mask_;
  index *= bishop_magic_params[square].magic_number_;
  index >>= bishop_magic_params[square].shift_bits_;
#else
  uint64_t index =
      _pext_u64(pieces.as_int(), bishop_magic_params[square].mask_);
#endif

  // Return attacks bitboard.
  return bishop_magic_params[square].attacks_table_[index];
}

}  // namespace

void InitializeMagicBitboards() {
#if defined(NO_PEXT)
  // Set magic numbers for all board squares.
  for (unsigned square = 0; square < 64; square++) {
    rook_magic_params[square].magic_number_ =
        kRookMagicNumbers[square].as_int();
    bishop_magic_params[square].magic_number_ =
        kBishopMagicNumbers[square].as_int();
  }
#endif

  // Build attacks tables.
  BuildAttacksTable(rook_magic_params, rook_attacks_table, kRookDirections);
  BuildAttacksTable(bishop_magic_params, bishop_attacks_table,
                    kBishopDirections);
}

MoveList ChessBoard::GeneratePseudolegalMoves() const {
  MoveList result;
  result.reserve(60);
  for (auto source : our_pieces_) {
    // King
    if (source == our_king_) {
      for (const auto& delta : kKingMoves) {
        const auto dst_row = source.row() + delta.first;
        const auto dst_col = source.col() + delta.second;
        if (!BoardSquare::IsValid(dst_row, dst_col)) continue;
        const BoardSquare destination(dst_row, dst_col);
        if (our_pieces_.get(destination)) continue;
        if (IsUnderAttack(destination)) continue;
        result.emplace_back(source, destination);
      }
      // Castlings.
      auto walk_free = [this](int from, int to, int rook, int king) {
        for (int i = from; i <= to; ++i) {
          if (i == rook || i == king) continue;
          if (our_pieces_.get(i) || their_pieces_.get(i)) return false;
        }
        return true;
      };
      // @From may be less or greater than @to. @To is not included in check
      // unless it is the same with @from.
      auto range_attacked = [this](int from, int to) {
        if (from == to) return IsUnderAttack(from);
        const int increment = from < to ? 1 : -1;
        while (from != to) {
          if (IsUnderAttack(from)) return true;
          from += increment;
        }
        return false;
      };
      const uint8_t king = source.col();
      // For castlings we don't check destination king square for checks, it
      // will be done in legal move check phase.
      if (castlings_.we_can_000()) {
        const uint8_t qrook = castlings_.our_queenside_rook();
        if (walk_free(std::min(static_cast<uint8_t>(C1), qrook),
                      std::max(static_cast<uint8_t>(D1), king), qrook, king) &&
            !range_attacked(king, C1)) {
          result.emplace_back(source, BoardSquare(RANK_1, qrook));
        }
      }
      if (castlings_.we_can_00()) {
        const uint8_t krook = castlings_.our_kingside_rook();
        if (walk_free(std::min(static_cast<uint8_t>(F1), king),
                      std::max(static_cast<uint8_t>(G1), krook), krook, king) &&
            !range_attacked(king, G1)) {
          result.emplace_back(source, BoardSquare(RANK_1, krook));
        }
      }
      continue;
    }
    bool processed_piece = false;
    // Rook (and queen)
    if (rooks_.get(source)) {
      processed_piece = true;
      BitBoard attacked =
          GetRookAttacks(source, our_pieces_ | their_pieces_) - our_pieces_;

      for (const auto& destination : attacked) {
        result.emplace_back(source, destination);
      }
    }
    // Bishop (and queen)
    if (bishops_.get(source)) {
      processed_piece = true;
      BitBoard attacked =
          GetBishopAttacks(source, our_pieces_ | their_pieces_) - our_pieces_;

      for (const auto& destination : attacked) {
        result.emplace_back(source, destination);
      }
    }
    if (processed_piece) continue;
    // Pawns.
    if ((pawns_ & kPawnMask).get(source)) {
      // Moves forward.
      {
        const auto dst_row = source.row() + 1;
        const auto dst_col = source.col();
        const BoardSquare destination(dst_row, dst_col);

        if (!our_pieces_.get(destination) && !their_pieces_.get(destination)) {
          if (dst_row != RANK_8) {
            result.emplace_back(source, destination);
            if (dst_row == RANK_3) {
              // Maybe it'll be possible to move two squares.
              if (!our_pieces_.get(RANK_4, dst_col) &&
                  !their_pieces_.get(RANK_4, dst_col)) {
                result.emplace_back(source, BoardSquare(RANK_4, dst_col));
              }
            }
          } else {
            // Promotions
            for (auto promotion : kPromotions) {
              result.emplace_back(source, destination, promotion);
            }
          }
        }
      }
      // Captures.
      {
        for (auto direction : {-1, 1}) {
          const auto dst_row = source.row() + 1;
          const auto dst_col = source.col() + direction;
          if (dst_col < 0 || dst_col >= 8) continue;
          const BoardSquare destination(dst_row, dst_col);
          if (their_pieces_.get(destination)) {
            if (dst_row == RANK_8) {
              // Promotion.
              for (auto promotion : kPromotions) {
                result.emplace_back(source, destination, promotion);
              }
            } else {
              // Ordinary capture.
              result.emplace_back(source, destination);
            }
          } else if (dst_row == RANK_6 && pawns_.get(RANK_8, dst_col)) {
            // En passant.
            // "Pawn" on opponent's file 8 means that en passant is possible.
            // Those fake pawns are reset in ApplyMove.
            result.emplace_back(source, destination);
          }
        }
      }
      continue;
    }
    // Knight.
    {
      for (const auto destination :
           kKnightAttacks[source.as_int()] - our_pieces_) {
        result.emplace_back(source, destination);
      }
    }
  }
  return result;
}  // namespace lczero

bool ChessBoard::ApplyMove(Move move) {
  const auto& from = move.from();
  const auto& to = move.to();
  const auto from_row = from.row();
  const auto from_col = from.col();
  const auto to_row = to.row();
  const auto to_col = to.col();

  // Castlings.
  if (from == our_king_) {
    castlings_.reset_we_can_00();
    castlings_.reset_we_can_000();
    auto do_castling = [this](int king_dst, int rook_src, int rook_dst) {
      // Remove en passant flags.
      pawns_ &= kPawnMask;
      our_pieces_.reset(our_king_);
      our_pieces_.reset(rook_src);
      rooks_.reset(rook_src);
      our_pieces_.set(king_dst);
      our_pieces_.set(rook_dst);
      rooks_.set(rook_dst);
      our_king_ = king_dst;
    };
    if (from_row == RANK_1 && to_row == RANK_1) {
      const auto our_rooks = rooks() & our_pieces_;
      if (our_rooks.get(to)) {
        // Castling.
        if (to_col > from_col) {
          // Kingside.
          do_castling(G1, to.as_int(), F1);
        } else {
          // Queenside.
          do_castling(C1, to.as_int(), D1);
        }
        return false;
      } else if (from_col == FILE_E && to_col == FILE_G) {
        // Non FRC-style e1g1 castling (as opposed to e1h1).
        do_castling(G1, H1, F1);
        return false;
      } else if (from_col == FILE_E && to_col == FILE_C) {
        // Non FRC-style e1c1 castling (as opposed to e1a1).
        do_castling(C1, A1, D1);
        return false;
      }
    }
  }

  // Move in our pieces.
  our_pieces_.reset(from);
  our_pieces_.set(to);

  // Remove captured piece.
  bool reset_50_moves = their_pieces_.get(to);
  their_pieces_.reset(to);
  rooks_.reset(to);
  bishops_.reset(to);
  pawns_.reset(to);
  if (to.as_int() == A8 + castlings_.their_kingside_rook()) {
    castlings_.reset_they_can_00();
  }
  if (to.as_int() == A8 + castlings_.their_queenside_rook()) {
    castlings_.reset_they_can_000();
  }

  // En passant.
  if (from_row == RANK_5 && pawns_.get(from) && from_col != to_col &&
      pawns_.get(RANK_8, to_col)) {
    pawns_.reset(RANK_5, to_col);
    their_pieces_.reset(RANK_5, to_col);
  }

  // Remove en passant flags.
  pawns_ &= kPawnMask;

  // If pawn was moved, reset 50 move draw counter.
  reset_50_moves |= pawns_.get(from);

  // King, non-castling move
  if (from == our_king_) {
    our_king_ = to;
    return reset_50_moves;
  }

  // Promotion.
  if (to_row == RANK_8 && pawns_.get(from)) {
    switch (move.promotion()) {
      case Move::Promotion::Rook:
        rooks_.set(to);
        break;
      case Move::Promotion::Bishop:
        bishops_.set(to);
        break;
      case Move::Promotion::Queen:
        rooks_.set(to);
        bishops_.set(to);
        break;
      default:;
    }
    pawns_.reset(from);
    return true;
  }

  // Reset castling rights.
  if (from_row == RANK_1 && rooks_.get(from)) {
    if (from_col == castlings_.our_queenside_rook()) {
      castlings_.reset_we_can_000();
    }
    if (from_col == castlings_.our_kingside_rook()) {
      castlings_.reset_we_can_00();
    }
  }

  // Ordinary move.
  rooks_.set_if(to, rooks_.get(from));
  bishops_.set_if(to, bishops_.get(from));
  pawns_.set_if(to, pawns_.get(from));
  rooks_.reset(from);
  bishops_.reset(from);
  pawns_.reset(from);

  // Set en passant flag.
  if (to_row - from_row == 2 && pawns_.get(to)) {
    BoardSquare ep_sq(to_row - 1, to_col);
    if (kPawnAttacks[ep_sq.as_int()].intersects(their_pieces_ & pawns_)) {
      pawns_.set(0, to_col);
    }
  }
  return reset_50_moves;
}

bool ChessBoard::IsUnderAttack(BoardSquare square) const {
  const int row = square.row();
  const int col = square.col();
  // Check king.
  {
    const int krow = their_king_.row();
    const int kcol = their_king_.col();
    if (std::abs(krow - row) <= 1 && std::abs(kcol - col) <= 1) return true;
  }
  // Check rooks (and queens).
  if (GetRookAttacks(square, our_pieces_ | their_pieces_)
          .intersects(their_pieces_ & rooks_)) {
    return true;
  }
  // Check bishops.
  if (GetBishopAttacks(square, our_pieces_ | their_pieces_)
          .intersects(their_pieces_ & bishops_)) {
    return true;
  }
  // Check pawns.
  if (kPawnAttacks[square.as_int()].intersects(their_pieces_ & pawns_)) {
    return true;
  }
  // Check knights.
  {
    if (kKnightAttacks[square.as_int()].intersects(their_pieces_ - their_king_ -
                                                   rooks_ - bishops_ -
                                                   (pawns_ & kPawnMask))) {
      return true;
    }
  }
  return false;
}

bool ChessBoard::IsSameMove(Move move1, Move move2) const {
  // If moves are equal, it's the same move.
  if (move1 == move2) return true;
  // Explicitly check all legacy castling moves. Need to check for king, for
  // e.g. rook e1a1 and e1c1 are different moves.
  if (move1.from() != move2.from() || move1.from() != E1 ||
      our_king_ != move1.from()) {
    return false;
  }
  if (move1.to() == A1 && move2.to() == C1) return true;
  if (move1.to() == C1 && move2.to() == A1) return true;
  if (move1.to() == G1 && move2.to() == H1) return true;
  if (move1.to() == H1 && move2.to() == G1) return true;
  return false;
}

Move ChessBoard::GetLegacyMove(Move move) const {
  if (our_king_ != move.from() || !our_pieces_.get(move.to())) {
    return move;
  }
  if (move == Move(E1, H1)) return Move(E1, G1);
  if (move == Move(E1, A1)) return Move(E1, C1);
  return move;
}

Move ChessBoard::GetModernMove(Move move) const {
  if (our_king_ != E1 || move.from() != E1) return move;
  if (move == Move(E1, G1) && !our_pieces_.get(G1)) return Move(E1, H1);
  if (move == Move(E1, C1) && !our_pieces_.get(C1)) return Move(E1, A1);
  return move;
}

KingAttackInfo ChessBoard::GenerateKingAttackInfo() const {
  KingAttackInfo king_attack_info;

  // Number of attackers that give check (used for double check detection).
  unsigned num_king_attackers = 0;

  const int row = our_king_.row();
  const int col = our_king_.col();
  // King checks are unnecessary, as kings cannot give check.
  // Check rooks (and queens).
  if (kRookAttacks[our_king_.as_int()].intersects(their_pieces_ & rooks_)) {
    for (const auto& direction : kRookDirections) {
      auto dst_row = row;
      auto dst_col = col;
      BitBoard attack_line(0);
      bool possible_pinned_piece_found = false;
      BoardSquare possible_pinned_piece;
      while (true) {
        dst_row += direction.first;
        dst_col += direction.second;
        if (!BoardSquare::IsValid(dst_row, dst_col)) break;
        const BoardSquare destination(dst_row, dst_col);
        if (our_pieces_.get(destination)) {
          if (possible_pinned_piece_found) {
            // No pieces pinned.
            break;
          } else {
            // This is a possible pinned piece.
            possible_pinned_piece_found = true;
            possible_pinned_piece = destination;
          }
        }
        if (!possible_pinned_piece_found) {
          attack_line.set(destination);
        }
        if (their_pieces_.get(destination)) {
          if (rooks_.get(destination)) {
            if (possible_pinned_piece_found) {
              // Store the pinned piece.
              king_attack_info.pinned_pieces_.set(possible_pinned_piece);
            } else {
              // Update attack lines.
              king_attack_info.attack_lines_ =
                  king_attack_info.attack_lines_ | attack_line;
              num_king_attackers++;
            }
          }
          break;
        }
      }
    }
  }
  // Check bishops.
  if (kBishopAttacks[our_king_.as_int()].intersects(their_pieces_ & bishops_)) {
    for (const auto& direction : kBishopDirections) {
      auto dst_row = row;
      auto dst_col = col;
      BitBoard attack_line(0);
      bool possible_pinned_piece_found = false;
      BoardSquare possible_pinned_piece;
      while (true) {
        dst_row += direction.first;
        dst_col += direction.second;
        if (!BoardSquare::IsValid(dst_row, dst_col)) break;
        const BoardSquare destination(dst_row, dst_col);
        if (our_pieces_.get(destination)) {
          if (possible_pinned_piece_found) {
            // No pieces pinned.
            break;
          } else {
            // This is a possible pinned piece.
            possible_pinned_piece_found = true;
            possible_pinned_piece = destination;
          }
        }
        if (!possible_pinned_piece_found) {
          attack_line.set(destination);
        }
        if (their_pieces_.get(destination)) {
          if (bishops_.get(destination)) {
            if (possible_pinned_piece_found) {
              // Store the pinned piece.
              king_attack_info.pinned_pieces_.set(possible_pinned_piece);
            } else {
              // Update attack lines.
              king_attack_info.attack_lines_ =
                  king_attack_info.attack_lines_ | attack_line;
              num_king_attackers++;
            }
          }
          break;
        }
      }
    }
  }
  // Check pawns.
  const BitBoard attacking_pawns =
      kPawnAttacks[our_king_.as_int()] & their_pieces_ & pawns_;
  king_attack_info.attack_lines_ =
      king_attack_info.attack_lines_ | attacking_pawns;

  if (attacking_pawns.as_int()) {
    // No more than one pawn can give check.
    num_king_attackers++;
  }

  // Check knights.
  const BitBoard attacking_knights =
      kKnightAttacks[our_king_.as_int()] &
      (their_pieces_ - their_king_ - rooks_ - bishops_ - (pawns_ & kPawnMask));
  king_attack_info.attack_lines_ =
      king_attack_info.attack_lines_ | attacking_knights;

  if (attacking_knights.as_int()) {
    // No more than one knight can give check.
    num_king_attackers++;
  }

  assert(num_king_attackers <= 2);
  king_attack_info.double_check_ = (num_king_attackers == 2);

  return king_attack_info;
}

bool ChessBoard::IsLegalMove(Move move,
                             const KingAttackInfo& king_attack_info) const {
  const auto& from = move.from();
  const auto& to = move.to();

  // En passant. Complex but rare. Just apply
  // and check that we are not under check.
  if (from.row() == 4 && pawns_.get(from) && from.col() != to.col() &&
      pawns_.get(7, to.col())) {
    ChessBoard board(*this);
    board.ApplyMove(move);
    return !board.IsUnderCheck();
  }

  // Check if we are already under check.
  if (king_attack_info.in_check()) {
    // King move.
    if (from == our_king_) {
      // Just apply and check that we are not under check.
      ChessBoard board(*this);
      board.ApplyMove(move);
      return !board.IsUnderCheck();
    }

    // Pinned pieces can never resolve a check.
    if (king_attack_info.is_pinned(from)) {
      return false;
    }

    // The piece to move is no king and is not pinned.
    if (king_attack_info.in_double_check()) {
      // Only a king move can resolve the double check.
      return false;
    } else {
      // Only one attacking piece gives check.
      // Our piece is free to move (not pinned). Check if the attacker is
      // captured or interposed after the piece has moved to its destination
      // square.
      return king_attack_info.is_on_attack_line(to);
    }
  }

  // King moves.
  if (from == our_king_) {
    if (from.row() != 0 || to.row() != 0 ||
        (abs(from.col() - to.col()) == 1 && !our_pieces_.get(to))) {
      // Non-castling move. Already checked during movegen.
      return true;
    }
    // Checking whether king is under check after castling.
    ChessBoard board(*this);
    board.ApplyMove(move);
    return !board.IsUnderCheck();
  }

  // If we get here, we are not under check.
  // If the piece is not pinned, it is free to move anywhere.
  if (!king_attack_info.is_pinned(from)) return true;

  // The piece is pinned. Now check that it stays on the same line w.r.t. the
  // king.
  const int dx_from = from.col() - our_king_.col();
  const int dy_from = from.row() - our_king_.row();
  const int dx_to = to.col() - our_king_.col();
  const int dy_to = to.row() - our_king_.row();

  if (dx_from == 0 || dx_to == 0) {
    return (dx_from == dx_to);
  } else {
    return (dx_from * dy_to == dx_to * dy_from);
  }
}

MoveList ChessBoard::GenerateLegalMoves() const {
  const KingAttackInfo king_attack_info = GenerateKingAttackInfo();
  MoveList result = GeneratePseudolegalMoves();
  result.erase(
      std::remove_if(result.begin(), result.end(),
                     [&](Move m) { return !IsLegalMove(m, king_attack_info); }),
      result.end());
  return result;
}

void ChessBoard::SetFromFen(std::string fen, int* rule50_ply, int* moves) {
  Clear();
  int row = 7;
  int col = 0;

  // Remove any trailing whitespaces to detect eof after the last field.
  fen.erase(std::find_if(fen.rbegin(), fen.rend(),
                         [](char c) { return !std::isspace(c); })
                .base(),
            fen.end());

  std::istringstream fen_str(fen);
  std::string board;
  fen_str >> board;
  std::string who_to_move = "w";
  if (!fen_str.eof()) fen_str >> who_to_move;
  // Assume no castling rights. Other engines, e.g., Stockfish, assume kings and
  // rooks on their initial rows can each castle with the outer-most rook.  Our
  // implementation currently supports 960 castling where white and black rooks
  // have matching columns, so it's unclear which rights to assume.
  std::string castlings = "-";
  if (!fen_str.eof()) fen_str >> castlings;
  std::string en_passant = "-";
  if (!fen_str.eof()) fen_str >> en_passant;
  int rule50_halfmoves = 0;
  if (!fen_str.eof()) fen_str >> rule50_halfmoves;
  int total_moves = 1;
  if (!fen_str.eof()) fen_str >> total_moves;
  if (!fen_str) throw Exception("Bad fen string: " + fen);

  for (char c : board) {
    if (c == '/') {
      --row;
      if (row < 0) throw Exception("Bad fen string (too many rows): " + fen);
      col = 0;
      continue;
    }
    if (std::isdigit(c)) {
      col += c - '0';
      continue;
    }
    if (col >= 8) throw Exception("Bad fen string (too many columns): " + fen);

    if (std::isupper(c)) {
      // White piece.
      our_pieces_.set(row, col);
    } else {
      // Black piece.
      their_pieces_.set(row, col);
    }

    if (c == 'K') {
      our_king_.set(row, col);
    } else if (c == 'k') {
      their_king_.set(row, col);
    } else if (c == 'R' || c == 'r') {
      rooks_.set(row, col);
    } else if (c == 'B' || c == 'b') {
      bishops_.set(row, col);
    } else if (c == 'Q' || c == 'q') {
      rooks_.set(row, col);
      bishops_.set(row, col);
    } else if (c == 'P' || c == 'p') {
      if (row == 7 || row == 0) {
        throw Exception("Bad fen string (pawn in first/last row): " + fen);
      }
      pawns_.set(row, col);
    } else if (c == 'N' || c == 'n') {
      // Do nothing
    } else {
      throw Exception("Bad fen string: " + fen);
    }
    ++col;
  }

  if (castlings != "-") {
    uint8_t our_left_rook = FILE_A;
    uint8_t our_right_rook = FILE_H;
    uint8_t their_left_rook = FILE_A;
    uint8_t their_right_rook = FILE_H;
    for (char c : castlings) {
      const bool is_black = std::islower(c);
      const int king_col = (is_black ? their_king_ : our_king_).col();
      const auto rooks =
          (is_black ? their_pieces_ : our_pieces_) & ChessBoard::rooks();
      auto find_rook = [rooks, king_col, fen](bool forward, uint8_t rank) {
        uint8_t rook;
        for (rook = forward ? FILE_A : FILE_H; rook != king_col;
             rook += 2 * forward - 1) {
          if (rooks.get(rank, rook)) break;
        }
        if (rook == king_col) {
          throw Exception("Bad fen string (missing rook): " + fen);
        }
        return rook;
      };
      if (c == 'K') {
        // Finding rightmost rook.
        our_right_rook = find_rook(false, RANK_1);
        castlings_.set_we_can_00();
      } else if (c == 'Q') {
        // Finding leftmost rook.
        our_left_rook = find_rook(true, RANK_1);
        castlings_.set_we_can_000();
      } else if (c >= 'A' && c <= 'H') {
        int rook_col = c - 'A';
        if (rook_col < king_col) {
          our_left_rook = rook_col;
          castlings_.set_we_can_000();
        } else {
          our_right_rook = rook_col;
          castlings_.set_we_can_00();
        }
      } else if (c == 'k') {
        // Finding rightmost rook.
        their_right_rook = find_rook(false, RANK_8);
        castlings_.set_they_can_00();
      } else if (c == 'q') {
        // Finding leftmost rook.
        their_left_rook = find_rook(true, RANK_8);
        castlings_.set_they_can_000();
      } else if (c >= 'a' && c <= 'h') {
        int rook_col = c - 'a';
        if (rook_col < king_col) {
          their_left_rook = rook_col;
          castlings_.set_they_can_000();
        } else {
          their_right_rook = rook_col;
          castlings_.set_they_can_00();
        }
      } else {
        throw Exception("Bad fen string (unexpected casting symbol): " + fen);
      }
    }
    castlings_.SetRookPositions(our_left_rook, our_right_rook, their_left_rook,
                                their_right_rook);
  }

  if (en_passant != "-") {
    auto square = BoardSquare(en_passant);
    if (square.row() != RANK_3 && square.row() != RANK_6)
      throw Exception("Bad fen string: " + fen + " wrong en passant rank");
    pawns_.set((square.row() == RANK_3) ? RANK_1 : RANK_8, square.col());
  }

  if (who_to_move == "b" || who_to_move == "B") {
    Mirror();
  } else if (who_to_move != "w" && who_to_move != "W") {
    throw Exception("Bad fen string (side to move): " + fen);
  }
  if (rule50_ply) *rule50_ply = rule50_halfmoves;
  if (moves) *moves = total_moves;
}

bool ChessBoard::HasMatingMaterial() const {
  if (!rooks_.empty() || !pawns_.empty()) {
    return true;
  }

  if ((our_pieces_ | their_pieces_).count() < 4) {
    // K v K, K+B v K, K+N v K.
    return false;
  }
  if (!(knights().empty())) {
    return true;
  }

  // Only kings and bishops remain.

  constexpr BitBoard kLightSquares(0x55AA55AA55AA55AAULL);
  constexpr BitBoard kDarkSquares(0xAA55AA55AA55AA55ULL);

  const bool light_bishop = bishops_.intersects(kLightSquares);
  const bool dark_bishop = bishops_.intersects(kDarkSquares);
  return light_bishop && dark_bishop;
}

std::string ChessBoard::DebugString() const {
  std::string result;
  for (int i = 7; i >= 0; --i) {
    for (int j = 0; j < 8; ++j) {
      if (!our_pieces_.get(i, j) && !their_pieces_.get(i, j)) {
        if (i == 2 && pawns_.get(0, j))
          result += '*';
        else if (i == 5 && pawns_.get(7, j))
          result += '*';
        else
          result += '.';
        continue;
      }
      if (our_king_ == i * 8 + j) {
        result += 'K';
        continue;
      }
      if (their_king_ == i * 8 + j) {
        result += 'k';
        continue;
      }
      char c = '?';
      if ((pawns_ & kPawnMask).get(i, j)) {
        c = 'p';
      } else if (bishops_.get(i, j)) {
        if (rooks_.get(i, j))
          c = 'q';
        else
          c = 'b';
      } else if (rooks_.get(i, j)) {
        c = 'r';
      } else {
        c = 'n';
      }
      if (our_pieces_.get(i, j)) c = std::toupper(c);
      result += c;
    }
    if (i == 0) {
      result += " " + castlings_.DebugString();
      result += flipped_ ? " (from black's eyes)" : " (from white's eyes)";
      result += " Hash: " + std::to_string(Hash());
    }
    result += '\n';
  }
  return result;
}

namespace {
const uint64_t Zorbist[781] = {
    0x9D39247E33776D41ULL, 0x2AF7398005AAA5C7ULL, 0x44DB015024623547ULL,
    0x9C15F73E62A76AE2ULL, 0x75834465489C0C89ULL, 0x3290AC3A203001BFULL,
    0x0FBBAD1F61042279ULL, 0xE83A908FF2FB60CAULL, 0x0D7E765D58755C10ULL,
    0x1A083822CEAFE02DULL, 0x9605D5F0E25EC3B0ULL, 0xD021FF5CD13A2ED5ULL,
    0x40BDF15D4A672E32ULL, 0x011355146FD56395ULL, 0x5DB4832046F3D9E5ULL,
    0x239F8B2D7FF719CCULL, 0x05D1A1AE85B49AA1ULL, 0x679F848F6E8FC971ULL,
    0x7449BBFF801FED0BULL, 0x7D11CDB1C3B7ADF0ULL, 0x82C7709E781EB7CCULL,
    0xF3218F1C9510786CULL, 0x331478F3AF51BBE6ULL, 0x4BB38DE5E7219443ULL,
    0xAA649C6EBCFD50FCULL, 0x8DBD98A352AFD40BULL, 0x87D2074B81D79217ULL,
    0x19F3C751D3E92AE1ULL, 0xB4AB30F062B19ABFULL, 0x7B0500AC42047AC4ULL,
    0xC9452CA81A09D85DULL, 0x24AA6C514DA27500ULL, 0x4C9F34427501B447ULL,
    0x14A68FD73C910841ULL, 0xA71B9B83461CBD93ULL, 0x03488B95B0F1850FULL,
    0x637B2B34FF93C040ULL, 0x09D1BC9A3DD90A94ULL, 0x3575668334A1DD3BULL,
    0x735E2B97A4C45A23ULL, 0x18727070F1BD400BULL, 0x1FCBACD259BF02E7ULL,
    0xD310A7C2CE9B6555ULL, 0xBF983FE0FE5D8244ULL, 0x9F74D14F7454A824ULL,
    0x51EBDC4AB9BA3035ULL, 0x5C82C505DB9AB0FAULL, 0xFCF7FE8A3430B241ULL,
    0x3253A729B9BA3DDEULL, 0x8C74C368081B3075ULL, 0xB9BC6C87167C33E7ULL,
    0x7EF48F2B83024E20ULL, 0x11D505D4C351BD7FULL, 0x6568FCA92C76A243ULL,
    0x4DE0B0F40F32A7B8ULL, 0x96D693460CC37E5DULL, 0x42E240CB63689F2FULL,
    0x6D2BDCDAE2919661ULL, 0x42880B0236E4D951ULL, 0x5F0F4A5898171BB6ULL,
    0x39F890F579F92F88ULL, 0x93C5B5F47356388BULL, 0x63DC359D8D231B78ULL,
    0xEC16CA8AEA98AD76ULL, 0x5355F900C2A82DC7ULL, 0x07FB9F855A997142ULL,
    0x5093417AA8A7ED5EULL, 0x7BCBC38DA25A7F3CULL, 0x19FC8A768CF4B6D4ULL,
    0x637A7780DECFC0D9ULL, 0x8249A47AEE0E41F7ULL, 0x79AD695501E7D1E8ULL,
    0x14ACBAF4777D5776ULL, 0xF145B6BECCDEA195ULL, 0xDABF2AC8201752FCULL,
    0x24C3C94DF9C8D3F6ULL, 0xBB6E2924F03912EAULL, 0x0CE26C0B95C980D9ULL,
    0xA49CD132BFBF7CC4ULL, 0xE99D662AF4243939ULL, 0x27E6AD7891165C3FULL,
    0x8535F040B9744FF1ULL, 0x54B3F4FA5F40D873ULL, 0x72B12C32127FED2BULL,
    0xEE954D3C7B411F47ULL, 0x9A85AC909A24EAA1ULL, 0x70AC4CD9F04F21F5ULL,
    0xF9B89D3E99A075C2ULL, 0x87B3E2B2B5C907B1ULL, 0xA366E5B8C54F48B8ULL,
    0xAE4A9346CC3F7CF2ULL, 0x1920C04D47267BBDULL, 0x87BF02C6B49E2AE9ULL,
    0x092237AC237F3859ULL, 0xFF07F64EF8ED14D0ULL, 0x8DE8DCA9F03CC54EULL,
    0x9C1633264DB49C89ULL, 0xB3F22C3D0B0B38EDULL, 0x390E5FB44D01144BULL,
    0x5BFEA5B4712768E9ULL, 0x1E1032911FA78984ULL, 0x9A74ACB964E78CB3ULL,
    0x4F80F7A035DAFB04ULL, 0x6304D09A0B3738C4ULL, 0x2171E64683023A08ULL,
    0x5B9B63EB9CEFF80CULL, 0x506AACF489889342ULL, 0x1881AFC9A3A701D6ULL,
    0x6503080440750644ULL, 0xDFD395339CDBF4A7ULL, 0xEF927DBCF00C20F2ULL,
    0x7B32F7D1E03680ECULL, 0xB9FD7620E7316243ULL, 0x05A7E8A57DB91B77ULL,
    0xB5889C6E15630A75ULL, 0x4A750A09CE9573F7ULL, 0xCF464CEC899A2F8AULL,
    0xF538639CE705B824ULL, 0x3C79A0FF5580EF7FULL, 0xEDE6C87F8477609DULL,
    0x799E81F05BC93F31ULL, 0x86536B8CF3428A8CULL, 0x97D7374C60087B73ULL,
    0xA246637CFF328532ULL, 0x043FCAE60CC0EBA0ULL, 0x920E449535DD359EULL,
    0x70EB093B15B290CCULL, 0x73A1921916591CBDULL, 0x56436C9FE1A1AA8DULL,
    0xEFAC4B70633B8F81ULL, 0xBB215798D45DF7AFULL, 0x45F20042F24F1768ULL,
    0x930F80F4E8EB7462ULL, 0xFF6712FFCFD75EA1ULL, 0xAE623FD67468AA70ULL,
    0xDD2C5BC84BC8D8FCULL, 0x7EED120D54CF2DD9ULL, 0x22FE545401165F1CULL,
    0xC91800E98FB99929ULL, 0x808BD68E6AC10365ULL, 0xDEC468145B7605F6ULL,
    0x1BEDE3A3AEF53302ULL, 0x43539603D6C55602ULL, 0xAA969B5C691CCB7AULL,
    0xA87832D392EFEE56ULL, 0x65942C7B3C7E11AEULL, 0xDED2D633CAD004F6ULL,
    0x21F08570F420E565ULL, 0xB415938D7DA94E3CULL, 0x91B859E59ECB6350ULL,
    0x10CFF333E0ED804AULL, 0x28AED140BE0BB7DDULL, 0xC5CC1D89724FA456ULL,
    0x5648F680F11A2741ULL, 0x2D255069F0B7DAB3ULL, 0x9BC5A38EF729ABD4ULL,
    0xEF2F054308F6A2BCULL, 0xAF2042F5CC5C2858ULL, 0x480412BAB7F5BE2AULL,
    0xAEF3AF4A563DFE43ULL, 0x19AFE59AE451497FULL, 0x52593803DFF1E840ULL,
    0xF4F076E65F2CE6F0ULL, 0x11379625747D5AF3ULL, 0xBCE5D2248682C115ULL,
    0x9DA4243DE836994FULL, 0x066F70B33FE09017ULL, 0x4DC4DE189B671A1CULL,
    0x51039AB7712457C3ULL, 0xC07A3F80C31FB4B4ULL, 0xB46EE9C5E64A6E7CULL,
    0xB3819A42ABE61C87ULL, 0x21A007933A522A20ULL, 0x2DF16F761598AA4FULL,
    0x763C4A1371B368FDULL, 0xF793C46702E086A0ULL, 0xD7288E012AEB8D31ULL,
    0xDE336A2A4BC1C44BULL, 0x0BF692B38D079F23ULL, 0x2C604A7A177326B3ULL,
    0x4850E73E03EB6064ULL, 0xCFC447F1E53C8E1BULL, 0xB05CA3F564268D99ULL,
    0x9AE182C8BC9474E8ULL, 0xA4FC4BD4FC5558CAULL, 0xE755178D58FC4E76ULL,
    0x69B97DB1A4C03DFEULL, 0xF9B5B7C4ACC67C96ULL, 0xFC6A82D64B8655FBULL,
    0x9C684CB6C4D24417ULL, 0x8EC97D2917456ED0ULL, 0x6703DF9D2924E97EULL,
    0xC547F57E42A7444EULL, 0x78E37644E7CAD29EULL, 0xFE9A44E9362F05FAULL,
    0x08BD35CC38336615ULL, 0x9315E5EB3A129ACEULL, 0x94061B871E04DF75ULL,
    0xDF1D9F9D784BA010ULL, 0x3BBA57B68871B59DULL, 0xD2B7ADEEDED1F73FULL,
    0xF7A255D83BC373F8ULL, 0xD7F4F2448C0CEB81ULL, 0xD95BE88CD210FFA7ULL,
    0x336F52F8FF4728E7ULL, 0xA74049DAC312AC71ULL, 0xA2F61BB6E437FDB5ULL,
    0x4F2A5CB07F6A35B3ULL, 0x87D380BDA5BF7859ULL, 0x16B9F7E06C453A21ULL,
    0x7BA2484C8A0FD54EULL, 0xF3A678CAD9A2E38CULL, 0x39B0BF7DDE437BA2ULL,
    0xFCAF55C1BF8A4424ULL, 0x18FCF680573FA594ULL, 0x4C0563B89F495AC3ULL,
    0x40E087931A00930DULL, 0x8CFFA9412EB642C1ULL, 0x68CA39053261169FULL,
    0x7A1EE967D27579E2ULL, 0x9D1D60E5076F5B6FULL, 0x3810E399B6F65BA2ULL,
    0x32095B6D4AB5F9B1ULL, 0x35CAB62109DD038AULL, 0xA90B24499FCFAFB1ULL,
    0x77A225A07CC2C6BDULL, 0x513E5E634C70E331ULL, 0x4361C0CA3F692F12ULL,
    0xD941ACA44B20A45BULL, 0x528F7C8602C5807BULL, 0x52AB92BEB9613989ULL,
    0x9D1DFA2EFC557F73ULL, 0x722FF175F572C348ULL, 0x1D1260A51107FE97ULL,
    0x7A249A57EC0C9BA2ULL, 0x04208FE9E8F7F2D6ULL, 0x5A110C6058B920A0ULL,
    0x0CD9A497658A5698ULL, 0x56FD23C8F9715A4CULL, 0x284C847B9D887AAEULL,
    0x04FEABFBBDB619CBULL, 0x742E1E651C60BA83ULL, 0x9A9632E65904AD3CULL,
    0x881B82A13B51B9E2ULL, 0x506E6744CD974924ULL, 0xB0183DB56FFC6A79ULL,
    0x0ED9B915C66ED37EULL, 0x5E11E86D5873D484ULL, 0xF678647E3519AC6EULL,
    0x1B85D488D0F20CC5ULL, 0xDAB9FE6525D89021ULL, 0x0D151D86ADB73615ULL,
    0xA865A54EDCC0F019ULL, 0x93C42566AEF98FFBULL, 0x99E7AFEABE000731ULL,
    0x48CBFF086DDF285AULL, 0x7F9B6AF1EBF78BAFULL, 0x58627E1A149BBA21ULL,
    0x2CD16E2ABD791E33ULL, 0xD363EFF5F0977996ULL, 0x0CE2A38C344A6EEDULL,
    0x1A804AADB9CFA741ULL, 0x907F30421D78C5DEULL, 0x501F65EDB3034D07ULL,
    0x37624AE5A48FA6E9ULL, 0x957BAF61700CFF4EULL, 0x3A6C27934E31188AULL,
    0xD49503536ABCA345ULL, 0x088E049589C432E0ULL, 0xF943AEE7FEBF21B8ULL,
    0x6C3B8E3E336139D3ULL, 0x364F6FFA464EE52EULL, 0xD60F6DCEDC314222ULL,
    0x56963B0DCA418FC0ULL, 0x16F50EDF91E513AFULL, 0xEF1955914B609F93ULL,
    0x565601C0364E3228ULL, 0xECB53939887E8175ULL, 0xBAC7A9A18531294BULL,
    0xB344C470397BBA52ULL, 0x65D34954DAF3CEBDULL, 0xB4B81B3FA97511E2ULL,
    0xB422061193D6F6A7ULL, 0x071582401C38434DULL, 0x7A13F18BBEDC4FF5ULL,
    0xBC4097B116C524D2ULL, 0x59B97885E2F2EA28ULL, 0x99170A5DC3115544ULL,
    0x6F423357E7C6A9F9ULL, 0x325928EE6E6F8794ULL, 0xD0E4366228B03343ULL,
    0x565C31F7DE89EA27ULL, 0x30F5611484119414ULL, 0xD873DB391292ED4FULL,
    0x7BD94E1D8E17DEBCULL, 0xC7D9F16864A76E94ULL, 0x947AE053EE56E63CULL,
    0xC8C93882F9475F5FULL, 0x3A9BF55BA91F81CAULL, 0xD9A11FBB3D9808E4ULL,
    0x0FD22063EDC29FCAULL, 0xB3F256D8ACA0B0B9ULL, 0xB03031A8B4516E84ULL,
    0x35DD37D5871448AFULL, 0xE9F6082B05542E4EULL, 0xEBFAFA33D7254B59ULL,
    0x9255ABB50D532280ULL, 0xB9AB4CE57F2D34F3ULL, 0x693501D628297551ULL,
    0xC62C58F97DD949BFULL, 0xCD454F8F19C5126AULL, 0xBBE83F4ECC2BDECBULL,
    0xDC842B7E2819E230ULL, 0xBA89142E007503B8ULL, 0xA3BC941D0A5061CBULL,
    0xE9F6760E32CD8021ULL, 0x09C7E552BC76492FULL, 0x852F54934DA55CC9ULL,
    0x8107FCCF064FCF56ULL, 0x098954D51FFF6580ULL, 0x23B70EDB1955C4BFULL,
    0xC330DE426430F69DULL, 0x4715ED43E8A45C0AULL, 0xA8D7E4DAB780A08DULL,
    0x0572B974F03CE0BBULL, 0xB57D2E985E1419C7ULL, 0xE8D9ECBE2CF3D73FULL,
    0x2FE4B17170E59750ULL, 0x11317BA87905E790ULL, 0x7FBF21EC8A1F45ECULL,
    0x1725CABFCB045B00ULL, 0x964E915CD5E2B207ULL, 0x3E2B8BCBF016D66DULL,
    0xBE7444E39328A0ACULL, 0xF85B2B4FBCDE44B7ULL, 0x49353FEA39BA63B1ULL,
    0x1DD01AAFCD53486AULL, 0x1FCA8A92FD719F85ULL, 0xFC7C95D827357AFAULL,
    0x18A6A990C8B35EBDULL, 0xCCCB7005C6B9C28DULL, 0x3BDBB92C43B17F26ULL,
    0xAA70B5B4F89695A2ULL, 0xE94C39A54A98307FULL, 0xB7A0B174CFF6F36EULL,
    0xD4DBA84729AF48ADULL, 0x2E18BC1AD9704A68ULL, 0x2DE0966DAF2F8B1CULL,
    0xB9C11D5B1E43A07EULL, 0x64972D68DEE33360ULL, 0x94628D38D0C20584ULL,
    0xDBC0D2B6AB90A559ULL, 0xD2733C4335C6A72FULL, 0x7E75D99D94A70F4DULL,
    0x6CED1983376FA72BULL, 0x97FCAACBF030BC24ULL, 0x7B77497B32503B12ULL,
    0x8547EDDFB81CCB94ULL, 0x79999CDFF70902CBULL, 0xCFFE1939438E9B24ULL,
    0x829626E3892D95D7ULL, 0x92FAE24291F2B3F1ULL, 0x63E22C147B9C3403ULL,
    0xC678B6D860284A1CULL, 0x5873888850659AE7ULL, 0x0981DCD296A8736DULL,
    0x9F65789A6509A440ULL, 0x9FF38FED72E9052FULL, 0xE479EE5B9930578CULL,
    0xE7F28ECD2D49EECDULL, 0x56C074A581EA17FEULL, 0x5544F7D774B14AEFULL,
    0x7B3F0195FC6F290FULL, 0x12153635B2C0CF57ULL, 0x7F5126DBBA5E0CA7ULL,
    0x7A76956C3EAFB413ULL, 0x3D5774A11D31AB39ULL, 0x8A1B083821F40CB4ULL,
    0x7B4A38E32537DF62ULL, 0x950113646D1D6E03ULL, 0x4DA8979A0041E8A9ULL,
    0x3BC36E078F7515D7ULL, 0x5D0A12F27AD310D1ULL, 0x7F9D1A2E1EBE1327ULL,
    0xDA3A361B1C5157B1ULL, 0xDCDD7D20903D0C25ULL, 0x36833336D068F707ULL,
    0xCE68341F79893389ULL, 0xAB9090168DD05F34ULL, 0x43954B3252DC25E5ULL,
    0xB438C2B67F98E5E9ULL, 0x10DCD78E3851A492ULL, 0xDBC27AB5447822BFULL,
    0x9B3CDB65F82CA382ULL, 0xB67B7896167B4C84ULL, 0xBFCED1B0048EAC50ULL,
    0xA9119B60369FFEBDULL, 0x1FFF7AC80904BF45ULL, 0xAC12FB171817EEE7ULL,
    0xAF08DA9177DDA93DULL, 0x1B0CAB936E65C744ULL, 0xB559EB1D04E5E932ULL,
    0xC37B45B3F8D6F2BAULL, 0xC3A9DC228CAAC9E9ULL, 0xF3B8B6675A6507FFULL,
    0x9FC477DE4ED681DAULL, 0x67378D8ECCEF96CBULL, 0x6DD856D94D259236ULL,
    0xA319CE15B0B4DB31ULL, 0x073973751F12DD5EULL, 0x8A8E849EB32781A5ULL,
    0xE1925C71285279F5ULL, 0x74C04BF1790C0EFEULL, 0x4DDA48153C94938AULL,
    0x9D266D6A1CC0542CULL, 0x7440FB816508C4FEULL, 0x13328503DF48229FULL,
    0xD6BF7BAEE43CAC40ULL, 0x4838D65F6EF6748FULL, 0x1E152328F3318DEAULL,
    0x8F8419A348F296BFULL, 0x72C8834A5957B511ULL, 0xD7A023A73260B45CULL,
    0x94EBC8ABCFB56DAEULL, 0x9FC10D0F989993E0ULL, 0xDE68A2355B93CAE6ULL,
    0xA44CFE79AE538BBEULL, 0x9D1D84FCCE371425ULL, 0x51D2B1AB2DDFB636ULL,
    0x2FD7E4B9E72CD38CULL, 0x65CA5B96B7552210ULL, 0xDD69A0D8AB3B546DULL,
    0x604D51B25FBF70E2ULL, 0x73AA8A564FB7AC9EULL, 0x1A8C1E992B941148ULL,
    0xAAC40A2703D9BEA0ULL, 0x764DBEAE7FA4F3A6ULL, 0x1E99B96E70A9BE8BULL,
    0x2C5E9DEB57EF4743ULL, 0x3A938FEE32D29981ULL, 0x26E6DB8FFDF5ADFEULL,
    0x469356C504EC9F9DULL, 0xC8763C5B08D1908CULL, 0x3F6C6AF859D80055ULL,
    0x7F7CC39420A3A545ULL, 0x9BFB227EBDF4C5CEULL, 0x89039D79D6FC5C5CULL,
    0x8FE88B57305E2AB6ULL, 0xA09E8C8C35AB96DEULL, 0xFA7E393983325753ULL,
    0xD6B6D0ECC617C699ULL, 0xDFEA21EA9E7557E3ULL, 0xB67C1FA481680AF8ULL,
    0xCA1E3785A9E724E5ULL, 0x1CFC8BED0D681639ULL, 0xD18D8549D140CAEAULL,
    0x4ED0FE7E9DC91335ULL, 0xE4DBF0634473F5D2ULL, 0x1761F93A44D5AEFEULL,
    0x53898E4C3910DA55ULL, 0x734DE8181F6EC39AULL, 0x2680B122BAA28D97ULL,
    0x298AF231C85BAFABULL, 0x7983EED3740847D5ULL, 0x66C1A2A1A60CD889ULL,
    0x9E17E49642A3E4C1ULL, 0xEDB454E7BADC0805ULL, 0x50B704CAB602C329ULL,
    0x4CC317FB9CDDD023ULL, 0x66B4835D9EAFEA22ULL, 0x219B97E26FFC81BDULL,
    0x261E4E4C0A333A9DULL, 0x1FE2CCA76517DB90ULL, 0xD7504DFA8816EDBBULL,
    0xB9571FA04DC089C8ULL, 0x1DDC0325259B27DEULL, 0xCF3F4688801EB9AAULL,
    0xF4F5D05C10CAB243ULL, 0x38B6525C21A42B0EULL, 0x36F60E2BA4FA6800ULL,
    0xEB3593803173E0CEULL, 0x9C4CD6257C5A3603ULL, 0xAF0C317D32ADAA8AULL,
    0x258E5A80C7204C4BULL, 0x8B889D624D44885DULL, 0xF4D14597E660F855ULL,
    0xD4347F66EC8941C3ULL, 0xE699ED85B0DFB40DULL, 0x2472F6207C2D0484ULL,
    0xC2A1E7B5B459AEB5ULL, 0xAB4F6451CC1D45ECULL, 0x63767572AE3D6174ULL,
    0xA59E0BD101731A28ULL, 0x116D0016CB948F09ULL, 0x2CF9C8CA052F6E9FULL,
    0x0B090A7560A968E3ULL, 0xABEEDDB2DDE06FF1ULL, 0x58EFC10B06A2068DULL,
    0xC6E57A78FBD986E0ULL, 0x2EAB8CA63CE802D7ULL, 0x14A195640116F336ULL,
    0x7C0828DD624EC390ULL, 0xD74BBE77E6116AC7ULL, 0x804456AF10F5FB53ULL,
    0xEBE9EA2ADF4321C7ULL, 0x03219A39EE587A30ULL, 0x49787FEF17AF9924ULL,
    0xA1E9300CD8520548ULL, 0x5B45E522E4B1B4EFULL, 0xB49C3B3995091A36ULL,
    0xD4490AD526F14431ULL, 0x12A8F216AF9418C2ULL, 0x001F837CC7350524ULL,
    0x1877B51E57A764D5ULL, 0xA2853B80F17F58EEULL, 0x993E1DE72D36D310ULL,
    0xB3598080CE64A656ULL, 0x252F59CF0D9F04BBULL, 0xD23C8E176D113600ULL,
    0x1BDA0492E7E4586EULL, 0x21E0BD5026C619BFULL, 0x3B097ADAF088F94EULL,
    0x8D14DEDB30BE846EULL, 0xF95CFFA23AF5F6F4ULL, 0x3871700761B3F743ULL,
    0xCA672B91E9E4FA16ULL, 0x64C8E531BFF53B55ULL, 0x241260ED4AD1E87DULL,
    0x106C09B972D2E822ULL, 0x7FBA195410E5CA30ULL, 0x7884D9BC6CB569D8ULL,
    0x0647DFEDCD894A29ULL, 0x63573FF03E224774ULL, 0x4FC8E9560F91B123ULL,
    0x1DB956E450275779ULL, 0xB8D91274B9E9D4FBULL, 0xA2EBEE47E2FBFCE1ULL,
    0xD9F1F30CCD97FB09ULL, 0xEFED53D75FD64E6BULL, 0x2E6D02C36017F67FULL,
    0xA9AA4D20DB084E9BULL, 0xB64BE8D8B25396C1ULL, 0x70CB6AF7C2D5BCF0ULL,
    0x98F076A4F7A2322EULL, 0xBF84470805E69B5FULL, 0x94C3251F06F90CF3ULL,
    0x3E003E616A6591E9ULL, 0xB925A6CD0421AFF3ULL, 0x61BDD1307C66E300ULL,
    0xBF8D5108E27E0D48ULL, 0x240AB57A8B888B20ULL, 0xFC87614BAF287E07ULL,
    0xEF02CDD06FFDB432ULL, 0xA1082C0466DF6C0AULL, 0x8215E577001332C8ULL,
    0xD39BB9C3A48DB6CFULL, 0x2738259634305C14ULL, 0x61CF4F94C97DF93DULL,
    0x1B6BACA2AE4E125BULL, 0x758F450C88572E0BULL, 0x959F587D507A8359ULL,
    0xB063E962E045F54DULL, 0x60E8ED72C0DFF5D1ULL, 0x7B64978555326F9FULL,
    0xFD080D236DA814BAULL, 0x8C90FD9B083F4558ULL, 0x106F72FE81E2C590ULL,
    0x7976033A39F7D952ULL, 0xA4EC0132764CA04BULL, 0x733EA705FAE4FA77ULL,
    0xB4D8F77BC3E56167ULL, 0x9E21F4F903B33FD9ULL, 0x9D765E419FB69F6DULL,
    0xD30C088BA61EA5EFULL, 0x5D94337FBFAF7F5BULL, 0x1A4E4822EB4D7A59ULL,
    0x6FFE73E81B637FB3ULL, 0xDDF957BC36D8B9CAULL, 0x64D0E29EEA8838B3ULL,
    0x08DD9BDFD96B9F63ULL, 0x087E79E5A57D1D13ULL, 0xE328E230E3E2B3FBULL,
    0x1C2559E30F0946BEULL, 0x720BF5F26F4D2EAAULL, 0xB0774D261CC609DBULL,
    0x443F64EC5A371195ULL, 0x4112CF68649A260EULL, 0xD813F2FAB7F5C5CAULL,
    0x660D3257380841EEULL, 0x59AC2C7873F910A3ULL, 0xE846963877671A17ULL,
    0x93B633ABFA3469F8ULL, 0xC0C0F5A60EF4CDCFULL, 0xCAF21ECD4377B28CULL,
    0x57277707199B8175ULL, 0x506C11B9D90E8B1DULL, 0xD83CC2687A19255FULL,
    0x4A29C6465A314CD1ULL, 0xED2DF21216235097ULL, 0xB5635C95FF7296E2ULL,
    0x22AF003AB672E811ULL, 0x52E762596BF68235ULL, 0x9AEBA33AC6ECC6B0ULL,
    0x944F6DE09134DFB6ULL, 0x6C47BEC883A7DE39ULL, 0x6AD047C430A12104ULL,
    0xA5B1CFDBA0AB4067ULL, 0x7C45D833AFF07862ULL, 0x5092EF950A16DA0BULL,
    0x9338E69C052B8E7BULL, 0x455A4B4CFE30E3F5ULL, 0x6B02E63195AD0CF8ULL,
    0x6B17B224BAD6BF27ULL, 0xD1E0CCD25BB9C169ULL, 0xDE0C89A556B9AE70ULL,
    0x50065E535A213CF6ULL, 0x9C1169FA2777B874ULL, 0x78EDEFD694AF1EEDULL,
    0x6DC93D9526A50E68ULL, 0xEE97F453F06791EDULL, 0x32AB0EDB696703D3ULL,
    0x3A6853C7E70757A7ULL, 0x31865CED6120F37DULL, 0x67FEF95D92607890ULL,
    0x1F2B1D1F15F6DC9CULL, 0xB69E38A8965C6B65ULL, 0xAA9119FF184CCCF4ULL,
    0xF43C732873F24C13ULL, 0xFB4A3D794A9A80D2ULL, 0x3550C2321FD6109CULL,
    0x371F77E76BB8417EULL, 0x6BFA9AAE5EC05779ULL, 0xCD04F3FF001A4778ULL,
    0xE3273522064480CAULL, 0x9F91508BFFCFC14AULL, 0x049A7F41061A9E60ULL,
    0xFCB6BE43A9F2FE9BULL, 0x08DE8A1C7797DA9BULL, 0x8F9887E6078735A1ULL,
    0xB5B4071DBFC73A66ULL, 0x230E343DFBA08D33ULL, 0x43ED7F5A0FAE657DULL,
    0x3A88A0FBBCB05C63ULL, 0x21874B8B4D2DBC4FULL, 0x1BDEA12E35F6A8C9ULL,
    0x53C065C6C8E63528ULL, 0xE34A1D250E7A8D6BULL, 0xD6B04D3B7651DD7EULL,
    0x5E90277E7CB39E2DULL, 0x2C046F22062DC67DULL, 0xB10BB459132D0A26ULL,
    0x3FA9DDFB67E2F199ULL, 0x0E09B88E1914F7AFULL, 0x10E8B35AF3EEAB37ULL,
    0x9EEDECA8E272B933ULL, 0xD4C718BC4AE8AE5FULL, 0x81536D601170FC20ULL,
    0x91B534F885818A06ULL, 0xEC8177F83F900978ULL, 0x190E714FADA5156EULL,
    0xB592BF39B0364963ULL, 0x89C350C893AE7DC1ULL, 0xAC042E70F8B383F2ULL,
    0xB49B52E587A1EE60ULL, 0xFB152FE3FF26DA89ULL, 0x3E666E6F69AE2C15ULL,
    0x3B544EBE544C19F9ULL, 0xE805A1E290CF2456ULL, 0x24B33C9D7ED25117ULL,
    0xE74733427B72F0C1ULL, 0x0A804D18B7097475ULL, 0x57E3306D881EDB4FULL,
    0x4AE7D6A36EB5DBCBULL, 0x2D8D5432157064C8ULL, 0xD1E649DE1E7F268BULL,
    0x8A328A1CEDFE552CULL, 0x07A3AEC79624C7DAULL, 0x84547DDC3E203C94ULL,
    0x990A98FD5071D263ULL, 0x1A4FF12616EEFC89ULL, 0xF6F7FD1431714200ULL,
    0x30C05B1BA332F41CULL, 0x8D2636B81555A786ULL, 0x46C9FEB55D120902ULL,
    0xCCEC0A73B49C9921ULL, 0x4E9D2827355FC492ULL, 0x19EBB029435DCB0FULL,
    0x4659D2B743848A2CULL, 0x963EF2C96B33BE31ULL, 0x74F85198B05A2E7DULL,
    0x5A0F544DD2B1FB18ULL, 0x03727073C2E134B1ULL, 0xC7F6AA2DE59AEA61ULL,
    0x352787BAA0D7C22FULL, 0x9853EAB63B5E0B35ULL, 0xABBDCDD7ED5C0860ULL,
    0xCF05DAF5AC8D77B0ULL, 0x49CAD48CEBF4A71EULL, 0x7A4C10EC2158C4A6ULL,
    0xD9E92AA246BF719EULL, 0x13AE978D09FE5557ULL, 0x730499AF921549FFULL,
    0x4E4B705B92903BA4ULL, 0xFF577222C14F0A3AULL, 0x55B6344CF97AAFAEULL,
    0xB862225B055B6960ULL, 0xCAC09AFBDDD2CDB4ULL, 0xDAF8E9829FE96B5FULL,
    0xB5FDFC5D3132C498ULL, 0x310CB380DB6F7503ULL, 0xE87FBB46217A360EULL,
    0x2102AE466EBB1148ULL, 0xF8549E1A3AA5E00DULL, 0x07A69AFDCC42261AULL,
    0xC4C118BFE78FEAAEULL, 0xF9F4892ED96BD438ULL, 0x1AF3DBE25D8F45DAULL,
    0xF5B4B0B0D2DEEEB4ULL, 0x962ACEEFA82E1C84ULL, 0x046E3ECAAF453CE9ULL,
    0xF05D129681949A4CULL, 0x964781CE734B3C84ULL, 0x9C2ED44081CE5FBDULL,
    0x522E23F3925E319EULL, 0x177E00F9FC32F791ULL, 0x2BC60A63A6F3B3F2ULL,
    0x222BBFAE61725606ULL, 0x486289DDCC3D6780ULL, 0x7DC7785B8EFDFC80ULL,
    0x8AF38731C02BA980ULL, 0x1FAB64EA29A2DDF7ULL, 0xE4D9429322CD065AULL,
    0x9DA058C67844F20CULL, 0x24C0E332B70019B0ULL, 0x233003B5A6CFE6ADULL,
    0xD586BD01C5C217F6ULL, 0x5E5637885F29BC2BULL, 0x7EBA726D8C94094BULL,
    0x0A56A5F0BFE39272ULL, 0xD79476A84EE20D06ULL, 0x9E4C1269BAA4BF37ULL,
    0x17EFEE45B0DEE640ULL, 0x1D95B0A5FCF90BC6ULL, 0x93CBE0B699C2585DULL,
    0x65FA4F227A2B6D79ULL, 0xD5F9E858292504D5ULL, 0xC2B5A03F71471A6FULL,
    0x59300222B4561E00ULL, 0xCE2F8642CA0712DCULL, 0x7CA9723FBB2E8988ULL,
    0x2785338347F2BA08ULL, 0xC61BB3A141E50E8CULL, 0x150F361DAB9DEC26ULL,
    0x9F6A419D382595F4ULL, 0x64A53DC924FE7AC9ULL, 0x142DE49FFF7A7C3DULL,
    0x0C335248857FA9E7ULL, 0x0A9C32D5EAE45305ULL, 0xE6C42178C4BBB92EULL,
    0x71F1CE2490D20B07ULL, 0xF1BCC3D275AFE51AULL, 0xE728E8C83C334074ULL,
    0x96FBF83A12884624ULL, 0x81A1549FD6573DA5ULL, 0x5FA7867CAF35E149ULL,
    0x56986E2EF3ED091BULL, 0x917F1DD5F8886C61ULL, 0xD20D8C88C8FFE65FULL,
    0x31D71DCE64B2C310ULL, 0xF165B587DF898190ULL, 0xA57E6339DD2CF3A0ULL,
    0x1EF6E6DBB1961EC9ULL, 0x70CC73D90BC26E24ULL, 0xE21A6B35DF0C3AD7ULL,
    0x003A93D8B2806962ULL, 0x1C99DED33CB890A1ULL, 0xCF3145DE0ADD4289ULL,
    0xD0E4427A5514FB72ULL, 0x77C621CC9FB3A483ULL, 0x67A34DAC4356550BULL,
    0xF8D626AAAF278509ULL,
};

}  // namespace

uint64_t ChessBoard::HashPiece(BitBoard in, int offset) const {
  uint64_t hash = 0;
  if (!flipped()) {
    for (auto square : theirs() & in) {
      hash ^= Zorbist[offset + square.as_int()];
    }
    for (auto square : ours() & in) {
      hash ^= Zorbist[offset + square.as_int() + 64];
    }
  } else {
    for (auto square : ours() & in) {
      hash ^= Zorbist[offset + 8 * (7 - square.row()) + square.col()];
    }
    for (auto square : theirs() & in) {
      hash ^= Zorbist[offset + 8 * (7 - square.row()) + square.col() + 64];
    }
  }
  return hash;
}

uint64_t ChessBoard::Hash() const {
  uint64_t hash = HashPiece(pawns(), 0);
  hash ^= HashPiece(knights(), 2 * 64);
  hash ^= HashPiece(bishops(), 4 * 64);
  hash ^= HashPiece(rooks(), 6 * 64);
  hash ^= HashPiece(queens(), 8 * 64);
  hash ^= HashPiece(kings(), 10 * 64);
  if (castlings().we_can_00()) hash ^= Zorbist[768 + 2 * flipped()];
  if (castlings().we_can_000()) hash ^= Zorbist[769 + 2 * flipped()];
  if (castlings().they_can_00()) hash ^= Zorbist[770 - 2 * flipped()];
  if (castlings().they_can_000()) hash ^= Zorbist[771 - 2 * flipped()];

  for (auto ep : en_passant()) {
    if (ep.row() != 7) break;
    for (auto p : pawns() & ours()) {
      if (p.row() == 4 && abs(p.col() - ep.col()) == 1) {
        hash ^= Zorbist[772 + ep.col()];
        break;
      }
    }
  }

  if (!flipped()) hash ^= Zorbist[780];
  return hash;
}

}  // namespace lczero
