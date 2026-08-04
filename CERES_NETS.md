# Ceres nets in lc0

How this branch loads and evaluates Ceres networks, what the TPG input format
is, which backend settings are mandatory, and what has been verified against
upstream Ceres.

Everything in the "Verified" sections was measured on this branch against
Ceres 2.50 running the same nets on the same GPU (RTX 5080, ONNX Runtime
1.24.2, TensorRT 10.15, CUDA 13). Statements not marked verified are read from
source.

---

## 1. What a Ceres net is

A Ceres net is a **raw `.onnx` file**, not an lc0 `.pb.gz` weights file. It uses
a completely different input encoding — TPG (Transformer Position Graph), a
`[batch, 64, 137]` per-square byte record — instead of lc0's
`[batch, 112, 8, 8]` planes. Policy output is still the standard lc0
1858-entry move table.

Consequences:

* Only the **ONNX backends** can run them (`onnx-cpu`, `onnx-cuda`, `onnx-trt`,
  `onnx-dml`). Every other backend receives `AddInputBytes()` and throws
  `"AddInputBytes not supported by this backend."` from `network.h`, at first
  evaluation rather than at load.
* lc0 recognises them purely by file extension (`IsOnnxFile()` in
  `loader.cc`: filename ends in `.onnx`).

```bash
lc0 --backend=onnx-cuda --weights=/path/to/C1-256-10.onnx
```

---

## 2. Net inventory

Measured directly from the ONNX graphs. All outputs are `FLOAT16` in every net.

| Net | Input tensor | Input dtype | Batch dim name | `action` head |
|---|---|---|---|---|
| C1-256-10 | `squares` | FLOAT16 | `batch_size` | `[b,1]` vestigial |
| C1-512-15 | `squares` | FLOAT16 | `batch_size` | `[b,1]` vestigial |
| C1-640-34 | `squares` | FLOAT16 | `batch_size` | `[b,1]` vestigial |
| C1-BIG (`C1-ULTRA-I8.zip`) | `squares_byte` | UINT8 | `batch_size` | `[b,1]` vestigial |
| C2-384-12-I8 | `squares_byte` | UINT8 | `batch` | `[b,1858,3]` real |
| C3-768-30-pre8-I8 | `squares_byte` | UINT8 | `batch` (outputs unnamed) | `[b,1858,3]` real |

Two things bite here, and both are handled in code:

* **Input dtype varies.** `squares` is FLOAT16; `squares_byte` is UINT8. The
  name is only a convention, so the element type is read from the session.
* **The batch dimension has two different names.** C1 nets call it
  `batch_size`, C2/C3 call it `batch`. A free-dimension override that sets only
  one name silently does nothing for the other family.

Heads present in **all** nets: `policy [b,1858]`, `value [b,3]`, `mlh [b,1]`,
`unc [b,1]`, `value2 [b,3]`, `q_deviation_lower`, `q_deviation_upper`,
`uncertainty_policy`. C2/C3 additionally expose `state_out`, `piece_move`,
`piece_capture`; C3 adds `punim_self`, `punim_opponent`; C1 has `prior_state`
and `action_uncertainty`.

lc0 binds only `policy`, `value`, `mlh`, plus `value2` and (when real)
`action`. The rest are ignored.

**Architecture note.** C1 nets carry LayerNorm *decomposed* into
`Pow`/`ReduceMean`/`Sqrt`/`Div` (31 of each in C1-256-10) with **zero `Cast`
nodes** — no in-graph FP32 protection at all. C2/C3 use the native
`RMSNormalization` op (85 in C2). This is why the two families react
differently to graph optimisation (§6).

---

## 3. TPG input format

`64` squares × `137` bytes. Constants live in `neural/network.h`
(`kCeresTPGSquares`, `kCeresTPGBytesPerSquare`, `kCeresTPGTotalBytes = 8768`).

### Square record layout

| Offset | Size | Field |
|---|---|---|
| 0–103 | 8 × 13 | Piece one-hot per history position |
| 104–111 | 8 | Repetition flag per history position |
| 112 | 1 | `CanOO` (our kingside) |
| 113 | 1 | `CanOOO` (our queenside) |
| 114 | 1 | `OpponentCanOO` |
| 115 | 1 | `OpponentCanOOO` |
| 116 | 1 | `Move50Count` |
| 117 | 1 | `PlySinceLastMove` |
| 118 | 1 | `IsEnPassant` |
| 119 | 1 | `QPositiveBlunders` |
| 120 | 1 | `QNegativeBlunders` |
| 121–128 | 8 | Rank one-hot |
| 129–136 | 8 | File one-hot |

The 13-byte piece one-hot is ordered: `empty`, our `P N B R Q K`, their
`P N B R Q K`.

### Encoding rules

* **ByteScaled**: every byte is `clamp(round(value * 100), 0, 255)`. The scale
  factor 100 is `ByteScaled.SCALING_FACTOR` in Ceres and
  `kCeresTPGByteDivisor` in lc0. For FLOAT16-input nets lc0 converts back with
  `byte / 100`.
* `Move50Count` = `min(rule50_ply, 100) / 50`.
* `PlySinceLastMove` = **0 at inference.** Ceres only populates it when
  `emitPlySinceLastMovePerSquare` is set, and
  `TPGRecord.EMIT_PLY_SINCE_LAST_MOVE_PER_SQUARE = false` with the ONNX
  evaluator leaving `LastMovePlies` unset. Writing 0 is correct.
* Blunder planes default to `0.03` (`DEFAULT_Q_BLUNDER`) → byte 3.
* Repetition is clamped to 1 (`min(reps, 1)`), matching Ceres, because LC0
  training data only ever encoded on/off.

### Square ordering (the subtle part)

Ceres writes the record for board square *i* into slot `63 - i` when Black is to
move — a full 180° rotation. lc0's board is *already* rank-flipped for Black,
so in lc0 coordinates the equivalent is a **file flip**, `sq ^ 7`:

```
Ceres slot = true_sq ^ 63        (180° rotation)
lc0 sq     = true_sq ^ 56        (rank flip, already applied)
=> slot    = lc0_sq ^ 7          (file flip)
```

The rank/file one-hot is a separate matter: Ceres uses `Square.Reversed`,
which is `FromFileAndRank(File, 7 - Rank)` — a **rank flip only**. That equals
lc0's already-flipped `sq`, so the one-hot is written from the unmirrored
index. The two look contradictory in the source and are both correct.

**Policy indexing needs no transform, for either colour.** Ceres maps legal
moves to the 1858 table in mover-relative coordinates, which is the frame lc0's
moves are already in. The `63 - i` reordering affects only the order of the
input sequence, and the transformer is invariant to that. An earlier
`FlipTransform` here made lc0 read the file-mirrored move's logit for every
Black move (d2d4's logit for e7e5); it has been removed.

---

## 4. Load path

`LoadWeightsFromFile()` → `IsOnnxFile()` → `LoadRawOnnxFile()`, which
synthesises a `WeightsFile` around the raw model:

* `input = INPUT_CERES_TPG`, `output = OUTPUT_WDL`, `network = NETWORK_ONNX`,
  `moves_left = MOVES_LEFT_V1`
* `data_type = FLOAT16` (correct: every net's outputs are FP16)
* input name **parsed out of the graph** (`pblczero::ModelProto` →
  `graph().input(0).name()`), with a `"squares_byte"` substring probe only as a
  fallback. This matters because the TensorRT provider builds its optimisation
  profile keys from this value *before any session exists*.
* head names hardcoded to `policy` / `value` / `mlh`

`OnnxNetwork`'s constructor then validates against the live session:

* input element type must be `UINT8` or `FLOAT16` → sets `ceres_tpg_float_`
* input shape must be `N x 64 x 137`
* `policy`, `value`, `mlh` must exist, else the error lists what the graph
  actually has
* `value2` present → dual value head
* `action` present → bound **only if** its per-position element count is
  `1858 * 3`. C1's vestigial `[b,1]` output shares the name and must be
  skipped.

---

## 5. Evaluation path

1. `wrapper.cc :: AddInput` — `EncodePositionForCeresTPG(history, 8, fill)`
   produces 8768 bytes; `transform = 0`.
2. `AddInputBytes` — rejects anything that isn't exactly 8768 bytes.
3. `StageCeresInput` — zeroes the batch, then either `memcpy`s the bytes
   (UINT8 nets) or writes `byte / 100` as FP16 (FLOAT16 nets).
4. Bind + `Run`, outputs read as FP16.
5. **Value**: softmax(`value` / `v1temp`) and softmax(`value2` / `v2temp`),
   then `W` and `L` blended linearly by `v2frac`, with `D` as the residual
   `1 - W - L`, then renormalised. This is formula-identical to Ceres's
   `CalcWLWithTemperature` plus its blend loop.
6. **Policy**: softmax over legal moves, then power-law flattening
   `prob^(1/PolicyTemperature)`. (`softmax_policy_temperature_` is stored as
   `1/PolicyTemperature`, so the exponent is applied as intended.) This differs
   from the standard lc0 path, which applies temperature in logit space.
7. **Action head** (C2/C3 only): `GetActionWDL()` returns per-move W/D/L.

### Options

| Option | Default | Meaning |
|---|---|---|
| `ceres_v2frac` | 0.4 | Weight of the second value head |
| `ceres_v1temp` | 0.55 | Softmax temperature for `value` |
| `ceres_v2temp` | 1.5 | Softmax temperature for `value2` |

Set via `--backend-opts=ceres_v2frac=0.4,ceres_v1temp=0.55,ceres_v2temp=1.5`.
These defaults match what Ceres prints as `OVERRIDDEN V2FRAC/V1TEMP/V2TEMP`.

---

## 6. Backend configuration — required, not optional

Four settings are load-bearing. Each was a real bug.

### Graph optimisation is clamped to `ORT_ENABLE_EXTENDED`

`optimize` defaults to 3 → `ORT_ENABLE_ALL`, which **aborts session init** on
every C1 net:

```
FAIL : Exception during initialization: graph_utils.cc:30 GetIndexFromName
itr != node_args.end() ... InsertedPrecisionFreeCast_/transformer_layer.8/ln1/
Constant_output_0 for node: /embedding_norm/Mul/SimplifiedLayerNormFusion/
```

`SimplifiedLayerNormFusion` removes a node arg and then looks it up. Upstream
Ceres never runs above `ORT_ENABLE_EXTENDED` either, deliberately
(`ONNXExecutor.cs`: *"Use of ORT_ENABLE_ALL might possibly exacerbate the
nondeterminism of engine generation and inconsistent request for FP16
precision"*). C2/C3 use native `RMSNormalization` and are unaffected.

> **Do not** "fix" this with
> `AddConfigEntry("optimization.disable_specified_optimizers", "SimplifiedLayerNormFusion")`.
> It works on CPU and silently breaks CUDA and TensorRT — the session runs but
> returns a **fixed evaluation for every position**. This was tried and
> reverted; the level clamp is the correct fix.

### TensorRT must have FP16 enabled

`trt_fp16_enable` was gated on `optimize >= 6`, so by default TensorRT built
FP16 graphs with FP16 **off**. That does not merely cost speed — it produces
badly wrong evaluations. Now forced on for Ceres FP16 nets, matching Ceres
(`trt_fp16_enable = precisionNumBits == 16`).

Measured on C1-256-10, 200 positions:

| TRT config | mean dev | max dev | same bestmove |
|---|---|---|---|
| FP16 off (old default) | 29.6‰ | 857‰ | 183/200 |
| FP16 on | **0.89‰** | **10‰** | 199/200 |

> **TRT engine cache caveat.** lc0's cache key is model hash + batch + optimize
> level and does **not** include provider flags. An engine built before this
> change is silently reused and stays wrong. Delete `build/trt_cache` once.

### Free-dimension override must set both names

C1 nets name the batch dim `batch_size`; setting only `batch` is a no-op for
them. Both are set — overriding a name the graph doesn't use is harmless.

### `DisableMemPattern`

Kept for Ceres nets, to avoid ORT memory-planner issues with varying internal
tensor shapes.

---

## 7. Verified parity with Ceres

Test set: **200 positions** with real move history from startpos (median 59
plies, up to 120), 100 white-to-move / 100 black-to-move, 63 with castling
rights, 23 in check, 3–32 pieces, plus curated en-passant, all
castling-combination, and promotion cases. Both engines driven over UCI with
`go nodes 1`, comparing reported WDL (per mille) and bestmove.

### Encoder — byte-identical

TPG dump hooks compiled into **both** engines, records diffed byte by byte:
**0 of 8768 bytes differ**, on bare FENs, short lines, and a 92-ply history.

### End-to-end

Both engines on **TensorRT**:

| Net | mean dev | max dev | identical bestmove |
|---|---|---|---|
| C1-256-10 | 0.89‰ | 10‰ | 198/200 |
| C1-512-15 | 0.42‰ | 3‰ | 200/200 |
| C1-640-34 | 0.81‰ | 7‰ | 198/200 |
| C2-384-12-I8 | 0.78‰ | 5‰ | 200/200 |
| C3-768-30-pre8-I8 | 0.70‰ | 7‰ | 199/200 |

Both engines on **CUDA**:

| Net | mean dev | max dev | identical bestmove |
|---|---|---|---|
| C1-256-10 | 0.62‰ | 3‰ | 200/200 |
| C1-512-15 | 0.45‰ | 3‰ | 200/200 |
| C1-640-34 | 0.59‰ | 3‰ | 200/200 |
| C2-384-12-I8 | 0.68‰ | 4‰ | 200/200 |
| C3-768-30-pre8-I8 | 0.56‰ | 3‰ | 200/200 |

lc0 on CPU vs Ceres on TRT: mean 0.77‰, max 7‰, 199/200 bestmove.

Residual of a few per mille is FP16 kernel-selection noise plus the per-mille
granularity of the reported WDL, not a semantic difference.

---

## 8. Known issues

* **C1-BIG does not run.** It loads (`readyok`) then the process exits during
  the first evaluation — in **Ceres too**, identically. Not an lc0 defect;
  unproven and unfixed. Suspected memory.
* **Small run-to-run variation on GPU.** Re-evaluating the same position in one
  process can shift a few per mille on `onnx-cuda`. A standard BT4 net does the
  same while being stable on CPU, so this is pre-existing lc0/ORT GPU
  behaviour, unrelated to Ceres support.
* **Non-ONNX backends fail at eval time, not load time.** Pairing a Ceres net
  with `--backend=cuda` throws on the first position rather than reporting a
  clear error at startup.

---

## 9. Building and testing

ONNX-only build (no CUDA compilation needed; the CUDA EP comes from ONNX
Runtime):

```bash
meson setup build-onnx --backend ninja --buildtype release -Donnx=true -Donnx_libdir=<ORT>/runtimes/win-x64/native -Donnx_include=<ORT>/buildTransitive/native/include -Dplain_cuda=false -Dcudnn=false -Dgtest=false -Dnvcc=false -Ddefault_library=static
```

Then `ninja -C build-onnx`. Copy `onnxruntime.dll`,
`onnxruntime_providers_shared.dll`, `onnxruntime_providers_cuda.dll` and (for
TRT) `onnxruntime_providers_tensorrt.dll` next to `lc0.exe`; TensorRT's own
`bin/` must be on `PATH` for `onnx-trt`.

Quick correctness check — a won K+3P endgame must not read as a draw:

```bash
lc0 --backend=onnx-trt --weights=C1-256-10.onnx
```
```
setoption name UCI_ShowWDL value true
position fen 8/2k5/8/8/8/8/5PPP/6K1 w - - 0 60
go nodes 1
```

Expect `wdl 998 2 0`. A constant `wdl 49 450 501` across different positions
means the graph is being mangled (see §6).

---

## 10. Ceres source cross-reference

| Topic | Ceres file |
|---|---|
| Square record layout and writer | `Ceres.Chess/NNEvaluators/Ceres/TPG/TPGSquareRecord.cs` |
| Byte scaling constant | `Ceres.Base/DataTypes/ByteScaled.cs` (`SCALING_FACTOR = 100`) |
| Move50 / plies-since encodings | `Ceres.Chess/NNEvaluators/Ceres/TPG/TPGRecordEncoding.cs` |
| `EMIT_PLY_SINCE_LAST_MOVE_PER_SQUARE` | `Ceres.Chess/NNEvaluators/Ceres/TPG/TPGRecord.cs` |
| Flat conversion / divisor | `Ceres.Chess/NNEvaluators/Ceres/TPG/TPGConvertersToFlat.cs` |
| Session options, EP setup, TRT flags | `Ceres.Chess/NNBackends/ONNXRuntime/ONNXExecutor.cs` |
| Value head temperatures and blending | `Ceres.Chess/NNEvaluators/Batch/PositionEvaluationBatch.cs` |
| Default V2FRAC / V1TEMP / V2TEMP | `Ceres.Chess/NNEvaluators/Ceres/NNEvaluatorOptionsCeres.cs` |
| Square reversal helper | `Ceres.Chess/Basic/Square.cs` (`Reversed`) |

Select a backend in Ceres with the device spec suffix, e.g.
`GPU:0#TensorRT` (`#` introduces the execution engine override).
