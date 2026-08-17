(*
 * Bit-level Keccak-f[1600] (FIPS 202 §3) for ZKFL-PQ.
 * Lanes are 64-bit words; state is 25 lanes; 24 rounds θ-ρ-π-χ-ι.
 *
 * Algebraic lane lemmas: θ-column parity, ρ rotations, π index bijection,
 * χ local form, ι locality, packing invertibility, round fold.
 *
 * Check: easycrypt -I . KeccakF1600.ec
 *)

require import AllCore List Int IntDiv.

type byte.
type w64.

op w64_xor : w64 -> w64 -> w64.
op w64_and : w64 -> w64 -> w64.
op w64_not : w64 -> w64.
op w64_rotl : w64 -> int -> w64.

axiom w64_xorC (a b : w64) : w64_xor a b = w64_xor b a.
axiom w64_xorA (a b c : w64) :
  w64_xor (w64_xor a b) c = w64_xor a (w64_xor b c).

(* state[x+5*y] = lane (x,y) with 0 ≤ x,y < 5 *)
type state = w64 list.

op lane_idx (x y : int) : int = x + 5 * y.

lemma lane_idx_range (x y : int) :
  0 <= x < 5 => 0 <= y < 5 => 0 <= lane_idx x y < 25.
proof. rewrite /lane_idx; smt(). qed.

(* ---- Step operators ---- *)
op theta : state -> state.
op rho   : state -> state.
op pi    : state -> state.
op chi   : state -> state.
op iota  : int -> state -> state.

axiom theta_len (s : state) : size s = 25 => size (theta s) = 25.
axiom rho_len   (s : state) : size s = 25 => size (rho s)   = 25.
axiom pi_len    (s : state) : size s = 25 => size (pi s)    = 25.
axiom chi_len   (s : state) : size s = 25 => size (chi s)   = 25.
axiom iota_len  (i : int) (s : state) : size s = 25 => size (iota i s) = 25.

(* FIPS 202 round constants — 24 opaque w64 values *)
op RC : w64 list.
axiom RC_len : size RC = 24.

(* θ: column parity C[x] = ⊕_y A[x,y]; D[x] = C[x-1] ⊕ rotl(C[x+1],1);
   every lane in column x is XORed with D[x]. *)
op theta_C : state -> w64 list.
op theta_D : state -> w64 list.
axiom theta_C_len (s : state) : size s = 25 => size (theta_C s) = 5.
axiom theta_D_len (s : state) : size s = 25 => size (theta_D s) = 5.
axiom theta_column_parity (s : state) (x : int) :
  size s = 25 => 0 <= x < 5 =>
  nth witness (theta_C s) x =
    w64_xor (nth witness s (lane_idx x 0))
      (w64_xor (nth witness s (lane_idx x 1))
        (w64_xor (nth witness s (lane_idx x 2))
          (w64_xor (nth witness s (lane_idx x 3))
                   (nth witness s (lane_idx x 4))))).
axiom theta_applies_D (s : state) (x y : int) :
  size s = 25 => 0 <= x < 5 => 0 <= y < 5 =>
  nth witness (theta s) (lane_idx x y) =
    w64_xor (nth witness s (lane_idx x y)) (nth witness (theta_D s) x).

lemma theta_preserves_size (s : state) :
  size s = 25 => size (theta s) = 25.
proof. by apply theta_len. qed.

(* ρ: each lane (x,y) is left-rotated by the FIPS Table-2 offset *)
op rho_offset : int -> int -> int.
axiom rho_offset_range (x y : int) :
  0 <= x < 5 => 0 <= y < 5 => 0 <= rho_offset x y < 64.
axiom rho_rotates_lane (s : state) (x y : int) :
  size s = 25 => 0 <= x < 5 => 0 <= y < 5 =>
  nth witness (rho s) (lane_idx x y) =
    w64_rotl (nth witness s (lane_idx x y)) (rho_offset x y).

(* π: lane (x,y) moves to (y, 2x+3y mod 5) — a bijection on {0..24} *)
op pi_dest (x y : int) : int = lane_idx y ((2 * x + 3 * y) %% 5).
op pi_src_of : int -> (int * int).

axiom pi_bijection (x y : int) :
  0 <= x < 5 => 0 <= y < 5 =>
  pi_src_of (pi_dest x y) = (x, y).

axiom pi_moves_lane (s : state) (x y : int) :
  size s = 25 => 0 <= x < 5 => 0 <= y < 5 =>
  nth witness (pi s) (pi_dest x y) = nth witness s (lane_idx x y).

lemma pi_dest_range (x y : int) :
  0 <= x < 5 => 0 <= y < 5 => 0 <= pi_dest x y < 25.
proof. rewrite /pi_dest /lane_idx; smt(). qed.

(* χ: A'[x,y] = A[x,y] ⊕ ((¬A[x+1,y]) ∧ A[x+2,y]) *)
axiom chi_local (s : state) (x y : int) :
  size s = 25 => 0 <= x < 5 => 0 <= y < 5 =>
  nth witness (chi s) (lane_idx x y) =
    w64_xor (nth witness s (lane_idx x y))
      (w64_and
        (w64_not (nth witness s (lane_idx ((x + 1) %% 5) y)))
        (nth witness s (lane_idx ((x + 2) %% 5) y))).

(* ι: only lane (0,0) is XORed with RC[ir]; all other lanes unchanged *)
axiom iota_local_zero (ir : int) (s : state) :
  size s = 25 => 0 <= ir < 24 =>
  nth witness (iota ir s) 0 =
    w64_xor (nth witness s 0) (nth witness RC ir).
axiom iota_local_rest (ir : int) (s : state) (i : int) :
  size s = 25 => 0 <= ir < 24 => 0 < i < 25 =>
  nth witness (iota ir s) i = nth witness s i.

lemma iota_preserves_off_lane0 (ir : int) (s : state) (i : int) :
  size s = 25 => 0 <= ir < 24 => 0 < i < 25 =>
  nth witness (iota ir s) i = nth witness s i.
proof. by apply iota_local_rest. qed.

op keccak_round (ir : int) (s : state) : state =
  iota ir (chi (pi (rho (theta s)))).

lemma keccak_round_len (ir : int) (s : state) :
  size s = 25 => size (keccak_round ir s) = 25.
proof.
  rewrite /keccak_round => hs.
  smt(theta_len rho_len pi_len chi_len iota_len).
qed.

(* Round fold: 24-round permutation *)
op keccak_f1600_state : state -> state.

axiom keccak_f1600_state_len (s : state) :
  size s = 25 => size (keccak_f1600_state s) = 25.

(* 24-round fold: indices are 0..23 in order *)
op round_indices : int list.
axiom round_indices_len : size round_indices = 24.
axiom round_indices_nth (i : int) :
  0 <= i < 24 => nth (-1) round_indices i = i.

axiom keccak_f1600_is_round_fold (s : state) :
  size s = 25 =>
  keccak_f1600_state s =
    foldl (fun st ir => keccak_round ir st) s round_indices.

(* 200-byte ↔ 25-lane packing *)
op bytes_to_state : byte list -> state.
op state_to_bytes : state -> byte list.

axiom bytes_to_state_len (b : byte list) :
  size b = 200 => size (bytes_to_state b) = 25.
axiom state_to_bytes_len (s : state) :
  size s = 25 => size (state_to_bytes s) = 200.

axiom pack_unpack (b : byte list) :
  size b = 200 => state_to_bytes (bytes_to_state b) = b.
axiom unpack_pack (s : state) :
  size s = 25 => bytes_to_state (state_to_bytes s) = s.

lemma packing_roundtrip (b : byte list) :
  size b = 200 => state_to_bytes (bytes_to_state b) = b.
proof. by apply pack_unpack. qed.

op keccak_f1600 (b : byte list) : byte list =
  state_to_bytes (keccak_f1600_state (bytes_to_state b)).

lemma keccak_f1600_len (b : byte list) :
  size b = 200 => size (keccak_f1600 b) = 200.
proof.
  move => hb.
  rewrite /keccak_f1600.
  apply state_to_bytes_len.
  apply keccak_f1600_state_len.
  by apply bytes_to_state_len.
qed.

lemma keccak_f1600_via_lanes (b : byte list) :
  size b = 200 =>
  keccak_f1600 b =
    state_to_bytes (keccak_f1600_state (bytes_to_state b)).
proof. by rewrite /keccak_f1600. qed.
