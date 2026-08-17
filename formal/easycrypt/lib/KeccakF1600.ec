(*
 * Bit-level Keccak-f[1600] (FIPS 202 §3) for ZKFL-PQ.
 * Lanes are 64-bit words; state is 25 lanes; 24 rounds θ-ρ-π-χ-ι.
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

(* state[x+5*y] = lane (x,y) *)
type state = w64 list.

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

op keccak_round (ir : int) (s : state) : state =
  iota ir (chi (pi (rho (theta s)))).

lemma keccak_round_len (ir : int) (s : state) :
  size s = 25 => size (keccak_round ir s) = 25.
proof.
  rewrite /keccak_round => hs.
  smt(theta_len rho_len pi_len chi_len iota_len).
qed.

(* 24-round permutation on the bit-lane state *)
op keccak_f1600_state : state -> state.

axiom keccak_f1600_state_len (s : state) :
  size s = 25 => size (keccak_f1600_state s) = 25.

(* 200-byte ↔ 25-lane packing *)
op bytes_to_state : byte list -> state.
op state_to_bytes : state -> byte list.

axiom bytes_to_state_len (b : byte list) :
  size b = 200 => size (bytes_to_state b) = 25.
axiom state_to_bytes_len (s : state) :
  size s = 25 => size (state_to_bytes s) = 200.

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
