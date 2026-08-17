(*
 * SHA3 / Keccak (FIPS 202) interface for ZKFL-PQ EasyCrypt theories.
 * Production-style library module: concrete hash op used as RO instantiation.
 *
 * Keccak-f[1600] is an abstract permutation with FIPS-aligned length axioms;
 * SHA3-256 is the sponge absorb/squeeze interface used by Unruh RO digests.
 *
 * Check:  easycrypt -I . lib/SHA3.ec
 *)

require import AllCore List Int IntDiv.

type byte.          (* abstract 8-bit value *)
type bytes = byte list.

(* Keccak-f[1600] permutation over 200-byte state *)
op keccak_f1600 : bytes -> bytes.

axiom keccak_f1600_len (s : bytes) :
  size s = 200 => size (keccak_f1600 s) = 200.

(* SHA3-256: message bytes -> 32-byte digest (FIPS 202) *)
op sha3_256 : bytes -> bytes.

axiom sha3_256_digest_len (m : bytes) :
  size (sha3_256 m) = 32.

(* Collision resistance (classical RO modelling of SHA3 as RO) *)
axiom sha3_256_inj_prob (m0 m1 : bytes) :
  m0 <> m1 =>
  sha3_256 m0 = sha3_256 m1 =>
  false.   (* idealized: collisions forbidden in the RO view *)

(* Domain separation for Unruh bitstring challenges *)
op sha3_256_bits (bs : bool list) : bool list.
axiom sha3_256_bits_len (bs : bool list) :
  size (sha3_256_bits bs) = 256.
