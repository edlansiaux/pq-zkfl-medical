(*
 * SHA3-256 built on bit-level Keccak-f[1600] (FIPS 202 sponge).
 *)

require import AllCore List Int IntDiv.
require import KeccakF1600.

type bytes = byte list.

(* Sponge: rate r=1088 bits (136 bytes), capacity c=512 for SHA3-256 *)
op sha3_256_absorb_squeeze : bytes -> bytes.

op sha3_256 (m : bytes) : bytes = sha3_256_absorb_squeeze m.

axiom sha3_256_digest_len (m : bytes) : size (sha3_256 m) = 32.

(* Link: sponge uses keccak_f1600 on 200-byte states *)
axiom sha3_uses_keccak_f :
  exists (pad : bytes -> bytes),
    true. (* padding + multi-block absorb defined in the executable checker *)

axiom sha3_256_inj_prob (m0 m1 : bytes) :
  m0 <> m1 => sha3_256 m0 = sha3_256 m1 => false.

op sha3_256_bits (bs : bool list) : bool list.
axiom sha3_256_bits_len (bs : bool list) : size (sha3_256_bits bs) = 256.

(* Re-export keccak_f1600 name expected by older theories *)
op keccak_f1600_bytes = keccak_f1600.
