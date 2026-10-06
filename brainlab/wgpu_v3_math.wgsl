// OpenLibm src/e_exp.c 9fbeafcd4f1b6ef6aa3946c1c8faead50f38a94d
// Copyright (C) 2004 by Sun Microsystems, Inc. All rights reserved.
// Permission to use, copy, modify, and distribute this
// software is freely granted, provided that this notice
// is preserved.
// Integer division copied from accepted G3C3 7be963f.
fn f(bits: u64) -> f64 { return bitcast<f64>(bits); }

fn openlibm_exp_nonpositive(x0: f64) -> f64 {
  let bits = bitcast<u64>(x0);
  let ax = bits & 0x7ffffffffffffffflu;
  if (ax == 0lu) { return f(0x3ff0000000000000lu); }

  let high = u32(ax >> 32u);
  let u_threshold = f(0xc0874910d52d3051lu);
  let twom1000 = f(0x0170000000000000lu);
  if (high >= 0x40862e42u && x0 < u_threshold) {
    return twom1000 * twom1000;
  }

  let one = f(0x3ff0000000000000lu);
  var r = x0;
  var hi = f(0lu);
  var lo = f(0lu);
  var k: i32 = 0;
  let ln2hi = f(0x3fe62e42fee00000lu);
  let ln2lo = f(0x3dea39ef35793c76lu);
  let invln2 = f(0x3ff71547652b82felu);

  if (high > 0x3fd62e42u) {
    if (high < 0x3ff0a2b2u) {
      hi = r + ln2hi;
      lo = -ln2lo;
      k = -1;
    } else {
      k = i32(invln2 * r - f(0x3fe0000000000000lu));
      let tk = f64(k);
      hi = r - tk * ln2hi;
      lo = tk * ln2lo;
    }
    r = hi - lo;
  } else if (high < 0x3e300000u) {
    return one + r;
  }

  let p1 = f(0x3fc555555555553elu);
  let p2 = f(0xbf66c16c16bebd93lu);
  let p3 = f(0x3f11566aaf25de2clu);
  let p4 = f(0xbebbbd41c5d26bf1lu);
  let p5 = f(0x3e66376972bea4d0lu);
  let t = r * r;
  let c = r - t * (p1 + t * (p2 + t * (p3 + t * (p4 + t * p5))));

  var twopk: f64;
  if (k >= -1021) {
    twopk = bitcast<f64>(u64(1023 + k) << 52u);
  } else {
    twopk = bitcast<f64>(u64(1023 + k + 1000) << 52u);
  }
  if (k == 0) {
    return one - ((r * c) / (c - f(0x4000000000000000lu)) - r);
  }
  let y = one - ((lo - (r * c) / (f(0x4000000000000000lu) - c)) - hi);
  if (k >= -1021) {
    return y * twopk;
  }
  return y * twopk * twom1000;
}


fn round_significand_quotient_rne(quotient: u64, remainder: u64, divisor: u64) -> u64 {
  // Registered bounds: quotient < 2^53, remainder < divisor < 2^53.
  // Doubling the remainder is < 2^54; no 106-bit scaled numerator exists.
  let twice_remainder = remainder << 1u;
  if (twice_remainder > divisor ||
      (twice_remainder == divisor && (quotient & 1lu) != 0lu)) {
    return quotient + 1lu;
  }
  return quotient;
}

fn divide_normal_f64_rne(numerator: f64, denominator: f64) -> f64 {
  let a = bitcast<u64>(numerator);
  let b = bitcast<u64>(denominator);
  let sign = (a ^ b) & 0x8000000000000000lu;
  let ae = (a >> 52u) & 0x7fflu;
  let be = (b >> 52u) & 0x7fflu;
  let af = a & 0x000ffffffffffffflu;
  let bf = b & 0x000ffffffffffffflu;
  // Explicit diagnostic refusal: denominator must be finite normal; numerator
  // must be finite normal or signed zero. Unsupported results use this qNaN.
  if (be == 0lu || be == 0x7fflu || ae == 0x7fflu ||
      (ae == 0lu && af != 0lu)) {
    return bitcast<f64>(0x7ff8000000000001lu);
  }
  if (ae == 0lu) { return bitcast<f64>(sign); }
  let am = (1lu << 52u) | af;
  let bm = (1lu << 52u) | bf;
  var output_exponent = i32(ae) - i32(be) + 1023;
  var remainder: u64;
  if (am < bm) {
    // am * 2 < 2^54; normalized numerator is in [bm, 2*bm).
    remainder = (am << 1u) - bm;
    output_exponent = output_exponent - 1;
  } else {
    remainder = am - bm;
  }
  if (output_exponent < 1 || output_exponent > 2046) {
    return bitcast<f64>(0x7ff8000000000001lu);
  }
  var quotient = 1lu;
  // Long division emits the other 52 significand bits. Each doubled remainder
  // is < 2*bm < 2^54; each quotient is < 2^53. u64 arithmetic never overflows.
  for (var bit = 0u; bit < 52u; bit = bit + 1u) {
    remainder = remainder << 1u;
    quotient = quotient << 1u;
    if (remainder >= bm) {
      remainder = remainder - bm;
      quotient = quotient | 1lu;
    }
  }
  quotient = round_significand_quotient_rne(quotient, remainder, bm);
  if (quotient == (1lu << 53u)) {
    quotient = quotient >> 1u;
    output_exponent = output_exponent + 1;
  }
  if (output_exponent > 2046) {
    return bitcast<f64>(0x7ff8000000000001lu);
  }
  return bitcast<f64>(sign | (u64(output_exponent) << 52u) |
                      (quotient - (1lu << 52u)));
}
