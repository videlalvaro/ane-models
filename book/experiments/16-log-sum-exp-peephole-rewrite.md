---
layout: default
title: "Experiment 16 - Log-Sum-Exp Peephole Rewrite"
---

<nav class="experiment-nav"><a href="../08-experiments.html">Back to Experiment Index</a> | <a href="17-fused-apl-style-inner-product.html">Next: Experiment 17</a></nav>

# Experiment 16 - Log-Sum-Exp Peephole Rewrite

**Sources**: Concrete Mathematics Ch. 9 (Asymptotics) + Dragon Book (Peephole Optimization)

**Mathematical basis**:
The matmul accumulator currently does: `acc = ln(exp(acc) + exp(new_term))` — that's
2 exps + 1 ln per accumulation step. The log-sum-exp identity rewrites this as:

$$
\ln(e^a + e^b) = \max(a,b) + \ln\left(1 + e^{-|a-b|}\right)
$$

This is 1 exp + 1 ln + 1 add — saving 1 transcendental per accumulation step.

**Provenance**:
- The identity itself is standard in numerical computing, but the *framing* as a 
  peephole rewrite (scan a window of 3–5 EML operations, pattern-match, replace) 
  comes directly from the Dragon Book's treatment of peephole optimization (§8.7 
  in 2nd edition).
- The asymptotic analysis of why this matters at scale (\(O(K)\) savings per dot
  product where \(K = 896\)) is Concrete Mathematics Ch. 9 thinking.

**Expected impact**: ~50% fewer transcendentals in the accumulation loop.

---


<nav class="experiment-nav"><a href="../08-experiments.html">Back to Experiment Index</a> | <a href="17-fused-apl-style-inner-product.html">Next: Experiment 17</a></nav>
