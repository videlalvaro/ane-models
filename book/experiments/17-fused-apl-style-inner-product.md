---
layout: default
title: "Experiment 17 - Fused APL-Style Inner Product"
---

<nav class="experiment-nav"><a href="../08-experiments.html">Back to Experiment Index</a> | <a href="16-log-sum-exp-peephole-rewrite.html">Previous: Experiment 16</a> | <a href="18-constraint-propagation-for-realness.html">Next: Experiment 18</a></nav>

# Experiment 17 - Fused APL-Style Inner Product

**Sources**: Iverson's *A Programming Language* — inner product operator `+.×`

**Mathematical basis**:
APL treats `A +.× B` (matmul) as a single fused operator. For EML:

$$
\operatorname{dot}(a,b) = \ln\left(\sum_j \exp\left(\ln(a_j) + \ln(b_j)\right)\right)
$$

Using the log-sum-exp trick with a running max:

$$
m = \max_j\left(\ln(a_j) + \ln(b_j)\right)
$$

$$
\operatorname{dot}(a,b) = m + \ln\left(\sum_j \exp\left(\ln(a_j) + \ln(b_j) - m\right)\right)
$$

This is K exps + 1 ln for the whole dot product instead of K exps + K lns 
for element-wise accumulation. Cuts lns by factor of K (896).

**Provenance**:
- Iverson's key insight: "think of the whole array operation as a single entity, 
  not a loop over scalars." His inner product operator `+.×` fuses reduction with 
  element-wise application.
- APL idiom recognition (from the APL implementation literature): detect 
  `+/A×B` patterns and evaluate as a single fused operation.
- The running-max numerically-stable variant is standard in ML (used in softmax), 
  but applying it to EML's log-domain matmul accumulation is novel.

**Expected impact**: Potentially reduces lns from \(O(K)\) to \(O(1)\) per dot product.
Combined with Exp 16, this is the most promising direction.

---


<nav class="experiment-nav"><a href="../08-experiments.html">Back to Experiment Index</a> | <a href="16-log-sum-exp-peephole-rewrite.html">Previous: Experiment 16</a> | <a href="18-constraint-propagation-for-realness.html">Next: Experiment 18</a></nav>
