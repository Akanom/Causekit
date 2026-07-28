/*
Cross-language parity for CausalKit's known-score matching inference.

Run from the causalkit repository root:

    stata -b do benchmarks/validate_matching_stata.do

or open this file in Stata and choose Do. The script prints machine-readable
key=value lines. It validates a fixed logit score with one match, replacement,
no caliper, no support trimming, no bias correction, no boundary ties, and two
same-arm neighbors for the Abadie-Imbens conditional-variance estimate. Stata
requires nn() to be at least 2 even though CausalKit also supports 1.

This deliberately uses teffects nnmatch on logit_score. Do not replace it with
teffects psmatch: that command estimates a treatment model and therefore targets
the separate estimated-propensity variance contract that CausalKit still refuses.
*/

clear all
set more off

display as text "stata_version=" c(stata_version)
display as text "stata_flavor=" c(flavor)

input str2 id double outcome byte treatment double logit_score
"c0"  0 0  0
"c1"  2 0  4
"c2"  5 0 10
"t0"  3 1  1
"t1"  8 1  6
"t2" 12 1  9
end

isid id
assert inlist(treatment, 0, 1)

/* ATT: Stata calls this ATET. */
quietly teffects nnmatch (outcome logit_score) (treatment), atet ///
    nneighbor(1) metric(euclidean) vce(robust, nn(2))
matrix b_att = e(b)
matrix V_att = e(V)
scalar att_estimate = el(b_att, 1, 1)
scalar att_standard_error = sqrt(el(V_att, 1, 1))

/* ATE: bidirectional nearest-neighbor imputation. */
quietly teffects nnmatch (outcome logit_score) (treatment), ate ///
    nneighbor(1) metric(euclidean) vce(robust, nn(2))
matrix b_ate = e(b)
matrix V_ate = e(V)
scalar ate_estimate = el(b_ate, 1, 1)
scalar ate_standard_error = sqrt(el(V_ate, 1, 1))

/*
Stata exposes ATE and ATET, but not ATC directly. After reversing treatment,
ATET estimates E[Y(0)-Y(1) | original treatment=0], so negate its coefficient.
The variance and standard error are unchanged by that sign reversal.
*/
generate byte treatment_reversed = 1 - treatment
quietly teffects nnmatch (outcome logit_score) (treatment_reversed), atet ///
    nneighbor(1) metric(euclidean) vce(robust, nn(2))
matrix b_atc_reversed = e(b)
matrix V_atc_reversed = e(V)
scalar atc_estimate = -el(b_atc_reversed, 1, 1)
scalar atc_standard_error = sqrt(el(V_atc_reversed, 1, 1))

/* Hand-computed CausalKit contract values. */
scalar expected_estimate = 16 / 3
scalar expected_att_atc_se = sqrt(26 / 27)
scalar expected_ate_se = sqrt(133 / 27)

if abs(att_estimate - expected_estimate) > 1e-10 {
    display as error "ATT estimate parity failed"
    exit 9
}
if abs(atc_estimate - expected_estimate) > 1e-10 {
    display as error "ATC estimate parity failed"
    exit 9
}
if abs(ate_estimate - expected_estimate) > 1e-10 {
    display as error "ATE estimate parity failed"
    exit 9
}
if abs(att_standard_error - expected_att_atc_se) > 1e-10 {
    display as error "ATT standard-error parity failed"
    exit 9
}
if abs(atc_standard_error - expected_att_atc_se) > 1e-10 {
    display as error "ATC standard-error parity failed"
    exit 9
}
if abs(ate_standard_error - expected_ate_se) > 1e-10 {
    display as error "ATE standard-error parity failed"
    exit 9
}

display as result "contract=fixed_known_logit_score_no_ties"
display as result "neighbors=1"
display as result "variance_neighbors=2"
display as result "att_estimate=" %21.17g att_estimate
display as result "att_standard_error=" %21.17g att_standard_error
display as result "atc_estimate=" %21.17g atc_estimate
display as result "atc_standard_error=" %21.17g atc_standard_error
display as result "ate_estimate=" %21.17g ate_estimate
display as result "ate_standard_error=" %21.17g ate_standard_error
display as result "parity_status=pass"

exit 0
