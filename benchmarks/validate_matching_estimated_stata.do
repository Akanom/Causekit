/*
Cross-language parity for CausalKit's estimated-propensity matching inference.

Run from the causalkit repository root:

    do "benchmarks/validate_matching_estimated_stata.do"

The fixture has a finite full-sample unpenalized Logit MLE and no score or
covariate boundary ties. Stata estimates that first step internally. CausalKit
consumes the corresponding fitted-result protocol instead of owning a duplicate
binary Logit estimator. Both implementations use one effect match, two local
neighbors for conditional covariance/variance estimation, one local outcome-
regression neighbor, no caliper, no support trimming, and no bias correction.
*/

clear all
set more off

display as text "stata_version=" c(stata_version)
display as text "stata_flavor=" c(flavor)

input str3 id double outcome byte treatment double x
"u00"  0 0 -2.60
"u01"  1 0 -2.00
"u02"  3 1 -1.35
"u03"  2 0 -0.95
"u04"  5 1 -0.62
"u05"  4 0 -0.18
"u06"  7 1  0.13
"u07"  6 0  0.49
"u08"  9 1  0.91
"u09" 11 1  1.38
"u10" 10 0  1.92
"u11" 14 1  2.57
end

isid id
assert inlist(treatment, 0, 1)

/* ATT: Stata calls this ATET. */
quietly teffects psmatch (outcome) (treatment x, logit), atet ///
    nneighbor(1) vce(robust, nn(2))
matrix b_att = e(b)
matrix V_att = e(V)
scalar att_estimate = el(b_att, 1, 1)
scalar att_standard_error = sqrt(el(V_att, 1, 1))

/* ATE: bidirectional nearest-neighbor imputation on the fitted propensity. */
quietly teffects psmatch (outcome) (treatment x, logit), ate ///
    nneighbor(1) vce(robust, nn(2))
matrix b_ate = e(b)
matrix V_ate = e(V)
scalar ate_estimate = el(b_ate, 1, 1)
scalar ate_standard_error = sqrt(el(V_ate, 1, 1))

/* Reverse treatment to obtain ATC, then restore the Y(1)-Y(0) sign. */
generate byte treatment_reversed = 1 - treatment
quietly teffects psmatch (outcome) (treatment_reversed x, logit), atet ///
    nneighbor(1) vce(robust, nn(2))
matrix b_atc_reversed = e(b)
matrix V_atc_reversed = e(V)
scalar atc_estimate = -el(b_atc_reversed, 1, 1)
scalar atc_standard_error = sqrt(el(V_atc_reversed, 1, 1))

/* Hand-computed CausalKit contract values. */
scalar expected_att_estimate = 2.5
scalar expected_att_se = 0.8286781699117679
scalar expected_atc_estimate = 13 / 6
scalar expected_atc_se = 0.5402752634692672
scalar expected_ate_estimate = 7 / 3
scalar expected_ate_se = 0.8689422298747904
scalar tolerance = 1e-8

if abs(att_estimate - expected_att_estimate) > tolerance {
    display as error "ATT estimate parity failed"
    exit 9
}
if abs(att_standard_error - expected_att_se) > tolerance {
    display as error "ATT standard-error parity failed"
    exit 9
}
if abs(atc_estimate - expected_atc_estimate) > tolerance {
    display as error "ATC estimate parity failed"
    exit 9
}
if abs(atc_standard_error - expected_atc_se) > tolerance {
    display as error "ATC standard-error parity failed"
    exit 9
}
if abs(ate_estimate - expected_ate_estimate) > tolerance {
    display as error "ATE estimate parity failed"
    exit 9
}
if abs(ate_standard_error - expected_ate_se) > tolerance {
    display as error "ATE standard-error parity failed"
    exit 9
}

display as result "contract=full_sample_unpenalized_logit_mle_no_ties"
display as result "neighbors=1"
display as result "variance_neighbors=2"
display as result "first_step_covariance_neighbors=2"
display as result "first_step_regression_neighbors=1"
display as result "first_step_covariate_neighbors=1"
display as result "att_estimate=" %21.17g att_estimate
display as result "att_standard_error=" %21.17g att_standard_error
display as result "atc_estimate=" %21.17g atc_estimate
display as result "atc_standard_error=" %21.17g atc_standard_error
display as result "ate_estimate=" %21.17g ate_estimate
display as result "ate_standard_error=" %21.17g ate_standard_error
display as result "parity_status=pass"

exit 0
