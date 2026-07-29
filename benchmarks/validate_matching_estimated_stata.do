/*
Cross-language parity for CausalKit's estimated-propensity matching inference.

Run from the causalkit repository root:

    do "benchmarks/validate_matching_estimated_stata.do"

The script always replaces
benchmarks/validate_matching_estimated_stata_output.txt before running its
parity assertions. This preserves diagnostics even when the script exits with
r(9), and lets CausalKit's maintainers inspect a manually generated result
without launching Stata.

The fixture has a finite full-sample unpenalized Logit MLE and no score or
covariate boundary ties. Stata estimates that first step internally. CausalKit
consumes the corresponding fitted-result protocol instead of owning a duplicate
binary Logit estimator. Both implementations use one effect match, one
leave-own-out same-arm neighbor for the known-score variance, two observations
for local first-step covariance moments, two leave-own-out same-arm score
neighbors for local outcome regressions, one opposite-arm covariate neighbor,
no caliper, no support trimming, and no bias correction.

Stata's vce(robust, nn(2)) counts the focal observation in its local PSM
variance set. Its nocorrection variance therefore maps to CausalKit's
variance_neighbors=1. The nocorrection option also provides the valid
within-psmatch decomposition of Stata's first-step adjustment.
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
matrix b_propensity = e(bps)
matrix V_propensity = e(Vps)
scalar att_estimate = el(b_att, 1, 1)
scalar att_standard_error = sqrt(el(V_att, 1, 1))

/* Stata-internal validation baseline: same PSM calculation, no first-step correction. */
quietly teffects psmatch (outcome) (treatment x, logit), atet ///
    nneighbor(1) vce(robust, nn(2)) nocorrection
matrix V_att_ps_uncorrected = e(V)
scalar att_ps_uncorrected_variance = el(V_att_ps_uncorrected, 1, 1)
scalar att_ps_first_step_adjustment = el(V_att, 1, 1) - att_ps_uncorrected_variance

/* ATE: bidirectional nearest-neighbor imputation on the fitted propensity. */
quietly teffects psmatch (outcome) (treatment x, logit), ate ///
    nneighbor(1) vce(robust, nn(2))
matrix b_ate = e(b)
matrix V_ate = e(V)
scalar ate_estimate = el(b_ate, 1, 1)
scalar ate_standard_error = sqrt(el(V_ate, 1, 1))

quietly teffects psmatch (outcome) (treatment x, logit), ate ///
    nneighbor(1) vce(robust, nn(2)) nocorrection
matrix V_ate_ps_uncorrected = e(V)
scalar ate_ps_uncorrected_variance = el(V_ate_ps_uncorrected, 1, 1)
scalar ate_ps_first_step_adjustment = el(V_ate, 1, 1) - ate_ps_uncorrected_variance

/* Reverse treatment to obtain ATC, then restore the Y(1)-Y(0) sign. */
generate byte treatment_reversed = 1 - treatment
quietly teffects psmatch (outcome) (treatment_reversed x, logit), atet ///
    nneighbor(1) vce(robust, nn(2))
matrix b_atc_reversed = e(b)
matrix V_atc_reversed = e(V)
scalar atc_estimate = -el(b_atc_reversed, 1, 1)
scalar atc_standard_error = sqrt(el(V_atc_reversed, 1, 1))

quietly teffects psmatch (outcome) (treatment_reversed x, logit), atet ///
    nneighbor(1) vce(robust, nn(2)) nocorrection
matrix V_atc_ps_uncorrected = e(V)
scalar atc_ps_uncorrected_variance = el(V_atc_ps_uncorrected, 1, 1)
scalar atc_ps_first_step_adjustment = ///
    el(V_atc_reversed, 1, 1) - atc_ps_uncorrected_variance

/* Hand-computed CausalKit contract values. */
scalar expected_att_estimate = 2.5
scalar expected_att_se = 0.49507487173864956
scalar expected_att_known_variance = 0.763888888888889
scalar expected_att_first_step_adj = -0.5187897602618486
scalar expected_atc_estimate = 13 / 6
scalar expected_atc_se = 0.4566897818895616
scalar expected_atc_known_variance = 0.35648148148148145
scalar expected_atc_first_step_adj = -0.14791592459914613
scalar expected_ate_estimate = 7 / 3
scalar expected_ate_se = 0.6975256843595473
scalar expected_ate_known_variance = 0.7824074074074074
scalar expected_ate_first_step_adj = -0.2958653270661526
scalar tolerance = 1e-8

/* Print observed values before assertions so a failed contract remains diagnosable. */
display as result "observed_att_estimate=" %21.17g att_estimate
display as result "observed_att_standard_error=" %21.17g att_standard_error
display as result "observed_atc_estimate=" %21.17g atc_estimate
display as result "observed_atc_standard_error=" %21.17g atc_standard_error
display as result "observed_ate_estimate=" %21.17g ate_estimate
display as result "observed_ate_standard_error=" %21.17g ate_standard_error

/*
Persist diagnostics before any assertion can exit. Run this do-file from the
repository root so the artifact lands at the stable path documented above.
*/
local att_estimate_status = cond( ///
    abs(att_estimate - expected_att_estimate) <= tolerance, "pass", "fail")
local att_standard_error_status = cond( ///
    abs(att_standard_error - expected_att_se) <= tolerance, "pass", "fail")
local atc_estimate_status = cond( ///
    abs(atc_estimate - expected_atc_estimate) <= tolerance, "pass", "fail")
local atc_standard_error_status = cond( ///
    abs(atc_standard_error - expected_atc_se) <= tolerance, "pass", "fail")
local ate_estimate_status = cond( ///
    abs(ate_estimate - expected_ate_estimate) <= tolerance, "pass", "fail")
local ate_standard_error_status = cond( ///
    abs(ate_standard_error - expected_ate_se) <= tolerance, "pass", "fail")
local att_known_variance_status = cond( ///
    abs(att_ps_uncorrected_variance - expected_att_known_variance) <= tolerance, ///
    "pass", "fail")
local atc_known_variance_status = cond( ///
    abs(atc_ps_uncorrected_variance - expected_atc_known_variance) <= tolerance, ///
    "pass", "fail")
local ate_known_variance_status = cond( ///
    abs(ate_ps_uncorrected_variance - expected_ate_known_variance) <= tolerance, ///
    "pass", "fail")
local att_first_step_status = cond( ///
    abs(att_ps_first_step_adjustment - expected_att_first_step_adj) <= tolerance, ///
    "pass", "fail")
local atc_first_step_status = cond( ///
    abs(atc_ps_first_step_adjustment - expected_atc_first_step_adj) <= tolerance, ///
    "pass", "fail")
local ate_first_step_status = cond( ///
    abs(ate_ps_first_step_adjustment - expected_ate_first_step_adj) <= tolerance, ///
    "pass", "fail")
local parity_status = cond( ///
    "`att_estimate_status'" == "pass" & ///
    "`att_standard_error_status'" == "pass" & ///
    "`atc_estimate_status'" == "pass" & ///
    "`atc_standard_error_status'" == "pass" & ///
    "`ate_estimate_status'" == "pass" & ///
    "`ate_standard_error_status'" == "pass" & ///
    "`att_known_variance_status'" == "pass" & ///
    "`atc_known_variance_status'" == "pass" & ///
    "`ate_known_variance_status'" == "pass" & ///
    "`att_first_step_status'" == "pass" & ///
    "`atc_first_step_status'" == "pass" & ///
    "`ate_first_step_status'" == "pass", "pass", "fail")

local results_path "benchmarks/validate_matching_estimated_stata_output.txt"
tempname results_file
file open `results_file' using "`results_path'", write text replace
file write `results_file' "contract=full_sample_unpenalized_logit_mle_no_ties" _n
file write `results_file' "stata_version=`c(stata_version)'" _n
file write `results_file' "stata_flavor=`c(flavor)'" _n
file write `results_file' "neighbors=1" _n
file write `results_file' "variance_neighbors=1" _n
file write `results_file' "stata_robust_neighbors=2" _n
file write `results_file' "first_step_covariance_neighbors=2" _n
file write `results_file' "first_step_regression_neighbors=2" _n
file write `results_file' "first_step_covariate_neighbors=1" _n
file write `results_file' "tolerance=" %21.17g (tolerance) _n
file write `results_file' "propensity_x_coefficient=" %21.17g (el(b_propensity, 1, 1)) _n
file write `results_file' "propensity_intercept=" %21.17g (el(b_propensity, 1, 2)) _n
file write `results_file' "propensity_x_variance=" %21.17g (el(V_propensity, 1, 1)) _n
file write `results_file' "propensity_intercept_variance=" %21.17g ///
    (el(V_propensity, 2, 2)) _n
file write `results_file' "propensity_x_intercept_covariance=" %21.17g ///
    (el(V_propensity, 1, 2)) _n
file write `results_file' "att_estimate=" %21.17g (att_estimate) _n
file write `results_file' "expected_att_estimate=" %21.17g (expected_att_estimate) _n
file write `results_file' "att_estimate_absolute_difference=" ///
    %21.17g (abs(att_estimate - expected_att_estimate)) _n
file write `results_file' "att_estimate_status=`att_estimate_status'" _n
file write `results_file' "att_standard_error=" %21.17g (att_standard_error) _n
file write `results_file' "expected_att_standard_error=" %21.17g (expected_att_se) _n
file write `results_file' "att_standard_error_absolute_difference=" ///
    %21.17g (abs(att_standard_error - expected_att_se)) _n
file write `results_file' "att_standard_error_status=`att_standard_error_status'" _n
file write `results_file' "att_ps_uncorrected_variance=" ///
    %21.17g (att_ps_uncorrected_variance) _n
file write `results_file' "expected_att_known_score_variance=" ///
    %21.17g (expected_att_known_variance) _n
file write `results_file' "att_known_variance_status=`att_known_variance_status'" _n
file write `results_file' "att_ps_first_step_adjustment=" ///
    %21.17g (att_ps_first_step_adjustment) _n
file write `results_file' "expected_att_first_step_adjustment=" ///
    %21.17g (expected_att_first_step_adj) _n
file write `results_file' "att_first_step_status=`att_first_step_status'" _n
file write `results_file' "atc_estimate=" %21.17g (atc_estimate) _n
file write `results_file' "expected_atc_estimate=" %21.17g (expected_atc_estimate) _n
file write `results_file' "atc_estimate_absolute_difference=" ///
    %21.17g (abs(atc_estimate - expected_atc_estimate)) _n
file write `results_file' "atc_estimate_status=`atc_estimate_status'" _n
file write `results_file' "atc_standard_error=" %21.17g (atc_standard_error) _n
file write `results_file' "expected_atc_standard_error=" %21.17g (expected_atc_se) _n
file write `results_file' "atc_standard_error_absolute_difference=" ///
    %21.17g (abs(atc_standard_error - expected_atc_se)) _n
file write `results_file' "atc_standard_error_status=`atc_standard_error_status'" _n
file write `results_file' "atc_ps_uncorrected_variance=" ///
    %21.17g (atc_ps_uncorrected_variance) _n
file write `results_file' "expected_atc_known_score_variance=" ///
    %21.17g (expected_atc_known_variance) _n
file write `results_file' "atc_known_variance_status=`atc_known_variance_status'" _n
file write `results_file' "atc_ps_first_step_adjustment=" ///
    %21.17g (atc_ps_first_step_adjustment) _n
file write `results_file' "expected_atc_first_step_adjustment=" ///
    %21.17g (expected_atc_first_step_adj) _n
file write `results_file' "atc_first_step_status=`atc_first_step_status'" _n
file write `results_file' "ate_estimate=" %21.17g (ate_estimate) _n
file write `results_file' "expected_ate_estimate=" %21.17g (expected_ate_estimate) _n
file write `results_file' "ate_estimate_absolute_difference=" ///
    %21.17g (abs(ate_estimate - expected_ate_estimate)) _n
file write `results_file' "ate_estimate_status=`ate_estimate_status'" _n
file write `results_file' "ate_standard_error=" %21.17g (ate_standard_error) _n
file write `results_file' "expected_ate_standard_error=" %21.17g (expected_ate_se) _n
file write `results_file' "ate_standard_error_absolute_difference=" ///
    %21.17g (abs(ate_standard_error - expected_ate_se)) _n
file write `results_file' "ate_standard_error_status=`ate_standard_error_status'" _n
file write `results_file' "ate_ps_uncorrected_variance=" ///
    %21.17g (ate_ps_uncorrected_variance) _n
file write `results_file' "expected_ate_known_score_variance=" ///
    %21.17g (expected_ate_known_variance) _n
file write `results_file' "ate_known_variance_status=`ate_known_variance_status'" _n
file write `results_file' "ate_ps_first_step_adjustment=" ///
    %21.17g (ate_ps_first_step_adjustment) _n
file write `results_file' "expected_ate_first_step_adjustment=" ///
    %21.17g (expected_ate_first_step_adj) _n
file write `results_file' "ate_first_step_status=`ate_first_step_status'" _n
file write `results_file' "parity_status=`parity_status'" _n
file close `results_file'
display as result "results_file=`results_path'"

if "`parity_status'" != "pass" {
    display as error "Estimated-propensity matching parity failed; inspect `results_path'."
    exit 9
}

display as result "contract=full_sample_unpenalized_logit_mle_no_ties"
display as result "neighbors=1"
display as result "variance_neighbors=1"
display as result "stata_robust_neighbors=2"
display as result "first_step_covariance_neighbors=2"
display as result "first_step_regression_neighbors=2"
display as result "first_step_covariate_neighbors=1"
display as result "att_estimate=" %21.17g att_estimate
display as result "att_standard_error=" %21.17g att_standard_error
display as result "atc_estimate=" %21.17g atc_estimate
display as result "atc_standard_error=" %21.17g atc_standard_error
display as result "ate_estimate=" %21.17g ate_estimate
display as result "ate_standard_error=" %21.17g ate_standard_error
display as result "parity_status=`parity_status'"

exit 0
