version 17.0
clear all
set more off

/* Independent Stata reconstruction of CauseKit's repeated-cross-section 2x2 contract. */
input str3 observation double outcome byte time double treatment_time
tp0 1 1 2
tp1 3 1 2
tq0 6 2 2
tq1 8 2 2
cp0 2 1 .
cp1 4 1 .
cq0 3 2 .
cq1 5 2 .
end

quietly count
scalar n_observations = r(N)
quietly summarize outcome if treatment_time == 2 & time == 1, meanonly
scalar mean_treated_pre = r(mean)
scalar n_treated_pre = r(N)
quietly summarize outcome if treatment_time == 2 & time == 2, meanonly
scalar mean_treated_post = r(mean)
scalar n_treated_post = r(N)
quietly summarize outcome if missing(treatment_time) & time == 1, meanonly
scalar mean_control_pre = r(mean)
scalar n_control_pre = r(N)
quietly summarize outcome if missing(treatment_time) & time == 2, meanonly
scalar mean_control_post = r(mean)
scalar n_control_post = r(N)

scalar estimate = (mean_treated_post - mean_treated_pre) - ///
    (mean_control_post - mean_control_pre)
generate double influence = 0
replace influence = influence + n_observations / n_treated_post * ///
    (outcome - mean_treated_post) if treatment_time == 2 & time == 2
replace influence = influence - n_observations / n_treated_pre * ///
    (outcome - mean_treated_pre) if treatment_time == 2 & time == 1
replace influence = influence - n_observations / n_control_post * ///
    (outcome - mean_control_post) if missing(treatment_time) & time == 2
replace influence = influence + n_observations / n_control_pre * ///
    (outcome - mean_control_pre) if missing(treatment_time) & time == 1
generate double influence_squared = influence ^ 2
quietly summarize influence_squared, meanonly
scalar influence_sum_squares = r(mean) * r(N)
scalar standard_error_hc1 = sqrt( ///
    influence_sum_squares / (n_observations * (n_observations - 1)))

scalar expected_estimate = 4
scalar expected_standard_error = sqrt(128 / 56)
scalar tolerance = 1e-12
local estimate_status = cond(abs(estimate - expected_estimate) <= tolerance, "pass", "fail")
local standard_error_status = cond( ///
    abs(standard_error_hc1 - expected_standard_error) <= tolerance, "pass", "fail")
local parity_status = cond( ///
    "`estimate_status'" == "pass" & "`standard_error_status'" == "pass", "pass", "fail")

local results_path "benchmarks/validate_did_rcs_stata_output.txt"
tempname results_file
file open `results_file' using "`results_path'", write text replace
file write `results_file' "contract=repeated_cross_section_2x2_observation_hc1" _n
file write `results_file' "stata_version=`c(stata_version)'" _n
file write `results_file' "estimate=" %21.17g (estimate) _n
file write `results_file' "standard_error_hc1=" %21.17g (standard_error_hc1) _n
file write `results_file' "estimate_status=`estimate_status'" _n
file write `results_file' "standard_error_status=`standard_error_status'" _n
file write `results_file' "parity_status=`parity_status'" _n
file close `results_file'

display as result "results_file=`results_path'"
display as result "estimate=" %21.17g estimate
display as result "standard_error_hc1=" %21.17g standard_error_hc1
display as result "parity_status=`parity_status'"
if "`parity_status'" != "pass" {
    display as error "Repeated-cross-section hand parity failed; inspect `results_path'."
    exit 9
}
