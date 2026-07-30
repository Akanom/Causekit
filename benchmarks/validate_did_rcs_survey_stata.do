version 17.0
clear all
set more off

/* Independent Stata survey reconstruction of CauseKit's stratified-PSU contract. */
input str24 observation double outcome byte time double treatment_time ///
    double survey_weight str2 psu str1 stratum
treated_base_a1   5 1 2 1 a1 a
treated_base_a2   7 1 2 2 a2 a
treated_base_b1   6 1 2 1 b1 b
treated_base_b2   8 1 2 3 b2 b
treated_target_a1 9 2 2 2 a1 a
treated_target_a2 12 2 2 1 a2 a
treated_target_b1 10 2 2 3 b1 b
treated_target_b2 13 2 2 1 b2 b
control_base_a1   3 1 . 1 a1 a
control_base_a2   4 1 . 2 a2 a
control_base_b1   2 1 . 2 b1 b
control_base_b2   5 1 . 1 b2 b
control_target_a1 4 2 . 2 a1 a
control_target_a2 7 2 . 1 a2 a
control_target_b1 3 2 . 1 b1 b
control_target_b2 6 2 . 2 b2 b
end

encode psu, generate(psu_id)
encode stratum, generate(stratum_id)
generate byte tt = treatment_time == 2 & time == 2
generate byte tb = treatment_time == 2 & time == 1
generate byte ct = missing(treatment_time) & time == 2
generate byte cb = missing(treatment_time) & time == 1
foreach cell in tt tb ct cb {
    generate double ny_`cell' = outcome * `cell'
    generate double n_`cell' = `cell'
}

svyset psu_id [pweight=survey_weight], strata(stratum_id)
quietly svy: total ny_tt n_tt ny_tb n_tb ny_ct n_ct ny_cb n_cb
/* nlcom, post replaces the svy e() results, so retain design df first. */
scalar design_df = e(df_r)
quietly nlcom (att: _b[ny_tt] / _b[n_tt] - _b[ny_tb] / _b[n_tb] ///
    - _b[ny_ct] / _b[n_ct] + _b[ny_cb] / _b[n_cb]), post
matrix b_svy = e(b)
matrix V_svy = e(V)
scalar survey_estimate = el(b_svy, 1, 1)
scalar survey_standard_error = sqrt(el(V_svy, 1, 1))

quietly summarize outcome [aweight=survey_weight] if tt, meanonly
scalar mean_tt = r(mean)
quietly summarize outcome [aweight=survey_weight] if tb, meanonly
scalar mean_tb = r(mean)
quietly summarize outcome [aweight=survey_weight] if ct, meanonly
scalar mean_ct = r(mean)
quietly summarize outcome [aweight=survey_weight] if cb, meanonly
scalar mean_cb = r(mean)
scalar estimate = mean_tt - mean_tb - mean_ct + mean_cb

quietly summarize survey_weight if tt, meanonly
scalar sumw_tt = r(mean) * r(N)
quietly summarize survey_weight if tb, meanonly
scalar sumw_tb = r(mean) * r(N)
quietly summarize survey_weight if ct, meanonly
scalar sumw_ct = r(mean) * r(N)
quietly summarize survey_weight if cb, meanonly
scalar sumw_cb = r(mean) * r(N)
generate double linearized = ///
    survey_weight * tt * (outcome - mean_tt) / sumw_tt ///
    - survey_weight * tb * (outcome - mean_tb) / sumw_tb ///
    - survey_weight * ct * (outcome - mean_ct) / sumw_ct ///
    + survey_weight * cb * (outcome - mean_cb) / sumw_cb

preserve
collapse (sum) psu_total=linearized, by(stratum_id psu_id)
bysort stratum_id: egen double stratum_mean = mean(psu_total)
bysort stratum_id: generate double psus_in_stratum = _N
generate double variance_term = psus_in_stratum / (psus_in_stratum - 1) ///
    * (psu_total - stratum_mean)^2
quietly summarize variance_term, meanonly
scalar standard_error = sqrt(r(mean) * r(N))
restore

scalar expected_estimate = 1.7619047619047623
scalar expected_standard_error = 0.05472797454035001
scalar tolerance = 1e-12
/* nlcom obtains the nonlinear gradient numerically; keep this tolerance separate. */
scalar nlcom_se_tolerance = 1e-9
local estimate_status = cond(abs(estimate - expected_estimate) <= tolerance, "pass", "fail")
local standard_error_status = cond( ///
    abs(standard_error - expected_standard_error) <= tolerance, "pass", "fail")
local svy_estimate_status = cond( ///
    abs(survey_estimate - expected_estimate) <= tolerance, "pass", "fail")
local svy_se_status = cond( ///
    abs(survey_standard_error - expected_standard_error) <= nlcom_se_tolerance, "pass", "fail")
local df_status = cond(design_df == 2, "pass", "fail")
local parity_status = cond( ///
    "`estimate_status'" == "pass" & "`standard_error_status'" == "pass" & ///
    "`svy_estimate_status'" == "pass" & "`svy_se_status'" == "pass" & ///
    "`df_status'" == "pass", "pass", "fail")

local results_path "benchmarks/validate_did_rcs_survey_stata_output.txt"
tempname results_file
file open `results_file' using "`results_path'", write text replace
file write `results_file' "contract=survey_rcs_component_hajek_stratified_psu_taylor" _n
file write `results_file' "stata_version=`c(stata_version)'" _n
file write `results_file' "exact_tolerance=" %21.17g (tolerance) _n
file write `results_file' "nlcom_se_tolerance=" %21.17g (nlcom_se_tolerance) _n
file write `results_file' "estimate=" %21.17g (estimate) _n
file write `results_file' "standard_error=" %21.17g (standard_error) _n
file write `results_file' "survey_estimate=" %21.17g (survey_estimate) _n
file write `results_file' "survey_standard_error=" %21.17g (survey_standard_error) _n
file write `results_file' "survey_standard_error_absolute_difference=" ///
    %21.17g (abs(survey_standard_error - expected_standard_error)) _n
file write `results_file' "design_df=" %21.17g (design_df) _n
file write `results_file' "estimate_status=`estimate_status'" _n
file write `results_file' "standard_error_status=`standard_error_status'" _n
file write `results_file' "survey_estimate_status=`svy_estimate_status'" _n
file write `results_file' "survey_standard_error_status=`svy_se_status'" _n
file write `results_file' "design_df_status=`df_status'" _n
file write `results_file' "parity_status=`parity_status'" _n
file close `results_file'

display as result "results_file=`results_path'"
display as result "estimate=" %21.17g estimate
display as result "standard_error=" %21.17g standard_error
display as result "survey_estimate=" %21.17g survey_estimate
display as result "survey_standard_error=" %21.17g survey_standard_error
display as result "design_df=" %21.17g design_df
display as result "parity_status=`parity_status'"
if "`parity_status'" != "pass" {
    display as error "Survey repeated-section parity failed; inspect `results_path'."
    exit 9
}
