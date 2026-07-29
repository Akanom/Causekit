version 17.0
clear all
set more off

/* Run from the CauseKit repository root after generating the fixed input with:
   python benchmarks/prepare_rlearner_stata.py */
local input_path "benchmarks/rlearner_stata_input.csv"
local results_path "benchmarks/validate_rlearner_stata_output.txt"
capture confirm file "`input_path'"
if _rc {
    display as error "Missing fixed R-learner parity input: `input_path'"
    display as error "Run: python benchmarks/prepare_rlearner_stata.py"
    exit 601
}

import delimited using "`input_path'", clear varnames(1) asdouble
scalar tolerance = 1e-8

generate double r_error = outcome_residual - treatment_residual * cate
generate double const_error = outcome_residual - treatment_residual * constant_effect
generate double r_error_sq = r_error^2
generate double const_error_sq = const_error^2
quietly summarize r_error_sq, meanonly
scalar r_loss = r(mean)
quietly summarize const_error_sq, meanonly
scalar const_loss = r(mean)
scalar loss_gain = 1 - r_loss / const_loss

generate double weighted_cate = treatment_residual^2 * cate
generate double residual_weight = treatment_residual^2
quietly summarize weighted_cate, meanonly
scalar weighted_cate_sum = r(sum)
quietly summarize residual_weight, meanonly
scalar residual_weight_sum = r(sum)
scalar cal_center = weighted_cate_sum / residual_weight_sum
generate double cal_level_x = treatment_residual
generate double cal_hetero_x = treatment_residual * (cate - cal_center)
quietly regress outcome_residual cal_level_x cal_hetero_x, noconstant vce(robust)
matrix cal_b = e(b)
matrix cal_v = e(V)
/* Avoid scalar names that Stata can resolve as abbreviations of cal_*_x variables. */
scalar cal_beta_level = el(cal_b, 1, 1)
scalar cal_beta_hetero = el(cal_b, 1, 2)
scalar cal_v_level = el(cal_v, 1, 1)
scalar cal_v_hetero = el(cal_v, 2, 2)
scalar cal_cov = el(cal_v, 1, 2)

forvalues group_number = 1/3 {
    generate double group_x`group_number' = treatment_residual * (group == `group_number')
}
quietly regress outcome_residual group_x1 group_x2 group_x3, noconstant vce(robust)
matrix group_b = e(b)
matrix group_v = e(V)
forvalues group_number = 1/3 {
    scalar g`group_number' = el(group_b, 1, `group_number')
    scalar g`group_number'_var = el(group_v, `group_number', `group_number')
}

scalar exp_r_loss_s = exp_r_loss[1]
scalar exp_const_loss_s = exp_const_loss[1]
scalar exp_gain_s = exp_gain[1]
scalar exp_cal_level_s = exp_cal_level[1]
scalar exp_cal_hetero_s = exp_cal_hetero[1]
scalar exp_cal_v_level_s = exp_cal_v_level[1]
scalar exp_cal_v_hetero_s = exp_cal_v_hetero[1]
scalar exp_cal_cov_s = exp_cal_cov[1]
forvalues group_number = 1/3 {
    scalar exp_g`group_number'_s = exp_g`group_number'[1]
    scalar exp_g`group_number'_var_s = exp_g`group_number'_var[1]
}

local parity_status "pass"
if abs(r_loss - exp_r_loss_s) > tolerance local parity_status "fail"
if abs(const_loss - exp_const_loss_s) > tolerance local parity_status "fail"
if abs(loss_gain - exp_gain_s) > tolerance local parity_status "fail"
if abs(cal_beta_level - exp_cal_level_s) > tolerance local parity_status "fail"
if abs(cal_beta_hetero - exp_cal_hetero_s) > tolerance local parity_status "fail"
if abs(cal_v_level - exp_cal_v_level_s) > tolerance local parity_status "fail"
if abs(cal_v_hetero - exp_cal_v_hetero_s) > tolerance local parity_status "fail"
if abs(cal_cov - exp_cal_cov_s) > tolerance local parity_status "fail"
forvalues group_number = 1/3 {
    if abs(g`group_number' - exp_g`group_number'_s) > tolerance local parity_status "fail"
    if abs(g`group_number'_var - exp_g`group_number'_var_s) > tolerance local parity_status "fail"
}

tempname results_file
file open `results_file' using "`results_path'", write text replace
file write `results_file' "contract=fixed_honest_rlearner_evaluation_hc1" _n
file write `results_file' "stata_version=`c(stata_version)'" _n
file write `results_file' "stata_flavor=`c(flavor)'" _n
file write `results_file' "tolerance=" %21.17g (tolerance) _n
file write `results_file' "honest_r_loss=" %21.17g (r_loss) _n
file write `results_file' "honest_constant_r_loss=" %21.17g (const_loss) _n
file write `results_file' "r_loss_gain=" %21.17g (loss_gain) _n
file write `results_file' "calibration_center=" %21.17g (cal_center) _n
file write `results_file' "calibration_level=" %21.17g (cal_beta_level) _n
file write `results_file' "calibration_heterogeneity=" %21.17g (cal_beta_hetero) _n
file write `results_file' "calibration_level_variance=" %21.17g (cal_v_level) _n
file write `results_file' "calibration_heterogeneity_variance=" %21.17g (cal_v_hetero) _n
file write `results_file' "calibration_covariance=" %21.17g (cal_cov) _n
forvalues group_number = 1/3 {
    file write `results_file' "group_`group_number'_effect=" %21.17g (g`group_number') _n
    file write `results_file' "group_`group_number'_variance=" %21.17g (g`group_number'_var) _n
}
file write `results_file' "parity_status=`parity_status'" _n
file close `results_file'
display as result "results_file=`results_path'"
display as result "parity_status=`parity_status'"

if "`parity_status'" != "pass" {
    display as error "R-learner evaluation parity failed; inspect `results_path'."
    exit 9
}
