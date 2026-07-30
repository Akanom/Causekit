version 17
clear all
set more off

local input_path "benchmarks/drlearner_stata_input.csv"
local results_path "benchmarks/validate_drlearner_stata_output.txt"
local tolerance = 1e-8

import delimited using "`input_path'", clear varnames(1) asdouble

quietly summarize dr_score
scalar nobs = r(N)
generate double dr_error = dr_score - cate
generate double const_error = dr_score - constant_effect
quietly summarize dr_error
scalar dr_loss = r(Var) * (r(N) - 1) / r(N) + r(mean)^2
quietly summarize const_error
scalar const_loss = r(Var) * (r(N) - 1) / r(N) + r(mean)^2
scalar loss_gain = 1 - dr_loss / const_loss

quietly summarize cate
scalar cal_center = r(mean)
generate double cal_level_x = 1
generate double cal_hetero_x = cate - cal_center
quietly regress dr_score cal_level_x cal_hetero_x, noconstant vce(robust)
matrix cal_b = e(b)
matrix cal_v = e(V)
scalar cal_beta_level = el(cal_b, 1, 1)
scalar cal_beta_hetero = el(cal_b, 1, 2)
scalar cal_var_level = el(cal_v, 1, 1)
scalar cal_var_hetero = el(cal_v, 2, 2)
scalar cal_cov = el(cal_v, 1, 2)

generate double group_x1 = group == 1
generate double group_x2 = group == 2
generate double group_x3 = group == 3
quietly regress dr_score group_x1 group_x2 group_x3, noconstant vce(robust)
matrix group_b = e(b)
matrix group_v = e(V)

scalar expected_dr_loss = exp_dr_loss[1]
scalar expected_const_loss = exp_const_loss[1]
scalar expected_gain = exp_gain[1]
scalar expected_cal_center = exp_cal_center[1]
scalar expected_cal_level = exp_cal_level[1]
scalar expected_cal_hetero = exp_cal_hetero[1]
scalar expected_cal_v_level = exp_cal_v_level[1]
scalar expected_cal_v_hetero = exp_cal_v_hetero[1]
scalar expected_cal_cov = exp_cal_cov[1]

local parity_status "pass"
foreach pair in dr_loss const_loss gain cal_center cal_level cal_hetero cal_v_level cal_v_hetero cal_cov {
    if "`pair'" == "dr_loss" scalar observed = dr_loss
    if "`pair'" == "const_loss" scalar observed = const_loss
    if "`pair'" == "gain" scalar observed = loss_gain
    if "`pair'" == "cal_center" scalar observed = cal_center
    if "`pair'" == "cal_level" scalar observed = cal_beta_level
    if "`pair'" == "cal_hetero" scalar observed = cal_beta_hetero
    if "`pair'" == "cal_v_level" scalar observed = cal_var_level
    if "`pair'" == "cal_v_hetero" scalar observed = cal_var_hetero
    if "`pair'" == "cal_cov" scalar observed = cal_cov
    scalar expected = expected_`pair'
    if abs(observed - expected) > `tolerance' local parity_status "fail"
}
forvalues group_number = 1/3 {
    scalar observed_group = el(group_b, 1, `group_number')
    scalar expected_group = exp_g`group_number'[1]
    scalar observed_group_var = el(group_v, `group_number', `group_number')
    scalar expected_group_var = exp_g`group_number'_var[1]
    if abs(observed_group - expected_group) > `tolerance' local parity_status "fail"
    if abs(observed_group_var - expected_group_var) > `tolerance' local parity_status "fail"
}

tempname results_file
file open `results_file' using "`results_path'", write text replace
file write `results_file' "contract=fixed_honest_drlearner_evaluation_hc1" _n
file write `results_file' "stata_version=`c(stata_version)'" _n
file write `results_file' "stata_flavor=`c(flavor)'" _n
file write `results_file' "tolerance=" %21.17g (`tolerance') _n
file write `results_file' "honest_dr_loss=" %21.17g (dr_loss) _n
file write `results_file' "honest_constant_dr_loss=" %21.17g (const_loss) _n
file write `results_file' "dr_loss_gain=" %21.17g (loss_gain) _n
file write `results_file' "calibration_center=" %21.17g (cal_center) _n
file write `results_file' "calibration_level=" %21.17g (cal_beta_level) _n
file write `results_file' "calibration_heterogeneity=" %21.17g (cal_beta_hetero) _n
file write `results_file' "calibration_level_variance=" %21.17g (cal_var_level) _n
file write `results_file' "calibration_heterogeneity_variance=" %21.17g (cal_var_hetero) _n
file write `results_file' "calibration_covariance=" %21.17g (cal_cov) _n
forvalues group_number = 1/3 {
    file write `results_file' "group_`group_number'_effect=" %21.17g (el(group_b, 1, `group_number')) _n
    file write `results_file' "group_`group_number'_variance=" %21.17g (el(group_v, `group_number', `group_number')) _n
}
file write `results_file' "parity_status=`parity_status'" _n
file close `results_file'

display as result "results_file=`results_path'"
display as result "parity_status=`parity_status'"
if "`parity_status'" != "pass" {
    display as error "DR-learner evaluation parity failed; inspect `results_path'."
    exit 9
}
