version 17
clear all
set more off

/*
CauseKit partially linear DML2 residual-stage parity.

Run manually from the CauseKit repository root:
    do "benchmarks/validate_dml_stata.do"

The nuisance values are fixed out-of-fold predictions from the hand contract. Stata is
used only for the aligned no-intercept HC1 residual regression; it does not implement or
run CauseKit's cross-fitting orchestration.
*/

input double outcome_residual treatment_residual
-1.5033333333333334 -1
 1.3033333333333332  1
-3.3066666666666666 -2
 3.7066666666666666  2
-2.755              -1.5
 3.0549999999999997  1.5
-0.9516666666666667 -0.5
 0.45166666666666666 0.5
end

quietly regress outcome_residual treatment_residual, noconstant vce(robust)
matrix dml_b = e(b)
matrix dml_v = e(V)
scalar dml_est = el(dml_b, 1, 1)
scalar dml_se = sqrt(el(dml_v, 1, 1))
scalar exp_est = 1.75
scalar exp_se = 0.07410103213156319
scalar tol = 1e-10

local est_status "fail"
if abs(scalar(dml_est) - scalar(exp_est)) <= scalar(tol) {
    local est_status "pass"
}
local se_status "fail"
if abs(scalar(dml_se) - scalar(exp_se)) <= scalar(tol) {
    local se_status "pass"
}
local parity_status "fail"
if "`est_status'" == "pass" & "`se_status'" == "pass" {
    local parity_status "pass"
}

local results_path "benchmarks/validate_dml_stata_output.txt"
tempname results_file
file open `results_file' using "`results_path'", write text replace
file write `results_file' "artifact=causekit_partially_linear_dml_stata_parity" _n
file write `results_file' "causekit_version=0.7.0a1" _n
file write `results_file' "contract=fixed_oof_nuisance_residual_regression_hc1" _n
file write `results_file' "stata_version=`c(stata_version)'" _n
file write `results_file' "stata_flavor=`c(flavor)'" _n
file write `results_file' "nobs=" %21.17g (e(N)) _n
file write `results_file' "tolerance=" %21.17g (scalar(tol)) _n
file write `results_file' "estimate=" %21.17g (scalar(dml_est)) _n
file write `results_file' "expected_estimate=" %21.17g (scalar(exp_est)) _n
file write `results_file' "estimate_status=`est_status'" _n
file write `results_file' "standard_error=" %21.17g (scalar(dml_se)) _n
file write `results_file' "expected_standard_error=" %21.17g (scalar(exp_se)) _n
file write `results_file' "standard_error_status=`se_status'" _n
file write `results_file' "parity_status=`parity_status'" _n
file close `results_file'

display as result "results_file=`results_path'"
display as result "estimate=" %21.17g dml_est
display as result "standard_error=" %21.17g dml_se
display as result "parity_status=`parity_status'"

if "`parity_status'" != "pass" {
    display as error "DML residual-stage parity failed; inspect `results_path'."
    exit 9
}
