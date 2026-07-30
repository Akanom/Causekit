/*
Fixed-bandwidth sharp/fuzzy RD parity against official Stata rdrobust.

Run from the CauseKit repository root:

    do "benchmarks/validate_rd_stata.do"

If the precompiled Mata library cannot be loaded, pass the local site created by
prepare_rd_stata.py. The harness then compiles the pinned official Mata source
with the running Stata version inside that cache:

    do "benchmarks/validate_rd_stata.do" "C:\path\to\rdrobust-stata-11.1.0"

If rdrobust is missing or its ado/Mata files are out of sync, run this once
in a fresh Stata session and rerun the do-file:

    capture ado uninstall rdrobust
    net install rdrobust, from(https://raw.githubusercontent.com/rdpackages/rdrobust/main/stata) replace
    discard
    mata: mata clear
    mata: mata mlib index

If Stata's Java certificate store cannot validate GitHub, first run
`python benchmarks/prepare_rd_stata.py` in PowerShell. Then replace the HTTPS
source above with the hash-verified local_stata_site printed by that command.

The output file is written before assertions so CauseKit can inspect a failed
manual run without launching Stata.
*/

args rdrobust_source_site
clear all
set more off

capture which rdrobust
if _rc {
    display as error "Official rdrobust is required. Run the net install command in the header."
    exit 499
}

quietly findfile rdrobust.ado
local rdrobust_path "`r(fn)'"
local rdrobust_version_line "unknown"
tempname rdrobust_ado
file open `rdrobust_ado' using "`rdrobust_path'", read text
forvalues line_number = 1/12 {
    file read `rdrobust_ado' ado_line
    if strpos(strtrim("`ado_line'"), "*! version") == 1 {
        local rdrobust_version_line = strtrim("`ado_line'")
    }
}
file close `rdrobust_ado'

/* A stale in-memory Mata index can combine a new ado with an old mlib. */
mata: mata clear
quietly mata: mata mlib index
capture mata: mata describe rdrobust_vce_qq_cluster()
if _rc & `"`rdrobust_source_site'"' != "" {
    local parity_working_directory `"`c(pwd)'"'
    local rdrobust_compile_directory `"`rdrobust_source_site'\compiled"'
    capture mkdir `"`rdrobust_compile_directory'"'
    capture quietly cd `"`rdrobust_compile_directory'"'
    if _rc {
        display as error "The supplied rdrobust compile directory cannot be opened."
        exit 601
    }
    capture noisily do `"`rdrobust_source_site'\rdrobust_functions.do"'
    local rdrobust_compile_rc = _rc
    quietly cd `"`parity_working_directory'"'
    if `rdrobust_compile_rc' {
        display as error "Compiling the pinned official rdrobust Mata source failed."
        exit `rdrobust_compile_rc'
    }
    capture mata: mata describe rdrobust_vce_qq_cluster()
}
if _rc {
    display as error "The official rdrobust Mata library is missing or stale."
    display as error "Run prepare_rd_stata.py and pass its local site as the do-file argument."
    exit 499
}

set obs 800
generate long position = mod(_n - 1, 400)
generate byte right = _n > 400
generate double running = cond(right == 0, ///
    -2 + position * (1.98 / 399), 0.02 + position * (1.98 / 399))
generate byte assignment = running >= 0
generate double sharp_outcome = 0.5 + 0.7 * running + 0.3 * running^2 + ///
    1.8 * assignment + 0.4 * sin(17 * running) + 0.2 * cos(31 * running)
generate byte treatment = cond(right == 0, mod(position, 5) == 0, mod(position, 5) <= 2)
generate double fuzzy_outcome = 0.5 + 0.4 * running + 0.2 * running^2 + ///
    2.2 * treatment + 0.35 * sin(13 * running) + 0.15 * cos(29 * running)

quietly rdrobust sharp_outcome running, c(0) h(1.2 1.4) b(1.6 1.7) ///
    p(1) q(2) kernel(triangular) vce(hc1) stdvars(off) precision(double)
scalar obs_sharp_cl = e(tau_cl)
scalar obs_sharp_bc = e(tau_bc)
scalar obs_sharp_se = e(se_tau_rb)

quietly rdrobust fuzzy_outcome running, c(0) fuzzy(treatment) ///
    h(1.2 1.4) b(1.6 1.7) p(1) q(2) kernel(triangular) vce(hc1) ///
    stdvars(off) precision(double)
scalar obs_fuzzy_cl = e(tau_cl)
scalar obs_fuzzy_bc = e(tau_bc)
scalar obs_fuzzy_se = e(se_tau_rb)
scalar obs_fuzzy_d_bc = e(tau_T_bc)

scalar exp_sharp_cl = 2.012628078790254
scalar exp_sharp_bc = 2.149509125727007
scalar exp_sharp_se = 0.06729310580994985
scalar exp_fuzzy_cl = 2.7836636780089483
scalar exp_fuzzy_bc = 3.0987154201530087
scalar exp_fuzzy_se = 0.19880345063046992
scalar exp_fuzzy_d_bc = 0.4360805850270855
scalar tolerance = 1e-8

local sharp_conventional_status = cond(abs(obs_sharp_cl - exp_sharp_cl) <= ///
    tolerance, "pass", "fail")
local sharp_bias_corrected_status = cond(abs(obs_sharp_bc - exp_sharp_bc) <= ///
    tolerance, "pass", "fail")
local sharp_robust_se_status = cond(abs(obs_sharp_se - exp_sharp_se) <= ///
    tolerance, "pass", "fail")
local fuzzy_conventional_status = cond(abs(obs_fuzzy_cl - exp_fuzzy_cl) <= ///
    tolerance, "pass", "fail")
local fuzzy_bias_corrected_status = cond(abs(obs_fuzzy_bc - exp_fuzzy_bc) <= ///
    tolerance, "pass", "fail")
local fuzzy_robust_se_status = cond(abs(obs_fuzzy_se - exp_fuzzy_se) <= ///
    tolerance, "pass", "fail")
local fuzzy_treatment_jump_status = cond(abs(obs_fuzzy_d_bc - exp_fuzzy_d_bc) <= ///
    tolerance, "pass", "fail")
local parity_status = cond("`sharp_conventional_status'" == "pass" & ///
    "`sharp_bias_corrected_status'" == "pass" & ///
    "`sharp_robust_se_status'" == "pass" & ///
    "`fuzzy_conventional_status'" == "pass" & ///
    "`fuzzy_bias_corrected_status'" == "pass" & ///
    "`fuzzy_robust_se_status'" == "pass" & ///
    "`fuzzy_treatment_jump_status'" == "pass", "pass", "fail")

local results_path "benchmarks/validate_rd_stata_output.txt"
tempname results_file
file open `results_file' using "`results_path'", write text replace
file write `results_file' "contract=fixed_bandwidth_triangular_p1_q2_hc1" _n
file write `results_file' "stata_version=`c(stata_version)'" _n
file write `results_file' "stata_flavor=`c(flavor)'" _n
file write `results_file' "rdrobust_version_line=`rdrobust_version_line'" _n
file write `results_file' "expected_official_source_version=11.1.0_22may2026" _n
file write `results_file' "stdvars=off" _n
file write `results_file' "precision=double" _n
file write `results_file' "nobs=800" _n
file write `results_file' "bandwidth_left=1.2" _n
file write `results_file' "bandwidth_right=1.4" _n
file write `results_file' "bias_bandwidth_left=1.6" _n
file write `results_file' "bias_bandwidth_right=1.7" _n
file write `results_file' "tolerance=" %21.17g (tolerance) _n
file write `results_file' "sharp_conventional_estimate=" ///
    %21.17g (obs_sharp_cl) _n
file write `results_file' "sharp_conventional_estimate_status=`sharp_conventional_status'" _n
file write `results_file' "sharp_bias_corrected_estimate=" ///
    %21.17g (obs_sharp_bc) _n
file write `results_file' "sharp_bias_corrected_estimate_status=`sharp_bias_corrected_status'" _n
file write `results_file' "sharp_robust_standard_error=" ///
    %21.17g (obs_sharp_se) _n
file write `results_file' "sharp_robust_standard_error_status=`sharp_robust_se_status'" _n
file write `results_file' "fuzzy_conventional_estimate=" ///
    %21.17g (obs_fuzzy_cl) _n
file write `results_file' "fuzzy_conventional_estimate_status=`fuzzy_conventional_status'" _n
file write `results_file' "fuzzy_bias_corrected_estimate=" ///
    %21.17g (obs_fuzzy_bc) _n
file write `results_file' "fuzzy_bias_corrected_estimate_status=`fuzzy_bias_corrected_status'" _n
file write `results_file' "fuzzy_robust_standard_error=" ///
    %21.17g (obs_fuzzy_se) _n
file write `results_file' "fuzzy_robust_standard_error_status=`fuzzy_robust_se_status'" _n
file write `results_file' "fuzzy_bias_corrected_treatment_jump=" ///
    %21.17g (obs_fuzzy_d_bc) _n
file write `results_file' "fuzzy_bias_corrected_treatment_jump_status=`fuzzy_treatment_jump_status'" _n
file write `results_file' "parity_status=`parity_status'" _n
file close `results_file'

display as result "results_file=`results_path'"
display as result "parity_status=`parity_status'"
if "`parity_status'" != "pass" {
    display as error "RD parity failed; inspect `results_path'."
    exit 9
}
