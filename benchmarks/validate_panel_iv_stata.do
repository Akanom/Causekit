version 17
clear all
set more off
set type double

/*
Run the verified fixture preparer from the CauseKit repository first:

    python benchmarks/prepare_panel_iv_stata.py --download

Then, from the repository root in Stata:

    do "benchmarks/validate_panel_iv_stata.do"

This harness uses official ivregress 2sls with explicit entity/time indicators. The
CauseKit runtime absorbs those effects compactly; the slope and CR1 covariance contracts
are algebraically identical without materializing the dummy matrix in production.
*/

local cache_root : environment LOCALAPPDATA
if `"`cache_root'"' == "" {
    local cache_root : environment TEMP
}
local data_path `"`cache_root'/causekit/parity/panel_iv_wage_v1.csv"'
capture confirm file `"`data_path'"'
if _rc {
    display as error "Missing verified Panel IV fixture: `data_path'"
    display as error "Run: python benchmarks/prepare_panel_iv_stata.py --download"
    exit 601
}

quietly import delimited using `"`data_path'"', clear varnames(1)
quietly count
if r(N) != 3815 {
    display as error "Panel IV fixture row count changed; expected 3815."
    exit 459
}
quietly levelsof nr, local(entity_levels)
local n_entities : word count `entity_levels'
if `n_entities' != 545 {
    display as error "Panel IV fixture entity count changed; expected 545."
    exit 459
}
quietly levelsof year, local(time_levels)
local n_periods : word count `time_levels'
if `n_periods' != 7 {
    display as error "Panel IV fixture period count changed; expected 7."
    exit 459
}
isid nr year

/* Exact LSDV reference to CauseKit's compact two-way within estimator. */
quietly ivregress 2sls lwage hours_1000 married i.nr i.year ///
    (union = union_lag), vce(cluster nr) small

scalar h_est = _b[hours_1000]
scalar h_se = _se[hours_1000]
scalar m_est = _b[married]
scalar m_se = _se[married]
scalar u_est = _b[union]
scalar u_se = _se[union]
scalar expected_h_est = -0.15962613778805484
scalar expected_h_se = 0.02416893230287323
scalar expected_m_est = 0.06533125127699804
scalar expected_m_se = 0.02417901182155533
scalar expected_u_est = 0.24962753388340345
scalar expected_u_se = 0.31401132417362376
scalar tolerance = 1e-8

local h_est_status = cond(abs(h_est - expected_h_est) <= tolerance, "pass", "fail")
local h_se_status = cond(abs(h_se - expected_h_se) <= tolerance, "pass", "fail")
local m_est_status = cond(abs(m_est - expected_m_est) <= tolerance, "pass", "fail")
local m_se_status = cond(abs(m_se - expected_m_se) <= tolerance, "pass", "fail")
local u_est_status = cond(abs(u_est - expected_u_est) <= tolerance, "pass", "fail")
local u_se_status = cond(abs(u_se - expected_u_se) <= tolerance, "pass", "fail")
local parity_status = "pass"
foreach status in h_est_status h_se_status m_est_status m_se_status u_est_status u_se_status {
    if "``status''" != "pass" {
        local parity_status = "fail"
    }
}

local results_path "benchmarks/validate_panel_iv_stata_output.txt"
tempname results_file
file open `results_file' using "`results_path'", write text replace
file write `results_file' "contract=wage_panel_two_way_fe_entity_clustered_cr1" _n
file write `results_file' "stata_version=`c(stata_version)'" _n
file write `results_file' "stata_flavor=`c(flavor)'" _n
file write `results_file' "estimator=official_ivregress_2sls_explicit_lsdv" _n
file write `results_file' "precision=double" _n
file write `results_file' "nobs=" %12.0f (e(N)) _n
file write `results_file' "n_entities=`n_entities'" _n
file write `results_file' "n_periods=`n_periods'" _n
file write `results_file' "tolerance=" %21.17g (tolerance) _n
file write `results_file' "hours_1000_estimate=" %21.17g (h_est) _n
file write `results_file' "hours_1000_expected_estimate=" %21.17g (expected_h_est) _n
file write `results_file' "hours_1000_estimate_absolute_difference=" ///
    %21.17g (abs(h_est - expected_h_est)) _n
file write `results_file' "hours_1000_estimate_status=`h_est_status'" _n
file write `results_file' "hours_1000_standard_error=" %21.17g (h_se) _n
file write `results_file' "hours_1000_expected_standard_error=" %21.17g (expected_h_se) _n
file write `results_file' "hours_1000_standard_error_absolute_difference=" ///
    %21.17g (abs(h_se - expected_h_se)) _n
file write `results_file' "hours_1000_standard_error_status=`h_se_status'" _n
file write `results_file' "married_estimate=" %21.17g (m_est) _n
file write `results_file' "married_expected_estimate=" %21.17g (expected_m_est) _n
file write `results_file' "married_estimate_absolute_difference=" ///
    %21.17g (abs(m_est - expected_m_est)) _n
file write `results_file' "married_estimate_status=`m_est_status'" _n
file write `results_file' "married_standard_error=" %21.17g (m_se) _n
file write `results_file' "married_expected_standard_error=" %21.17g (expected_m_se) _n
file write `results_file' "married_standard_error_absolute_difference=" ///
    %21.17g (abs(m_se - expected_m_se)) _n
file write `results_file' "married_standard_error_status=`m_se_status'" _n
file write `results_file' "union_estimate=" %21.17g (u_est) _n
file write `results_file' "union_expected_estimate=" %21.17g (expected_u_est) _n
file write `results_file' "union_estimate_absolute_difference=" ///
    %21.17g (abs(u_est - expected_u_est)) _n
file write `results_file' "union_estimate_status=`u_est_status'" _n
file write `results_file' "union_standard_error=" %21.17g (u_se) _n
file write `results_file' "union_expected_standard_error=" %21.17g (expected_u_se) _n
file write `results_file' "union_standard_error_absolute_difference=" ///
    %21.17g (abs(u_se - expected_u_se)) _n
file write `results_file' "union_standard_error_status=`u_se_status'" _n
file write `results_file' "parity_status=`parity_status'" _n
file close `results_file'

display as result "results_file=`results_path'"
display as result "parity_status=`parity_status'"
if "`parity_status'" != "pass" {
    display as error "Panel IV Stata parity failed; inspect `results_path'."
    exit 9
}
