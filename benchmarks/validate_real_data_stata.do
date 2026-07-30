/*
Real-data Stata parity for CauseKit 0.6.0a4.

Run from the CauseKit repository root after preparing the hash-verified CSVs:

    python benchmarks/prepare_real_data.py --download

Then, inside Stata:

    do "benchmarks/validate_real_data_stata.do"

The script never launches Stata from Python. It writes all observed values and statuses
to benchmarks/validate_real_data_stata_output.txt before asserting, so a failed run is
diagnosable by CauseKit without access to Stata.

Native Stata commands certify IV2SLS, randomized HC1 regression, and fixed-score nearest-
neighbor matching. Supplied-nuisance IPW/AIPW and conventional group-time DiD use their
declared influence equations because Stata's teffects first-step correction and
didregress aggregation target different uncertainty/aggregation contracts. No aligned
Stata implementation of Chen-Sant'Anna-Xie efficient DiD is claimed.
*/

version 17
clear all
set more off

local local_app_data : environment LOCALAPPDATA
/* Forward slashes avoid Stata treating a Windows backslash before a macro as
   an escape that suppresses expansion of the following local macro. */
local data_root "`local_app_data'/causekit/parity/real_data_v1"
foreach file in hsng.csv nsw_mixtape.csv cattaneo2.csv hospdd_panel.csv {
    capture confirm file "`data_root'/`file'"
    if _rc {
        display as error "Missing verified parity file: `data_root'/`file'"
        display as error "Run: python benchmarks/prepare_real_data.py --download"
        exit 601
    }
}

scalar linear_tol = 0.000002
scalar nuisance_tol = 0.0005
scalar matching_tol = 0.0000001
scalar did_tol = 0.0000000001

/* ------------------------------------------------------------------------- */
/* Real IV2SLS: 1980 Census state housing data. */
import delimited using "`data_root'/hsng.csv", clear varnames(1)
generate byte reg2 = region == 2
generate byte reg3 = region == 3
generate byte reg4 = region == 4
/* CauseKit's HC1 kernel applies n/(n-k); Stata's small option requests that
   finite-sample scaling for robust IV2SLS covariance. */
quietly ivregress 2sls rent pcturban (hsngval = faminc reg2 reg3 reg4), ///
    vce(robust) small
scalar iv_const = _b[_cons]
scalar iv_const_se = _se[_cons]
scalar iv_pcturban = _b[pcturban]
scalar iv_pcturban_se = _se[pcturban]
scalar iv_hsngval = _b[hsngval]
scalar iv_hsngval_se = _se[hsngval]

scalar exp_iv_const = 120.70651454296397
scalar exp_iv_const_se = 15.734805096771982
scalar exp_iv_pcturban = 0.08151596812623473
scalar exp_iv_pcturban_se = 0.45856387111843344
scalar exp_iv_hsngval = 0.00223983298436843
scalar exp_iv_hsngval_se = 0.000693117708017762

scalar iv_max_diff = max( ///
    abs(iv_const - exp_iv_const), ///
    abs(iv_const_se - exp_iv_const_se), ///
    abs(iv_pcturban - exp_iv_pcturban), ///
    abs(iv_pcturban_se - exp_iv_pcturban_se), ///
    abs(iv_hsngval - exp_iv_hsngval), ///
    abs(iv_hsngval_se - exp_iv_hsngval_se) ///
)
local iv_status "pass"
if iv_max_diff > linear_tol local iv_status "fail"

/* ------------------------------------------------------------------------- */
/* Real randomized ATE: National Supported Work experiment. */
import delimited using "`data_root'/nsw_mixtape.csv", clear varnames(1)
quietly regress re78 treat, vce(robust)
scalar rand_raw = _b[treat]
scalar rand_raw_se = _se[treat]

foreach variable in age educ black hisp marr nodegree re74 re75 {
    quietly summarize `variable', meanonly
    generate double c_`variable' = `variable' - r(mean)
    generate double t_`variable' = treat * c_`variable'
}
quietly regress re78 treat c_age c_educ c_black c_hisp c_marr c_nodegree ///
    c_re74 c_re75 t_age t_educ t_black t_hisp t_marr t_nodegree t_re74 t_re75, ///
    vce(robust)
scalar rand_lin = _b[treat]
scalar rand_lin_se = _se[treat]

scalar exp_rand_raw = 1794.3423818501026
scalar exp_rand_raw_se = 670.8244907669412
scalar exp_rand_lin = 1621.5830818957436
scalar exp_rand_lin_se = 689.3676649693389
scalar rand_max_diff = max( ///
    abs(rand_raw - exp_rand_raw), ///
    abs(rand_raw_se - exp_rand_raw_se), ///
    abs(rand_lin - exp_rand_lin), ///
    abs(rand_lin_se - exp_rand_lin_se) ///
)
local randomized_status "pass"
if rand_max_diff > linear_tol local randomized_status "fail"

/* ------------------------------------------------------------------------- */
/* Real supplied-nuisance IPW/AIPW and fixed-score matching: Cattaneo data. */
import delimited using "`data_root'/cattaneo2.csv", clear varnames(1)
sort source_row
quietly logit mbsmoke mmarried mage medu fbaby, iterate(200)
predict double propensity, pr
quietly regress bweight mmarried mage medu fbaby if mbsmoke == 0
predict double mu0, xb
quietly regress bweight mmarried mage medu fbaby if mbsmoke == 1
predict double mu1, xb
scalar obs_n = _N

/* IPW ATE. */
generate double ipw_ate_s = mbsmoke * bweight / propensity - ///
    (1 - mbsmoke) * bweight / (1 - propensity)
quietly summarize ipw_ate_s, meanonly
scalar ipw_ate = r(mean)
/* scalar() prevents Stata variable-abbreviation lookup from resolving ipw_ate
   to the existing ipw_ate_s variable. */
generate double ipw_ate_i = ipw_ate_s - scalar(ipw_ate)
generate double ipw_ate_i2 = ipw_ate_i^2
quietly summarize ipw_ate_i2, meanonly
scalar ipw_ate_se = sqrt(r(sum) / (obs_n * (obs_n - 1)))

/* Shared normalized weights and outcome totals for ATT/ATC. */
generate double wt_att = mbsmoke
generate double wc_att = (1 - mbsmoke) * propensity / (1 - propensity)
generate double wt_atc = mbsmoke * (1 - propensity) / propensity
generate double wc_atc = 1 - mbsmoke
foreach role in wt_att wc_att wt_atc wc_atc {
    generate double `role'_y = `role' * bweight
    quietly summarize `role', meanonly
    scalar sum_`role' = r(sum)
    quietly summarize `role'_y, meanonly
    scalar mean_`role' = r(sum) / sum_`role'
}

/* IPW ATT. */
scalar ipw_att = mean_wt_att - mean_wc_att
generate double ipw_att_i = ///
    wt_att * (bweight - mean_wt_att) / (sum_wt_att / obs_n) - ///
    wc_att * (bweight - mean_wc_att) / (sum_wc_att / obs_n)
generate double ipw_att_i2 = ipw_att_i^2
quietly summarize ipw_att_i2, meanonly
scalar ipw_att_se = sqrt(r(sum) / (obs_n * (obs_n - 1)))

/* IPW ATC. */
scalar ipw_atc = mean_wt_atc - mean_wc_atc
generate double ipw_atc_i = ///
    wt_atc * (bweight - mean_wt_atc) / (sum_wt_atc / obs_n) - ///
    wc_atc * (bweight - mean_wc_atc) / (sum_wc_atc / obs_n)
generate double ipw_atc_i2 = ipw_atc_i^2
quietly summarize ipw_atc_i2, meanonly
scalar ipw_atc_se = sqrt(r(sum) / (obs_n * (obs_n - 1)))

/* AIPW ATE. */
generate double aipw_ate_s = mu1 - mu0 + ///
    mbsmoke * (bweight - mu1) / propensity - ///
    (1 - mbsmoke) * (bweight - mu0) / (1 - propensity)
quietly summarize aipw_ate_s, meanonly
scalar aipw_ate = r(mean)
generate double aipw_ate_i = aipw_ate_s - scalar(aipw_ate)
generate double aipw_ate_i2 = aipw_ate_i^2
quietly summarize aipw_ate_i2, meanonly
scalar aipw_ate_se = sqrt(r(sum) / (obs_n * (obs_n - 1)))

/* AIPW ATT. */
generate double aipw_att_num = mbsmoke * (bweight - mu0) - ///
    (1 - mbsmoke) * propensity / (1 - propensity) * (bweight - mu0)
quietly summarize aipw_att_num, meanonly
scalar aipw_att = r(mean) / (sum_wt_att / obs_n)
generate double aipw_att_i = ///
    (aipw_att_num - scalar(aipw_att) * mbsmoke) / (sum_wt_att / obs_n)
generate double aipw_att_i2 = aipw_att_i^2
quietly summarize aipw_att_i2, meanonly
scalar aipw_att_se = sqrt(r(sum) / (obs_n * (obs_n - 1)))

/* AIPW ATC. */
generate double aipw_atc_num = ///
    mbsmoke * (1 - propensity) / propensity * (bweight - mu1) - ///
    (1 - mbsmoke) * (bweight - mu1)
quietly summarize aipw_atc_num, meanonly
scalar aipw_atc = r(mean) / (sum_wc_atc / obs_n)
generate double aipw_atc_i = ///
    (aipw_atc_num - scalar(aipw_atc) * (1 - mbsmoke)) / (sum_wc_atc / obs_n)
generate double aipw_atc_i2 = aipw_atc_i^2
quietly summarize aipw_atc_i2, meanonly
scalar aipw_atc_se = sqrt(r(sum) / (obs_n * (obs_n - 1)))

scalar exp_ipw_ate = -343.29412016382497
scalar exp_ipw_ate_se = 134.45000024415245
scalar exp_ipw_att = -220.78871551375187
scalar exp_ipw_att_se = 23.32735352439394
scalar exp_ipw_atc = -234.60863252983836
scalar exp_ipw_atc_se = 25.222067085426
scalar exp_aipw_ate = -232.19435978296843
scalar exp_aipw_ate_se = 23.274103869448513
scalar exp_aipw_att = -224.70439780218382
scalar exp_aipw_att_se = 23.458642849537455
scalar exp_aipw_atc = -233.90725738789118
scalar exp_aipw_atc_se = 24.22118869867583

scalar obs_max_diff = max( ///
    abs(ipw_ate - exp_ipw_ate), abs(ipw_ate_se - exp_ipw_ate_se), ///
    abs(ipw_att - exp_ipw_att), abs(ipw_att_se - exp_ipw_att_se), ///
    abs(ipw_atc - exp_ipw_atc), abs(ipw_atc_se - exp_ipw_atc_se), ///
    abs(aipw_ate - exp_aipw_ate), abs(aipw_ate_se - exp_aipw_ate_se), ///
    abs(aipw_att - exp_aipw_att), abs(aipw_att_se - exp_aipw_att_se), ///
    abs(aipw_atc - exp_aipw_atc), abs(aipw_atc_se - exp_aipw_atc_se) ///
)
local observational_status "pass"
if obs_max_diff > nuisance_tol local observational_status "fail"

/* Deterministic 200-treated/400-control real-data matching subset. */
generate long treated_rank = sum(mbsmoke == 1)
generate long control_rank = sum(mbsmoke == 0)
keep if (mbsmoke == 1 & treated_rank <= 200) | ///
    (mbsmoke == 0 & control_rank <= 400)
quietly teffects nnmatch (bweight propensity) (mbsmoke), atet ///
    nneighbor(1) metric(euclidean) vce(robust, nn(2))
matrix match_b_att = e(b)
scalar match_att = el(match_b_att, 1, 1)
quietly teffects nnmatch (bweight propensity) (mbsmoke), ate ///
    nneighbor(1) metric(euclidean) vce(robust, nn(2))
matrix match_b_ate = e(b)
scalar match_ate = el(match_b_ate, 1, 1)
generate byte mbsmoke_rev = 1 - mbsmoke
quietly teffects nnmatch (bweight propensity) (mbsmoke_rev), atet ///
    nneighbor(1) metric(euclidean) vce(robust, nn(2))
matrix match_b_atc_rev = e(b)
scalar match_atc = -el(match_b_atc_rev, 1, 1)

scalar exp_match_att = -340.1200932539682
scalar exp_match_ate = -278.8296421957672
scalar exp_match_atc = -248.18441666666666
scalar match_max_diff = max( ///
    abs(match_att - exp_match_att), ///
    abs(match_ate - exp_match_ate), ///
    abs(match_atc - exp_match_atc) ///
)
local matching_status "pass"
if match_max_diff > matching_tol local matching_status "fail"

/* ------------------------------------------------------------------------- */
/* Real conventional group-time DiD: hospital procedure adoption. */
import delimited using "`data_root'/hospdd_panel.csv", clear varnames(1)
reshape wide outcome treated, i(hospital treatment_time) j(month)
generate byte did_treated = treatment_time == 4
generate byte did_control = treatment_time == 0
quietly summarize did_treated, meanonly
scalar did_pt = r(mean)
quietly summarize did_control, meanonly
scalar did_pc = r(mean)

forvalues period = 4/7 {
    generate double did_ch`period' = outcome`period' - outcome3
    quietly summarize did_ch`period' if did_treated == 1, meanonly
    scalar did_mt`period' = r(mean)
    quietly summarize did_ch`period' if did_control == 1, meanonly
    scalar did_mc`period' = r(mean)
    scalar did_a`period' = did_mt`period' - did_mc`period'
    generate double did_i`period' = ///
        did_treated / did_pt * (did_ch`period' - did_mt`period') - ///
        did_control / did_pc * (did_ch`period' - did_mc`period')
    generate double did_i2_`period' = did_i`period'^2
    quietly summarize did_i2_`period', meanonly
    scalar did_s`period' = sqrt(r(sum) / (_N * (_N - 1)))
}
scalar did_estimate = (did_a4 + did_a5 + did_a6 + did_a7) / 4
generate double did_overall_i = (did_i4 + did_i5 + did_i6 + did_i7) / 4
generate double did_overall_i2 = did_overall_i^2
quietly summarize did_overall_i2, meanonly
scalar did_se = sqrt(r(sum) / (_N * (_N - 1)))

scalar exp_did_estimate = 0.8615139751207261
scalar exp_did_se = 0.0468766018680398
scalar did_max_diff = max( ///
    abs(did_a4 - 0.8166715985252745), abs(did_s4 - 0.05110555841205596), ///
    abs(did_a5 - 0.9035346489104013), abs(did_s5 - 0.04778500634961714), ///
    abs(did_a6 - 0.8342092614325267), abs(did_s6 - 0.0656803669073146), ///
    abs(did_a7 - 0.891640391614702), abs(did_s7 - 0.05697409651031751), ///
    abs(did_estimate - exp_did_estimate), abs(did_se - exp_did_se) ///
)
local did_status "pass"
if did_max_diff > did_tol local did_status "fail"

/* ------------------------------------------------------------------------- */
local parity_status "pass"
if "`iv_status'" != "pass" local parity_status "fail"
if "`randomized_status'" != "pass" local parity_status "fail"
if "`observational_status'" != "pass" local parity_status "fail"
if "`matching_status'" != "pass" local parity_status "fail"
if "`did_status'" != "pass" local parity_status "fail"

local results_path "benchmarks/validate_real_data_stata_output.txt"
tempname results_file
file open `results_file' using "`results_path'", write text replace
file write `results_file' "artifact=causekit_real_data_stata_parity" _n
file write `results_file' "causekit_version=0.6.0a4" _n
file write `results_file' "stata_version=`c(stata_version)'" _n
file write `results_file' "stata_flavor=`c(flavor)'" _n
file write `results_file' "data_schema=causekit_real_data_parity_v1" _n
file write `results_file' "hsng_source_sha256=d19cd25299af57569d93d8f16b4f72d5ffc7f897c9247d8a8e5fddddef43ad11" _n
file write `results_file' "nsw_source_sha256=fc424cfc9d7861f4b95a6612f27c7e842671fea5a8612edcfe0273ee62e6f0a4" _n
file write `results_file' "cattaneo_source_sha256=631e926eb9981828ba2e542b32c16ae08f336b9efa10621651a8a185405e0577" _n
file write `results_file' "hospdd_source_sha256=e3ae6451e89cb915c546ab772410046726f280ad7d117611376beb4f46a521bb" _n
file write `results_file' "linear_tolerance=" %21.17g (linear_tol) _n
file write `results_file' "nuisance_tolerance=" %21.17g (nuisance_tol) _n
file write `results_file' "matching_tolerance=" %21.17g (matching_tol) _n
file write `results_file' "did_tolerance=" %21.17g (did_tol) _n
file write `results_file' "iv_const_estimate=" %21.17g (iv_const) _n
file write `results_file' "iv_const_standard_error=" %21.17g (iv_const_se) _n
file write `results_file' "iv_pcturban_estimate=" %21.17g (iv_pcturban) _n
file write `results_file' "iv_pcturban_standard_error=" %21.17g (iv_pcturban_se) _n
file write `results_file' "iv_hsngval_estimate=" %21.17g (iv_hsngval) _n
file write `results_file' "iv_hsngval_standard_error=" %21.17g (iv_hsngval_se) _n
file write `results_file' "iv_maximum_absolute_difference=" %21.17g (iv_max_diff) _n
file write `results_file' "iv_status=`iv_status'" _n
file write `results_file' "randomized_raw_estimate=" %21.17g (rand_raw) _n
file write `results_file' "randomized_raw_standard_error=" %21.17g (rand_raw_se) _n
file write `results_file' "randomized_lin_estimate=" %21.17g (rand_lin) _n
file write `results_file' "randomized_lin_standard_error=" %21.17g (rand_lin_se) _n
file write `results_file' "randomized_maximum_absolute_difference=" %21.17g (rand_max_diff) _n
file write `results_file' "randomized_status=`randomized_status'" _n
foreach estimator in ipw aipw {
    foreach estimand in ate att atc {
        file write `results_file' "`estimator'_`estimand'_estimate=" ///
            %21.17g (`estimator'_`estimand') _n
        file write `results_file' "`estimator'_`estimand'_standard_error=" ///
            %21.17g (`estimator'_`estimand'_se) _n
    }
}
file write `results_file' "observational_maximum_absolute_difference=" %21.17g (obs_max_diff) _n
file write `results_file' "observational_status=`observational_status'" _n
file write `results_file' "matching_att_estimate=" %21.17g (match_att) _n
file write `results_file' "matching_ate_estimate=" %21.17g (match_ate) _n
file write `results_file' "matching_atc_estimate=" %21.17g (match_atc) _n
file write `results_file' "matching_maximum_absolute_difference=" %21.17g (match_max_diff) _n
file write `results_file' "matching_status=`matching_status'" _n
file write `results_file' "did_group_time_estimates=" ///
    %21.17g (did_a4) "," %21.17g (did_a5) "," %21.17g (did_a6) "," %21.17g (did_a7) _n
file write `results_file' "did_group_time_standard_errors=" ///
    %21.17g (did_s4) "," %21.17g (did_s5) "," %21.17g (did_s6) "," %21.17g (did_s7) _n
file write `results_file' "did_esavg_estimate=" %21.17g (did_estimate) _n
file write `results_file' "did_esavg_standard_error=" %21.17g (did_se) _n
file write `results_file' "did_maximum_absolute_difference=" %21.17g (did_max_diff) _n
file write `results_file' "did_status=`did_status'" _n
file write `results_file' "efficient_did_stata_status=unavailable_no_aligned_estimator" _n
file write `results_file' "parity_status=`parity_status'" _n
file close `results_file'

display as result "results_file=`results_path'"
display as result "iv_status=`iv_status'"
display as result "randomized_status=`randomized_status'"
display as result "observational_status=`observational_status'"
display as result "matching_status=`matching_status'"
display as result "did_status=`did_status'"
display as result "efficient_did_stata_status=unavailable_no_aligned_estimator"
display as result "parity_status=`parity_status'"

if "`parity_status'" != "pass" {
    display as error "Real-data parity failed; inspect `results_path'."
    exit 9
}

exit 0
