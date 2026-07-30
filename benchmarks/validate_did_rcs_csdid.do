/* Estimator-level repeated-cross-section parity against Stata csdid.

Run manually from the CauseKit repository root:

    do "benchmarks/validate_did_rcs_csdid.do"

The output is always written before the final assertion.  Group-time standard
errors are accepted only when they match either did 2.5.0's analytical convention
or the explicit n/(n-1) HC1 mapping used by CauseKit.  Aggregate standard errors
remain diagnostic because csdid estimates period-specific cell-share influence,
whereas CauseKit and did use pooled cohort-share influence.
*/

version 17
clear all
set more off

local results_path "benchmarks/validate_did_rcs_csdid_output.txt"
tempname results_file

capture which csdid
if _rc {
    file open `results_file' using "`results_path'", write text replace
    file write `results_file' "contract=csdid_repeated_cross_section_reg_long2" _n
    file write `results_file' "stata_version=`c(stata_version)'" _n
    file write `results_file' "parity_status=missing_csdid" _n
    file close `results_file'
    display as error "csdid is unavailable. Install/update csdid and drdid, then rerun."
    exit 199
}
capture which drdid
if _rc {
    file open `results_file' using "`results_path'", write text replace
    file write `results_file' "contract=csdid_repeated_cross_section_reg_long2" _n
    file write `results_file' "stata_version=`c(stata_version)'" _n
    file write `results_file' "parity_status=missing_drdid" _n
    file close `results_file'
    display as error "drdid is unavailable. Install/update drdid, then rerun."
    exit 199
}
import delimited using "benchmarks/did_rcs_parity_input.csv", clear varnames(1)
expand 3

/* Never-treated controls; omitting ivar() requests repeated cross sections. */
quietly csdid outcome, time(time) gvar(treatment_time) method(reg) long2
matrix never_attgt_b = e(b_attgt)
matrix never_attgt_V = e(V_attgt)

scalar n_g22_col = colnumb(never_attgt_b, "g2:t_1_2")
scalar n_g23_col = colnumb(never_attgt_b, "g2:t_1_3")
scalar n_g33_col = colnumb(never_attgt_b, "g3:t_2_3")
scalar n_g22_b = el(never_attgt_b, 1, n_g22_col)
scalar n_g23_b = el(never_attgt_b, 1, n_g23_col)
scalar n_g33_b = el(never_attgt_b, 1, n_g33_col)
scalar n_g22_se = sqrt(el(never_attgt_V, n_g22_col, n_g22_col))
scalar n_g23_se = sqrt(el(never_attgt_V, n_g23_col, n_g23_col))
scalar n_g33_se = sqrt(el(never_attgt_V, n_g33_col, n_g33_col))

quietly estat event
matrix never_event_b = r(b)
matrix never_event_V = r(V)
scalar n_event0_col = colnumb(never_event_b, "Tp0")
scalar n_event1_col = colnumb(never_event_b, "Tp1")
scalar n_esavg_col = colnumb(never_event_b, "Post_avg")
scalar n_event0_b = el(never_event_b, 1, n_event0_col)
scalar n_event1_b = el(never_event_b, 1, n_event1_col)
scalar n_esavg_b = el(never_event_b, 1, n_esavg_col)
scalar n_event0_se = sqrt(el(never_event_V, n_event0_col, n_event0_col))
scalar n_event1_se = sqrt(el(never_event_V, n_event1_col, n_event1_col))
scalar n_esavg_se = sqrt(el(never_event_V, n_esavg_col, n_esavg_col))

quietly estat calendar
matrix never_calendar_b = r(b)
matrix never_calendar_V = r(V)
scalar n_calendar2_col = colnumb(never_calendar_b, "T2")
scalar n_calendar3_col = colnumb(never_calendar_b, "T3")
scalar n_calendar2_b = el(never_calendar_b, 1, n_calendar2_col)
scalar n_calendar3_b = el(never_calendar_b, 1, n_calendar3_col)
scalar n_calendar2_se = sqrt(el(never_calendar_V, n_calendar2_col, n_calendar2_col))
scalar n_calendar3_se = sqrt(el(never_calendar_V, n_calendar3_col, n_calendar3_col))

/* Repeat with not-yet-treated cohorts added to each admissible comparison. */
quietly csdid outcome, time(time) gvar(treatment_time) method(reg) long2 notyet
matrix notyet_attgt_b = e(b_attgt)
matrix notyet_attgt_V = e(V_attgt)

scalar y_g22_col = colnumb(notyet_attgt_b, "g2:t_1_2")
scalar y_g23_col = colnumb(notyet_attgt_b, "g2:t_1_3")
scalar y_g33_col = colnumb(notyet_attgt_b, "g3:t_2_3")
scalar y_g22_b = el(notyet_attgt_b, 1, y_g22_col)
scalar y_g23_b = el(notyet_attgt_b, 1, y_g23_col)
scalar y_g33_b = el(notyet_attgt_b, 1, y_g33_col)
scalar y_g22_se = sqrt(el(notyet_attgt_V, y_g22_col, y_g22_col))
scalar y_g23_se = sqrt(el(notyet_attgt_V, y_g23_col, y_g23_col))
scalar y_g33_se = sqrt(el(notyet_attgt_V, y_g33_col, y_g33_col))

quietly estat event
matrix notyet_event_b = r(b)
matrix notyet_event_V = r(V)
scalar y_event0_col = colnumb(notyet_event_b, "Tp0")
scalar y_event1_col = colnumb(notyet_event_b, "Tp1")
scalar y_esavg_col = colnumb(notyet_event_b, "Post_avg")
scalar y_event0_b = el(notyet_event_b, 1, y_event0_col)
scalar y_event1_b = el(notyet_event_b, 1, y_event1_col)
scalar y_esavg_b = el(notyet_event_b, 1, y_esavg_col)
scalar y_event0_se = sqrt(el(notyet_event_V, y_event0_col, y_event0_col))
scalar y_event1_se = sqrt(el(notyet_event_V, y_event1_col, y_event1_col))
scalar y_esavg_se = sqrt(el(notyet_event_V, y_esavg_col, y_esavg_col))

quietly estat calendar
matrix notyet_calendar_b = r(b)
matrix notyet_calendar_V = r(V)
scalar y_calendar2_col = colnumb(notyet_calendar_b, "T2")
scalar y_calendar3_col = colnumb(notyet_calendar_b, "T3")
scalar y_calendar2_b = el(notyet_calendar_b, 1, y_calendar2_col)
scalar y_calendar3_b = el(notyet_calendar_b, 1, y_calendar3_col)
scalar y_calendar2_se = sqrt(el(notyet_calendar_V, y_calendar2_col, y_calendar2_col))
scalar y_calendar3_se = sqrt(el(notyet_calendar_V, y_calendar3_col, y_calendar3_col))

matrix observed_b = ( ///
    n_g22_b, n_g23_b, n_g33_b, n_event0_b, n_event1_b, n_esavg_b, ///
    n_calendar2_b, n_calendar3_b, ///
    y_g22_b, y_g23_b, y_g33_b, y_event0_b, y_event1_b, y_esavg_b, ///
    y_calendar2_b, y_calendar3_b ///
)
matrix observed_se = ( ///
    n_g22_se, n_g23_se, n_g33_se, n_event0_se, n_event1_se, n_esavg_se, ///
    n_calendar2_se, n_calendar3_se, ///
    y_g22_se, y_g23_se, y_g33_se, y_event0_se, y_event1_se, y_esavg_se, ///
    y_calendar2_se, y_calendar3_se ///
)
matrix expected_b = (2, 4, 6, 4, 4, 4, 2, 5, 2, 4, 6, 4, 4, 4, 2, 5)
matrix expected_se_did = ( ///
    .81649658092772603, .81649658092772603, .81649658092772603, ///
    .60092521257733178, .81649658092772603, .61801654059130529, ///
    .81649658092772603, .66666666666666674, ///
    2.1602468994692869, .81649658092772603, .81649658092772603, ///
    1.2018504251546638, .81649658092772603, .79494933451412131, ///
    2.1602468994692869, .66666666666666674 ///
)
matrix expected_se_hc1 = ( ///
    .82416338369213415, .82416338369213415, .82416338369213415, ///
    .60656782662937792, .82416338369213415, .62381964011741931, ///
    .82416338369213415, .67292658491045321, ///
    2.1805313529348935, .82416338369213415, .82416338369213415, ///
    1.2131356532587561, .82416338369213415, .80241381127713762, ///
    2.1805313529348935, .67292658491045321 ///
)
matrix observed_group_time_se = ( ///
    n_g22_se, n_g23_se, n_g33_se, y_g22_se, y_g23_se, y_g33_se ///
)
matrix expected_group_time_se_did = ( ///
    .81649658092772603, .81649658092772603, .81649658092772603, ///
    2.1602468994692869, .81649658092772603, .81649658092772603 ///
)
matrix expected_group_time_se_hc1 = ( ///
    .82416338369213415, .82416338369213415, .82416338369213415, ///
    2.1805313529348935, .82416338369213415, .82416338369213415 ///
)
scalar tolerance = 1e-8

mata: st_numscalar("max_estimate_difference", ///
    max(abs(st_matrix("observed_b") :- st_matrix("expected_b"))))
mata: st_numscalar("max_all_se_did_difference", ///
    max(abs(st_matrix("observed_se") :- st_matrix("expected_se_did"))))
mata: st_numscalar("max_all_se_hc1_difference", ///
    max(abs(st_matrix("observed_se") :- st_matrix("expected_se_hc1"))))
mata: st_numscalar("max_group_se_did_difference", ///
    max(abs(st_matrix("observed_group_time_se") :- ///
    st_matrix("expected_group_time_se_did"))))
mata: st_numscalar("max_group_se_hc1_difference", ///
    max(abs(st_matrix("observed_group_time_se") :- ///
    st_matrix("expected_group_time_se_hc1"))))

local estimate_status "fail"
if max_estimate_difference <= tolerance local estimate_status "pass"

local group_time_se_status "fail"
local se_convention "mismatch"
if max_group_se_did_difference <= tolerance {
    local group_time_se_status "pass"
    local se_convention "did_2.5_analytic"
}
else if max_group_se_hc1_difference <= tolerance {
    local group_time_se_status "pass"
    local se_convention "causekit_observation_hc1"
}
local aggregate_se_status "non_comparable_period_specific_cell_share_influence"
local parity_scope "group_time_estimates_and_standard_errors_plus_aggregate_points"

local parity_status "fail"
if "`estimate_status'" == "pass" & "`group_time_se_status'" == "pass" {
    local parity_status "pass"
}

file open `results_file' using "`results_path'", write text replace
file write `results_file' "contract=csdid_repeated_cross_section_reg_long2" _n
file write `results_file' "stata_version=`c(stata_version)'" _n
file write `results_file' "stata_flavor=`c(flavor)'" _n
file write `results_file' "csdid_available=true" _n
file write `results_file' "drdid_available=true" _n
file write `results_file' "reviewed_csdid_source_commit=738defceb5413face7051754b259c82798c67e52" _n
file write `results_file' "method=reg" _n
file write `results_file' "panel=false" _n
file write `results_file' "base_period=long2" _n
file write `results_file' "source_observations=18" _n
file write `results_file' "deterministic_replications=3" _n
file write `results_file' "n_observations=" %9.0g (_N) _n
file write `results_file' "tolerance=" %21.17g (tolerance) _n
file write `results_file' "max_estimate_difference=" %21.17g (max_estimate_difference) _n
file write `results_file' "estimate_status=`estimate_status'" _n
file write `results_file' "max_all_se_did_difference=" %21.17g (max_all_se_did_difference) _n
file write `results_file' "max_all_se_hc1_difference=" %21.17g (max_all_se_hc1_difference) _n
file write `results_file' "max_group_time_se_did_difference=" ///
    %21.17g (max_group_se_did_difference) _n
file write `results_file' "max_group_time_se_hc1_difference=" ///
    %21.17g (max_group_se_hc1_difference) _n
file write `results_file' "se_convention=`se_convention'" _n
file write `results_file' "group_time_se_status=`group_time_se_status'" _n
file write `results_file' "aggregate_se_status=`aggregate_se_status'" _n
file write `results_file' "parity_scope=`parity_scope'" _n

local labels ///
    never_g2_t2 never_g2_t3 never_g3_t3 never_event0 never_event1 never_esavg ///
    never_calendar2 never_calendar3 notyet_g2_t2 notyet_g2_t3 notyet_g3_t3 ///
    notyet_event0 notyet_event1 notyet_esavg notyet_calendar2 notyet_calendar3
local position = 0
foreach label of local labels {
    local ++position
    file write `results_file' "`label'_estimate=" %21.17g (el(observed_b, 1, `position')) _n
    file write `results_file' "`label'_standard_error=" ///
        %21.17g (el(observed_se, 1, `position')) _n
}
file write `results_file' "parity_status=`parity_status'" _n
file close `results_file'

display as result "results_file=`results_path'"
display as result "estimate_status=`estimate_status'"
display as result "se_convention=`se_convention'"
display as result "group_time_se_status=`group_time_se_status'"
display as result "aggregate_se_status=`aggregate_se_status'"
display as result "parity_status=`parity_status'"

if "`parity_status'" != "pass" {
    display as error "CSDID aligned group-time parity failed; inspect `results_path'."
    exit 9
}
