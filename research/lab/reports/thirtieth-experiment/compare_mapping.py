"""Prespecified raw CT correspondence decision; no model scores or reference labels."""
import numpy as np


def compare(candidates, controls, observed, plan):
    predicted=np.asarray(candidates,dtype=np.float64)
    control=np.asarray(controls,dtype=np.float64)
    target=np.asarray(observed,dtype=np.float64)
    if predicted.shape!=(6,7,3,9) or control.shape!=predicted.shape or target.shape!=(7,3,9):
        raise ValueError('Exact frozen seven-site, three-layer, nine-pixel layout required')
    if not all(np.isfinite(a).all() for a in (predicted,control,target)):
        raise ValueError('Every selected CT pair must be finite and complete')
    error=np.abs(predicted-target)
    control_error=np.abs(control-target)
    cal=error[:,:3].mean(axis=(1,2,3))
    order=np.argsort(cal,kind='stable')
    chosen=int(order[0]); runner=int(order[1]); a=plan['acceptance']
    check_errors=error[chosen,3:]
    check_mae=float(check_errors.mean())
    wrong_mae=float(control_error[chosen,3:].mean())
    checks=dict(
        calibration_mae=float(cal[chosen])<=a['calibration_mae_max'],
        calibration_max_error=float(error[chosen,:3].max())<=a['calibration_max_error_max'],
        identified_from_calibration=float(cal[runner]-cal[chosen])>=a['runner_up_mae_gap_min'],
        every_check_site_mae=bool((check_errors.mean(axis=(1,2))<=a['each_check_site_mae_max']).all()),
        every_check_site_max_error=bool((check_errors.max(axis=(1,2))<=a['each_check_site_max_error_max']).all()),
        informative_check_sites=int((np.ptp(target[3:],axis=(1,2))>=a['informative_check_range_min']).sum())>=a['informative_check_sites_min'],
        wrong_coordinate_absolute=wrong_mae>=a['wrong_coordinate_check_mae_min'],
        wrong_coordinate_relative=wrong_mae>=a['wrong_coordinate_mae_ratio_min']*check_mae)
    recipes=[]
    for i,name in enumerate(plan['hypothesis_order']):
        recipes.append(dict(name=name,calibration_mae=float(cal[i]),
            calibration_max_error=float(error[i,:3].max()),check_mae=float(error[i,3:].mean()),
            check_max_error=float(error[i,3:].max()),check_site_mae=error[i,3:].mean(axis=(1,2)).tolist(),
            check_site_max_error=error[i,3:].max(axis=(1,2)).tolist(),
            wrong_coordinate_check_mae=float(control_error[i,3:].mean())))
    return dict(status='MAPPING_REPRODUCED_ON_FIXED_PROBES' if all(checks.values()) else 'MAPPING_NOT_QUALIFIED',
        selected_by_calibration=plan['hypothesis_order'][chosen],selected_index=chosen,
        calibration_runner_up=plan['hypothesis_order'][runner],calibration_gap=float(cal[runner]-cal[chosen]),
        selected_check_mae=check_mae,selected_check_max_error=float(check_errors.max()),
        selected_wrong_coordinate_check_mae=wrong_mae,criteria=checks,
        check_site_observed_ranges=np.ptp(target[3:],axis=(1,2)).tolist(),
        recipes=recipes,observed_site_layer_ranges=np.ptp(target,axis=2).tolist(),
        evaluated_CT_pairs=int(target.size),calibration_CT_pairs=int(target[:3].size),
        check_CT_pairs=int(target[3:].size),ink_accuracy_claim=False,model_forwards=0,
        scope='Exact selected source, seven neighborhoods and three layers only; no global/optical/anatomical authentication')
