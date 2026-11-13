"""Source-bound pairing metadata checks, never landing-label alignment."""
from __future__ import annotations
import math

def finite(x):return isinstance(x,(int,float)) and not isinstance(x,bool) and math.isfinite(x)
def validate_calibration(calibration,view_a,view_b):
    if (view_a['group']!=view_b['group'] or view_a['number']!=view_b['number']
            or view_a['view']!='A' or view_b['view']!='B'):
        raise ValueError('Cross-file pairing is not the declared A/B pair')
    for side,row in [('A',view_a),('B',view_b)]:
        if calibration.get('source_sha256_'+side)!=row['sha256']:raise ValueError('Calibration source hash differs')
    if calibration.get('landing_reference_used') is not False:raise ValueError('Reference-derived timing is forbidden')
    if calibration.get('recorded_before_predictions') is not True:raise ValueError('Post-hoc alignment is not a declared calibration')
    if calibration.get('same_rally_verified') is not True:raise ValueError('Filename pairing alone does not confirm the same rally')
    if not calibration.get('independent_source_cues'):raise ValueError('Independent source timing evidence is required')
    scale,offset,error=(calibration.get(k) for k in ['scale','offset_seconds','uncertainty_seconds'])
    if not all(finite(x) for x in [scale,offset,error]) or scale<=0 or error<0:raise ValueError('Invalid affine clock calibration')
    method=calibration.get('method')
    if method not in {'audio_cross_correlation','automatic_shared_nonlanding_actions','operator_shared_nonlanding_actions'}:raise ValueError('Unsupported/undeclared calibration method')
    assisted=method.startswith('operator_') or calibration.get('assistance_used') is True
    return {'contract_valid':True,'empirical_synchronization_independently_proven':False,'assistance_used':assisted,'automatic_evaluation_eligible':not assisted,'scale':scale,'offset_seconds':offset,'uncertainty_seconds':error,'limitations':'A valid metadata contract does not prove the cue matching, target identity or contact. Source evidence must be independently checked.'}

def temporal_consistency(interval_a,interval_b,calibration_status):
    for interval in [interval_a,interval_b]:
        if len(interval)!=2 or not all(finite(x) for x in interval) or interval[1]<interval[0]:raise ValueError('Invalid measured/inferred event interval')
    scale=calibration_status['scale'];offset=calibration_status['offset_seconds'];error=calibration_status['uncertainty_seconds']
    mapped=[scale*interval_a[0]+offset-error,scale*interval_a[1]+offset+error]
    return {'compatible_time_windows':max(mapped[0],interval_b[0])<=min(mapped[1],interval_b[1]),'mapped_a_interval_in_b_seconds':mapped,'b_interval_seconds':list(interval_b),'proves_same_shuttle':False,'proves_floor_contact':False,'cross_view_pixel_positions_compared':False}
