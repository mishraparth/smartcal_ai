# utils/predictor.py

import numpy as np
import pandas as pd

def smartcalc_predict(load=None, units=None, bill=None,
                      model_bill=None, scaler_bill=None,
                      model_units=None, scaler_units=None,
                      model_load=None, scaler_load=None,
                      feature_means=None, full_features=None):

    base = feature_means.copy()

    if load is not None:
        base['LOAD'] = load
        base['LOAD_EFFICIENCY'] = units / (load + 0.01) if units else base['LOAD_EFFICIENCY']
    if units is not None:
        base['CONSUMPTION_KWH'] = units
        base['CONSUMPTION_TREND'] = units - base['CONSUMPTION_PREV_MNTH']
        base['CONSUMPTION_AVG_3M'] = np.mean([
            units,
            base['CONSUMPTION_PREV_MNTH'],
            base['CONSUMPTION_PREV_TO_PREV_MNTH']
        ])
    if bill is not None:
        base['BILLED_AMOUNT'] = bill

    input_df = pd.DataFrame([base])

    if load is not None and units is not None:
        scaled = scaler_bill.transform(input_df[full_features])
        bill_pred = model_bill.predict(scaled)[0]
        return {'LOAD': round(load, 2), 'CONSUMPTION_KWH': round(units, 2), 'BILLED_AMOUNT': round(bill_pred, 2)}

    elif load is not None and bill is not None:
        scaled = scaler_units.transform(input_df[full_features])
        units_pred = model_units.predict(scaled)[0]
        return {'LOAD': round(load, 2), 'CONSUMPTION_KWH': round(units_pred, 2), 'BILLED_AMOUNT': round(bill, 2)}

    elif units is not None and bill is not None:
        scaled = scaler_load.transform(input_df[full_features])
        load_pred = model_load.predict(scaled)[0]
        return {'LOAD': round(load_pred, 2), 'CONSUMPTION_KWH': round(units, 2), 'BILLED_AMOUNT': round(bill, 2)}

    else:
        return "Please provide any two values to predict the third."