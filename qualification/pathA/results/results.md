# Path A results (contract sha256 cd69cb6db2f2ba825dc804e762ea78c5651f3d61ec962955126c3fec02aa1c97)

## A1: PASS
- P1: `{"values": [0.0, 63.5625, 87.0625, 109.0625, 123.625, 134.375], "monotone": true, "robust": {"100": 109.0625, "200": 134.375}, "holds": true}`
- P2: `{"ratios": {"100": 0.0, "200": 0.0009302325581395349}, "holds": true}`
- S1: `{"monotone": {"aDN1": true, "aDN2": true}, "holds": true}`
- S2: `{"ratios": {"aDN1@100": 0.0, "aDN1@200": 0.0, "aDN2@100": 0.0, "aDN2@200": 0.0}, "holds": true}`
- S3: `{"monotone": true, "positive": {"A1_JOCE_200": 5.722222222222221, "A1_JOF_200": 0.2839506172839506}, "holds": true}`
- control_irrelevant: `{"ratios": {"aBN1@100": 0.0, "aBN1@200": 0.0, "aDN1@100": 0.0, "aDN1@200": 0.0, "aDN2@100": 0.0, "aDN2@200": 0.0}, "valid": true}`
- counterfactual_shuffled: `{"label": "LABELLED OFFLINE COUNTERFACTUAL", "ratios": {"A1_JOCE_200_SHUFFLED": 0.0}}`

## A2: FAIL
- P1: `{"series": {"A2_LC4_{r}": {"values": [0.0, 89.1875, 142.625, 193.25, 220.8125, 238.4375], "monotone": true, "at200": 238.4375, "holds": true}, "A2_LPLC2_{r}": {"values": [0.0, 110.875, 139.5625, 199.125, 221.6875, 235.8125], "monotone": true, "at200": 235.8125, "holds": true}}, "holds": true}`
- P2: `{"ttmn_over_gf": {"A2_LC4_200": 0.2943643512450852, "A2_LPLC2_200": 0.2536443148688047}, "holds": false}`
- S1: `{"ratios": {"A2_LC4_100_silence_GF": 0.47494780793319413, "A2_LC4_200_silence_GF": 0.691006233303651, "A2_LPLC2_100_silence_GF": 0.10384068278805121, "A2_LPLC2_200_silence_GF": 0.47335423197492166}, "holds": false}`
- control_irrelevant: `{"ratios": {"GF@100": 0.14974126778783958, "GF@200": 0.18296199213630407, "TTMn@100": 0.006263048016701462, "TTMn@200": 0.032056990204808546}, "valid": false}`
- counterfactual_shuffled: `{"label": "LABELLED OFFLINE COUNTERFACTUAL", "ratios": {"A2_LC4_200_SHUFFLED": 0.012581913499344692, "A2_LPLC2_200_SHUFFLED": 0.0018552875695732839}}`

## CAL: CONSISTENT
- C1: `{"values": [0.0, 0.875, 16.1875, 57.5625, 71.25, 74.3125, 75.3125], "monotone": true, "fraction_of_max_at_100": 0.946058091286307, "holds": true}`

## Dose-response (seed-mean Hz; levels per contract)
- A1 JO-CE -> aBN1: [0.0, 63.56, 87.06, 109.06, 123.62, 134.38]
- A1 JO-F -> aBN1: [0.0, 0.0, 0.0, 0.0, 0.0, 0.12]
- A1 JO-CE -> aDN1: [0.0, 87.88, 103.31, 110.94, 113.19, 114.94]
- A1 JO-F -> aDN1: [0.0, 0.0, 0.0, 0.0, 0.0, 4.0]
- A1 JO-CE -> aDN2: [0.0, 73.38, 85.94, 86.56, 81.12, 81.25]
- A1 JO-F -> aDN2: [0.0, 0.0, 0.0, 0.0, 0.0, 4.06]
- A1 JO-CE -> ProLN-MN: [0.0, 4.31, 5.07, 5.57, 5.59, 5.72]
- A1 JO-F -> ProLN-MN: [0.0, 0.07, 0.1, 0.09, 0.06, 0.28]
- A2 LC4 -> GF: [0.0, 89.19, 142.62, 193.25, 220.81, 238.44]
- A2 LC4 -> TTMn: [0.0, 12.38, 39.31, 59.88, 68.0, 70.19]
- A2 LPLC2 -> GF: [0.0, 110.88, 139.56, 199.12, 221.69, 235.81]
- A2 LPLC2 -> TTMn: [0.0, 25.81, 28.06, 43.94, 54.31, 59.81]
- CAL sugar -> MN9: [0.0, 0.88, 16.19, 57.56, 71.25, 74.31, 75.31]

## All conditions (mean +- SD over seeds, Hz)
- A1_IRR_ORN_100 (n=8, activated 82.7 Hz, firing neurons 8255): aBN1 0.00+-0.00; aDN1 0.00+-0.00; aDN2 0.00+-0.00; ProLN-MN 0.23+-0.04; JO-CE 0.00+-0.00; JO-F 0.00+-0.00
- A1_IRR_ORN_200 (n=8, activated 140.8 Hz, firing neurons 8552): aBN1 0.00+-0.00; aDN1 0.00+-0.00; aDN2 0.00+-0.00; ProLN-MN 0.25+-0.03; JO-CE 0.00+-0.00; JO-F 0.00+-0.00
- A1_JOCE_0 (n=8, activated 0.0 Hz, firing neurons 0): aBN1 0.00+-0.00; aDN1 0.00+-0.00; aDN2 0.00+-0.00; ProLN-MN 0.00+-0.00; JO-CE 0.00+-0.00; JO-F 0.00+-0.00
- A1_JOCE_100 (n=8, activated 82.7 Hz, firing neurons 4741): aBN1 109.06+-1.05; aDN1 110.94+-4.52; aDN2 86.56+-6.53; ProLN-MN 5.57+-0.30; JO-CE 82.65+-0.47; JO-F 0.00+-0.00
- A1_JOCE_100_silence_aBN1 (n=8, activated 82.7 Hz, firing neurons 4264): aBN1 0.00+-0.00; aDN1 0.00+-0.00; aDN2 0.00+-0.00; ProLN-MN 0.01+-0.01; JO-CE 82.65+-0.47; JO-F 0.00+-0.00
- A1_JOCE_150 (n=8, activated 114.2 Hz, firing neurons 5068): aBN1 123.62+-1.83; aDN1 113.19+-3.03; aDN2 81.12+-4.72; ProLN-MN 5.59+-0.18; JO-CE 114.22+-0.36; JO-F 0.00+-0.00
- A1_JOCE_200 (n=8, activated 140.8 Hz, firing neurons 5173): aBN1 134.38+-1.13; aDN1 114.94+-3.98; aDN2 81.25+-4.30; ProLN-MN 5.72+-0.18; JO-CE 140.78+-0.47; JO-F 0.00+-0.00
- A1_JOCE_200_SHUFFLED (n=8, activated 140.8 Hz, firing neurons 8961): aBN1 0.00+-0.00; aDN1 0.00+-0.00; aDN2 0.00+-0.00; ProLN-MN 0.26+-0.17; JO-CE 140.79+-0.47; JO-F 0.04+-0.06
- A1_JOCE_200_silence_aBN1 (n=8, activated 140.8 Hz, firing neurons 6140): aBN1 0.00+-0.00; aDN1 0.00+-0.00; aDN2 0.00+-0.00; ProLN-MN 0.14+-0.15; JO-CE 140.78+-0.47; JO-F 0.00+-0.00
- A1_JOCE_25 (n=8, activated 23.9 Hz, firing neurons 4274): aBN1 63.56+-5.70; aDN1 87.88+-10.57; aDN2 73.38+-9.11; ProLN-MN 4.31+-0.61; JO-CE 23.85+-0.32; JO-F 0.00+-0.00
- A1_JOCE_50 (n=8, activated 45.3 Hz, firing neurons 4500): aBN1 87.06+-3.88; aDN1 103.31+-6.17; aDN2 85.94+-6.06; ProLN-MN 5.07+-0.25; JO-CE 45.32+-0.43; JO-F 0.00+-0.00
- A1_JOF_100 (n=8, activated 82.1 Hz, firing neurons 2742): aBN1 0.00+-0.00; aDN1 0.00+-0.00; aDN2 0.00+-0.00; ProLN-MN 0.09+-0.03; JO-CE 0.00+-0.00; JO-F 82.13+-0.84
- A1_JOF_150 (n=8, activated 113.8 Hz, firing neurons 2965): aBN1 0.00+-0.00; aDN1 0.00+-0.00; aDN2 0.00+-0.00; ProLN-MN 0.06+-0.03; JO-CE 0.00+-0.00; JO-F 113.76+-1.06
- A1_JOF_200 (n=8, activated 140.8 Hz, firing neurons 3102): aBN1 0.12+-0.35; aDN1 4.00+-11.31; aDN2 4.06+-11.49; ProLN-MN 0.28+-0.61; JO-CE 0.00+-0.00; JO-F 140.79+-0.67
- A1_JOF_25 (n=8, activated 24.0 Hz, firing neurons 3126): aBN1 0.00+-0.00; aDN1 0.00+-0.00; aDN2 0.00+-0.00; ProLN-MN 0.07+-0.04; JO-CE 0.00+-0.00; JO-F 23.95+-0.72
- A1_JOF_50 (n=8, activated 45.4 Hz, firing neurons 3129): aBN1 0.00+-0.00; aDN1 0.00+-0.00; aDN2 0.00+-0.00; ProLN-MN 0.10+-0.04; JO-CE 0.00+-0.00; JO-F 45.37+-0.62
- A2_IRR_LC15_100 (n=8, activated 82.4 Hz, firing neurons 4196): GF 28.94+-1.97; TTMn 0.38+-0.35; LC4 0.00+-0.00; LPLC2 0.00+-0.00
- A2_IRR_LC15_200 (n=8, activated 141.0 Hz, firing neurons 4382): GF 43.62+-3.35; TTMn 2.25+-0.71; LC4 0.00+-0.00; LPLC2 0.00+-0.00
- A2_LC4_0 (n=8, activated 0.0 Hz, firing neurons 0): GF 0.00+-0.00; TTMn 0.00+-0.00; LC4 0.00+-0.00; LPLC2 0.00+-0.00
- A2_LC4_100 (n=8, activated 83.3 Hz, firing neurons 2240): GF 193.25+-2.46; TTMn 59.88+-1.60; LC4 83.31+-0.85; LPLC2 0.00+-0.00
- A2_LC4_100_silence_GF (n=8, activated 83.2 Hz, firing neurons 2234): GF 0.00+-0.00; TTMn 28.44+-3.08; LC4 83.22+-0.87; LPLC2 0.00+-0.00
- A2_LC4_150 (n=8, activated 116.6 Hz, firing neurons 2838): GF 220.81+-1.67; TTMn 68.00+-1.41; LC4 116.58+-0.66; LPLC2 0.00+-0.00
- A2_LC4_200 (n=8, activated 143.5 Hz, firing neurons 3092): GF 238.44+-1.52; TTMn 70.19+-1.83; LC4 143.49+-0.59; LPLC2 0.00+-0.00
- A2_LC4_200_SHUFFLED (n=8, activated 141.0 Hz, firing neurons 8504): GF 3.00+-8.29; TTMn 0.44+-0.86; LC4 141.03+-0.63; LPLC2 0.10+-0.14
- A2_LC4_200_silence_GF (n=8, activated 143.5 Hz, firing neurons 3104): GF 0.00+-0.00; TTMn 48.50+-1.31; LC4 143.46+-0.57; LPLC2 0.00+-0.00
- A2_LC4_25 (n=8, activated 23.6 Hz, firing neurons 1408): GF 89.19+-2.93; TTMn 12.38+-2.45; LC4 23.63+-0.37; LPLC2 0.00+-0.00
- A2_LC4_50 (n=8, activated 45.3 Hz, firing neurons 1749): GF 142.62+-4.49; TTMn 39.31+-2.40; LC4 45.30+-0.68; LPLC2 0.00+-0.00
- A2_LPLC2_100 (n=8, activated 86.5 Hz, firing neurons 3914): GF 199.12+-2.05; TTMn 43.94+-2.04; LC4 0.00+-0.00; LPLC2 86.50+-0.54
- A2_LPLC2_100_silence_GF (n=8, activated 86.4 Hz, firing neurons 4044): GF 0.00+-0.00; TTMn 4.56+-2.93; LC4 0.00+-0.00; LPLC2 86.45+-0.48
- A2_LPLC2_150 (n=8, activated 121.7 Hz, firing neurons 4562): GF 221.69+-2.27; TTMn 54.31+-1.81; LC4 0.01+-0.01; LPLC2 121.70+-0.62
- A2_LPLC2_200 (n=8, activated 149.4 Hz, firing neurons 4744): GF 235.81+-0.53; TTMn 59.81+-2.00; LC4 0.09+-0.03; LPLC2 149.38+-0.57
- A2_LPLC2_200_SHUFFLED (n=8, activated 140.6 Hz, firing neurons 8780): GF 0.44+-0.90; TTMn 0.12+-0.23; LC4 0.01+-0.01; LPLC2 140.58+-0.56
- A2_LPLC2_200_silence_GF (n=8, activated 149.4 Hz, firing neurons 4989): GF 0.00+-0.00; TTMn 28.31+-2.69; LC4 0.07+-0.03; LPLC2 149.36+-0.57
- A2_LPLC2_25 (n=8, activated 23.8 Hz, firing neurons 2485): GF 110.88+-3.45; TTMn 25.81+-1.98; LC4 0.00+-0.00; LPLC2 23.84+-0.33
- A2_LPLC2_50 (n=8, activated 45.1 Hz, firing neurons 3001): GF 139.56+-2.41; TTMn 28.06+-2.62; LC4 0.00+-0.00; LPLC2 45.10+-0.62
- CAL_SUGAR_0 (n=8, activated 0.0 Hz, firing neurons 0): MN9 0.00+-0.00; sugarGRN-cal 0.00+-0.00
- CAL_SUGAR_10 (n=8, activated 9.7 Hz, firing neurons 1342): MN9 0.88+-0.95; sugarGRN-cal 9.67+-0.62
- CAL_SUGAR_100 (n=8, activated 82.7 Hz, firing neurons 2814): MN9 71.25+-4.55; sugarGRN-cal 82.68+-1.08
- CAL_SUGAR_150 (n=8, activated 114.5 Hz, firing neurons 3026): MN9 74.31+-4.92; sugarGRN-cal 114.50+-1.55
- CAL_SUGAR_200 (n=8, activated 140.0 Hz, firing neurons 2979): MN9 75.31+-3.33; sugarGRN-cal 139.96+-1.73
- CAL_SUGAR_25 (n=8, activated 23.4 Hz, firing neurons 1899): MN9 16.19+-12.73; sugarGRN-cal 23.44+-0.93
- CAL_SUGAR_50 (n=8, activated 45.6 Hz, firing neurons 2315): MN9 57.56+-11.35; sugarGRN-cal 45.59+-1.54
