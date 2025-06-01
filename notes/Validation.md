# Validation

Created: May 2, 2025 8:58 PM
Reviewed: No

- Ohio T1DM dataset
    - this seems to be perfect, but it’s restricted
    - DUA has to be signed by PI, not by student
- DIaTrend Dataset: restricted (Synapse)
- T1GDUJA Dataset: open (Zenodo)
    - doesn’t have meal times, so not appropriate
- ShanghaiT1DM: open (Zenodo)
    - seems to have meal data?
    - but it’s quite course, hard to plug into the simulator directly
- T1DiabetesGranada: restricted (Zenodo)
    - also, not appropriate since it doesn’t have meal data
- REPLACE-BG: open (JCHR)
    - doesn’t seem to have any kind of meal data
- D1NAMO
    - seems to be better than the Shanghai dataset, meals are annotated with calories, we could manually carb-count some traces and validate against those, but it will be hard to automate on a large number of traces

“Few datasets are found in the literature that meet some of the necessary
 requirements for making realistic predictions: “REPLACE-BG”[19](https://www.nature.com/articles/s41597-023-02737-4#ref-CR19),
 a public dataset collecting real CGM data during 182 days from 226 T1D 
patients with well-controlled DM; “The D1NAMO Open Dataset”[20](https://www.nature.com/articles/s41597-023-02737-4#ref-CR20), an open dataset collecting real CGM data during approximately 30 days from 9 T1D patients; “The OhioT1DM Dataset”[21](https://www.nature.com/articles/s41597-023-02737-4#ref-CR21), an on-request dataset collecting real CGM data during 56 days from 12 T1D patients; and “ShanghaiT1DM”[22](https://www.nature.com/articles/s41597-023-02737-4#ref-CR22)
 a public dataset collecting real CGM data during 14 days from 12 T1D 
patients. However, these four datasets are characterized by a relatively
 small sample size and short study duration. Very recently, a 
contribution has been made towards increasing the study duration: 
“DiaTrend”[23](https://www.nature.com/articles/s41597-023-02737-4#ref-CR23), a public dataset collecting real CGM data during an average of 510 days from 54 T1D patients.”