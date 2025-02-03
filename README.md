# cocovax_submit

- **cocovax_methods__analyse.py**
  - Main analysis module containing the CDS class
  - Implements sequence deoptimization algorithms
  - Provides tools for sequence analysis and evaluation
- **cocovax_methods__calculate_cps_and_cai.py**
  - Utilities for calculating Codon Pair Scores (CPS)
  - Functions for computing CAI weights ($lnw$)
- **prompt.py**
  - prompt for the Reference Library
- **Configuration Files**:
  - `cai_choice.json`: Species-specific preferred codon choices for deoptimization
  - `cai_lnw.json`: Logarithmic weights for CAI calculation
  - `cps_dict.json`: Comprehensive codon pair scores dictionary