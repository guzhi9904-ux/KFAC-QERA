# L10 full-budget paired extension

See protocol.md for the explicit L31-to-L10 delta. This version is independent
of qer_kronecker_sketch_sample_v1; historical outputs must not be modified.

Stages: configure.py; run.sh CONFIG ROOT prepare; run.sh CONFIG ROOT pilot;
run.sh CONFIG ROOT run. The pilot must pass before formal work. Resume uses the
same command and verifies the complete code/config identity.

Output: RESULTS.md, READOUT.md, verification.json, summary/*.csv and
resource_usage.json. Parallel timing rows overlap; sum of rows is not wall time.

Validation: test_math.py, test_resources.py, test_pipeline.py, test_exact.py.
The full pipeline fixture exercises all 224/256/15/240 counts with real toy
autograd. Real pilot compares grouped contractions to the original dense-S path.

No precision reduction, fewer methods, early evaluation selection or relaxed
teacher/input/autograd thresholds. No H100 and no additional layer is launched.
