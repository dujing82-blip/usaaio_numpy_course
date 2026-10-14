# Validation report

97 pytest checks passed. All 9 student and 9 instructor notebooks executed successfully through IPython on CPU. Student answer stubs produced FAIL feedback without uncaught errors; all instructor answers passed.

The local host prohibits Jupyter socket connections, so local end-to-end validation used `python -m scripts.run_all --folder both --in-process`. CI uses nbclient with separate Jupyter kernels. Installation/network time is excluded; these measured host timings are not a Colab speed guarantee.

| Notebook | Student seconds | Instructor seconds |
|---|---:|---:|
| 00_setup_and_how_to_use.ipynb | 0.83 | 0.01 |
| 01_arrays_and_shapes.ipynb | 0.99 | 0.04 |
| 02_elementwise_and_broadcasting.ipynb | 0.03 | 0.04 |
| 03_dot_matmul_transpose.ipynb | 0.03 | 0.03 |
| 04_linear_algebra_toolbox.ipynb | 0.04 | 0.05 |
| 05_where_it_shows_up_in_ai.ipynb | 0.06 | 0.09 |
| 06_mock_exam_A.ipynb | 0.14 | 0.21 |
| 06_mock_exam_B.ipynb | 1.66 | 0.09 |
| 07_pytorch_bridge.ipynb | 1.32 | 0.29 |
