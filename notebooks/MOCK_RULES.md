# Mock-exam rules

Use a CPU runtime. You have 60 minutes for each mock.

- Run the provided setup cell first. No other imports, including temporary imports.
- Edit solution stubs only; do not edit graders or read reference implementations.
- A no-loop task bans for/while and all comprehensions. An explicit iteration/epoch loop is allowed only where stated.
- PURE tasks ban loops, np.linalg, @, dot/matmul/einsum-style substitutes, and transpose operations.
- Return the exact shape and integer dtype. Keep seeded randomness and requested seed resets.
- Print only requested summaries, with four decimal places when stated.
- GQA must broadcast K/V rather than copying them.
- Write your shape prediction before running code. Review written reasoning and plots after grading.

These are adapted practice exams, not official USAAIO exam papers. Static checks catch common mistakes, not deliberate circumvention. The code score gives each problem's points only when all its components pass; explanations and plot styling require self-review. Each paper sums to 100 points.
