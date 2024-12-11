import sys

sys.path.append("/usr/local/NVIDIA-Nsight-Compute/extras/python/")

import ncu_report
from test_fused import BENCHMARK_SIZES


def main():
    ctx = ncu_report.load_report("profile_fused.ncu-rep")
    ncu_range = ctx.range_by_idx(0)
    num_actions = ncu_range.num_actions()
    i = 0
    for m, n, k in BENCHMARK_SIZES[1:]:
        while (
            i < num_actions
            and ncu_range.action_by_idx(i).name() != "matmul_hadamard_kernel"
        ):
            i += 1
        assert i < num_actions
        action = ncu_range.action_by_idx(i)
        fused_duration_ns = action.metric_by_name("gpu__time_duration.min").as_double()
        i += 1

        while (
            i < num_actions
            and ncu_range.action_by_idx(i).name() != "fast_hadamard_transform_kernel"
        ):
            i += 1
        assert i < num_actions
        unfused_duration_ns = 0
        while i < num_actions:
            action = ncu_range.action_by_idx(i)
            unfused_duration_ns += action.metric_by_name(
                "gpu__time_duration.min"
            ).as_double()
            if action.name() == "sym_dequantize_i32_f16_kernel":
                break
            i += 1

        def format_time(t_ns):
            if t_ns < 1e3:
                return f"\\qty{{{t_ns:.2f}}}{{\\nano\\second}}"
            elif t_ns < 1e6:
                return f"\\qty{{{t_ns/1e3:.2f}}}{{\\micro\\second}}"
            else:
                return f"\\qty{{{t_ns/1e6:.2f}}}{{\\milli\\second}}"

        speedup = unfused_duration_ns / fused_duration_ns
        print(
            f"{m:4} & {n:4} & {k:4} & {format_time(unfused_duration_ns)} & {format_time(fused_duration_ns)} & {speedup:.2f}\\\\"
        )
        i += 1


if __name__ == "__main__":
    main()
