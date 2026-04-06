from __future__ import annotations

from pathlib import Path

from src.experiment_runner import run_all_experiments


def main() -> int:
    data_path = Path("Homework1/Data4Regression.xlsx")
    output_dir = Path("Homework1/output")
    results = run_all_experiments(
        data_path,
        output_dir=output_dir,
        generate_report=False,
        generate_pdf=False,
    )

    print("Homework1 regression experiments completed.")
    for method_name, result in results["linear"].items():
        print(
            f"linear::{method_name} train_mse={result['train_mse']:.6f} "
            f"test_mse={result['test_mse']:.6f}"
        )

    for name in ("polynomial", "kernel", "trigonometric"):
        best = results[name]["best"]
        print(
            f"{name}::best train_mse={best['train_mse']:.6f} "
            f"test_mse={best['test_mse']:.6f}"
        )

    print(f"report_markdown={results.get('report_path')}")
    print(f"report_pdf={results.get('pdf_path')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
