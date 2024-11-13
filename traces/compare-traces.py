import argparse, csv, sys

from math import isclose

VG = 1.49

def compare_values(csv_py, csv_ml, prec, debug=False):

    entries = min(len(csv_py), len(csv_ml))

    consts = {
        "G": 0,
        "Gp": 0,
        "Gt": 0,
        "Il": 0,
        "Ip": 0,
        "I1": 0,
        "Id": 0,
        "Qsto1": 0,
        "Qsto2": 0,
        "Qgut": 0,
        "X": 0,
        "SRsH": 0,
        "H": 0,
        "XH": 0,
    }

    total_percentage_diff = {const: 0 for const in consts.keys()}
    total_diff_count = {const: 0 for const in consts.keys()}

    for i in range(entries):
        if debug: print(f"Iteration {i}")
        py_vals = csv_py[i]
        ml_vals = csv_ml[i]

        for const in consts.keys():
            if const == "G":
                py_val = float(py_vals[const])
                ml_val = float(ml_vals["Gp"]) / VG
            else:
                py_val = float(py_vals[const])
                ml_val = float(ml_vals[const])

            if isclose(py_val, ml_val, abs_tol=10**(-prec)):
                consts[const] += 1
                if debug: print(f"       {const}: {py_val} (py) == {ml_val} (ml)")
            else:
                diff_percentage = abs(py_val - ml_val) / py_val * 100
                total_percentage_diff[const] += diff_percentage
                total_diff_count[const] += 1
                if debug: print(f"[DIFF] {const}: {py_val} (py) != {ml_val} (ml) | Diff %: {diff_percentage:.2f}%")

    print("Time within range:")
    for const in consts.keys():
        if total_diff_count[const] > 0:
            avg_diff = total_percentage_diff[const] / total_diff_count[const]
            print(f"{(consts[const]/entries) * 100:6.1f}% | {const} (average {avg_diff:6.4f}% diff)")
        else:
            print(f"{(consts[const]/entries) * 100:6.1f}% | {const}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_py", type=str, help="Path to the Python file.")
    parser.add_argument("csv_ml", type=str, help="Path to the MATLAB file.")
    parser.add_argument("precision", type=int, help="The precision to use for comparison.")
    parser.add_argument("--debug", action="store_true", help="Enable debug printing.")

    args = parser.parse_args()

    with open(args.csv_py, "r") as file_py, open(args.csv_ml, "r") as file_ml:
        csv_py = list(csv.DictReader(file_py))
        csv_ml = list(csv.DictReader(file_ml))
        compare_values(csv_py, csv_ml, args.precision, args.debug)


main()
