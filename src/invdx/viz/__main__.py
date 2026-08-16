import argparse

from .plots import render_run


def main():
    p = argparse.ArgumentParser(
        description="Render figures from an invdx run directory.")
    p.add_argument("run_dir")
    p.add_argument("-o", "--out", default=None,
                   help="output dir (default: the run dir itself)")
    p.add_argument("--pdf", action="store_true",
                   help="also save vector PDFs next to each PNG (LaTeX-ready)")
    args = p.parse_args()
    made = render_run(args.run_dir, args.out, pdf=args.pdf)
    if not made:
        print(f"[viz] nothing renderable found in {args.run_dir}")
    for m in made:
        print(f"[viz] {m}")


if __name__ == "__main__":
    main()
