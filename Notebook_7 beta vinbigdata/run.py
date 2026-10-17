"""Explicit stage entrypoint for Notebook_7 beta (VinBigData 512); importing never starts training."""
from pathlib import Path
import argparse

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("download")
    p.add_argument("--slug", default="awsaf49/vinbigdata-512-image-dataset")
    p = commands.add_parser("audit")
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--output", type=Path, default=ROOT / "audit")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--limit", type=int, help="Feasibility check only; not an accepted complete audit")
    p = commands.add_parser("prepare")
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--audit-dir", type=Path, default=ROOT / "audit")
    p.add_argument("--work-root", type=Path, default=ROOT)
    p = commands.add_parser("train")
    p.add_argument("--data-dir", type=Path, default=ROOT / "selected_data")
    p.add_argument("--output", type=Path, default=ROOT / "runs/v7_beta_vinbigdata")
    p.add_argument("--device", default="cuda")
    p.add_argument("--smoke", action="store_true")
    p = commands.add_parser("smoke")
    p.add_argument("--device", default="cuda")
    p.add_argument("--output", type=Path, default=ROOT / ".runtime/smoke")
    args = parser.parse_args()
    if args.command == "download":
        from v7.colab import download_vinbigdata
        print(download_vinbigdata(args.slug))
    elif args.command == "audit":
        from v7.audit import audit
        audit(args.data_root, args.output, workers=args.workers, limit=args.limit)
    elif args.command == "prepare":
        from v7.prepare import prepare
        prepare(ROOT, args.data_root, args.audit_dir, args.work_root)
    elif args.command == "train":
        from v7.runner import run
        run(ROOT, args.data_dir, args.output, device=args.device, smoke=args.smoke)
    else:
        from v7.smoke import smoke
        smoke(ROOT, args.output, args.device)


if __name__ == "__main__":
    main()
