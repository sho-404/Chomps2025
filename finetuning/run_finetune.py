"""Launch, poll, and deploy Azure OpenAI fine-tuning jobs for the council judges.

Usage (run from the repo root with the project virtualenv):

    python code/finetuning/run_finetune.py launch --file <jsonl> --base gpt-4.1-mini-2025-04-14 --suffix fact-en-mini
    python code/finetuning/run_finetune.py status                  # list all jobs
    python code/finetuning/run_finetune.py deploy --job <job-id> --name <deployment-name>

Credentials come from the repo-root `.env` (AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY).
Job ids are appended to code/finetuning/jobs.jsonl so that every launched job is traceable.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import AzureOpenAI

REPO_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(REPO_ROOT / ".env")
JOBS_LOG = Path(__file__).resolve().parent / "jobs.jsonl"
API_VERSION = "2025-04-01-preview"

ENDPOINT = os.environ["AZURE_OPENAI_ENDPOINT"]
API_KEY = os.environ["AZURE_OPENAI_API_KEY"]
client = AzureOpenAI(api_version=API_VERSION, azure_endpoint=ENDPOINT, api_key=API_KEY)


def launch(args):
    path = Path(args.file).resolve()
    if not path.exists():
        sys.exit(f"training file not found: {path}")
    n_lines = sum(1 for _ in open(path, encoding="utf-8"))
    with open(path, "rb") as fh:
        uploaded = client.files.create(file=fh, purpose="fine-tune")
    # Wait until the file is processed; Azure rejects jobs on pending files.
    for _ in range(120):
        f = client.files.retrieve(uploaded.id)
        if f.status in ("processed", "succeeded"):
            break
        if f.status in ("error", "deleted"):
            sys.exit(f"file upload failed: {f}")
        time.sleep(5)
    else:
        sys.exit("file never reached 'processed' state")
    job = client.fine_tuning.jobs.create(
        training_file=uploaded.id,
        model=args.base,
        suffix=args.suffix,
        seed=args.seed,
    )
    record = {
        "job_id": job.id,
        "base": args.base,
        "suffix": args.suffix,
        "file": str(path.relative_to(REPO_ROOT)),
        "n_examples": n_lines,
        "seed": args.seed,
        "hyperparameters": _hp(job),
        "created_at": job.created_at,
    }
    with open(JOBS_LOG, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")
    print(json.dumps(record, indent=2))


def _hp(job):
    hp = getattr(job, "hyperparameters", None)
    if hp is None:
        return None
    return hp.model_dump() if hasattr(hp, "model_dump") else dict(hp)


def status(args):
    jobs = client.fine_tuning.jobs.list(limit=50)
    for j in jobs.data:
        print(f"{j.id}  {j.status:<12} base={j.model}  ft_model={j.fine_tuned_model}  "
              f"hp={_hp(j)}  trained_tokens={getattr(j, 'trained_tokens', None)}")
        if args.events:
            for e in client.fine_tuning.jobs.list_events(j.id, limit=5).data:
                print(f"      - {e.message}")


def deploy(args):
    job = client.fine_tuning.jobs.retrieve(args.job)
    if job.status != "succeeded" or not job.fine_tuned_model:
        sys.exit(f"job {args.job} is {job.status}; fine_tuned_model={job.fine_tuned_model}")
    # Deployment goes through ARM, not the data-plane API, so use the Azure CLI.
    cmd = [
        "az", "cognitiveservices", "account", "deployment", "create",
        "-g", args.resource_group, "-n", args.account,
        "--deployment-name", args.name,
        "--model-name", job.fine_tuned_model,
        "--model-version", "1",
        "--model-format", "OpenAI",
        "--sku-name", args.sku,
        "--sku-capacity", str(args.capacity),
        "-o", "json",
    ]
    out = subprocess.run(cmd, capture_output=True, text=True)
    if out.returncode != 0:
        sys.exit(out.stderr)
    d = json.loads(out.stdout)
    print(d["name"], d["properties"]["provisioningState"], "model=", job.fine_tuned_model)
    with open(JOBS_LOG, "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"job_id": job.id, "deployment": args.name,
                             "fine_tuned_model": job.fine_tuned_model}) + "\n")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("launch")
    a.add_argument("--file", required=True)
    a.add_argument("--base", required=True)
    a.add_argument("--suffix", required=True)
    a.add_argument("--seed", type=int, default=42)
    a.set_defaults(fn=launch)
    s = sub.add_parser("status")
    s.add_argument("--events", action="store_true")
    s.set_defaults(fn=status)
    d = sub.add_parser("deploy")
    d.add_argument("--job", required=True)
    d.add_argument("--name", required=True)
    d.add_argument("--resource-group", default="llms-chomps-rg")
    d.add_argument("--account", default="chomps-openai")
    d.add_argument("--capacity", type=int, default=250)  # K tokens/min; fine-tune quota is 500 per base model and SKU
    d.add_argument("--sku", default="Standard", choices=["Standard", "GlobalStandard", "DataZoneStandard"],
                   help="each SKU has its own fine-tuned quota (500 per base model in this subscription)")
    d.set_defaults(fn=deploy)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
