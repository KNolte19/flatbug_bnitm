import os
import json
import shutil
from idlelib.debugobj_r import remote_object_tree_item
from pathlib import Path
from zipfile import ZipFile
import hashlib

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

from cvat_sdk import make_client
from cvat_sdk.core.progress import ProgressReporter
from botocore.config import Config
import yaml

TARGET_DIR = Path("./test-sync")
FORMAT_NAME = "COCO 1.0"
# ------------------ Secrets ------------------

# ------------------ Load secrets from YAML ------------------
def load_secrets_yaml(path=".secrets.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

SECRETS = load_secrets_yaml()

CVAT_CFG = SECRETS["cvat"]
S3_CFG = SECRETS["s3"]

CVAT_HOST = CVAT_CFG.get("host", "https://app.cvat.ai")
USERNAME = CVAT_CFG["username"]
PASSWORD = CVAT_CFG["password"]
PROJECT_ID = int(CVAT_CFG["project_id"])
ORG_SLUG = CVAT_CFG.get("org_slug")

AWS_ACCESS_KEY_ID = S3_CFG["access_key"]
AWS_SECRET_ACCESS_KEY = S3_CFG["secret_key"]
AWS_REGION = S3_CFG["region"]
S3_ENDPOINT_URL = S3_CFG["endpoint_url"]
S3_BUCKET = S3_CFG["bucket"]
S3_PREFIX = S3_CFG.get("prefix", "").rstrip("/")

TARGET_DIR.mkdir(parents=True, exist_ok=True)


# ------------------ Helpers ------------------
def safe_segment(name: str) -> str:
    """
    Make a filesystem-safe folder name (keep common chars; replace others with underscore).
    """
    allowed = "-_.() abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    cleaned = "".join(c if c in allowed else "_" for c in name).strip()
    # avoid empty folder names
    return cleaned or "unnamed_task"


def md5_file(path: Path, chunk=1024 * 1024) -> str:
    """Compute MD5 hex digest of a file (for ETag comparison if single-part)."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def task_is_completed(task) -> bool:
    """
    Consider a task completed if either:
      - task.status == 'completed', OR
      - all of its jobs are in state == 'completed'
    """
    try:
        if getattr(task, "status", None) == "completed":
            return True
    except Exception:
        pass

    try:
        jobs = list(task.get_jobs())
        return len(jobs) > 0 and all(getattr(j, "state", None) == "completed" for j in jobs)
    except Exception:
        return False

def build_s3_client():
    session = boto3.session.Session(
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name=AWS_REGION,
    )
    return session.client(
        "s3",
        endpoint_url=S3_ENDPOINT_URL,
        config=Config(
            signature_version="s3v4",
            s3={"addressing_style": "virtual"},  # or "path" if your bucket has dots
        ),
    )

# def build_s3_client():
#     """
#     Linode/Akamai Object Storage (S3-compatible):
#       - point boto3 at the custom endpoint URL
#       - keep SigV4
#       - prefer virtual-hosted style (works with linodeobjects.com)
#     """
#     print(AWS_SECRET_ACCESS_KEY, AWS_ACCESS_KEY_ID)
#     session = boto3.session.Session(
#         aws_access_key_id=AWS_ACCESS_KEY_ID,
#         aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
#         region_name=AWS_REGION,  # e.g. 'eu-central-1'
#     )
#     endpoint_url = SECRETS.get("S3_ENDPOINT_URL")  # e.g. 'https://eu-central-1.linodeobjects.com'
#
#     return session.client(
#         "s3",
#         endpoint_url=endpoint_url,
#         config=Config(
#             signature_version="s3v4",
#             s3={"addressing_style": "virtual"}  # virtual-hosted style: <bucket>.<endpoint>/<key>
#         ),
#     )

def list_s3_objects_with_prefix(s3, bucket: str, prefix: str):
    """
    Yield dicts with 'Key', 'Size', 'ETag' (no quotes), and 'LastModified'.
    """
    continuation = None
    while True:
        kwargs = {"Bucket": bucket, "Prefix": prefix}
        if continuation:
            kwargs["ContinuationToken"] = continuation
        resp = s3.list_objects_v2(**kwargs)
        for item in resp.get("Contents", []):
            key = item["Key"]
            if key.endswith("/"):
                continue  # skip directory placeholders
            etag = item.get("ETag", "").strip('"')
            yield {
                "Key": key,
                "Size": item["Size"],
                "ETag": etag,
                "LastModified": item["LastModified"],
            }
        if resp.get("IsTruncated"):
            continuation = resp["NextContinuationToken"]
        else:
            break


def ensure_parent(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)


def sync_s3_prefix_to_local(s3, bucket: str, prefix: str, local_root: Path):
    """
    Sync a single S3 prefix (a "directory") to local_root:
      - download/replace files when missing or when ETag differs
      - delete local files that aren't present upstream
      - store ETag in sidecar <file>.etag for accurate repeat runs
    """
    local_root.mkdir(parents=True, exist_ok=True)

    # 1) Index upstream
    upstream = {}
    for obj in list_s3_objects_with_prefix(s3, bucket, prefix):
        rel = obj["Key"][len(prefix) :].lstrip("/")
        upstream[rel] = obj

    if not upstream:
        return False  # nothing upstream for this prefix

    # 2) Download/refresh
    for rel, meta in upstream.items():
        dest = local_root / rel
        etag_sidecar = dest.with_suffix(dest.suffix + ".etag")
        need_dl = False

        if not dest.exists():
            need_dl = True
        else:
            # Compare ETag if we have it, else fall back to size/MD5 (best effort)
            on_disk_etag = etag_sidecar.read_text().strip() if etag_sidecar.exists() else ""
            if on_disk_etag and on_disk_etag == meta["ETag"]:
                need_dl = False
            else:
                # Try quick size compare; if size differs, definitely re-download
                if dest.stat().st_size != meta["Size"]:
                    need_dl = True
                else:
                    # If ETag looks like simple MD5 (no '-') compare with local MD5
                    if "-" not in meta["ETag"] and meta["ETag"]:
                        need_dl = (md5_file(dest) != meta["ETag"])
                    else:
                        # Multipart upload: can't compute same ETag; trust upstream timestamp/etag difference
                        need_dl = True

        if need_dl:
            ensure_parent(dest)
            tmp = dest.with_suffix(dest.suffix + ".tmp")
            s3.download_file(bucket, meta["Key"], str(tmp))
            tmp.replace(dest)
            etag_sidecar.write_text(meta["ETag"])

    # 3) Delete locals not on upstream
    for local_path in local_root.rglob("*"):
        if local_path.is_dir():
            continue
        if local_path.suffix == ".etag":
            # Only keep sidecars for files that still exist
            base = local_path.with_suffix("")
            if not base.exists():
                local_path.unlink(missing_ok=True)
            continue
        rel = str(local_path.relative_to(local_root))
        if rel not in upstream:
            # remove file and its .etag if present
            local_path.unlink(missing_ok=True)
            side = local_path.with_suffix(local_path.suffix + ".etag")
            side.unlink(missing_ok=True)

    return True


# ------------------ COCO Export ------------------
def export_coco_annotations_for_task(task, output_json_path: Path):
    """
    Export task dataset (COCO 1.0, annotations only) to a temp zip,
    then extract the COCO annotations json to output_json_path.
    """
    tmp_zip = output_json_path.parent / f"task-{task.id}-export.tmp.zip"
    output_json_path.parent.mkdir(parents=True, exist_ok=True)

    task.export_dataset(
        FORMAT_NAME,
        filename=str(tmp_zip),
        include_images=False,

    )

    # Find the annotations JSON inside the zip (usually 'annotations/instances_default.json')
    with ZipFile(tmp_zip, "r") as zf:
        # Prefer the standard path; fallback to the first *instances*.json we find.
        preferred = "annotations/instances_default.json"
        info = None
        try:
            info = zf.getinfo(preferred)
        except KeyError:
            for zi in zf.infolist():
                name = zi.filename.replace("\\", "/")
                if name.lower().startswith("annotations/") and name.lower().endswith(".json") and "instance" in name.lower():
                    info = zi
                    break
        if info is None:
            # last resort: any annotations json
            for zi in zf.infolist():
                name = zi.filename.replace("\\", "/")
                if name.lower().startswith("annotations/") and name.lower().endswith(".json"):
                    info = zi
                    break

        if info is None:
            zf.close()
            tmp_zip.unlink(missing_ok=True)
            raise RuntimeError("Could not locate COCO annotations JSON inside the export zip.")

        with zf.open(info, "r") as src, open(output_json_path, "wb") as dst:
            shutil.copyfileobj(src, dst)

    tmp_zip.unlink(missing_ok=True)


# ------------------ Main ------------------
def main():
    # S3 client
    s3 = build_s3_client()
    resp = s3.list_objects_v2(Bucket=S3_BUCKET, MaxKeys=500)

    # if "Contents" not in resp:
    #     print("❌ No objects found under that prefix.")
    # else:
    #     print(f"✅ Found {len(resp['Contents'])} objects:")
    #     for obj in resp["Contents"]:
    #         key = obj["Key"]
    #         size = obj["Size"]
    #         print(f"  - {key} ({size / 1024:.1f} KB)")
    # return
    with make_client(host=CVAT_HOST, credentials=(USERNAME, PASSWORD)) as client:
        if ORG_SLUG:
            client.organization_slug = ORG_SLUG

        # List tasks for the given project
        tasks = [t for t in client.tasks.list() if t.project_id == PROJECT_ID]
        print(f"Found {len(tasks)} tasks in project {PROJECT_ID}")

        for task in tasks:
            task.fetch()

            if not task_is_completed(task):
                print(f"Skipping task {task.id} ({task.name}): not completed")
                continue

            # Local folder name = task name (sanitized)
            task_dir_name = safe_segment(task.name)
            task_dir = TARGET_DIR / task_dir_name
            task_dir.mkdir(parents=True, exist_ok=True)

            # S3 prefix for this task
            # Expecting eponymous folder: S3_PREFIX/<task_name>/
            if S3_PREFIX:
                prefix = f"{S3_PREFIX}/{task.name}/"
            else:
                prefix = f"{task.name}/"

            print(f"Checking S3 for '{prefix}' ...")
            has_upstream = False
            # do a lightweight listing to see if it exists
            try:
                # list at most 1 object to verify existence
                resp = s3.list_objects_v2(Bucket=S3_BUCKET, Prefix=prefix, MaxKeys=1)
                has_upstream = resp.get("KeyCount", 0) > 0
            except ClientError as e:
                print(f"  S3 error: {e}")
                has_upstream = False

            if has_upstream:
                print(f"Syncing S3 s3://{S3_BUCKET}/{prefix} -> {task_dir}")
                synced = sync_s3_prefix_to_local(s3, S3_BUCKET, prefix, task_dir)
                if not synced:
                    print(f"  Nothing to sync for {task.name}.")
            else:
                print(f"No S3 dataset found for task '{task.name}' (prefix '{prefix}')")

            # Export COCO annotations for this task
            coco_json_path = task_dir / "instances_default.json"
            print(f"Exporting COCO annotations -> {coco_json_path}")
            export_coco_annotations_for_task(task, coco_json_path)

    print("✅ Done.")


if __name__ == "__main__":
    main()