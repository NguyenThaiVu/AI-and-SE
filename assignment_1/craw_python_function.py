import requests, os, zipfile, shutil
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv
import time
import re
import ast
import numpy as np
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from typing import List, Dict, Iterable, Optional


def search_python_repos(n=1, min_star=50, header={}, allowed_licenses=None):
    """
    Search Java repos by stars with pagination support.
    Filters by license if allowed_licenses is provided.
    """
    url = "https://api.github.com/search/repositories"
    language = "Python"

    repos, page = [], 1
    while len(repos) < n:
        params = {
            "q": f"language:{language} stars:>{min_star}",
            "sort": "stars",
            "order": "desc",
            "per_page": 100,
            "page": page,
        }
        r = requests.get(url, headers=header, params=params, timeout=30)
        r.raise_for_status()
        items = r.json().get("items", [])
        if not items:
            break

        for repo in items:
            lic = repo.get("license")
            if allowed_licenses:
                if not lic or lic["key"].lower() not in allowed_licenses:
                    continue  # skip incompatible/missing licenses
            repos.append(repo)
            if len(repos) >= n:
                break

        page += 1

    return repos[:n]


def get_repo_files(owner, repo, header, branch="main", extension=".py", max_files=None):
    """
    Get all file paths in a repo.
    * Arguments:
        - owner: repo owner
        - repo: repo name
        - branch: branch name (default: main)
    * Returns:
        - list of file paths (strings)
    """

    url = f"https://api.github.com/repos/{owner}/{repo}/git/trees/{branch}?recursive=1"
    r = requests.get(url, headers=header, timeout=30)
    r.raise_for_status()
    tree = r.json().get("tree", [])

    result = [
        item["path"]
        for item in tree
        if item["type"] == "blob" and item["path"].endswith(extension)
    ]
    if max_files:
        result = result[:max_files]

    return result


def download_and_extract_repo(owner, repo, header, branch, dest):
    """Download a GitHub repo as zipball, extract locally, and remove the zip file."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)

    zip_path = dest / f"{owner}-{repo}-{branch}.zip"
    extract_dir = dest / f"{owner}-{repo}-{branch}"

    # Clean up old extracted folder if exists
    if extract_dir.exists():
        shutil.rmtree(extract_dir)

    url = f"https://api.github.com/repos/{owner}/{repo}/zipball/{branch}"
    print(f"[DOWNLOAD] {url}")

    try:
        # Download the zip file
        with requests.get(url, headers=header, stream=True, timeout=60) as r:
            r.raise_for_status()
            with open(zip_path, "wb") as f:
                for chunk in r.iter_content(1024 * 256):  # 256 KB chunks
                    f.write(chunk)

        # Extract contents
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(extract_dir)

        # Delete the zip file after extraction
        zip_path.unlink(missing_ok=True)
        print(f"[CLEANUP] Deleted zip: {zip_path}")

        # Return the extracted repo folder
        subfolders = list(extract_dir.iterdir())
        if len(subfolders) == 1 and subfolders[0].is_dir():
            return subfolders[0]
        return extract_dir

    except Exception as e:
        if zip_path.exists():
            zip_path.unlink(missing_ok=True)
        if extract_dir.exists():
            shutil.rmtree(extract_dir, ignore_errors=True)
        raise RuntimeError(
            f"Failed to download/extract repo {owner}/{repo}@{branch}: {e}"
        )


def deduplicate_methods(df):
    """Remove duplicate methods from the dataset."""
    return df.drop_duplicates(subset=["repo_name", "file_path", "method_code"])


def extract_python_methods(file_path):
    """Extract all Python method definitions from a file."""
    methods = []
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        code = f.read()
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                methods.append(ast.get_source_segment(code, node))
    except:
        pass
    return methods


def _worker(args) -> Optional[List[Dict]]:
    """
    Worker run in a separate process.
    Returns a list of result dicts for one file, or None if the file is skipped/errored.
    """
    file_path, local_repo_path, repo_name, repo_url = args
    full_path = os.path.join(local_repo_path, file_path)
    try:
        if not os.path.isfile(full_path):
            return None

        methods = extract_python_methods(full_path)
        if not methods:
            return []

        out = []
        for method in methods:
            out.append(
                {
                    "repo_name": repo_name,
                    "repo_url": repo_url,
                    "file_path": file_path,
                    "method_code": method,
                }
            )
        return out
    except Exception as e:
        return None


def extract_python_method_parallel(
    files: list,
    local_repo_path: str,
    repo_name: str,
    repo_url: str,
    max_workers: int | None = None,
    chunksize: int = 32,
) -> list:
    """
    Parallelize over files using processes (good for CPU-bound AST parsing).
    """
    all_results: List[Dict] = []
    tasks = ((fp, local_repo_path, repo_name, repo_url) for fp in files)

    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        for res in ex.map(_worker, tasks, chunksize=chunksize):
            if res:
                all_results.extend(res)

    return all_results


def _fetch_one_repos(repo):
    """Download, extract, and list files for one repo."""
    owner, name = repo["full_name"].split("/")
    repo_branch = repo["default_branch"]
    repo_url = repo["html_url"]

    print(f"\n[FETCH] {repo['full_name']} (branch={repo_branch})")
    local_repo_path = download_and_extract_repo(
        owner, name, HEADERS, repo_branch, DATA_DIR_RAW
    )
    print(f"[OK] Extracted -> {local_repo_path}")

    files = get_repo_files(
        owner,
        name,
        HEADERS,
        repo_branch,
        extension=FILE_EXTENSION,
        max_files=MAX_FILE_PER_REPO,
    )
    print(f"[OK] {repo['full_name']}: found {len(files)} {FILE_EXTENSION} files")
    time.sleep(1)
    return repo["full_name"], str(local_repo_path), files, repo_url


if __name__ == "__main__":
    load_dotenv()
    TOKEN = os.getenv("GITHUB_TOKEN")
    if not TOKEN:
        raise SystemExit("[ERROR] Please set GITHUB_TOKEN in your environment.")

    HEADERS = {
        "Authorization": f"Bearer {TOKEN}",
        "Accept": "application/vnd.github+json",
    }

    DATA_DIR_RAW = Path("/home/thaiv7/Desktop/WM/AI-and-SE/assignment_1/dataset/raw")

    N_REPOS = 500
    MIN_STAR_REPO = 50
    MAX_FILE_PER_REPO = 2_000
    ALLOWED_LICENSES = {"mit", "apache-2.0", "bsd-2-clause", "bsd-3-clause"}
    FILE_EXTENSION = ".py"
    NUM_THREADS = 6

    # 1. Search for Python repos
    repos = search_python_repos(
        n=N_REPOS,
        min_star=MIN_STAR_REPO,
        header=HEADERS,
        allowed_licenses=ALLOWED_LICENSES,
    )

    print(f"Found {len(repos)} Python repos.")

    # 2. Extract methods from each repo
    all_results = []

    # --- Stage 1: download repos in parallel ---
    list_downloaded_repos = []
    with ThreadPoolExecutor(max_workers=NUM_THREADS) as ex:
        future_map = {ex.submit(_fetch_one_repos, repo): repo for repo in repos}
        for fut in as_completed(future_map):
            repo = future_map[fut]
            try:
                list_downloaded_repos.append(fut.result())
            except Exception as e:
                print(
                    f"[ERROR] Fetch failed for {repo.get('full_name','<unknown>')}: {e}"
                )

    # --- Stage 2: parse per repo in parallel ---
    for idx, (repo_full_name, local_repo_path, files, repo_url) in enumerate(
        list_downloaded_repos
    ):
        if (idx + 1) % 10 == 0:
            df = pd.DataFrame(all_results)
            df = deduplicate_methods(df)
            df = df.reset_index(drop=True)

            output_raw_csv = os.path.join(DATA_DIR_RAW, f"python_methods_raw_{idx}.csv")

            df.to_csv(output_raw_csv, index=False)
            print(f"[SAVE] Dataset saved to {output_raw_csv} at {len(df)} samples")
            all_results = []

        try:
            if not files:
                continue

            res = extract_python_method_parallel(
                files,
                local_repo_path,
                repo_full_name,
                repo_url,
                max_workers=NUM_THREADS,
                chunksize=32,
            )
            print(f"[DONE] {repo_full_name}: extracted {len(res)} methods")
            all_results.extend(res)

            time.sleep(1)

        except Exception as e:
            print(f"[ERROR] Parse failed for {repo_full_name}: {e}")

    # 3. Save to CSV
    df = pd.DataFrame(all_results)
    df = deduplicate_methods(df)
    df = df.reset_index(drop=True)

    output_raw_csv = os.path.join(DATA_DIR_RAW, f"python_methods_raw_final.csv")
    df.to_csv(output_raw_csv, index=False)
    print("=================================")
    print(f"[SAVE] Dataset saved to {output_raw_csv} at {len(df)} samples")
