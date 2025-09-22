"""
In this file, I will crawl Java methods from GitHub repositories using GitHub's API.
"""

import requests, os, zipfile, shutil
from pathlib import Path
import javalang
import pandas as pd
from dotenv import load_dotenv
import time
import re

def search_java_repos(n=5, min_star=50, allowed_licenses=None):
    """
    Search Java repos by stars with pagination support.
    Filters by license if allowed_licenses is provided.
    """
    url = "https://api.github.com/search/repositories"
    repos, page = [], 1
    while len(repos) < n:
        params = {
            "q": f"language:Java stars:>{min_star}",
            "sort": "stars",
            "order": "desc",
            "per_page": 100,
            "page": page,
        }
        r = requests.get(url, headers=HEADERS, params=params, timeout=30)
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



def get_repo_files(owner, repo, branch="main", extension=".java"):
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
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    tree = r.json().get("tree", [])
    return [item["path"] for item in tree if item["type"] == "blob" and item["path"].endswith(extension)]


def get_last_commit(owner, repo, filepath, branch):
    """
    Get the last commit SHA for a given file.   
    
    Since each file can have multiple method, it is hard to track commit per method. 
    We assume the last commit of the file is the commit for all methods in that file.

    * Arguments:
        - owner: repo owner
        - repo: repo name
        - filepath: path to the file in the repo
    * Returns:
        - commit SHA (string) or None if not found
    """
    url = f"https://api.github.com/repos/{owner}/{repo}/commits"
    params = {"path": filepath, "sha": branch, "per_page": 1}  # pin to branch
    r = requests.get(url, headers=HEADERS, params=params, timeout=30)
    r.raise_for_status()
    commits = r.json()
    return commits[0]["sha"] if commits else None


def download_and_extract_repo(owner, repo, branch, dest):
    """Download repo as zipball and extract it locally."""
    dest.mkdir(parents=True, exist_ok=True)
    zip_path = dest / f"{owner}-{repo}-{branch}.zip"
    extract_dir = dest / f"{owner}-{repo}-{branch}"

    if extract_dir.exists():
        shutil.rmtree(extract_dir)

    url = f"https://api.github.com/repos/{owner}/{repo}/zipball/{branch}"
    r = requests.get(url, headers=HEADERS, stream=True, timeout=60)
    r.raise_for_status()

    with open(zip_path, "wb") as f:
        for chunk in r.iter_content(1024 * 256):
            f.write(chunk)

    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(extract_dir)

    # GitHub zip includes a single top-level folder, return that
    subfolders = list(extract_dir.iterdir())
    return subfolders[0] if subfolders else extract_dir

def _strip_java_comments_for_check(src: str) -> str:
    """Using regex to remove comments for validation checks."""
    return re.sub(r'//.*?$|/\*.*?\*/', '', src, flags=re.M|re.S)


def filter_invalid_methods(methods, min_lines=3, max_lines=100):
    """
    Keep comments in code for the dataset, but drop methods that:
    - Have no non-comment code
    - Are shorter than min_lines or longer than max_lines (after stripping comments)
    """
    cleaned = []
    for m in methods:
        # remove comments only for the check
        check_code = _strip_java_comments_for_check(m["original_code"])
        code_lines = [l for l in check_code.splitlines() if l.strip()]

        if len(code_lines) == 0:
            continue  # only comments, no executable code

        if len(code_lines) < min_lines or len(code_lines) > max_lines:
            continue

        cleaned.append(m)
    return cleaned


def tokenize_code(text):
    """
    Function tokenize Java code into variable ([A-Za-z_][A-Za-z0-9_]*), numbers (\d+),
                        double-quoted strings (".*?"), single-quoted strings ('.*?'), multi-char operators (|==|!=|<=|>=|&&|\|\|)
                        any single non-whitespace character ([^\s])
    """
    return re.findall(r'[A-Za-z_][A-Za-z0-9_]*|\d+|".*?"|\'.*?\'|==|!=|<=|>=|&&|\|\||[^\s]', text, flags=re.S)



def _brace_matched_end(lines, start_line):
    """
    Find the ending line of a method by matching braces.

    Args:
        lines (list[str]): The source code split into lines.
        start_line (int): 1-based line number where the method starts.

    Returns:
        int | None: The line number where the method ends.
    """

    i = start_line - 1
    # Find first '{' after the method declaration line(s)
    open_i = i
    while open_i < len(lines) and '{' not in lines[open_i]:
        open_i += 1
    if open_i >= len(lines): 
        return None
    depth = 0
    for j in range(open_i, len(lines)):
        depth += lines[j].count('{')
        depth -= lines[j].count('}')
        if depth == 0:
            return j + 1  # 1-based end line inclusive
    return None


def extract_methods_from_file(filepath):
    """Extract methods from a Java file using javalang."""
    try:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            code = f.read()
        tree = javalang.parse.parse(code)
    except Exception:
        return []

    results = []
    lines = code.splitlines()
    for _, node in tree.filter(javalang.tree.MethodDeclaration):
        method_name = node.name
        start_line = node.position.line if node.position else None
        end_line = _brace_matched_end(lines, start_line) if start_line else None
        if not start_line or not end_line:
            continue
        signature = f"{' '.join(node.modifiers)} {node.return_type} {method_name}({', '.join(str(p.type) for p in node.parameters)})"
        original_code = "\n".join(lines[start_line-1:end_line]) if start_line and end_line else ""
        # code_tokens = original_code.split()  # simple whitespace tokenization
        code_tokens = tokenize_code(original_code)
        results.append({
            "method_name": method_name,
            "start_line": start_line,
            "end_line": end_line,
            "signature": signature,
            "original_code": original_code,
            "code_tokens": code_tokens,
        })
    return results

def deduplicate_methods(df):
    """Remove duplicate methods from the dataset."""
    return df.drop_duplicates(
        subset=["repo_name", "file_path", "method_name", "original_code"]
    )

def build_dataset(n_repos=2, min_star=500, max_files=50, output_csv="java_functions_dataset.csv", allowed_licenses=None, max_num_samples=30_000):
    """
    The main function to build the dataset.
    * Arguments:
        - n_repos: number of repositories to crawl
        - min_star: minimum stars to filter repositories
        - max_files: maximum number of files to process per repository
        - output_csv: path to save the output CSV file
    * Returns:
        - pandas DataFrame containing the dataset
    """

    all_results = []
    repos = search_java_repos(n_repos, min_star, allowed_licenses)

    for repo in repos:
        if len(all_results) >= max_num_samples:
            break

        try:
            owner, name = repo["full_name"].split("/")
            branch = repo["default_branch"]
            repo_url = repo["html_url"]

            print(f"\nProcessing repo: {repo['full_name']} (branch={branch})")

            files = get_repo_files(owner, name, branch)
            print(f"   Found {len(files)} .java files")

            local_repo = download_and_extract_repo(owner, name, branch, DATA_DIR)

            for f in files[:max_files]:  # avoid crawling too many files
                file_path = local_repo / f
                if not file_path.exists():
                    continue

                methods = extract_methods_from_file(file_path)
                methods = filter_invalid_methods(methods)

                commit_sha = get_last_commit(owner, name, f, branch)
                if not commit_sha:
                    continue  # skip samples without a concrete SHA

                if not methods:
                    continue

                for m in methods:
                    all_results.append({
                        "repo_name": repo["full_name"],
                        "repo_url": repo_url,
                        "repo_license": repo["license"]["spdx_id"] if repo.get("license") else "NO-LICENSE",
                        "commit_sha": commit_sha,
                        "file_path": f,
                        **m
                    })
        except Exception as e:
            print(f"   [Error] Skipping repo {repo['full_name']}: {e}")
            continue
        
        time.sleep(1)  # to respect rate limits


    df = pd.DataFrame(all_results)
    df = deduplicate_methods(df)    
    df = df.reset_index(drop=True)

    df.to_csv(output_csv, index=False)
    print("=================================")
    print(f"[DONE] Dataset saved to {output_csv}, total samples: {len(df)}")
    return df


if __name__ == "__main__":
    load_dotenv()
    TOKEN = os.getenv("GITHUB_TOKEN")
    if not TOKEN:
        raise SystemExit("❌ Please set GITHUB_TOKEN in your environment.")

    HEADERS = {"Authorization": f"Bearer {TOKEN}", "Accept": "application/vnd.github+json"}
    ALLOWED_LICENSES = {"mit", "apache-2.0", "bsd-2-clause", "bsd-3-clause"}


    DATA_DIR = Path("java_repos") 

    NUM_REPOS = 20
    MIN_STAR = 50    
    MAX_FILES = 1_000
    OUTPUT_CSV = "java_functions_dataset.csv"
    MAX_NUM_SAMPLES = 30_000

    df = build_dataset(NUM_REPOS, MIN_STAR, MAX_FILES, OUTPUT_CSV, ALLOWED_LICENSES, MAX_NUM_SAMPLES)