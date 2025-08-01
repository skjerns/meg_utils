#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Helper to keep all meg_utils submodules in sync.

- Searches ~ for all folders with 'nextcloud' in the name (ignore case)
- Recursively finds 'meg_utils' repos under each detected Nextcloud directory
- Runs 'git pull' in each repo and prints a compact summary

Created on Fri Jun 27 08:58:58 2025
Author: simon
"""

import os
import subprocess
import textwrap


def find_nextcloud_dirs(home_dir):
    """Return absolute paths to directories in home_dir whose names contain 'nextcloud' (case-insensitive)."""
    candidates = []
    try:
        for name in os.listdir(home_dir):
            path = os.path.join(home_dir, name)
            if os.path.isdir(path) and "nextcloud" in name.lower():
                candidates.append(path)
    except FileNotFoundError:
        pass
    return candidates


def find_git_dirs(base_dir, target_folder):
    """Walk base_dir to find target_folder directories that are git repos (contain a .git)."""
    matches = []
    for root, dirs, files in os.walk(base_dir):
        if target_folder in dirs:
            full_path = os.path.join(root, target_folder)
            if os.path.exists(os.path.join(full_path, ".git")):
                matches.append(full_path)
    return matches


def pull_git_repo(repo_path):
    """Run 'git pull' in repo_path and return (success, output_or_error)."""
    try:
        result = subprocess.run(
            ["git", "-C", repo_path, "pull"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        return True, result.stdout.strip()
    except subprocess.CalledProcessError as e:
        return False, e.stderr.strip()


def summarize_git_pull_output_line(output):
    """Parse 'git pull' output to a compact single-line summary."""
    lines = output.strip().split('\n')
    summary = {
        "commit_range": lines[0] if lines else "No update info",
        "files_changed": 0,
        "insertions": 0,
        "deletions": 0,
        "created": [],
        "modified": [],
    }

    for line in lines:
        if "file changed" in line:
            parts = line.split(',')
            for part in parts:
                part = part.strip()
                if "file changed" in part or "files changed" in part:
                    try:
                        summary["files_changed"] = int(part.split()[0])
                    except ValueError:
                        pass
                elif "insertion" in part:
                    try:
                        summary["insertions"] = int(part.split()[0])
                    except ValueError:
                        pass
                elif "deletion" in part:
                    try:
                        summary["deletions"] = int(part.split()[0])
                    except ValueError:
                        pass
        elif "create mode" in line:
            summary["created"].append(line.split()[-1])
        elif "|" in line:
            summary["modified"].append(line.split("|")[0].strip())

    created = f"created: {', '.join(summary['created'])}" if summary["created"] else ""
    modified = f"modified: {', '.join(summary['modified'])}" if summary["modified"] else ""
    parts = [
        f"{summary['files_changed']} files",
        f"{summary['insertions']} inserts",
        f"{summary['deletions']} dels",
        created,
        modified
    ]
    return " | ".join(part for part in parts if part)


if __name__ == "__main__":
    home_dir = os.path.expanduser("~")
    target = "meg_utils"

    print("Scanning home directory for Nextcloud locations...\n")
    nextcloud_dirs = find_nextcloud_dirs(home_dir)

    if not nextcloud_dirs:
        raise Exception("No Nextcloud-like directories found in ~ (looked for names containing 'nextcloud').")

    for d in nextcloud_dirs:
        print(f"Found Nextcloud directory: {d}")
    print()

    # Search all detected Nextcloud directories for target repos
    repos_set = set()
    for base in nextcloud_dirs:
        for repo in find_git_dirs(base, target):
            repos_set.add(os.path.abspath(repo))

    repos = sorted(repos_set)
    if not repos:
        raise Exception(f"No '{target}' git repositories found under detected Nextcloud directories.")

    print(f"Searching for '{target}' repositories under detected Nextcloud directories...\n")
    for repo in repos:
        print(f"Found {repo}")
    input('\nContinue? [enter]')

    print()
    for repo in repos:
        rel = repo.replace(home_dir, "~")
        print(f"git pull ..{rel} -> ", end=' ')
        success, output = pull_git_repo(repo)
        if success:
            if "Already up to date" in output or "Already up-to-date" in output:
                print("ok\n")
            else:
                summary = textwrap.indent(summarize_git_pull_output_line(output), '   ')
                print(f"ok\n{summary}\n")
        else:
            print(f"ERROR\n{textwrap.indent(output, '    | ')}\n")
